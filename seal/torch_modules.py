import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from seal.mpc import MPC, MPCParameter, MPCState
from seal.nn.modules import MPCSolutionModule


def string_to_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError("Activation function not recognized.")


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    Taken (slightly modified) from gfox in https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/5 .
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str,
    use_activation_on_output: bool,
) -> nn.Sequential:
    """Returns a simple Multi-Layer Perceptron with orthogonal weight initialization.

    Parameters:
        input_dim: The dimension of the input.
        output_dim: The dimension of the output.
        hidden_dims: The dimensions of the hidden layers.
        activation: The type of activation function used in the hidden layers.
        use_activation_on_output: Whether to use an activation function on the output layer.
    """

    dims = [input_dim] + list(hidden_dims) + [output_dim]

    fcs = []
    for prev_d, d in zip(dims[:-1], dims[1:]):
        fcs.extend([nn.Linear(prev_d, d), string_to_activation(activation)])
    if use_activation_on_output:
        mlp = nn.Sequential(*fcs)
    else:
        mlp = nn.Sequential(*fcs[:-1])

    mlp.apply(weight_init)
    return mlp


def weight_init(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.orthogonal_(tensor.weight.data)
        tensor.bias.data.fill_(0.0)


class TanhNormal(nn.Module):
    """A torch module that takes the mean and standard deviation as input and builds a Normal distribution.
    The output is sampled from this distribution and then squashed with a tanh function.
    """

    def __init__(
        self,
        minimal_std: float = 1e-3,
        **kwargs,
    ):
        """
        Parameters:
            minimal_std: The minimal standard deviation of the distribution.
                Will always be added to the softplus of the std to form the actual std.
        """
        super().__init__(**kwargs)
        self.minimal_std = minimal_std

    def forward(
        self, mean: torch.Tensor, std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            an output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        stats = dict()
        std = F.softplus(std) + self.minimal_std
        stats["std"] = std

        if deterministic:
            action = mean
            log_prob = 1
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action)

        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        return real_action, real_log_prob


class TanhNormalNetwork(nn.Module):
    """Uses a Network to predict the mean and standard deviation of a TanhNormal distribution."""

    def __init__(
        self,
        mean_std_module: nn.Module,
        minimal_std: float = 1e-3,
    ):
        super().__init__()
        self.mean_std_module = mean_std_module
        self.tanh_normal = TanhNormal(minimal_std=minimal_std)

    def forward(
        self, obs: tuple[torch.Tensor, MPCParameter], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            obs: The input to the mean/std model.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            an output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        mean, std = self.mean_std_module(obs)
        return self.tanh_normal(mean, std, deterministic=deterministic)


class MeanStdMLP(nn.Module):
    """An MLP with a little syntactic sugar (being the splitting of the output in the correct dimensions),
    ment to predict the mean and standard deviation of a distribution."""

    def __init__(
        self,
        s_dim: int,
        mean_dim: int,
        std_dim: int,
        hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        self.mean_dim = mean_dim
        self.std_dim = std_dim
        self.mlp = create_mlp(s_dim, mean_dim + std_dim, hidden_dims, activation, False)

    def forward(
        self, obs: tuple[torch.Tensor, MPCParameter]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NOTE: Ignores MPCParameter"""
        return torch.split(self.mlp(obs[0]), [self.mean_dim, self.std_dim], dim=-1)  # type:ignore


class FOUMPCNetwork(nn.Module):
    """
    Uses a neural network to predict PARAMETERS of the MPC solver and a standard deviation for the ACTION(!) of the MPC.
    The MPC solver is then used to calculate the action which is used as mean in a tanh normal distribution,
    together with the standard deviation from the neural network.
    This actor is trained by differentiating through the MPC (by using sensitivities),
    hence treating the MPC as kind of "layer".
    NOTE: All learnable parameter must be global for this architecture to work, such that we can calculate the sensitivities for them.
    Also see the MPCSolutionModule documentation.
    """

    def __init__(
        self,
        mpc: MPC,
        param_mean_action_std_model: nn.Module,
        param_factor: np.ndarray,
        param_shift: np.ndarray,
        minimal_std: float = 1e-3,
    ):
        """
        Parameters:
            mpc: The MPC to be used for solving the OCP.
            param_mean_action_std_model: A model that predicts the parameters for the MPC and a standard deviation for the actions of the MPC.
                Possibly also outputs a stats dict (see forward).
            param_factor: A factor to scale the parameters, before putting them into the MPC.
            param_shift: A shift to add to the scaled parameters, before putting them into the MPC.
            minimal_std: The minimal standard deviation of the action distribution.
        """
        super().__init__()
        self.param_mean_action_std_model = param_mean_action_std_model

        self.mpc_layer = MPCSolutionModule(mpc)
        self.tanh_normal = TanhNormal(minimal_std=minimal_std)

        if not param_factor.shape == param_shift.shape:
            raise ValueError("param_scaling and param_shift must have the same shape.")

        self.param_factor = nn.Parameter(
            torch.tensor(param_factor), requires_grad=False
        )
        self.param_shift = nn.Parameter(torch.tensor(param_shift), requires_grad=False)

    def forward(
        self,
        obs: tuple[torch.Tensor, MPCParameter],
        param_transform_kwargs: dict | None = None,
        mpc_initialization: list[MPCState] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Parameters:
            obs: A tuple containing the input to the mean/std model and the stagewise Parameters to set in the MPC before solving.
                NOTE: Despite being of the class MPCParameter, if p_global is not None,
                it will be overridden, because p_global will be set by the prediction from the network.
            param_transform_kwargs: The keyword arguments for the final_param_transform_before_mpc (useful if overwritten).
            initialization: The initialization for the MPC. If not None, it should be a list of length batch_size.
            deterministic: If True, the output will just be tanh(MPC(mean)), no sampling is taking place.

        Returns the action, the log probability of the action, the status of the MPC and some statistics.
        """
        x, param = obs
        p_stagewise = MPCParameter(
            p_stagewise=param.p_stagewise.astype(np.float64)  # type:ignore
            if param.p_stagewise is not None
            else None,
            p_stagewise_sparse_idx=param.p_stagewise_sparse_idx,
        )
        net_out = self.param_mean_action_std_model(obs)
        if isinstance(net_out[-1], dict):
            stats = net_out[-1]
            net_out = net_out[:-1]
        else:
            stats = dict()
        params_unscaled, action_std = net_out

        param_transform_kwargs = (
            dict() if param_transform_kwargs is None else param_transform_kwargs
        )
        params_scaled = self.final_param_transform_before_mpc(
            params_unscaled, x, **param_transform_kwargs
        )
        stats["params_input_to_mpc"] = params_scaled

        action_mean, _, mpc_status = self.mpc_layer(
            x,
            p_global=params_scaled,
            p_stagewise=p_stagewise,
            initializations=mpc_initialization,
        )
        stats["mean_from_mpc"] = action_mean
        stats["std_from_net"] = action_std

        real_action, real_log_prob = self.tanh_normal(
            action_mean, action_std, deterministic=deterministic
        )
        # NOTE: One could use a TruncatedNormal instead of a TanhNormal to retain the constraint satisfaction of the MPC at test-time (deterministic mode),
        # but this results in much more training needed, probably due to bad gradients?
        # We did not check if this is also a problem if we just use a "differentiable clamp" in the manner described below to clamp the Normal distribution.
        # This could be another way to retain the constraint satisfaction of the MPC at test-time.
        # Scale the log prob for the clamp
        # if action < -Fmax:
        #     real_log_prob = torch.log(dist.cdf(torch.Tensor(-Fmax, device=self.device)))
        # elif action > Fmax:
        #     real_log_prob = torch.log(1 - dist.cdf(torch.Tensor(Fmax, device=self.device)))
        # else:
        #     real_log_prob = log_prob
        # real_action = dclamp(action, -self.main_mpc.Fmax, self.main_mpc.Fmax) / Fmax # Scale it to [-1, 1]

        # return_ac = real_action / Fmax  # Scale it to [-1, 1]

        return real_action, real_log_prob, mpc_status, stats

    def final_param_transform_before_mpc(
        self, params_unscaled: torch.Tensor, state: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """The final transformation of the parameters before they are used in the MPC.
        Amounts to a tanh transformation to squash the parameters between [-1, 1] followed by a linear transformation.
        This can be overridden in subclasses to fit your needs, but it is expected to be differentiable (meaning torch.autograd)!

        Parameters:
            params: The parameters from the neural network.
            state: The state of the system, e.g., needed for using deltas (param = x + delta) instead of explicitly predicting the parameters.
        """
        return F.tanh(params_unscaled) * self.param_factor + self.param_shift
