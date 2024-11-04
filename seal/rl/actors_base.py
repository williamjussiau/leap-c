from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from seal.mpc import MPC
from seal.nn.modules import MPCSolutionModule
from seal.torch_utils import MLP


class Actor(ABC):
    """Interface for all actors."""

    def __init__(self):
        super(Actor, self).__init__()

    @abstractmethod
    def forward(self, state: torch.Tensor, deterministic: bool):
        """Run a forward pass based on the state. If deterministic is True, the same state input should always produce the same output."""
        raise NotImplementedError()

    @abstractmethod
    def act(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Return the action based on the state. If deterministic is True, the same state input should always produce the same output."""
        raise NotImplementedError()

    @abstractmethod
    def train(
        self,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """
        Implements one training step.
        """
        raise NotImplementedError()


class StochasticMLP(nn.Module):
    """A torch module that uses an MLP to predict the mean and standard deviation of the action distribution.
    The action is sampled from this distribution and then squashed with a tanh function.
    """

    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_dims: list[int],
        activation: str,
        minimal_std: float = 1e-3,
        **kwargs,
    ):
        """
        Parameters:
            s_dim: The dimension of the state space.
            a_dim: The dimension of the action space.
            hidden_dims: The dimensions of the hidden layers of the neural network.
            activation: The name of the activation function of the neural network.
            minimal_std: The minimal standard deviation of the action distribution.
        """
        super().__init__(**kwargs)
        self.minimal_std = minimal_std
        self.mlp = MLP(
            input_dim=s_dim,
            output_dim=2 * a_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.a_dim = a_dim

        # TODO proper Logging
        self.latest_mu = None
        self.latest_std = None

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the action and the log probability of the action.
        """
        net_out = self.mlp(x)
        mu, std = torch.split(net_out, split_size_or_sections=self.a_dim, dim=-1)
        std = F.softplus(std) + self.minimal_std
        self.latest_std = std
        self.latest_mu = mu

        if deterministic:
            action = mu
            log_prob = 1
        else:
            dist = Normal(mu, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action)

        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        return real_action, real_log_prob


class MLPWithMPCHeadWithActionNoise(nn.Module):
    """A stochastic MLP that uses a neural network to predict parameters of the MPC solver.
    The MPC solver is then used to calculate the action which is used as mean in a tanh normal distribution,
    together with a standard deviation from the neural network.
    This actor is trained by differentiating through the MPC (by using sensitivities),
    hence treating the MPC as kind of "layer".
    """

    def __init__(
        self,
        mpc: MPC,
        s_dim: int,
        param_dim: int,
        hidden_dims: list[int],
        activation: str,
        param_factor: np.ndarray,
        param_shift: np.ndarray,
        minimal_std: float = 1e-3,
    ):
        """
        Parameters:
            mpc: The MPC which is used for solving the optimal control problem.
            s_dim: The dimension of the state space.
            param_dim: The dimension of the parameter space (the parameters that are inserted into the MPC).
            hidden_dims: The dimensions of the hidden layers of the neural network.
            activation: The name of the activation function of the neural network.
            minimal_std: The minimal standard deviation of the action distribution.
        """
        super().__init__()
        self.minimal_std = minimal_std
        self.activation = activation
        self.param_model = MLP(
            input_dim=s_dim,
            output_dim=2 * param_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.mpc_layer = MPCSolutionModule(mpc)

        if (
            not param_factor.shape == param_shift.shape
            and param_shift.shape[-1] == param_dim
        ):
            raise ValueError(
                "param_scaling and param_shift must have the same shape and match the parameter dimensions."
            )

        self.param_factor = torch.tensor(param_factor)
        self.param_shift = torch.tensor(param_shift)

        self.recent_acados_status = None
        self.dont_do_gradient_update = False
        self.missed_gradient_updates = 0
        self.valid_samples = 0

        # for logging #TODO How exactly to handle logging
        # Which part to do here, which part in the upper class?
        # Maybe move it into an easy logger class?
        self.recent_params = None
        self.recent_mu = None
        self.recent_std = None
        self.recent_log_prob = None
        self.sum_gradients = 0
        self.n_gradient_updates = 0
        self.sum_stds = 0
        self.sensitivities = None

    def forward(
        self,
        x: torch.Tensor,
        param_transform_kwargs: dict,
        initialization: list[dict[str, np.ndarray]] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the action, the log probability of the action and the status of the MPC.
        """
        params_unscaled, std = torch.split(
            self.param_model(x), split_size_or_sections=2, dim=-1
        )
        std = F.softplus(std) + self.minimal_std
        self.recent_std = std

        params_unscaled = F.tanh(params_unscaled)
        params_scaled = self.final_param_transform_before_mpc(
            params_unscaled, x, **param_transform_kwargs
        )
        self.recent_params = params_scaled

        # TODO: Figure out unlearned parameters.
        # Will introduce this properly with state dataclasses. Atm it can be done via the final_param_transform_before_mpc.
        mu, self.recent_acados_status = self.mpc_layer(
            x, p=params_scaled, initializations=initialization
        )
        self.recent_mu = mu

        if deterministic:
            action = mu
            log_prob = 1
        else:
            dist = Normal(mu, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action)

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
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        return real_action, real_log_prob, self.recent_acados_status

    def final_param_transform_before_mpc(
        self, params_unscaled: torch.Tensor, state: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """The final transformation of the parameters before they are used in the MPC,
        happening directly after the tanh transformation, i.e., the incoming parameters are assumed to be between [-1, 1].
        This can be overridden in subclasses to fit your needs, but it is expected to be differentiable (meaning torch.autograd)!

        Parameters:
            params: The parameters from the neural network.
            state: The state of the system, e.g., needed for using deltas (param = x + delta) instead of explicitly predicting the parameters.
        """
        return params_unscaled * self.param_factor + self.param_shift
