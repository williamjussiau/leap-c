import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
