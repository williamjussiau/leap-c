import numpy as np
import torch
import torch.nn as nn


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


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron with orthogonal weight initialization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        fcs = []
        for prev_d, d in zip(dims[:-1], dims[1:]):
            fcs.extend([nn.Linear(prev_d, d), string_to_activation(activation)])
        self.fcs = nn.Sequential(*fcs[:-1])
        self.output_dim = output_dim

        self.apply(weight_init)

    def forward(self, ob):
        output = self.fcs(ob)
        return output


def weight_init(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.orthogonal_(tensor.weight.data)
        tensor.bias.data.fill_(0.0)


class ParamsWithMLP(nn.Module):
    """A module containing an input-independent Parameter tensor and an MLP that takes an input and outputs a tensor."""

    def __init__(
        self,
        init_params: np.ndarray,
        mlp_input_dim: int,
        mlp_output_dim: int,
        mlp_hidden_dims: list[int],
        mlp_activation: str,
    ):
        super().__init__()
        self.params = nn.parameter.Parameter(torch.from_numpy(init_params))
        self.std_network = MLP(
            mlp_input_dim, mlp_output_dim, mlp_hidden_dims, mlp_activation
        )

    def forward(self, ob):
        if len(ob.shape) > 1:
            params_enhanced = self.params.unsqueeze(0).expand(ob.shape[0], -1).clone()
        else:
            params_enhanced = self.params
        return params_enhanced, self.std_network(ob)
