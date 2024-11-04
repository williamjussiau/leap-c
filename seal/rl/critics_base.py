import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from seal.torch_utils import MLP, string_to_activation


class QNet(nn.Module):
    """Base class for Q Networks."""

    optimizer: Optimizer

    def __init__(self, tau: float):
        """
        Parameters:
            tau: The soft update factor.
        """
        super(QNet, self).__init__()
        self.tau = tau
        # TODO Logging of the loss
        self.loss = None

    def train_net(
        self,
        target: torch.Tensor,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """
        Trains the network with the given mini_batch and target, using a smooth L1 loss.

        Parameters:
            target: The target to train against.
            mini_batch: The mini_batch containing the states, actions, rewards, next_states and dones.
        """
        s, a, r, s_prime, done = mini_batch
        pred = self.forward(s, a)
        loss = F.smooth_l1_loss(pred, target)
        self.optimizer.zero_grad()
        mean_loss = loss.mean()
        mean_loss.backward()
        self.optimizer.step()

    def soft_update(self, net_target: nn.Module):
        """Update the target network parameters with the current network parameters using a soft update."""
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )


class QNetMLP(QNet):
    """Q Network which is basically just an embedding for state and action each, followed by an MLP."""

    def __init__(
        self,
        learning_rate: float,
        tau: float,
        s_dim: int,
        a_dim: int,
        q_embed_size: int,
        hidden_dims: "list[int]",
        activation: str,
        **kwargs,
    ):
        """
        Parameters:
            learning_rate: The learning rate of the optimizer.
            tau: The soft update factor.
            s_dim: The dimension of the state space.
            a_dim: The dimension of the action space.
            q_embed_size: The state and the action will both be embedded into this dimension (using a Linear layer, respectively).
            hidden_dims: The dimensions of the hidden layers of the neural network. The first layer always has to be twice the size of the q_embed_size.
            activation: The name of the activation function of the neural network.
        """
        super(QNetMLP, self).__init__(tau=tau)

        if not q_embed_size * 2 == hidden_dims[0]:
            raise ValueError(
                "The first hidden layer must be twice the size of the q_embed_size."
            )
        self.fc_s = nn.Linear(s_dim, q_embed_size)
        self.fc_a = nn.Linear(a_dim, q_embed_size)
        self.fc_cat = MLP(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims[1:],
            activation=activation,
        )
        self.activation = string_to_activation(activation)

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h1 = self.activation(self.fc_s(x))
        h2 = self.activation(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = self.activation(self.fc_cat(cat))
        return q
