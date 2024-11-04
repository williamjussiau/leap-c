from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from seal.mpc import MPC
from seal.nn.modules import CleanseAndReducePerSampleLoss
from seal.rl.actors_base import Actor, MLPWithMPCHeadWithActionNoise, StochasticMLP
from seal.rl.critics_base import QNet
from seal.rl.replay_buffer import ReplayBuffer
from seal.rl.trainer_base import Trainer


class SACActor(nn.Module, Actor):
    """An interface to adhere to for defining the training of a SAC Actor.
    NOTE: This is thought of to be used in conjunction with an actual module (This class does not define a module, despite inheriting from nn.Module).
    """

    log_alpha: torch.Tensor
    optimizer: Optimizer
    log_alpha_optimizer: Optimizer

    def __init__(self, target_entropy: float, **kwargs):
        super().__init__(**kwargs)
        self.target_entropy = target_entropy

    @abstractmethod
    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the action and the log_prob of the action.

        Parameters:
        state: The state for which the action should be determined.
        deterministic: If True, the action is drawn deterministically.
        """
        raise NotImplementedError()

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return the action based on the state.

        Parameters:
        state: The state for which the action should be determined.
        deterministic: If True, the action is drawn deterministically.
        """
        return self.forward(state, deterministic=deterministic)[0]

    def train(
        self,
        q1: nn.Module,
        q2: nn.Module,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)

        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = (-min_q - entropy).mean()  # for gradient ascent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(
            self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class SACActorWithCleansingNonconvergence(nn.Module, Actor):
    """An interface to adhere to for defining the training of a SAC Actor,
    but with the possibility to cleanse the loss from nonconvergences introduced by the usage of an MPC.
    NOTE: This is thought of to be used in conjunction with an actual module (This class does not define a module, despite inheriting from nn.Module).
    """

    log_alpha: torch.Tensor
    optimizer: Optimizer
    log_alpha_optimizer: Optimizer

    def __init__(
        self,
        accept_n_nonconvergences_per_batch: int,
        target_entropy: float,
        num_batch_dimensions: int = 1,
    ):
        """
        Parameters:
        accept_n_nonconvergences_per_batch: If this number of nonconvergences is reached in a batch, no training will be performed with it. If the number is not reached, the nonconvergend samples will still be removed from the batch before training.
        target_entropy: The target entropy for the automatic entropy bonus scaling (here referred to as alpha).
        num_batch_dimensions: The number of batch dimensions in the input data.
        """
        self.cleanse_loss = CleanseAndReducePerSampleLoss(
            reduction="mean",
            num_batch_dimensions=num_batch_dimensions,
            n_nonconvergences_allowed=accept_n_nonconvergences_per_batch,
        )

        self.accept_n_nonconvergences_per_batch = accept_n_nonconvergences_per_batch
        self.target_entropy = target_entropy

        # TODO: Things that may be logworthy
        self.entropy = None
        self.loss = None
        self.alpha_loss = None
        self.log_prob = None
        self.sum_stds = 0
        self.sum_gradients = 0
        self.n_gradient_updates = 0
        self.valid_samples = 0
        self.missed_gradient_updates = 0

    @abstractmethod
    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the action, the log_prob of the action and the corresponding status of the MPC.

        Parameters:
        state: The state for which the action should be determined.
        deterministic: If True, the action is drawn deterministically.
        """
        raise NotImplementedError()

    def act(self, state, deterministic: bool = False) -> torch.Tensor:
        """Return the action based on the state.

        Parameters:
        state: The state for which the action should be determined.
        deterministic: If True, the action is drawn deterministically.
        """
        return self.forward(state, deterministic=deterministic)[0]

    def train(
        self,
        q1: nn.Module,
        q2: nn.Module,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> bool:
        s, _, _, _, _ = mini_batch
        a, log_prob, status = self.forward(s)  # type:ignore

        errors = status.to(dtype=torch.bool)
        if torch.sum(errors) > self.accept_n_nonconvergences_per_batch:
            # TODO FOR LOGGING WE WANT TO GIVE SOME INFO HERE
            return False
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        cleansed_loss = self.cleanse_loss(loss, status)

        self.optimizer.zero_grad()
        cleansed_loss.backward()
        self.optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach())
        cleansed_alpha_loss = self.cleanse_loss(alpha_loss, status)

        self.log_alpha_optimizer.zero_grad()
        cleansed_alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return True


class SACActorFOMPCWithActionNoise(SACActorWithCleansingNonconvergence):
    """An Actor for SAC which uses an MLP to predict the parameters of an MPC which produces the action,
    on top of which a TanhNormal distribution is built.
    Backpropagation is done through the MPC, hence using the MPC as kind of "layer".
    """

    def __init__(
        self,
        mpc: MPC,
        learning_rate: float,
        lr_alpha: float,
        init_alpha: float,
        target_entropy: float,
        hidden_dims: "list[int]",
        activation: str,
        s_dim: int,
        param_dim: int,
        param_factor: np.ndarray,
        param_shift: np.ndarray,
        accept_n_nonconvergences_per_batch: int,
        minimal_std: float = 1e-3,
    ):
        """
        Parameters:
        mpc: The MPC to use for the action prediction.
        learning_rate: The learning rate of the optimizer.
        lr_alpha: The learning rate of the optimizer for the entropy bonus scaling.
        init_alpha: The initial entropy bonus scaling.
        target_entropy: The target entropy for the entropy bonus scaling.
        hidden_dims: The number of nodes in the hidden layers of the MLP.
        activation: The name of the activation function of the MLP.
        s_dim: The dimension of the state space.
        param_dim: The dimension of the parameter space (the parameters that are inserted into the MPC).
        param_factor: The factor by which the parameters are scaled before being inserted to the MPC.
        param_shift: The shift by which the parameters are shifted after being scaled with param_factor and before being inserted to the MPC.
        accept_n_nonconvergences_per_batch: The number of nonconvergences that are allowed in a batch. If it is exceeded, no training is performed with it.
        minimal_std: The minimal standard deviation of the action distribution.
        """
        SACActorWithCleansingNonconvergence.__init__(
            self, accept_n_nonconvergences_per_batch, target_entropy
        )

        self.model = MLPWithMPCHeadWithActionNoise(
            mpc=mpc,
            s_dim=s_dim,
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            param_factor=param_factor,
            param_shift=param_shift,
            minimal_std=minimal_std,
        )

        self.log_alpha = torch.tensor(np.log(init_alpha))  # type:ignore
        self.log_alpha.requires_grad = True

        self.optimizer = Adam(self.param_model.parameters(), lr=learning_rate)

        self.log_alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(state, deterministic=deterministic)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(state, deterministic=deterministic)[0]


class SACActorMLP(SACActor):
    """An actor for SAC that uses a multi-layer perceptron to predict the mean and standard deviation of the action distribution.
    The action is sampled from this distribution and then squashed with a tanh function.
    """

    def __init__(
        self,
        learning_rate: float,
        lr_alpha: float,
        init_alpha: float,
        target_entropy: float,
        s_dim: int,
        a_dim: int,
        hidden_dims: list[int],
        activation: str,
        minimal_std: float = 1e-3,
        **kwargs,
    ):
        """
        Parameters:
            learning_rate: The learning rate of the optimizer.
            lr_alpha: The learning rate of the optimizer for the entropy bonus scaling.
            init_alpha: The initial entropy bonus scaling.
            target_entropy: The target entropy for the entropy bonus scaling.
            s_dim: The dimension of the state space.
            a_dim: The dimension of the action space.
            hidden_dims: The number of nodes of the hidden layers of the MLP.
            activation: The name of the activation function of the MLP.
            minimal_std: The minimal standard deviation of the action distribution.
        """
        super().__init__(target_entropy)

        self.model = StochasticMLP(
            s_dim=s_dim,
            a_dim=a_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            minimal_std=minimal_std,
        )

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.log_alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(state, deterministic=deterministic)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(state, deterministic=deterministic)[0]


class SACTrainer(Trainer):
    """A class to handle training of the different components in the SAC algorithm.
    NOTE: You might want to override the method normalize_transitions_before_training which is applied
    to a batch before training with it."""

    def __init__(
        self,
        actor: SACActor | SACActorWithCleansingNonconvergence,
        critic1: QNet,
        critic2: QNet,
        critic1_target: QNet,
        critic2_target: QNet,
        replay_buffer: ReplayBuffer,
        soft_update_frequency: int,
        gamma: float,
        batch_size: int,
        device: str,
    ):
        """
        Parameters:
            actor: The actor to train.
            critic1: The first critic to train.
            critic2: The second critic to train.
            critic1_target: The target network for the first critic.
            critic2_target: The target network for the second critic.
            replay_buffer: The replay buffer to sample from.
            soft_update_frequency: The target networks are updated every this many successfull training steps.
            gamma: The discount factor.
            batch_size: The size of the mini-batches.
            device: The device to run the training on.
        """
        super().__init__()
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_target = critic1_target
        self.critic2_target = critic2_target

        self.device = device
        self.actor = self.actor.to(self.device)
        self.critic1 = self.critic1.to(self.device)
        self.critic2 = self.critic2.to(self.device)
        self.critic1_target = self.critic1_target.to(self.device)
        self.critic2_target = self.critic2_target.to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.soft_update_frequency = soft_update_frequency
        self.soft_update_counter = 0

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Act based on the state. This is intended for rollouts (= interaction with the environment).
        Parameters:
            state: The state for which the action should be determined.
            deterministic: If True, the action is drawn deterministically.
        """
        added_batch_dim = False
        if len(state.shape) == 1:
            # Add batch dimension
            added_batch_dim = True
            state = state.unsqueeze(0)
        action = self.actor.act(state, deterministic=deterministic)
        if added_batch_dim:
            action = action.squeeze(0)
        return action

    def train(self):
        """
        Do a training step with the given mini_batch, updating the actor and the critics one time, respectively,
        also updating the target networks with a soft update every self.soft_update_frequency steps.

        NOTE: As it is implemented now, the Q updates are still done, even if the actor training fails
        (because we can still do them, since the gradient needs not be backpropagated through the q_target).
        """
        mini_batch = self.normalize_transitions_before_training(
            self.replay_buffer.sample(n=self.batch_size)
        )
        training_succeeded = self.actor.train(self.critic1, self.critic2, mini_batch)
        if training_succeeded is not None and not training_succeeded:
            # TODO Logging?
            pass
        self.soft_update_counter += 1
        q_target = self.calc_Q_target(mini_batch)
        self.critic1.train_net(q_target, mini_batch)
        self.critic2.train_net(q_target, mini_batch)
        if self.soft_update_counter == self.soft_update_frequency:
            self.critic1.soft_update(self.critic1_target)
            self.critic2.soft_update(self.critic2_target)
            self.soft_update_counter = 0

    def calc_Q_target(
        self,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            actor_fwd = self.actor.forward(s_prime)
            a_prime, log_prob = actor_fwd[0], actor_fwd[1]
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = -self.actor.log_alpha.exp() * log_prob
            q1_val, q2_val = (
                self.critic1(s_prime, a_prime),
                self.critic2(s_prime, a_prime),
            )
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self.gamma * (1 - done) * (min_q + entropy)

        return target

    def normalize_transitions_before_training(
        self,
        transition_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """Normalize the transition before training with it."""
        return transition_batch

    def save(self, save_directory: str):
        """Save the models in the given directory."""
        torch.save(self.actor.state_dict(), f"{save_directory}/actor.pth")
        torch.save(self.critic1.state_dict(), f"{save_directory}/critic1.pth")
        torch.save(self.critic2.state_dict(), f"{save_directory}/critic2.pth")
        torch.save(
            self.critic1_target.state_dict(), f"{save_directory}/critic1_target.pth"
        )
        torch.save(
            self.critic2_target.state_dict(), f"{save_directory}/critic2_target.pth"
        )

    def load(self, save_directory: str):
        """Load the models from the given directory. Is ment to be exactly compatible with save."""
        self.actor.load_state_dict(torch.load(f"{save_directory}/actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{save_directory}/critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"{save_directory}/critic2.pth"))
        self.critic1_target.load_state_dict(
            torch.load(f"{save_directory}/critic1_target.pth")
        )
        self.critic2_target.load_state_dict(
            torch.load(f"{save_directory}/critic2_target.pth")
        )
