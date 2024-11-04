from abc import ABC, abstractmethod

import torch

from seal.rl.replay_buffer import ReplayBuffer


class Trainer(ABC):
    """Interface for a trainer."""

    replay_buffer: ReplayBuffer
    device: str

    @abstractmethod
    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Act based on the state. This is intended for rollouts (= interaction with the environment).
        Parameters:
            state: The state for which the action should be determined.
            deterministic: If True, the action is drawn deterministically.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """One step of training the components."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, save_directory: str):
        """Save the models in the given directory."""
        raise NotImplementedError()

    def load(self, save_directory: str):
        """Load the models from the given directory. Is ment to be exactly compatible with save."""
        raise NotImplementedError()
