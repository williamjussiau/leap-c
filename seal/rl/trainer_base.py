import contextlib
import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ContextManager

import numpy as np
import torch

from seal.ocp_env import OCPEnv
from seal.rl.replay_buffer import ReplayBuffer
from seal.util import create_dir_if_not_exists


def very_crude_debug_memory_leak():
    debug_file = os.path.join(os.getcwd(), "debug.txt")
    with open(debug_file, "a") as f:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    f.write(str(type(obj)) + "\t" + str(obj.size()) + "\n")
            except:
                pass
        f.write("====End of Episode====" + "\n")


@dataclass(kw_only=True)
class BaseTrainerConfig:
    """Contains the necessary information for the training loop.
    Attributes:
        save_directory_path: The path to a directory where the models will be saved.
        save_interval: The interval in which the models will be saved, additional to models being saved when validation hits a new high.
        max_episodes: The maximum number of episodes to rollout.
        training_per_episode: The number of training steps per episode.
        max_eps_length: The maximum number of steps in an episode.
        dont_train_until_this_many_transitions: The number of transitions that should be gathered in the replay buffer before training starts.
        seed: The seed for reproducibility.
        val_interval: Every this many episodes a validation will be done.
        n_val_rollouts: Number of rollouts during validation.
        no_grad_during_rollout: If True, no gradients will be calculated during the rollout (for efficiency).
        crude_memory_debugging: If True, a very crude memory debugging will be run after every training and the results will be appended into "debug.txt" in the current working directory."""

    seed: int
    device: str
    save_directory_path: str

    max_episodes: int
    training_steps_per_episode: int
    max_eps_length: int  # TODO: Usually this should be in the env, right?
    dont_train_until_this_many_transitions: int
    val_interval: int
    n_val_rollouts: int
    save_interval: int
    no_grad_during_rollout: bool
    crude_memory_debugging: bool


class Trainer(ABC):
    """Interface for a trainer."""

    replay_buffer: ReplayBuffer
    device: str

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """Act based on the observation. This is intended for rollouts (= interaction with the environment).
        Parameters:
            obs: The observation for which the action should be determined.
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

    @abstractmethod
    def load(self, save_directory: str):
        """Load the models from the given directory. Is ment to be exactly compatible with save."""
        raise NotImplementedError()

    @abstractmethod
    def goal_reached(self, max_val_score: float) -> bool:
        """Returns True if the goal of training is reached and the training loop should be terminated
        (independent from whether or not the maximum iterations are reached).

        Parameters:
            max_val_score: The highest validation score so far.
        """
        raise NotImplementedError()

    def validate(
        self,
        ocp_env: OCPEnv,
        n_val_rollouts: int,
        config: BaseTrainerConfig,
    ):
        """Do a deterministic validation run of the policy and return the mean of the cumulative reward over all validation episodes."""
        scores = []
        for _ in range(n_val_rollouts):
            info = self.episode_rollout(ocp_env, True, torch.no_grad(), config)
            score = info["score"]
            scores.append(score)

        return sum(scores) / n_val_rollouts

    def episode_rollout(
        self,
        ocp_env: OCPEnv,
        validation: bool,
        grad_or_no_grad: ContextManager,
        config: BaseTrainerConfig,
    ) -> dict:
        """Rollout an episode (including putting transitions into the replay buffer) and return the cumulative reward.
        Parameters:
            ocp_env: The gym environment.
            validation: If True, the policy will act as if this is validation (e.g., turning off exploration).
            grad_or_no_grad: A context manager in which to perform the rollout. E.g., torch.no_grad().
            config: The configuration for the training loop.

        Returns:
            A dictionary containing information about the rollout, at least containing the key

            "score" for the cumulative score
        """
        score = 0
        obs, info = ocp_env.reset(seed=config.seed)

        terminated = False
        truncated = False
        count = 0
        with grad_or_no_grad:
            while count < config.max_eps_length and not terminated and not truncated:
                a = self.act(obs, deterministic=validation)
                obs_prime, r, terminated, truncated, info = ocp_env.step(a)
                self.replay_buffer.put((obs, a, r, obs_prime, terminated))
                score += r  # type: ignore
                obs = obs_prime
                count += 1
        return dict(score=score)

    def training_loop(
        self,
        ocp_env: OCPEnv,
        config: BaseTrainerConfig,
    ) -> float:
        """Call this function in your script to start the training loop.
        Saving works by calling the save method of the trainer object every
        save_interval many episodes or when validation returns a new best score.

        Parameters:
            ocp_env: The gym environment.
            config: The configuration for the training loop.
        """

        if config.no_grad_during_rollout:
            grad_or_no_grad = torch.no_grad()
        else:
            grad_or_no_grad = contextlib.nullcontext()
        max_val_score = -np.inf

        for n_epi in range(config.max_episodes):
            info = self.episode_rollout(ocp_env, False, grad_or_no_grad, config)
            score = info["score"]
            print("Episode rollout: ", n_epi, "Score: ", score)
            if (
                self.replay_buffer.size()
                > config.dont_train_until_this_many_transitions
            ):
                for i in range(config.training_steps_per_episode):
                    self.train()
                    print(f"Training step {i}")
                if config.crude_memory_debugging:
                    very_crude_debug_memory_leak()

                if n_epi % config.val_interval == 0:
                    # TODO Log this
                    avg_val_score = self.validate(
                        ocp_env=ocp_env,
                        n_val_rollouts=config.n_val_rollouts,
                        config=config,
                    )
                    print("avg_val_score: ", avg_val_score)
                    if avg_val_score > max_val_score:
                        save_directory_for_models = os.path.join(
                            config.save_directory_path,
                            "val_score_" + str(avg_val_score),
                        )
                        create_dir_if_not_exists(save_directory_for_models)
                        max_val_score = avg_val_score
                        self.save(save_directory_for_models)
                    if self.goal_reached(max_val_score):
                        # terminate training
                        break
            if n_epi % config.save_interval == 0:
                save_directory_for_models = os.path.join(
                    config.save_directory_path, "episode_" + str(n_epi)
                )
                create_dir_if_not_exists(save_directory_for_models)
                self.save(save_directory_for_models)
        ocp_env.close()
        return avg_val_score  # Return last validation score for testing purposes
