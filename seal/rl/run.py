import contextlib
import gc
import os
import random
from typing import Callable

import numpy as np
import torch

from seal.ocp_env import OCPEnv
from seal.rl.trainer_base import Trainer
from seal.util import create_dir_if_not_exists, tensor_to_numpy


def validate(
    ocp_env: OCPEnv,
    trainer: Trainer,
    map_ocp_env_state_to_trainer_input: Callable,
    map_policy_to_env: Callable[[np.ndarray], np.ndarray],
    n_val_rollouts: int,
    max_eps_length: int,
):
    """Do a deterministic validation run of the policy and return the mean of the cumulative reward over all validation episodes."""
    scores = []
    for _ in range(n_val_rollouts):
        mpc_input, info = ocp_env.reset()
        s = map_ocp_env_state_to_trainer_input(mpc_input)
        terminated = False
        truncated = False
        score = 0
        count = 0
        while count < max_eps_length and not terminated and not truncated:
            a = trainer.act(
                torch.tensor(s, device=trainer.device, dtype=torch.float32),
                deterministic=True,
            )
            a = tensor_to_numpy(a)
            s_prime, r, terminated, truncated, info = ocp_env.step(map_policy_to_env(a))
            s_prime = map_ocp_env_state_to_trainer_input(s_prime)
            s = s_prime
            score += r  # type:ignore
            count += 1
        scores.append(score)

    return sum(scores) / n_val_rollouts


def training_loop(
    ocp_env: OCPEnv,
    trainer: Trainer,
    map_ocp_env_state_to_trainer_input: Callable,
    map_policy_to_env: Callable[[np.ndarray], np.ndarray],
    save_directory_path: str,
    save_interval: int,
    max_episodes: int,
    training_steps_per_episode: int,
    max_eps_length: int,
    dont_train_until_this_many_transitions: int,
    seed: int,
    val_interval: int,
    n_val_rollouts: int,
    no_grad_during_rollout: bool = True,
    crude_memory_debugging: bool = False,
    **kwargs,
):
    """Call this function in your script to start the training loop.
    Saving works by calling the save method of the trainer object every
    save_interval many episodes or when validation returns a new best score.

    Parameters:
        env: The gym environment.
        trainer: The trainer object.
        map_policy_to_env: A function that maps the policy output to the environment action space.
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
        crude_memory_debugging: If True, a very crude memory debugging will be run after every training and the results will be appended into "debug.txt" in the current working directory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if no_grad_during_rollout:
        grad_or_no_grad = torch.no_grad()
    else:
        grad_or_no_grad = contextlib.nullcontext()

    score = 0.0
    max_val_score = 0.0

    for n_epi in range(max_episodes):
        mpc_input, info = ocp_env.reset(seed=seed)
        s = map_ocp_env_state_to_trainer_input(mpc_input)

        terminated = False
        truncated = False
        count = 0
        with grad_or_no_grad:
            while count < max_eps_length and not terminated and not truncated:
                a = trainer.act(
                    torch.tensor(s, device=trainer.device, dtype=torch.float32)
                )
                a = tensor_to_numpy(a)
                s_prime, r, terminated, truncated, info = ocp_env.step(
                    map_policy_to_env(a)
                )
                s_prime = map_ocp_env_state_to_trainer_input(s_prime)
                trainer.replay_buffer.put((s, a, r, s_prime, terminated))
                score += r  # type: ignore
                s = s_prime
                count += 1
        #        print("Episode " + str(n_epi) + " finished!")
        print("Episode rollout: ", n_epi, "Score: ", score)
        if trainer.replay_buffer.size() > dont_train_until_this_many_transitions:
            for i in range(training_steps_per_episode):
                trainer.train()
            if crude_memory_debugging:
                very_crude_debug_memory_leak()

            if n_epi % val_interval == 0:
                # TODO Log this
                avg_val_score = validate(
                    ocp_env=ocp_env,
                    trainer=trainer,
                    map_ocp_env_state_to_trainer_input=map_ocp_env_state_to_trainer_input,
                    map_policy_to_env=map_policy_to_env,
                    n_val_rollouts=n_val_rollouts,
                    max_eps_length=max_eps_length,
                )
                print("avg_val_score: ", avg_val_score)
                if avg_val_score > max_val_score:
                    save_here = os.path.join(
                        save_directory_path, "val_score_" + str(avg_val_score)
                    )
                    create_dir_if_not_exists(save_here)
                    max_val_score = avg_val_score
                    trainer.save(save_directory_path)
        if n_epi % save_interval == 0:
            save_here = os.path.join(save_directory_path, "episode_" + str(n_epi))
            create_dir_if_not_exists(save_here)
            trainer.save(save_directory_path)
        score = 0.0
    ocp_env.close()


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
