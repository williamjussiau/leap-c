from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable

import gin
import gymnasium
import numpy as np
import pandas as pd
from acados_template import AcadosSimSolver

from acados_il.ocp import MPCInput, MPCPlanner
from acados_il.utils import create_sim_from_ocp


class EvalEnv(gymnasium.Env, ABC):
    """Minimal interface to be used with the evaluate_policy function."""

    @property
    @abstractmethod
    def episode_id(self) -> str:
        """The id of the current episode."""
        ...

    @abstractmethod
    def write_npy(self, output_path: Path):
        """Write the current episode to a numpy file.

        Args:
            output_path: The path where the numpy files are stored.
        """
        ...

    @abstractmethod
    def compare_npy(self, comparison_path: Path) -> dict[str, float | np.ndarray]:
        """Compare the current episode with the reference episode.

        Args:
            comparison_path: The root path where the numpy files are stored.

        Returns:
            The MSE stats generated from the comparison.
        """
        ...


@gin.configurable
class OCPEnv(gymnasium.Env):
    def __init__(self, planner: MPCPlanner, dt: float = 0.1, max_time: float = 10.0):
        """A gym environment created from a MPCPlanner object.

        The gym environment could be used to generate new samples
        or to evaluate the performance of learned planners.

        The two methods that often need to be overloaded are:
        - _mpc_input: Provides the input for the planner, needed for parameterized
            OCP problems.
        - init_state: Initializes the state of the system. Default is to sample uniformly
            from the state space.

        Args:
            planner: The learnable MPC planner.
            dt: The time step of the environment.
            max_time: The maximum time per episode.
        """
        self.planner = planner
        self._sim_solver = None
        self._dt = dt
        self.options = {"max_time": max_time}

        self.action_space = planner.control_space
        self.observation_space = planner.input_space

        self.episode_idx = -1
        self.t = 0
        self.x: None | np.ndarray = None
        self.x_his = []  # needed for comparison
        self.u_his = []  # needed for comparison

    def init_state(self) -> np.ndarray:
        """Initializes the state of the system.

        Returns:
            The initial state of the system.
        """
        return self.planner.state_space.sample()

    def _mpc_input(self) -> MPCInput:
        """Provides the input for the planner.

        This needs to be overloaded if the planner needs additional
        parameters.

        Raises:
            ValueError: Raises when the env was not properly reset.

        Returns:
            mpc_input: Returns the input for the planner.
        """
        if self.x is None:
            raise ValueError("State is not set. Call reset first.")
        return {"x0_bar": self.x.copy()}

    def dynamics(
        self, x: np.ndarray, u: np.ndarray, mpc_input: dict | None = None
    ) -> np.ndarray:
        """The dynamics of the system.

        Args:
            x: The current state.
            u: The current control.
            mpc_input: The input for the MPC.

        Returns:
            next state: The next state of the system.
        """
        if self._sim_solver is None:
            sim = create_sim_from_ocp(self.planner.ocp, self.planner.export_dir / "sim")
            self._sim_solver = AcadosSimSolver(
                sim, str(self.planner.export_dir / "acados_ocp.json")
            )

        self._sim_solver.set("x", x)
        self._sim_solver.set("u", u)
        p = self.planner.fetch_params(mpc_input, 0)  # type: ignore
        if p is not None:
            self._sim_solver.set("p", p)

        self._sim_solver.solve()

        return self._sim_solver.get("x")  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[MPCInput, dict]:  # type: ignore
        """Resets the environment.

        Args:
            seed: The seed used to make the results reproducible.
            options: Options that can be used to modify the environment.

        Returns:
            Returns the current mpc_input and the initial state.
        """
        if options is not None:
            self.options.update(options)
        if seed is not None:
            super().reset(seed=seed)
            self.planner.state_space.seed(seed)
        if self._np_random is None:
            raise RuntimeError("The first reset needs to be called with a seed.")

        self.episode_idx += 1
        self.t = 0
        self.x = self.init_state()
        self.x_his = []
        self.u_his = []

        # TODO: Remove this!
        # self.planner.reset(self._mpc_input())

        return self._mpc_input(), {}

    def step(self, action: np.ndarray) -> tuple[MPCInput, float, bool, bool, dict]:
        """The step function of the environment.

        Args:
            action: The action taken by the policy.

        Returns:
            next state: The next state of the environment.
            cost: The cost of the current state and action.
            terminal: Whether the episode reached a terimal state => V(x)=0.
            truncated: Whether the episode is finished without reaching a terminal
                state.
            stats: Additional information that could be used for evaluation.
        """
        # evaluate cons
        info = self.planner.stage_cons(self.x, action, self._mpc_input())  # type: ignore

        # assert action in self.action_space
        if self.x is None:
            raise ValueError("State is not set. Call reset first.")

        self.t += self._dt
        self.x = self.dynamics(self.x, action, self._mpc_input())  # type: ignore
        mpc_input = self._mpc_input()

        self.x_his.append(self.x.copy())
        self.u_his.append(action.copy())
        self.planner

        truncated = False
        if self.t >= self.options["max_time"]:
            truncated = True
            self.x = None

        return mpc_input, 0.0, False, truncated, info

    @property
    def episode_id(self) -> str:
        return f"episode_{self.episode_idx}"

    def npy_path(self, output_path: Path, name: str):
        """Declares the path where the numpy files are stored.

        Args:
            output_path: The root path where the numpy files are stored.
            name: The name of the data.

        Returns:
            The path where the numpy files are stored.
        """
        return output_path / f"{name}_{self.episode_id}.npy"

    def write_npy(self, output_path: Path):
        """Write the current episode to a numpy file.

        Args:
            output_path: The path where the numpy files are stored.
        """
        np.save(self.npy_path(output_path, "state"), self.x_his)
        np.save(self.npy_path(output_path, "control"), self.u_his)

    def compare_npy(self, comparison_path: Path) -> dict[str, float | np.ndarray]:
        """Compare the current episode with the reference episode.

        Args:
            comparison_path: The root path where the numpy files are stored.

        Returns:
            The MSE stats generated from the comparison.
        """
        x_ref = np.load(self.npy_path(comparison_path, "state"))
        u_ref = np.load(self.npy_path(comparison_path, "control"))
        x_his = np.stack(self.x_his)
        u_his = np.stack(self.u_his)
        assert x_his.shape == x_ref.shape and u_his.shape == u_ref.shape

        mse_x = np.mean((self.x_his - x_ref) ** 2, axis=0)
        mse_u = np.mean((self.u_his - u_ref) ** 2, axis=0)

        return {"mse_x": mse_x, "mse_u": mse_u}


def evaluate_policy(
    policy_fn: Callable[[MPCInput], np.ndarray],
    env: EvalEnv,
    num_episodes: int | None = None,
    seed: int = 0,
    comparison_dir: Path | None = None,
    store_dir: Path | None = None,
) -> pd.DataFrame:
    """Closed-loop evaluation of a policy in terms of costs, constraint violations
    and approximation error to the original planner.

    Args:
        policy_fn: The function that takes the mpc_input and returns the mpc_output.
        env: The environment that is used to evaluate the planner.
        num_episodes: The number of episodes to evaluate the planner. If comparison_dir
            is given, the number of episodes is taken from the comparison_dir.
        seed: The seed used to make the results reproducible. Is used at the first
            reset.
        comparison_dir: The directory of trajectories from a different planner to
            evaluate against.
        store_dir: The directory for storing trajectories generated by the current
            planner.

    Returns:
        The statistics of the evaluation.
    """
    stats = []

    if comparison_dir is not None:
        assert num_episodes is None
        num_episodes = len(list(comparison_dir.glob("state_*.npy")))
    elif hasattr(env, "num_episodes"):
        num_episodes = env.num_episodes  # type: ignore

    assert num_episodes is not None

    for _ in range(num_episodes):
        episode_stats = defaultdict(list)
        episode_return = 0
        episode_length = 0

        # TODO Jasper: Check this again!
        mpc_input, _ = env.reset(seed=seed)
        terminal = False
        truncated = False

        while not terminal and not truncated:
            action = policy_fn(mpc_input)
            mpc_input, cost, terminal, truncated, info = env.step(action)
            episode_return += cost
            episode_length += 1

            for key, value in info.items():
                episode_stats[key].append(value)

            import pdb; pdb.set_trace()

        final_stats = {}

        # calculate the mean stats ignoring nans
        for key, value in episode_stats.items():
            final_stats[key] = np.nanmean(value)

        if store_dir is not None:
            env.write_npy(store_dir)

        if comparison_dir is not None:
            comp_stats = env.compare_npy(comparison_dir)
            # report each dimension separately
            for key, value in comp_stats.items():
                if isinstance(value, float):
                    final_stats[key] = value
                    continue
                # average over all dimensions
                final_stats[key] = np.mean(value)
                for idx in range(len(value)):  # type: ignore
                    final_stats[key + f"_{idx}"] = value[idx]  # type: ignore

        final_stats["return"] = episode_return
        final_stats["length"] = episode_length
        final_stats["terminal"] = terminal
        final_stats["truncated"] = truncated
        final_stats["episode_id"] = env.episode_id

        stats.append(final_stats)

    return pd.DataFrame(stats)
