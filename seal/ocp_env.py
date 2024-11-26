from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from seal.dynamics import create_dynamics_from_mpc
from seal.mpc import MPC, MPCParameter


class OCPEnv(gym.Env):
    def __init__(
        self,
        mpc: MPC,
        dt: float = 0.1,
        max_time: float = 10.0,
    ):
        """A gym environment created from an MPC object.

        The gym environment could be used to generate new samples
        or to evaluate the performance of learned planners.

        You probably want to overload methods like observation space, etc..


        Args:
            mpc: The learnable MPC planner.
            learnable_params: The parameters that can be learned.
            dt: The time discretization of the environment.
            max_time: The maximum time per episode.
        """
        self.mpc = mpc
        self.state_dynamics = create_dynamics_from_mpc(mpc)
        self._dt = dt
        self.options = {"max_time": max_time}

        self.t = 0
        self.x: None | np.ndarray = None
        self.env_params: None | MPCParameter = None

        self.action_space = self.derive_action_space()
        self.state_space = self.derive_state_space()
        self.observation_space = self.derive_observation_space()

        self.p_learnable_space = self.derive_p_learnable_space()

    def derive_action_space(self) -> spaces.Box:
        return spaces.Box(
            low=self.mpc.ocp.constraints.lbu.astype(np.float32),
            high=self.mpc.ocp.constraints.ubu.astype(np.float32),
            dtype=np.float32,
        )

    def derive_state_space(self) -> spaces.Box:
        return spaces.Box(
            low=self.mpc.ocp.constraints.lbx.astype(np.float32),
            high=self.mpc.ocp.constraints.ubx.astype(np.float32),
            dtype=np.float32,
        )

    def derive_observation_space(self) -> spaces.Space:
        if self.mpc.ocp.p_global_values is None:
            p_global_shape = (0,)
        else:
            p_global_shape = self.mpc.ocp.p_global_values.shape

        if len(p_global_shape) > 1:
            raise ValueError("Parameter vector should be flat.")

        p_global_space = spaces.Box(
            low=np.full(p_global_shape, -np.inf).astype(np.float32),
            high=np.full(p_global_shape, np.inf).astype(np.float32),
            dtype=np.float32,
        )

        if self.mpc.ocp.parameter_values is None:
            p_stagewise_shape = (0,)
        else:
            stages = self.mpc.N + 1
            shape = self.mpc.ocp.parameter_values.shape
            p_stagewise_shape = (stages, shape[0])

        if len(p_stagewise_shape) > 2:
            raise ValueError("Parameter vector should be flat.")

        p_stagewise_space = spaces.Box(
            low=np.full(p_stagewise_shape, -np.inf).astype(np.float32),
            high=np.full(p_stagewise_shape, np.inf).astype(np.float32),
            dtype=np.float32,
        )

        p_combined_space = spaces.Tuple([p_global_space, p_stagewise_space])

        return spaces.Tuple([self.state_space, p_combined_space])

    def derive_p_learnable_space(self) -> spaces.Box | None:
        """Per default all learnable parameters are global parameters."""
        if self.mpc.ocp.p_global_values is None:
            return None
        shape = self.mpc.ocp.p_global_values.shape

        if len(shape) > 1:
            raise ValueError("Parameter vector should be flat.")

        return spaces.Box(
            low=np.full(shape, -np.inf).astype(np.float32),
            high=np.full(shape, np.inf).astype(np.float32),
            dtype=np.float32,
        )

    def init_state(self) -> np.ndarray:
        """Initializes the state of the system.

        Returns:
            The initial state of the system.
        """
        return self.state_space.sample()

    def current_env_params(self) -> MPCParameter:
        """Returns the parameters of the system. Per default the parameter values are taken from the mpc.

        Returns:
            The current parameters of the system.
        """
        return MPCParameter(
            p_global=self.mpc.default_p_global.astype(np.float32)
            if self.mpc.default_p_global is not None
            else None,
            p_stagewise=self.mpc.default_p_stagewise.astype(np.float32)
            if self.mpc.default_p_stagewise is not None
            else None,
        )

    def combine_params(self, learned_params, env_obs) -> MPCParameter:
        """Returns a new MPCParameter instance where the env_parameters are overridden by learned_parameters."""
        if learned_params is not None:
            raise ValueError(
                "The learned parameters are p_global by default. Override this method if p_stagewise should be learned."
            )
        return MPCParameter(
            p_global=learned_params,
            p_stagewise=env_obs[1].p_stagewise,
            p_stagewise_sparse_idx=env_obs[1].p_stagewise_sparse_idx,
        )

    def current_observation(self):
        """Returns the current observation of the environment.

        Returns:
            The current observation of the environment.
        """
        if self.x is None or self.env_params is None:
            raise ValueError("State or Env_params is not set. Call reset first.")
        return self.x, self.env_params

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        """Resets the environment.

        Args:
            seed: The seed used to make the results reproducible.
            options: Options that can be used to modify the environment.

        Returns:
            Returns the initial observation and a dict containing information
        """
        if options is not None:
            self.options.update(options)
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        if self._np_random is None:
            raise RuntimeError("The first reset needs to be called with a seed.")

        self.t = 0
        self.x = self.init_state()
        self.env_params = self.current_env_params()

        return self.current_observation(), {}

    def stage_cost(
        self, x: np.ndarray, u: np.ndarray, env_params: MPCParameter
    ) -> float:
        """Return the stage cost of the current state and action."""
        return self.mpc.stage_cost(x, u, env_params)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """The step function of the environment.

        Args:
            action: The action taken by the policy.

        Returns:
            next state: The next state of the environment.
            reward: The reward of the current state and action.
            terminal: Whether the episode reached a terimal state => V(x)=0.
            truncated: Whether the episode is finished without reaching a terminal
                state.
            stats: Additional information that could be used for evaluation.
        """
        # evaluate cons
        x, env_params = self.current_observation()
        info = self.mpc.stage_cons(x, action, env_params)  # type: ignore

        # evaluate stage cost
        cost = self.stage_cost(x, action, env_params)  # type: ignore

        # assert action in self.action_space
        if self.x is None:
            raise ValueError("State is not set. Call reset first.")

        self.t += self._dt
        self.x = self.state_dynamics(x, action, env_params).astype(dtype=np.float32)  # type: ignore
        self.env_params = self.current_env_params()
        x_next, env_params_next = self.current_observation()

        truncated = False
        if self.t >= self.options["max_time"]:
            truncated = True
            self.x = None

        return (x_next, env_params_next), -cost, False, truncated, info
