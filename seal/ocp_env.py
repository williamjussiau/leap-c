from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from seal.dynamics import create_dynamics_from_mpc
from seal.mpc import MPC

MPCInput = tuple[np.ndarray, np.ndarray | None]


class ParamCreator(ABC):
    """Something that is used to create parameters for the MPC in every step of the Environment.
    #TODO: Could be part of Dynamics?
    """

    @property
    @abstractmethod
    def current_param(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod    
    def reset(self):
        raise NotImplementedError()
    
    

class ConstantParamCreator(ParamCreator):
    """A ParameterCreator that always returns the same parameter.
    """
    def __init__(self, param: np.ndarray):
        self.param = param.copy()
        
    @property
    def current_param(self) -> np.ndarray:
        return self.param
    
    def step(self):
        return self.param
    
    def reset(self):
        return self.param
    

class OCPEnv(gym.Env):
    def __init__(self, mpc: MPC, param_creator: ParamCreator, dt: float = 0.1, max_time: float = 10.0):
        """A gym environment created from an MPC object.

        The gym environment could be used to generate new samples
        or to evaluate the performance of learned planners.

        The two methods that often need to be overloaded are:
        - mpc_input: Provides the input for the planner, needed for parameterized
            OCP problems.
        - init_state: Initializes the state of the system. Default is to sample uniformly
            from the state space.

        Args:
            mpc: The learnable MPC planner.
            param_creator: Creates the parameters for the MPC in every step.
            dt: The time discretization of the environment.
            max_time: The maximum time per episode.
        """
        self.mpc = mpc
        self.dynamics = create_dynamics_from_mpc(mpc)
        self._dt = dt
        self.options = {"max_time": max_time}
        self.param_creator = param_creator

        self.episode_idx = -1
        self.t = 0
        self.x: None | np.ndarray = None
        self.x_his = []  # needed for comparison
        self.u_his = []  # needed for comparison

        self.action_space = self.derive_action_space()
        self.state_space = self.derive_state_space()
        self.parameter_space = self.derive_parameter_space()
        self.observation_space = self.derive_observation_space()

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

    def derive_parameter_space(self) -> spaces.Box | None:
        if self.mpc.ocp.p_global_values is None:
            return None

        shape = self.mpc.ocp.p_global_values.shape

        return spaces.Box(
            low=np.full(shape, -np.inf).astype(np.float32),
            high=np.full(shape, np.inf).astype(np.float32),
            dtype=np.float32,
        )

    def derive_observation_space(self) -> spaces.Space:
        if self.parameter_space is not None:
            return spaces.Tuple([self.state_space, self.parameter_space])
        return self.state_space

    def init_state(self) -> np.ndarray:
        """Initializes the state of the system.

        Returns:
            The initial state of the system.
        """
        return self.state_space.sample()

    def mpc_input(self) -> MPCInput:
        """Provides the input for the planner.

        This needs to be overloaded if the planner needs additional
        parameters.

        Raises:
            ValueError: Raises when the env was not properly reset.

        Returns:
            The state x and the parameters p.
        """
        if self.x is None:
            raise ValueError("State is not set. Call reset first.")
        return self.x.copy(), self.param_creator.current_param

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
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        if self._np_random is None:
            raise RuntimeError("The first reset needs to be called with a seed.")

        self.episode_idx += 1
        self.t = 0
        self.x = self.init_state()
        self.x_his = []
        self.u_his = []

        # TODO: Remove this!
        x, _ = self.mpc_input()
        self.mpc.reset(x)  # type: ignore
        p = self.param_creator.reset()

        return (x, p), {}

    def stage_cost(self, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
        """Return the stage cost of the current state and action."""
        # TODO: Implement this in the MPC class
        # self.mpc.stage_cost()
        return 0.0

    def step(self, action: np.ndarray) -> tuple[MPCInput, float, bool, bool, dict]:
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
        x, p = self.mpc_input()
        info = self.mpc.stage_cons(x, action, p)  # type: ignore

        # evaluate stage cost
        cost = self.stage_cost(x, action, p)  # type: ignore

        # assert action in self.action_space
        if self.x is None:
            raise ValueError("State is not set. Call reset first.")

        self.t += self._dt
        self.x = self.dynamics(x, action, p)  # type: ignore
        self.param_creator.step()
        x_next, p_next = self.mpc_input()

        self.x_his.append(self.x.copy())  # type: ignore
        self.u_his.append(action.copy())

        truncated = False
        if self.t >= self.options["max_time"]:
            truncated = True
            self.x = None

        return (x_next, p_next), -cost, False, truncated, info
