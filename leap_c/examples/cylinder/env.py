from pathlib import Path

import dolfin
import flowcontrol.flowsolverparameters as flowsolverparameters
import gymnasium as gym
import numpy as np
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint
from gymnasium import spaces


class CylinderEnv(gym.Env):
    """
    An environment of the flow past a cylinder
    meat to be stabilized around its unstable stationary equilibrium

    Observation Space:
    ------------------

    The observation is a `ndarray` with shape `(xxx,)` and dtype `np.float32`
    representing the state of the system.

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Pole Angle (theta)    | -2pi                | 2pi               |
    | 2   | Cart Velocity         | -Inf                | Inf               |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    ---> TODO: full field or sensors?

    Action Space:
    -------------

    The action is a `ndarray` with shape `(1,)` which can take values in the range (-umax, umax)
    indicating the amount of blowing (positive) or suction (negative) from the upper pole of the
    cylinder. The lower pole of the cylinder acts symetrically to maintain zero mass flux.


    Reward:
    -------
    Maximum reward is attained when maintaining the cylinder close to equilibrium.
    We can think of formalizing the reward in terms of Cl/Cd (force coefficients) as per Paris et al. (2020)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        Re: float = 100,
        Tf: float = 1,
        save_every: int = 0,
    ):
        self.Re = Re
        self.umax = 2

        # FlowSolver
        dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
        self.flowsolver = instantiate_flowsolver(
            Re=self.Re, Tf=Tf, save_every=save_every
        )
        initialize_flowsolver(self.flowsolver)

        # Action & observation spaces bounds
        # high = np.array(
        #     [
        #         self.x_threshold * 2,
        #         2 * np.pi,
        #         np.finfo(np.float32).max,
        #         np.finfo(np.float32).max,
        #     ],
        #     dtype=np.float32,
        # )

        self.action_space = spaces.Box(-self.umax, self.umax, dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, dtype=np.float32)  ########### TODO

        self.reset_needed = True
        self.t = 0
        self.max_time = Tf
        self.x = None

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        # self.pos_trajectory = None
        # self.pole_end_trajectory = None
        # self.x_trajectory = None
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step of the flow dynamics."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")

        y = self.flowsolver.step(
            u_ctrl=np.repeat(action, repeats=2, axis=0)
        )  # 1D action into 2D to feed 2 actuators
        self.x = y

        self.x_trajectory.append(self.x)  # type: ignore
        self.t = self.flowsolver.t

        r = -np.linalg.norm(y)  # bigger reward when smaller y norm

        # W: Stopping criteria?
        term = False
        trunc = False
        info = {}
        # if self.x[0] > self.x_threshold or self.x[0] < -self.x_threshold:
        #     term = True  # Just terminating should be enough punishment when reward is positive
        #     info = {"task": {"violation": True, "success": False}}
        if self.t > self.max_time:
            # check if the pole is upright in the last 10 steps
            if len(self.x_trajectory) >= 10:
                success = all(
                    np.linalg.norm(self.x_trajectory[i]) < 0.1 for i in range(-10, 0)
                )
            else:
                success = False  # Not enough data to determine success

            info = {"task": {"violation": False, "success": success}}
            trunc = True
        self.reset_needed = trunc or term

        return self.x, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        """W: Most likely resets environment?"""
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # reinit flowsolver
        self.flowsolver.load_steady_state()
        self.flowsolver.initialize_time_stepping(
            ic=None
        )  # initial state = base flow + ic

        # reset the rest
        self.t = 0
        self.x = self.flowsolver.y_meas  ########### TODO: what is state
        self.reset_needed = False

        self.x_trajectory = []
        return self.x, {}

    # def init_state(self, options: Optional[dict] = None) -> np.ndarray:
    #     return np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

    # def include_this_state_trajectory_to_rendering(self, state_trajectory: np.ndarray):
    #     """Meant for setting a state trajectory for rendering.
    #     If a state trajectory is not set before the next call of render,
    #     the rendering will not render a state trajectory.

    #     NOTE: The record_video wrapper of gymnasium will call render() AFTER every step.
    #     This means if you use the wrapper,
    #     make a step,
    #     calculate action and state trajectory from the observations,
    #     and input the state trajectory with this function before taking the next step,
    #     the picture being rendered after this next step will be showing the trajectory planned BEFORE DOING the step.
    #     """
    #     self.pos_trajectory = []
    #     # self.pole_end_trajectory = []
    #     for x in state_trajectory:
    #         self.pos_trajectory.append(x[0])
    #         # self.pole_end_trajectory.append(self.calc_pole_end(x[0], x[1], self.length))

    def render(self):
        pass

    def close(self):
        pass


def instantiate_flowsolver(Re, Tf, save_every):
    cwd = Path(__file__).parent
    dt = 0.005
    num_steps = int(np.ceil(Tf / dt))

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=dt, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=save_every, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "O1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart()

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    angular_size_deg = 10
    actuator_bc_1 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    actuator_bc_2 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=1,
    )

    return fs


def initialize_flowsolver(fs: CylinderFlowSolver):
    uctrl0 = [0.0, 0.0]

    if Path(fs.paths["U0"]).exists():
        fs.load_steady_state()
    else:
        fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)
        fs.compute_steady_state(
            method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
        )
        # Expected:
        # Newton iteration 4: r (abs) = 6.901e-14 (tol = 1.000e-10) r (rel) = 1.109e-11 (tol = 1.000e-09)

    # TODO: read attractor snapshot
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)


# class CylinderStabilizationEnv(CylinderEnv):
#     """What is that"""

#     def init_state(self, options: Optional[dict] = None) -> np.ndarray:
#         low, high = gym_utils.maybe_parse_reset_bounds(
#             options,
#             -0.05,
#             0.05,  # default low
#         )  # default high
#         return self.np_random.uniform(low=low, high=high, size=(4,))


if __name__ == "__main__":
    print("Instantiate CylinderFlowSolver.")
    env = CylinderEnv(render_mode="human", Re=100, Tf=0.05, save_every=0)

    print("Reset CylinderFlowSolver.")
    obs, info = env.reset(seed=44)

    terminated = False
    truncated = False
    total_reward = 0
    # env.render()

    for i in range(100):  # Increase steps for longer visualization
        action = env.action_space.sample()

        # goal_dir = env.goal.pos - obs[:2]
        # goal_dir_norm = np.linalg.norm(goal_dir)
        # if goal_dir_norm > 1e-3:
        #     proportional_force = (goal_dir / goal_dir_norm) * env.Fmax
        # else:
        #     proportional_force = np.zeros(2)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # print(f"Current action: {action}")

        #     env.render()

        if terminated or truncated:
            print(f"Episode finished after {i + 1} timesteps.")
            print(f"Termination: {terminated}, Truncation: {truncated}")
            print(f"Final state (pos): {obs}")
            # print(f"Goal position: {env.goal.pos}")
            # print(f"Distance to goal: {np.linalg.norm(obs[:2] - env.goal.pos):.3f}")
            print(f"Total reward: {total_reward:.2f}")
            if env.render_mode == "human":
                pass
                # plt.pause(5.0)  # Increased pause to see final state
            break  # Stop after one episode for this example

    # # Close the environment rendering window
    # env.close()
    print("Environment closed.")
