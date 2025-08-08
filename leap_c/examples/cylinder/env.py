from datetime import datetime
from pathlib import Path

import cylinder_renderer
import dolfin
import flowcontrol.flowsolverparameters as flowsolverparameters
import gymnasium as gym
import numpy as np
import utils.utils_flowsolver as flu
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
    | 0   | Vy(3, 0)              | -Inf                | Inf               |
    | 1   | Vy(3.1, 1)            | -Inf                | Inf               |
    | 2   | Vy(3.1, -1)           | -Inf                | Inf               |

    For now:
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
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
        render_method: str = "project",
        Re: float = 100,
        Tf: float = 10,
        save_every: int = 100,
    ):
        self.Re = Re
        self.umax = 2
        self.umin = -self.umax

        # FlowSolver
        dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
        self.flowsolver = instantiate_flowsolver(
            Re=self.Re, Tf=Tf, save_every=save_every
        )
        initialize_flowsolver(self.flowsolver)

        # Action, Observation...
        self.action_space = spaces.Box(low=self.umin, high=self.umax, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.reset_needed = True
        self.t = 0
        self.max_time = Tf
        self.x = None

        # For rendering
        self.check_render_mode(render_mode=render_mode)
        self.renderer = cylinder_renderer.CylinderRenderer(
            self.flowsolver, render_method=render_method, render_mode=render_mode
        )
        self.render_mode = render_mode

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step of the flow dynamics."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")

        # saturation: this form prefered over minmax for some reason
        if action > self.umax:
            action = self.umax
        if action < self.umin:
            action = self.umin

        y = self.flowsolver.step(
            u_ctrl=np.repeat(action, repeats=2, axis=0)
        )  # 1D action into 2D to feed 2 actuators
        self.x = y

        self.x_trajectory.append(self.x)  # type: ignore
        self.t = self.flowsolver.t

        r = -np.linalg.norm(y) * 1 / np.sqrt(self.max_time)
        # bigger reward when smaller y norm

        # W: Stopping criteria?
        term = False
        trunc = False
        info = {}

        if self.t >= self.max_time:
            if len(self.x_trajectory) >= 10:
                success = all(
                    np.linalg.norm(self.x_trajectory[i]) < 0.1 for i in range(-10, 0)
                )
            else:
                success = False  # Not enough data to determine success

            info = {"task": {"violation": False, "success": success}}
            trunc = True

        if np.isnan(r):
            info = {"task": {"violation": True, "success": False}}
            trunc = True

        self.reset_needed = trunc or term

        return self.x, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # reinit flowsolver
        self.flowsolver.load_steady_state()
        self.flowsolver.initialize_time_stepping(
            ic=None
        )  # initial state = base flow + ic
        # TODO initial state on attractor, rand phase in [0,2pi]
        self.flowsolver.paths["timeseries"] = (
            self.flowsolver.params_save.path_out
            / f"ts_{datetime.now().strftime('%m-%d_%H-%M-%S')}.csv"
        )

        # reset the rest
        self.t = 0
        self.x = self.flowsolver.y_meas
        self.reset_needed = False

        self.x_trajectory = []
        return self.x, {}

    ####################################################################################
    ####################################################################################
    def render(self):
        return self.renderer.render()

    def close(self):
        return self.renderer.close()

    def check_render_mode(self, render_mode):
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        return 1

    ####################################################################################
    ####################################################################################


def instantiate_flowsolver(Re, Tf, save_every):
    """Create CylinderFlowSolver object with mainly default parameters"""
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
        verbose=500,
    )

    return fs


def initialize_flowsolver(fs: CylinderFlowSolver):
    """Initialize CylinderFlowSolver before time-stepping
    TODO: here, we choose the initial state of the flow (e.g. somewhere
    close to equilibrium, or on attractor)"""
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

    # TODO: read attractor snapshot for init
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)
