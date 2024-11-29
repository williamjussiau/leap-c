import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel

from seal.mpc import MPC
from seal.ocp_env import OCPEnv
from casadi.tools import struct_symSX

from seal.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)

from typing import Any


class PendulumOnCartMPC(MPC):
    def __init__(
        self,
        params: dict[str, np.ndarray] = None,
        learnable_params: list[str] | None = None,
        discount_factor: float = 0.99,
        n_batch: int = 1,
    ):
        if params is None:
            params = {
                "M": np.array([1.0]),  # mass of the cart [kg]
                "m": np.array([0.1]),  # mass of the ball [kg]
                "g": np.array([9.81]),  # gravity constant [m/s^2]
                "l": np.array([0.8]),  # length of the rod [m]
                "Q": np.diag([2e3, 2e3, 1e-2, 1e-2]),  # state cost
                "R": np.diag([2e-1]),  # control cost
            }

        ocp = export_parametric_ocp(
            nominal_param=params, learnable_param=learnable_params
        )
        configure_ocp_solver(ocp)

        super().__init__(ocp=ocp, discount_factor=discount_factor, n_batch=n_batch)


class PendulumOnCartOcpEnv(OCPEnv):
    def __init__(
        self,
        mpc: PendulumOnCartMPC,
        dt: float = 0.1,
        max_time: float = 10.0,
    ):
        super().__init__(
            mpc,
            dt=dt,
            max_time=max_time,
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart and add random noise to the resulting cart velocity."""
        o, r, term, trunc, info = super().step(
            action
        )  # o is the next state as np.ndarray, next parameters as MPCParameter
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        noise = self._np_random.uniform(-0.1, 0)
        state = o[0].copy()
        state[2] += noise
        self.x = state
        o = (state, o[1])

        if state not in self.state_space:
            r -= 1e2
            term = True

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        return super().reset(seed=seed, options=options)

    def init_state(self):
        return np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)


def configure_ocp_solver(
    ocp: AcadosOcp,
):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True


def f_expl_expr(model: AcadosModel) -> ca.SX:
    p = find_param_in_p_or_p_global(["M", "m", "g", "l"], model)

    M = p["M"]
    m = p["m"]
    g = p["g"]
    l = p["l"]

    theta = model.x[1]
    v1 = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    return f_expl


def disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
    f_expl = f_expl_expr(model)

    x = model.x
    u = model.u

    # discrete dynamics via RK4
    p = ca.vertcat(
        *list(find_param_in_p_or_p_global(["M", "m", "g", "l"], model).values())
    )

    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def cost_expr_ext_cost(model: AcadosModel) -> ca.SX:
    x = model.x
    u = model.u

    p = find_param_in_p_or_p_global(["Q", "R"], model)
    Q = p["Q"]
    R = p["R"]

    return 0.5 * (
        ca.mtimes([ca.transpose(x), Q, x]) + ca.mtimes([ca.transpose(u), R, u])
    )


def cost_expr_ext_cost_0(model: AcadosModel) -> ca.SX:
    return cost_expr_ext_cost(model)


def cost_expr_ext_cost_e(model: AcadosModel) -> ca.SX:
    x = model.x
    Q = find_param_in_p_or_p_global(["Q"], model)["Q"]

    return 0.5 * (ca.mtimes([ca.transpose(x), Q, x]))


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    cost_type: str = "EXTERNAL",
    name: str = "pendulum_on_cart",
    learnable_param: list[str] | None = None,
    Fmax: float = 80.0,
    N_horizon: int = 50,
    tf: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = tf
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_param, ocp=ocp
    )

    ocp.model.disc_dyn_expr = disc_dyn_expr(model=ocp.model, dt=dt)

    ######## Cost ########
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = cost_type
        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model)

        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model)
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")
        # TODO: Implement NONLINEAR_LS with y_expr = sqrt(Q) * x and sqrt(R) * u

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-2.5])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp
