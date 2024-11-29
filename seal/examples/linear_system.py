"""
linear system
"""

from typing import Any

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import struct_symSX
from scipy.linalg import solve_discrete_are

from seal.mpc import MPC
from seal.ocp_env import OCPEnv

from seal.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)


class LinearSystemMPC(MPC):
    """MPC for a system with linear dynamics and quadratic cost functions.
    The equations are given by:
        Dynamics: x_next = Ax + Bu + b
        Initial_Cost: V0
        Stage_cost: J = 0.5 * (x'Qx + u'Ru + f'cat(x,u))
        Terminal Cost: x'SOLUTION_ARE(Q,R)x
        Hard constraints:
            lbx = 0, -1
            ubx = 1, 1
            lbu = -1
            ubu = 1
        Slack:
            Both entries of x are slacked linearly, i.e., the punishment in the cost is slack_weights^T*violation.
            The slack weights are [1e2, 1e2].
    """

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        discount_factor: float = 0.99,
        n_batch: int = 1,
    ):
        if params is None:
            params = {
                "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
                "B": np.array([[0.03125], [0.25]]),
                "Q": np.identity(2),
                "R": np.identity(1),
                "b": np.array([[0.0], [0.0]]),
                "f": np.array([[0.0], [0.0], [0.0]]),
                "V_0": np.array([1e-3]),
            }

        learnable_params = learnable_params if learnable_params is not None else []
        ocp = export_parametric_ocp(
            param=params, learnable_params=learnable_params, N_horizon=N_horizon
        )
        configure_ocp_solver(ocp)

        super().__init__(ocp=ocp, discount_factor=discount_factor, n_batch=n_batch)


class LinearSystemOcpEnv(OCPEnv):
    """The idea is that the linear system describes a point mass that is pushed by a force (noise)
    and the agent is required to learn to control the point mass in such a way that this force does not push
    the point mass over its boundaries (the constraints) while still minimizing the distance to the origin and
    minimizing control effort.
    """

    def __init__(
        self,
        mpc: LinearSystemMPC,
        dt: float = 0.1,
        max_time: float = 10.0,
    ):
        super().__init__(
            mpc,
            dt=dt,
            max_time=max_time,
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Execute the dynamics of the linear system and push the resulting state with a random noise."""
        o, r, term, trunc, info = super().step(
            action
        )  # o is the next state as np.ndarray, next parameters as MPCParameter
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        noise = self._np_random.uniform(-0.1, 0)
        state = o[0].copy()
        state[0] += noise
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
        return self.mpc.ocp.constraints.x0.astype(dtype=np.float32)


def disc_dyn_expr(model: AcadosModel):
    """
    Define the discrete dynamics function expression.
    """
    x = model.x
    u = model.u

    param = find_param_in_p_or_p_global(["A", "B", "b"], model)

    return param["A"] @ x + param["B"] @ u + param["b"]


def cost_expr_ext_cost(model: AcadosModel):
    """
    Define the external cost function expression.
    """
    x = model.x
    u = model.u
    param = find_param_in_p_or_p_global(["Q", "R", "f"], model)

    return 0.5 * (
        cs.transpose(x) @ param["Q"] @ x
        + cs.transpose(u) @ param["R"] @ u
        + cs.transpose(param["f"]) @ cs.vertcat(x, u)
    )


def cost_expr_ext_cost_0(model: AcadosModel):
    """
    Define the external cost function expression at stage 0.
    """
    param = find_param_in_p_or_p_global(["V_0"], model)

    return param["V_0"] + cost_expr_ext_cost(model)


def cost_expr_ext_cost_e(model: AcadosModel, param: dict[str, np.ndarray]):
    """
    Define the external cost function expression at the terminal stage as the solution of the discrete-time algebraic Riccati
    equation.
    """

    x = model.x

    return 0.5 * cs.mtimes(
        [
            cs.transpose(x),
            solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]),
            x,
        ]
    )


def configure_ocp_solver(
    ocp: AcadosOcp,
):
    ocp.solver_options.tf = ocp.solver_options.N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True

    # Set nominal parameters. Could be done at AcadosOcpSolver initialization?


def export_parametric_ocp(
    param: dict[str, np.ndarray],
    cost_type="EXTERNAL",
    name: str = "lti",
    learnable_params: list[str] | None = None,
    N_horizon=20,
) -> AcadosOcp:
    """
    Export a parametric optimal control problem (OCP) for a discrete-time linear time-invariant (LTI) system.

    Parameters:
        param:
            Dictionary containing the parameters of the system. Keys should include "A", "B", "b", "V_0", and "f".
        cost_type:
            Type of cost function to use. Options are "LINEAR_LS" or "EXTERNAL".
        name:
            Name of the model.
        learnable_params:
            List of parameters that should be learnable.
        N_horizon:
            The length of the prediction horizon.

    Returns:
        AcadosOcp
            An instance of the AcadosOcp class representing the optimal control problem.
    """
    if learnable_params is None:
        learnable_params = []
    ocp = AcadosOcp()

    ocp.model.name = name

    ocp.dims.nx = 2
    ocp.dims.nu = 1

    ocp.model.x = cs.SX.sym("x", ocp.dims.nx)  # type:ignore
    ocp.model.u = cs.SX.sym("u", ocp.dims.nu)  # type:ignore

    ocp.solver_options.N_horizon = N_horizon

    ocp = translate_learnable_param_to_p_global(
        nominal_param=param,
        learnable_param=learnable_params,
        ocp=ocp,
    )

    ocp.model.disc_dyn_expr = disc_dyn_expr(ocp.model)

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model)

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model, param)

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([-1.0, -1.0])
    ocp.constraints.ubx_0 = np.array([1.0, 1.0])

    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([-0.0, -1.0])
    ocp.constraints.ubx = np.array([+1.0, +1.0])
    ocp.constraints.x0 = np.array([0.5, 0.0])  # Default constraints.x0?

    # Only slack first dimension
    # ocp.constraints.idxsbx = np.array([0])
    # ocp.cost.zl = np.array([1e2])
    # ocp.cost.zu = np.array([1e2])
    # ocp.cost.Zl = np.diag([0])
    # ocp.cost.Zu = np.diag([0])
    # Slack both dimensions
    ocp.constraints.idxsbx = np.array([0, 1])
    ocp.cost.zl = np.ones((2,)) * 1e2
    ocp.cost.zu = np.ones((2,)) * 1e2
    ocp.cost.Zl = np.diag([0, 0])
    ocp.cost.Zu = np.diag([0, 0])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([+1.0])

    # TODO: Make a PR to acados to allow struct_symSX | struct_symMX in acados_template and then concatenate there
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []
    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp
