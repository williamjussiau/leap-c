"""
linear system
"""

import numpy as np
import casadi as cs
from scipy import linalg
from scipy.linalg import solve_discrete_are
from acados_template import AcadosOcp
from seal.mpc import MPC


class LinearSystemMPC(MPC):
    def __init__(self, param: dict[str, np.ndarray] | None = None, discount_factor: float | None = None):
        if param is None:
            param = {
                "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
                "B": np.array([[0.03125], [0.25]]),
                "Q": np.identity(2),
                "R": np.identity(1),
                "b": np.array([[0.0], [0.0]]),
                "f": np.array([[0.0], [0.0], [0.0]]),
                "V_0": np.array([1e-3]),
            }

        ocp = export_parametric_ocp(param)
        configure_ocp_solver(ocp)

        super().__init__(ocp, discount_factor)


def disc_dyn_expr(x, u, param):
    """
    Define the discrete dynamics function expression.
    """
    return param["A"] @ x + param["B"] @ u + param["b"]


def cost_expr_ext_cost(x, u, p):
    """
    Define the external cost function expression.
    """
    y = cs.vertcat(x, u)
    return 0.5 * (cs.mtimes([y.T, y])) + cs.mtimes([get_parameter("f", p).T, y])


def cost_expr_ext_cost_0(x, u, p):
    """
    Define the external cost function expression at stage 0.
    """
    return get_parameter("V_0", p) + cost_expr_ext_cost(x, u, p)


def cost_expr_ext_cost_e(x, param, N):
    """
    Define the external cost function expression at the terminal stage as the solution of the discrete-time algebraic Riccati
    equation.
    """

    return 0.5 * cs.mtimes(
        [x.T, solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]), x]
    )


def get_parameter(field, p) -> cs.DM:
    if field == "A":
        return cs.reshape(p[:4], 2, 2)
    elif field == "B":
        return cs.reshape(p[4:6], 2, 1)
    elif field == "b":
        return cs.reshape(p[6:8], 2, 1)
    elif field == "V_0":
        return p[8]
    elif field == "f":
        return cs.reshape(p[9:12], 3, 1)


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
    ocp.solver_options.with_solution_sens_wrt_params = True

    # Set nominal parameters. Could be done at AcadosOcpSolver initialization?


def export_parametric_ocp(
    param: dict,
    cost_type="EXTERNAL",
    name: str = "lti",
) -> AcadosOcp:
    """
    Export a parametric optimal control problem (OCP) for a discrete-time linear time-invariant (LTI) system.

    Parameters:
    -----------
    param : dict
        Dictionary containing the parameters of the system. Keys should include "A", "B", "b", "V_0", and "f".
    cost_type : str, optional
        Type of cost function to use. Options are "LINEAR_LS" or "EXTERNAL".
    name : str, optional
        Name of the model.

    Returns:
    --------
    AcadosOcp
        An instance of the AcadosOcp class representing the optimal control problem.
    """
    ocp = AcadosOcp()

    ocp.model.name = name

    ocp.model.x = cs.SX.sym("x", 2)
    ocp.model.u = cs.SX.sym("u", 1)

    ocp.solver_options.N_horizon = 40
    ocp.dims.nx = 2
    ocp.dims.nu = 1

    A = cs.SX.sym("A", 2, 2)
    B = cs.SX.sym("B", 2, 1)
    b = cs.SX.sym("b", 2, 1)
    V_0 = cs.SX.sym("V_0", 1, 1)
    f = cs.SX.sym("f", 3, 1)

    ocp.model.p_global = cs.vertcat(
        cs.reshape(A, -1, 1),
        cs.reshape(B, -1, 1),
        cs.reshape(b, -1, 1),
        V_0,
        cs.reshape(f, -1, 1),
    )

    ocp.p_global_values = np.concatenate(
        [param[key].T.reshape(-1, 1) for key in ["A", "B", "b", "V_0", "f"]]
    ).flatten()

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u + b

    if cost_type == "LINEAR_LS":
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.Vx_0 = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nx))
        ocp.cost.Vx_0[: ocp.dims.nx, : ocp.dims.nx] = np.identity(ocp.dims.nx)
        ocp.cost.Vu_0 = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nu))
        ocp.cost.Vu_0[-1, -1] = 1

        ocp.cost.Vx = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nx))
        ocp.cost.Vx[: ocp.dims.nx, : ocp.dims.nx] = np.identity(ocp.dims.nx)
        ocp.cost.Vu = np.zeros((ocp.dims.nx + ocp.dims.nu, ocp.dims.nu))
        ocp.cost.Vu[-1, -1] = 1
        ocp.cost.Vx_e = np.identity(ocp.dims.nx)

        ocp.cost.W_0 = linalg.block_diag(param["Q"], param["R"])
        ocp.cost.W = linalg.block_diag(param["Q"], param["R"])
        ocp.cost.W_e = param["Q"]

        ocp.cost.yref_0 = np.zeros(ocp.dims.nx + ocp.dims.nu)
        ocp.cost.yref = np.zeros(ocp.dims.nx + ocp.dims.nu)
        ocp.cost.yref_e = np.zeros(ocp.dims.nx)

    elif cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(
            ocp.model.x, ocp.model.u, ocp.model.p_global
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(
            ocp.model.x, ocp.model.u, ocp.model.p_global
        )

        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(
            ocp.model.x, param, ocp.solver_options.N_horizon
        )

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([-1.0, -1.0])
    ocp.constraints.ubx_0 = np.array([1.0, 1.0])

    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([-0.0, -1.0])
    ocp.constraints.ubx = np.array([+1.0, +1.0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = np.array([1e2])
    ocp.cost.zu = np.array([1e2])
    ocp.cost.Zl = np.diag([0])
    ocp.cost.Zu = np.diag([0])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([+1.0])

    return ocp
