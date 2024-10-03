"""
linear system
"""

import numpy as np
import casadi as cs
from scipy import linalg
from scipy.linalg import solve_discrete_are
from acados_template import AcadosOcp, AcadosOcpSolver
from seal.mpc import MPC


class LinearSystemMPC(MPC):
    """docstring for MPC."""

    ocp: AcadosOcp
    ocp_solver: AcadosOcpSolver
    ocp_sensitivity_solver: AcadosOcpSolver

    def __init__(self, param, discount_factor=0.99, **kwargs):
        super(LinearSystemMPC, self).__init__()

        ocp_solver_kwargs = kwargs["ocp_solver"] if "ocp_solver" in kwargs else {}

        ocp_sensitivity_solver_kwargs = kwargs["ocp_sensitivity_solver"] if "ocp_sensitivity_solver" in kwargs else {}

        self.ocp = export_parametric_ocp(param)

        self.ocp_solver = setup_ocp_solver(self.ocp, **ocp_solver_kwargs)
        self.ocp_sensitivity_solver = setup_ocp_sensitivity_solver(self.ocp, **ocp_sensitivity_solver_kwargs)


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

    return 0.5 * cs.mtimes([x.T, solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]), x])


def get_parameter(field, p) -> cs.DM:
    """
    Retrieves and reshapes a parameter vector based on the specified field.

    Args:
        field (str): The field name indicating which parameter to retrieve.
                     Possible values are "A", "B", "b", "V_0", and "f".
        p (cs.DM): A CasADi DM vector containing the parameters.

    Returns:
        cs.SX: The reshaped parameter corresponding to the specified field.
    """
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


def setup_ocp_solver(
    ocp: AcadosOcp,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    hessian_approx: str = "GAUSS_NEWTON",
    integrator_type: str = "DISCRETE",
    nlp_solver_type: str = "SQP",
    qp_solver_ric_alg: int = 0,
    **kwargs,
) -> AcadosOcpSolver:
    """
    Set up an Optimal Control Problem (OCP) solver using the provided parameters.

    Args:
        param (dict): Dictionary containing the nominal parameters for the OCP.
        qp_solver (str, optional): Quadratic programming solver to use.
        hessian_approx (str, optional): Method for Hessian approximation.
        integrator_type (str, optional): Type of integrator to use.
        nlp_solver_type (str, optional): Type of Nonlinear Programming (NLP) solver.
        name (str, optional): Name of the OCP.
        **kwargs: Additional keyword arguments for the OCP solver.

    Returns:
        AcadosOcpSolver: An instance of the AcadosOcpSolver configured with the provided parameters.
    """

    ocp.solver_options.tf = ocp.dims.N
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    ocp.solver_options.hessian_approx = hessian_approx
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.qp_solver_ric_alg = qp_solver_ric_alg
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True

    ocp_solver = AcadosOcpSolver(ocp, **kwargs)
    # Set nominal parameters. Could be done at AcadosOcpSolver initialization?
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp_solver.acados_ocp.parameter_values)

    return ocp_solver


def setup_ocp_sensitivity_solver(
    ocp: AcadosOcp,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    integrator_type: str = "DISCRETE",
    nlp_solver_type: str = "SQP",
    **kwargs,
) -> AcadosOcpSolver:
    ocp.model.name = f"{ocp.model.name}_sensitivity"

    ocp.solver_options.tf = ocp.dims.N
    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True

    ocp_solver = AcadosOcpSolver(ocp, **kwargs)
    # Set nominal parameters. Could be done at AcadosOcpSolver initialization?
    for stage in range(ocp_solver.acados_ocp.dims.N + 1):
        ocp_solver.set(stage, "p", ocp_solver.acados_ocp.parameter_values)

    return ocp_solver


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

    ocp.dims.N = 40
    ocp.dims.nx = 2
    ocp.dims.nu = 1

    A = cs.SX.sym("A", 2, 2)
    B = cs.SX.sym("B", 2, 1)
    b = cs.SX.sym("b", 2, 1)
    V_0 = cs.SX.sym("V_0", 1, 1)
    f = cs.SX.sym("f", 3, 1)

    ocp.model.p = cs.vertcat(
        cs.reshape(A, -1, 1),
        cs.reshape(B, -1, 1),
        cs.reshape(b, -1, 1),
        V_0,
        cs.reshape(f, -1, 1),
    )

    ocp.parameter_values = np.concatenate([param[key].T.reshape(-1, 1) for key in ["A", "B", "b", "V_0", "f"]])

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u + b
    # ocp.model.disc_dyn_expr = param["A"] @ ocp.model.x + param["B"] @ ocp.model.u + param["b"]

    # f_disc = cs.Function("f", [ocp.model.x, ocp.model.u], [ocp.model.disc_dyn_expr])

    # print(f_disc(np.array([0.5, 0.5], 0)))

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

    # :math:`l(x,u,z) = 0.5 \cdot || V_x \, x + V_u \, u + V_z \, z - y_\\text{ref}||^2_W`,

    elif cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model.x, ocp.model.u, ocp.model.p)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model.x, ocp.model.u, ocp.model.p)

        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model.x, param, ocp.dims.N)

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
