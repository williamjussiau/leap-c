import os
import matplotlib.pyplot as plt
import pytest
import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver
from seal.examples.linear_system import (
    export_parametric_ocp,
    setup_ocp_sensitivity_solver,
    setup_ocp_solver,
    LinearSystemMPC,
)

from seal.test import (
    run_test_v_update_for_varying_parameters,
    run_test_q_update_for_varying_parameters,
    run_test_pi_update_for_varying_parameters,
    set_up_test_parameters,
)


def get_test_param():
    """
    Returns a dictionary of parameters for a linear system.

    The dictionary contains the following keys:
    - "A": State transition matrix (2x2 numpy array).
    - "B": Control input matrix (2x1 numpy array).
    - "b": Offset vector (2x1 numpy array).
    - "V_0": Initial state covariance (1-element numpy array).
    - "f": State transition noise (3x1 numpy array).
    - "Q": State-cost weight matrix (2x2 identity matrix).
    - "R": Input-cost weight matrix (1x1 identity matrix).

    Returns:
        dict: A dictionary containing the parameters of the linear system.
    """
    return {
        "A": np.array([[1.0, 0.1], [0.0, 1.0]]),
        "B": np.array([[0.0], [0.1]]),
        "b": np.array([[0.0], [0.0]]),
        "V_0": np.array([0.0]),
        "f": np.array([[0.0], [0.0], [0.0]]),
        "Q": np.eye(2),
        "R": np.eye(1),
    }


def test_export_parametric_ocp_external(param: dict[np.ndarray] = None):
    """
    Test the export of a parametric optimal control problem (OCP) with an external cost type.

    This function tests the `export_parametric_ocp` function by passing a dictionary of parameters
    and specifying the cost type as "EXTERNAL". It asserts that the returned object is an instance
    of `AcadosOcp`.

    Parameters:
    param (dict[np.ndarray]): A dictionary where the keys are parameter names and the values are
                              numpy arrays representing the parameter values.

    Asserts:
    - The returned object from `export_parametric_ocp` is an instance of `AcadosOcp`.
    """

    if param is None:
        param = get_test_param()
    ocp = export_parametric_ocp(param, cost_type="EXTERNAL")
    assert isinstance(ocp, AcadosOcp)


def get_ocp_kwargs(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
):
    """
    Generate a dictionary of keyword arguments for OCP (Optimal Control Problem) solvers.

    This function creates a dictionary containing configurations for the OCP solver and
    OCP sensitivity solver. It checks if the JSON files specified by the `json_file_prefix`
    exist. If they do, it uses the provided `generate_code` and `build_code` flags. If not,
    it sets both `generate` and `build` to True.

    Args:
        generate_code (bool): Flag to indicate whether to generate code.
        build_code (bool): Flag to indicate whether to build code.
        json_file_prefix (str): Prefix for the JSON file names.

    Returns:
        dict: A dictionary containing the build/generate configurations for the OCP solver and OCP sensitivity solver.
    """
    kwargs = {
        "ocp_solver": {"json_file": f"{json_file_prefix}.json"},
        "ocp_sensitivity_solver": {"json_file": f"{json_file_prefix}_sensitivity.json"},
    }

    for val in kwargs.values():
        if os.path.isfile(val["json_file"]):
            val["generate"] = generate_code
            val["build"] = build_code
        else:
            val["generate"] = True
            val["build"] = True

    return kwargs


def test_setup_ocp_solver_default():
    """
    Test the setup of an OCP solver with default parameters.
    """
    ocp_solver_kwargs = get_ocp_kwargs()

    ocp_solver = setup_ocp_solver(
        ocp=export_parametric_ocp(param=get_test_param(), cost_type="EXTERNAL"),
        **ocp_solver_kwargs["ocp_solver"],
    )
    assert isinstance(ocp_solver, AcadosOcpSolver)


def test_setup_ocp_sensitivity_solver_default():
    """
    Test the setup of an OCP sensitivity solver with default parameters.
    """
    ocp_sensitivity_solver_kwargs = get_ocp_kwargs()

    ocp_sensitivity_solver = setup_ocp_sensitivity_solver(
        ocp=export_parametric_ocp(param=get_test_param(), cost_type="EXTERNAL"),
        **ocp_sensitivity_solver_kwargs["ocp_sensitivity_solver"],
    )
    assert isinstance(ocp_sensitivity_solver, AcadosOcpSolver)


def set_up_mpc(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
) -> LinearSystemMPC:
    return LinearSystemMPC(
        param=get_test_param(),
        **get_ocp_kwargs(generate_code, build_code, json_file_prefix),
    )


def test_v_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_v_update_for_varying_parameters(mpc, x0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_q_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    u0: np.ndarray = np.array([0.0]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    u0 = mpc.ocp_solver.solve_for_x0(x0)

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    absolute_difference = run_test_q_update_for_varying_parameters(mpc, x0, u0, test_param, plot)

    assert np.median(absolute_difference) <= 1e-1


def test_pi_update(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, np_test, varying_param_label=varying_param_label)

    pi, dpi_dp = mpc.pi_update(x0=x0, p=test_param[:, 0])

    assert not np.isnan(dpi_dp).any(), "Gradient of pi includes NaN."


def test_closed_loop(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.5, 0.5]),
    n_sim: int = 100,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    x = [x0]
    u = []

    for _ in range(n_sim):
        u.append(mpc.ocp_solver.solve_for_x0(x[-1]))
        x.append(mpc.ocp_solver.get(1, "x"))
        assert mpc.ocp_solver.get_status() == 0

    x = np.array(x)
    u = np.array(u)

    if plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.step(np.arange(n_sim + 1), x[:, 0], label="x_0")
        plt.subplot(3, 1, 2)
        plt.step(np.arange(n_sim + 1), x[:, 1], label="x_1")
        plt.subplot(3, 1, 3)
        plt.step(np.arange(n_sim), u, label="u")
        plt.legend()
        plt.show()

    assert np.median(x[-10:, 0]) <= 1e-1 and np.median(x[-10:, 1]) <= 1e-1 and np.median(u[-10:]) <= 1e-1


def test_open_loop(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.5, 0.5]),
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    mpc.ocp_solver.solve_for_x0(x0)

    k = np.arange(mpc.ocp_solver.acados_ocp.dims.N)
    u = np.array([mpc.ocp_solver.get(stage, "u") for stage in range(mpc.ocp_solver.acados_ocp.dims.N)])
    x = np.array([mpc.ocp_solver.get(stage, "x") for stage in range(mpc.ocp_solver.acados_ocp.dims.N)])

    if plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.step(k, x[:, 0], label="x_0")
        plt.subplot(3, 1, 2)
        plt.step(k, x[:, 1], label="x_1")
        plt.subplot(3, 1, 3)
        plt.step(k, u, label="u")
        plt.legend()
        plt.show()

    assert np.median(x[-10:, 0]) <= 1e-1 and np.median(x[-10:, 1]) <= 1e-1 and np.median(u[-10:]) <= 1e-1


if __name__ == "__main__":
    test_pi_update(build_code=True, generate_code=True, np_test=100, plot=True)
    # test_closed_loop(plot=True)
