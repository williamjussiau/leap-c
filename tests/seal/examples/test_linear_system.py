import matplotlib.pyplot as plt
import numpy as np
import pytest
from acados_template import AcadosOcp

from seal.examples.linear_system import (
    LinearSystemMPC,
    export_parametric_ocp,
)
from seal.test import (
    run_test_policy_for_varying_parameters,
    run_test_state_action_value_for_varying_parameters,
    run_test_state_value_for_varying_parameters,
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
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
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


def set_up_mpc(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    learnable_params: list[str] = [],
) -> LinearSystemMPC:
    return LinearSystemMPC(params=get_test_param(), learnable_params=learnable_params)


def test_state_value(
    generate_code: bool = True,
    build_code: bool = True,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    learnable_params: list[str] = ["A", "B", "Q", "R", "b", "f", "V_0"],
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix, learnable_params)

    test_param = set_up_test_parameters(
        mpc, np_test, varying_param_label=varying_param_label
    )

    absolute_difference = run_test_state_value_for_varying_parameters(
        mpc, x0, test_param, plot
    )

    assert np.median(absolute_difference) <= 1e-1


def test_state_action_value(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    u0: np.ndarray = np.array([0.0]),
    varying_param_label: str = "A_0",
    learnable_params: list[str] = ["A", "B", "Q", "R", "b", "f", "V_0"],
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix, learnable_params)

    u0 = mpc.ocp_solver.solve_for_x0(x0)

    test_param = set_up_test_parameters(
        mpc, np_test, varying_param_label=varying_param_label
    )

    absolute_difference = run_test_state_action_value_for_varying_parameters(
        mpc, x0, u0, test_param, plot
    )

    assert np.median(absolute_difference) <= 1e-1


def test_policy(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.1, 0.1]),
    varying_param_label: str = "A_0",
    learnable_params: list[str] = ["A", "B", "Q", "R", "b", "f", "V_0"],
    np_test: int = 10,
    plot: bool = False,
):
    mpc = set_up_mpc(
        generate_code=generate_code,
        build_code=build_code,
        json_file_prefix=json_file_prefix,
        learnable_params=learnable_params,
    )

    test_param = set_up_test_parameters(
        mpc, np_test, varying_param_label=varying_param_label
    )

    absolute_difference = run_test_policy_for_varying_parameters(
        mpc=mpc,
        x0=x0,
        test_param=test_param,
        use_adj_sens=False,
        plot=plot,
    )

    assert np.median(absolute_difference) <= 1e-1


def test_closed_loop(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.5, 0.5]),
    learnable_params: list[str] = ["A", "B", "Q", "R", "b", "f", "V_0"],
    n_sim: int = 100,
    plot: bool = False,
):
    mpc = set_up_mpc(
        generate_code=generate_code,
        build_code=build_code,
        json_file_prefix=json_file_prefix,
        learnable_params=learnable_params,
    )

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

    assert (
        np.median(x[-10:, 0]) <= 1e-1
        and np.median(x[-10:, 1]) <= 1e-1
        and np.median(u[-10:]) <= 1e-1
    )


def test_open_loop(
    generate_code: bool = False,
    build_code: bool = False,
    json_file_prefix: str = "acados_ocp_linear_system",
    x0: np.ndarray = np.array([0.5, 0.5]),
    plot: bool = False,
):
    mpc = set_up_mpc(
        generate_code, build_code, json_file_prefix, learnable_params=["A", "b", "V_0"]
    )

    mpc.ocp_solver.solve_for_x0(x0)

    k = np.arange(mpc.ocp_solver.acados_ocp.solver_options.N_horizon)
    u = np.array(
        [
            mpc.ocp_solver.get(stage, "u")
            for stage in range(mpc.ocp_solver.acados_ocp.solver_options.N_horizon)
        ]
    )
    x = np.array(
        [
            mpc.ocp_solver.get(stage, "x")
            for stage in range(mpc.ocp_solver.acados_ocp.solver_options.N_horizon)
        ]
    )

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

    assert (
        np.median(x[-10:, 0]) <= 1e-1
        and np.median(x[-10:, 1]) <= 1e-1
        and np.median(u[-10:]) <= 1e-1
    )


if __name__ == "__main__":
    pytest.main([__file__])
