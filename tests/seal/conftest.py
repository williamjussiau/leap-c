import pytest

from seal.examples.linear_system import LinearSystemMPC

from seal.mpc import MPC

import numpy as np

from seal.util import SX_to_labels, find_idx_for_labels


def set_up_test_parameters(
    mpc: MPC,
    np_test: int = 10,
    scale_low: float = 0.9,
    scale_high: float = 1.1,
    varying_param_label="A_0",
) -> np.ndarray:
    p_global_values = mpc.ocp_solver.acados_ocp.p_global_values

    test_param = np.repeat(p_global_values, np_test).reshape(len(p_global_values), -1)

    # Vary parameter along one dimension of p_label
    p_idx = find_idx_for_labels(
        mpc.ocp_solver.acados_ocp.model.p_global, varying_param_label
    )[0]
    test_param[p_idx, :] = np.linspace(
        scale_low * p_global_values[p_idx],
        scale_high * p_global_values[p_idx],
        np_test,
    ).flatten()

    return test_param


@pytest.fixture(scope="session")
def linear_mpc():
    return LinearSystemMPC()


@pytest.fixture(scope="session")
def learnable_linear_mpc() -> LinearSystemMPC:
    return LinearSystemMPC(learnable_params=["A", "B", "Q", "R", "b", "f", "V_0"])


@pytest.fixture(scope="session")
def linear_mpc_test_params(learnable_linear_mpc) -> list[np.ndarray]:
    """Set up test parameters for the linear system MPC.
    Only vary the parameters in p_global that are not zero."""
    params = []
    for val, label in zip(
        learnable_linear_mpc.ocp_solver.acados_ocp.p_global_values,
        SX_to_labels(learnable_linear_mpc.ocp_solver.acados_ocp.model.p_global),
    ):
        if abs(val) > 1e-6:
            params.append(
                set_up_test_parameters(
                    learnable_linear_mpc, 10, varying_param_label=label
                )
            )

    return params
