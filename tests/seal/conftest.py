import numpy as np
import pytest

from seal.examples.linear_system import LinearSystemMPC
from seal.mpc import MPC
from seal.util import find_idx_for_labels


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
def n_batch() -> int:
    return 4


@pytest.fixture(scope="session")
def learnable_linear_mpc(n_batch: int) -> LinearSystemMPC:
    return LinearSystemMPC(
        learnable_params=["A", "B", "Q", "R", "b", "f", "V_0"], n_batch=n_batch
    )


@pytest.fixture(scope="session")
def linear_mpc_test_params(
    learnable_linear_mpc: MPC, n_batch: int, delta: float = 0.05
) -> np.ndarray:
    """Set up test parameters for the linear system MPC.

    Args:
        learnable_linear_mpc (MPC): Linear system MPC with learnable parameters.
        n_batch (int): Number of test parameters to generate.
        delta (float, optional): Relative distance from nominal value.

    Returns:
    --------
    np.ndarray (n_batch, n_p_global, n_p_global): Test parameters for the linear system MPC where n_p_global is the number of
    global parameters. The last axis determines the parameter that is varied.
    """

    nominal = learnable_linear_mpc.ocp_solver.acados_ocp.p_global_values

    width = np.array([delta * p if np.abs(p) > 1e-6 else delta for p in nominal])

    # repeat mean into an array with shape (n_batch, n_param)
    params = np.tile(nominal, (n_batch, 1))

    # repeat mean along a third axis
    params = np.repeat(params[:, :, np.newaxis], nominal.shape[0], axis=2)

    for i, idx in enumerate(np.arange(len(nominal))):
        params[:, idx, i] = np.linspace(
            nominal[idx] - width[idx], nominal[idx] + width[idx], n_batch
        )

    return params
