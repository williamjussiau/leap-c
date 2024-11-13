import numpy as np
import numpy.testing as npt
from seal.mpc import MPC
import pytest
from seal.test import (
    run_test_policy_for_varying_parameters,
)


def test_stage_cost(linear_mpc: MPC):
    x = np.array([0.0, 0.0])
    u = np.array([0.0])

    stage_cost = linear_mpc.stage_cost(x, u)

    assert stage_cost == 0.0


def test_stage_cons(linear_mpc: MPC):
    x = np.array([2.0, 1.0])
    u = np.array([0.0])

    stage_cons = linear_mpc.stage_cons(x, u)

    npt.assert_array_equal(stage_cons["ubx"], np.array([1.0, 0.0]))


def test_policy_gradient_via_adjoint_sensitivity(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params,
):
    absolute_difference = run_test_policy_for_varying_parameters(
        mpc=learnable_linear_mpc,
        x0=np.array([0.1, 0.1]),
        test_param=linear_mpc_test_params,
        use_adj_sens=True,
        plot=False,
    )

    assert np.median(absolute_difference) <= 1e-1


if __name__ == "__main__":
    pytest.main([__file__])
