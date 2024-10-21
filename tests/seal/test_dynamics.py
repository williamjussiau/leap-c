import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from seal.mpc import MPC
from seal.dynamics import create_dynamics_from_mpc


def test_casadi_dynamics(linear_mpc: MPC):
    dynamics = create_dynamics_from_mpc(linear_mpc, "dynamics")

    # test forward single sample
    x = np.array([0.5, 0.5])
    u = np.array([0.5])
    p = linear_mpc.default_param

    x_next = dynamics(x, u, p)

    # test forward multiple samples
    x = np.array([[0.5, 0.5], [0.3, 0.2]])
    u = np.array([[0.5], [0.2]])
    p = np.stack([linear_mpc.default_param, linear_mpc.default_param])

    x_next_batch = dynamics(x, u, p)

    assert_almost_equal(x_next, x_next_batch[0])

    # test jacobian
    x = np.array([0.5, 0.5])
    u = np.array([0.5])
    p = linear_mpc.default_param

    x_next, Sx, Su = dynamics(x, u, p, with_sens=True)

    # test jacobian multiple samples
    x = np.array([[0.5, 0.5], [0.3, 0.2]])
    u = np.array([[0.5], [0.2]])
    p = np.stack([linear_mpc.default_param, linear_mpc.default_param])

    x_next_batch, Sx_batch, Su_batch = dynamics(x, u, p, with_sens=True)

    assert_almost_equal(x_next, x_next_batch[0])
    assert_almost_equal(Sx, Sx_batch[0])
    assert_almost_equal(Su, Su_batch[0])
