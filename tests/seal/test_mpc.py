import numpy as np
import numpy.testing as npt

from seal.examples.linear_system import (
    LinearSystemMPC,
)

def test_stage_cost():
    x = np.array([0.0, 0.0])
    u = np.array([0.0])

    mpc = LinearSystemMPC()

    stage_cost = mpc.stage_cost(x, u)

    assert stage_cost == 0.0


def test_stage_cons():
    x = np.array([2.0, 1.0])
    u = np.array([0.0])

    mpc = LinearSystemMPC()

    stage_cons = mpc.stage_cons(x, u)

    npt.assert_array_equal(stage_cons["ubx"], np.array([1.0, 0.0]))


