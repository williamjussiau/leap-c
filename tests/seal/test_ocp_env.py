import pytest
from seal.examples.linear_system import LinearSystemMPC
from seal.ocp_env import OCPEnv
import numpy as np


def test_env_reset(linear_mpc):
    env = OCPEnv(linear_mpc)

    (x, p), _ = env.reset(seed=0)
    assert x is not None
    (x_, p_), _ = env.reset(seed=0)

    assert np.allclose(x, x_)


def test_env_step(linear_mpc):
    env = OCPEnv(linear_mpc)

    env.reset(seed=0)
    action = np.array([0])
    (x_next, p_next), _, _, _, _ = env.step(action)

    env.reset(seed=0)
    action = np.array([0])
    (x_next_, p_next_), _, _, _, _ = env.step(action)

    assert np.allclose(x_next, x_next_)

