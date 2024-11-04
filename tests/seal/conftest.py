import numpy as np
import pytest

from seal.examples.linear_system import LinearSystemMPC
from seal.ocp_env import ConstantParamCreator


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


@pytest.fixture(scope="session")
def linear_mpc():
    return LinearSystemMPC(get_test_param())

@pytest.fixture(scope="session")
def linear_system_default_param_creator():
    params = get_test_param()
    params.pop("Q")
    params.pop("R")
    default_params = np.concatenate([val.flatten() for key, val in params.items()])
    return ConstantParamCreator(default_params)