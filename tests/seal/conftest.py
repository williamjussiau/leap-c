import numpy as np
import pytest

from seal.examples.linear_system import LinearSystemMPC
from seal.test import set_up_test_parameters


@pytest.fixture(scope="session")
def linear_mpc():
    return LinearSystemMPC()


@pytest.fixture(scope="session")
def learnable_linear_mpc():
    return LinearSystemMPC(learnable_params=["A", "B", "Q", "R", "b", "f", "V_0"])


@pytest.fixture(scope="session")
def linear_mpc_test_params(learnable_linear_mpc):
    test_param = set_up_test_parameters(
        learnable_linear_mpc, 10, varying_param_label="A_0"
    )

    return test_param
