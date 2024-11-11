import numpy as np
import pytest

from seal.examples.linear_system import LinearSystemMPC


@pytest.fixture(scope="session")
def linear_mpc():
    return LinearSystemMPC()

