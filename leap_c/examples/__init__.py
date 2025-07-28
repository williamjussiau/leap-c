from functools import partial
from pathlib import Path

from .cartpole.controller import CartPoleController
from .cartpole.env import CartPoleEnv
from .chain.controller import ChainController
from .chain.env import ChainEnv
from .cylinder.controller import CylinderController
from .cylinder.env import CylinderEnv
from .hvac.controller import HvacController
from .hvac.env import StochasticThreeStateRcEnv
from .pointmass.controller import PointMassController
from .pointmass.env import PointMassEnv

ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "chain": ChainEnv,
    "pointmass": PointMassEnv,
    "hvac": StochasticThreeStateRcEnv,
    "cylinder": CylinderEnv,
}


CONTROLLER_REGISTRY = {
    "cartpole": CartPoleController,
    "cartpole_stagewise": partial(CartPoleController, stagewise=True),
    "chain": ChainController,
    "chain_stagewise": partial(ChainController, stagewise=True),
    "pointmass": PointMassController,
    "pointmass_stagewise": partial(PointMassController, stagewise=True),
    "hvac": HvacController,
    "cylinder": CylinderController,
}


def create_env(env_name: str, **kw):
    """Create an environment based on the given name."""
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name](**kw)

    raise ValueError(f"Environment '{env_name}' is not registered.")


def create_controller(
    controller_name: str,
    reuse_code_base_dir: Path | None = None,
):
    """Create a controller.

    Args:
        controller_name: Name of the controller.
        reuse_code_base_dir: Directory to reuse code base from.
    """
    if controller_name not in CONTROLLER_REGISTRY:
        raise ValueError(
            f"Controller '{controller_name}' is not registered or does not exist."
        )

    controller_class = CONTROLLER_REGISTRY[controller_name]

    if reuse_code_base_dir is not None:
        export_directory = reuse_code_base_dir / f"{controller_name}"
        try:
            return controller_class(export_directory=export_directory)
        except TypeError:
            pass

    return controller_class()
