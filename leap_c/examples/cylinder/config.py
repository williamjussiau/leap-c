from dataclasses import dataclass


@dataclass(kw_only=True)
class CylinderParams:
    Re: float
    # W: What is this supposed to be? Is it the parameters of the Cylinder configuration,
    # or the parameters for the controller for the Cylinder? It seems to be used only
    # in controller.py, so I would go for the 2nd option. In that case, do not use Re
    # but rather the parameters of the controller like N_expansion...
    p: int
    theta: list


def make_default_cylinder_params() -> CylinderParams:
    """Returns a CylinderParams instance with default parameter values."""
    return CylinderParams(Re=100, p=1, theta=[0.0])
