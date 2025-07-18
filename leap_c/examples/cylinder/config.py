from dataclasses import dataclass


@dataclass(kw_only=True)
class CylinderParams:
    Re: float


def make_default_cylinder_params() -> CylinderParams:
    """Returns a CylinderParams instance with default parameter values."""
    return CylinderParams(Re=100)
