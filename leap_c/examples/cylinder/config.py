from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

import control
import flowcontrol.controller as flowcontroller
import numpy as np
import utils.youla_utils as flowconyu

DEFAULT_LAGUERRE_EXPANSION_SIZE = int(2)


@dataclass(kw_only=True)
class CylinderCfg:
    Re: float = 100.0
    dt: float = 0.005


@dataclass(kw_only=True)
class YoulaControllerCfg:
    G: Optional[flowcontroller.Controller] = None
    K0: Optional[flowcontroller.Controller] = None

    def __init__(self, G=None, K0=None):
        if G is None:
            G = flowcontroller.Controller.from_file(
                file=Path(
                    "leap_c",
                    "examples",
                    "cylinder",
                    "data_input",
                    "sysid_o16_d=3_ssest.mat",
                ),
                x0=None,
            )
        if K0 is None:
            Qx = np.diag(np.ones(shape=(G.nstates,)))
            Qu = 1
            Qw = np.diag(np.ones(shape=(G.nstates,)))
            Qv = 1
            K0, _, _ = flowconyu.lqg_regulator(G=G, Qx=Qx, Qu=Qu, Qw=Qw, Qv=Qv)
            # positive feedback with G

        # K0 = flowcontroller.Controller.from_file(
        #     file=Path(
        #         "leap_c ", " examples ", " cylinder ", " data_input ", " Kopt_reduced13.mat"
        #     ),
        #     x0=None,
        # )
        self.G = G
        self.K0 = K0
        self.Ghat = flowconyu.control.feedback(self.G, self.K0, sign=+1)
        self.Ghatnorm = flowconyu.norm(self.Ghat, p=np.inf)
        self.log_rho0 = 0
        self.log_rho_scale = 2
        alpha = 10
        self.theta_scale = alpha * np.sqrt(0.5 * 10**self.log_rho0) * 1 / self.Ghatnorm


def default_ss():
    return flowcontroller.Controller.from_matrices(A=1, B=1, C=1, D=0, x0=None)


@dataclass(kw_only=True)
class CylinderParams:
<<<<<<< HEAD
    p: float  # single real pole of Laguerre basis, float
    theta: np.array  # coordinates of Q in Laguerre basis, list[float]
    # G: control.StateSpace
    # K0: flowcontroller.Controller


def make_default_cylinder_params(stagewise: bool = False) -> CylinderParams:
    """Returns a CylinderParams instance with default parameter values."""
<<<<<<< HEAD
    p = 1.0
    theta = np.zeros(
        DEFAULT_LAGUERRE_EXPANSION_SIZE,
    )
    return CylinderParams(p=p, theta=theta)


class FlowControlCtx(NamedTuple):
    """Context for FlowControl controller
    Contains the LTI controller internal state x_t"""

    controller_state: np.ndarray
    controller_order: np.ndarray

    @classmethod
    def default_flowcontrol_context(cls, controller_order):
        return cls(
            controller_order=controller_order,
            controller_state=np.zeros(
                controller_order,
            ),
        )


def collate_flowcontrol_ctx(
    batch: Sequence[FlowControlCtx],
    collate_fn_map: Optional[dict[str, Callable]] = None,
) -> FlowControlCtx:
    """Collates a batch of FlowControlCtx objects into a single object."""
    # TODO?
    nbatch = len(batch)
    nstates = batch[0].controller_state.shape[0]
    controller_order_batch = np.empty(shape=(nbatch,))
    controller_state_batch = np.empty(shape=(nbatch, nstates))
    for ctx, ii in enumerate(batch):
        controller_order_batch[ii] = ctx.controller_order
        controller_state_batch[ii, :] = ctx.controller_state

    return FlowControlCtx(
        controller_order=controller_order_batch, controller_state=controller_state_batch
    )
=======
    return CylinderParams(Re=100, p=1, theta=[0.0])
>>>>>>> fcfc07c (Redefine CylinderParams to include Controller parameters)
