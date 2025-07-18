from dataclasses import asdict
from typing import Any, Optional

import control
import flowcontrol.controller as flowcon
import gymnasium as gym
import numpy as np
import torch
import utils.youla_utils as yu

from leap_c.controller import ParameterizedController
from leap_c.examples.cylinder.config import CylinderParams, make_default_cylinder_param


class CylinderController(ParameterizedController):
    """
    LTI controller for the flow past a cylinder
    Youla parametrization with Laguerre basis expansion
    """

    def __init__(
        self,
        params: Optional[CylinderParams] = None,
        G: Optional[control.StateSpace] = None,
        K0: Optional[flowcon.Controller] = None,
        N_expansion: int = 4,
    ):
        """
        Args:
            G [control.StateSpace]: ROM of linearised flow
            K0 [flowcontrol.controller.Controller]: initial controller stabilizing G
            N_expansion [int]: size of Laguerre expansion
        """
        super().__init__()
        self.params = make_default_cylinder_param() if params is None else params
        tuple_params = tuple(asdict(self.params).values())

        self.G = G
        self.K0 = K0
        # Make Youla controller as flowcon.Controller
        Ky = yu.youla_laguerre(G, K0, p=1, theta=[0, 0])  ### make Youla controller
        self.Ky = flowcon.Controller.from_matrices(Ky.A, Ky.B, Ky.C, Ky.D, x0=None)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        # x0 = torch.as_tensor(obs, dtype=torch.float64)
        # p_global = torch.as_tensor(param, dtype=torch.float64)
        # ctx, u0, x, u, value = self.diff_mpc(
        #     x0.unsqueeze(0), p_global=p_global.unsqueeze(0), ctx=ctx
        # )
        u0 = self.Ky.step(y=obs, dt=0.005)  ####### get from param?
        # how is the internal state managed? Is it reset sometimes?
        ctx = None
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    def param_space(self) -> gym.Space:
        # TODO: can't determine the param space because it depends on the learnable parameters
        # we need to define boundaries for every parameter and based on that create a gym.Space
        raise NotImplementedError

    def default_param(self) -> np.ndarray:
        return np.concatenate(
            [asdict(self.params)[p].flatten() for p in self.learnable_params]
        )
