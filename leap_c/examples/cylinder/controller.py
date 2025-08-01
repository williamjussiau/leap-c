from typing import Any, Optional

import flowcontrol.controller as flowcon
import gymnasium as gym
import numpy as np
import torch
import utils.youla_utils as flowconyu

from leap_c.controller import ParameterizedController
from leap_c.examples.cylinder.config import (
    DEFAULT_LAGUERRE_EXPANSION_SIZE,
    CylinderCfg,
    CylinderParams,
    FlowControlCtx,
    YoulaControllerCfg,
    collate_flowcontrol_ctx,
    make_default_cylinder_params,
)


class CylinderController(ParameterizedController):
    collate_fn_map = {FlowControlCtx: collate_flowcontrol_ctx}

    def __init__(
        self,
        params: Optional[CylinderParams] = None,
        cylinderConfig: Optional[CylinderCfg] = None,
        youlaControllerConfig: Optional[YoulaControllerCfg] = None,
        N_expansion: int = DEFAULT_LAGUERRE_EXPANSION_SIZE,
        log_rho0: float = 0,
        stagewise: bool = False,
    ):
        """
        Args:
            params: A dict with the parameters of the ocp, together with their default values.
                For a description of the parameters, see the docstring of the class.
            G
            K0
            N_expansion
        """
        super().__init__()
        self.params = (
            make_default_cylinder_params(stagewise=stagewise)
            if params is None
            else params
        )

        if youlaControllerConfig is None:
            youlaControllerConfig = YoulaControllerCfg()

        if cylinderConfig is None:
            cylinderConfig = CylinderCfg()

        self.youlaControllerConfig = youlaControllerConfig
        self.cylinderConfig = cylinderConfig
        self.controller_order = (
            2 * self.youlaControllerConfig.K0.nstates
            + self.youlaControllerConfig.G.nstates
            + N_expansion
        )
        self.N_expansion = N_expansion
        self.log_rho0 = log_rho0

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        # No batch
        assert obs.shape[0] == param.shape[0] == 1

        if ctx is None:
            ctx = FlowControlCtx.default_flowcontrol_context(
                controller_order=self.controller_order
            )

        # here: log-transform p if necessary TODO
        p_pow = 10 ** param[0][0]
        Ky = flowconyu.youla_laguerre(
            G=self.youlaControllerConfig.G,
            K0=self.youlaControllerConfig.K0,
            p=p_pow,
            theta=param[0][1:],
        )
        Ky = flowcon.Controller.from_matrices(
            Ky.A, Ky.B, Ky.C, Ky.D, x0=ctx.controller_state
        )

        # no batch yet
        # index depends on where is feedback sensor
        u0 = Ky.step(y=np.asarray(obs[0, 0]), dt=self.cylinderConfig.dt)
        ctx = FlowControlCtx(
            controller_order=self.controller_order, controller_state=Ky.x
        )
        # TODO try saturation

        return ctx, torch.from_numpy(u0)

    # def jacobian_action_param(self, ctx) -> np.ndarray:
    #     return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")
    ## keep unimplemented for now

    @property
    def param_space(self) -> gym.Space:
        # theta
        theta_scale = self.youlaControllerConfig.theta_scale / self.N_expansion
        low = -theta_scale * np.ones(
            1 + self.N_expansion,
        )
        high = theta_scale * np.ones(
            1 + self.N_expansion,
        )
        # rho (log transformed)
        low[0] = (
            self.youlaControllerConfig.log_rho0
            - self.youlaControllerConfig.log_rho_scale
        )
        high[0] = (
            self.youlaControllerConfig.log_rho0
            + self.youlaControllerConfig.log_rho_scale
        )
        low[0] = self.log_rho0 - 2  # log(rho)
        high[0] = self.log_rho0 + 2  # log(rho)
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)  # type:ignore

    @property
    def default_param(self) -> np.ndarray:
        defpar = np.concatenate((np.array([1.0]), np.zeros(self.N_expansion)), axis=0)
        return defpar
