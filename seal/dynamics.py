from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimOptions,
    AcadosSimSolver,
)

from seal.mpc import MPC
from seal.util import AcadosFileManager


def create_dynamics_from_mpc(
    mpc: MPC, export_dir: None | Path = None, sens_forw=False, dt=None
) -> "Dynamics":
    """Create a dynamics object from an MPC object."""
    ocp: AcadosOcp = mpc.ocp

    # if there is a discrete dynamics function, use it
    if ocp.model.disc_dyn_expr is not None:
        inputs = [ocp.model.x, ocp.model.u]
        if ocp.model.p_global is not None:
            inputs.append(ocp.model.p_global)

        expr = ocp.model.disc_dyn_expr

        return CasadiDynamics(expr, inputs)

    # otherwise we create a AcadosSim object
    assert ocp.model.f_expl_expr is not None

    # create sim opts
    sim_opts = AcadosSimOptions()

    # assert sim_method_num_stages all equal
    # does this makes sense?
    sim_opts.num_steps = ocp.solver_options.sim_method_num_steps
    sim_opts.newton_iter = ocp.solver_options.sim_method_newton_iter
    sim_opts.integrator_type = ocp.solver_options.integrator_type

    # create sim
    sim = AcadosSim()
    sim.solver_options = sim_opts
    sim.solver_options.sens_forw = sens_forw
    sim.model = deepcopy(ocp.model)
    sim.parameter_values = ocp.parameter_values

    if dt is None:
        dt = ocp.solver_options.tf / ocp.solver_options.N_horizon  # type: ignore

    sim.solver_options.T = dt

    return SimDynamics(sim, export_dir=export_dir)


class Dynamics(ABC):
    @abstractmethod
    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray | None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state.
        """
        ...


class SimDynamics(Dynamics):
    def __init__(
        self, sim: AcadosSim, export_dir: Path | None = None, cleanup: bool = True
    ):
        self.sim = sim
        afm = AcadosFileManager(export_dir, cleanup=cleanup)
        self._sim_solver_fn = partial(afm.setup_acados_sim_solver, sim)
        self._sim_solver = None

    @property
    def sim_solver(self) -> AcadosSimSolver:
        if self._sim_solver is None:
            self._sim_solver = self._sim_solver_fn()
        return self._sim_solver

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray | None = None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state and the sensitivity if with_sens is True.
        """
        self.sim_solver.set("x", x)
        self.sim_solver.set("u", u)

        if p is not None:
            self.sim_solver.set("p", p)

        self.sim_solver.solve()

        if not with_sens:
            return self.sim_solver.get("x")  # type: ignore

        Sx = self.sim_solver.get("Sx")
        Su = self.sim_solver.get("Su")

        return self.sim_solver.get("x"), Sx, Su  # type: ignore


class CasadiDynamics(Dynamics):
    def __init__(self, expr: ca.SX, inputs: list[ca.SX]):
        self.expr = expr
        self.inputs = inputs

        self.dyn_fn = ca.Function(
            "dyn", inputs, [expr], ["x", "u", "p"], ["x_next"]
        )

        # generate the jacobian
        inputs_cat = ca.vertcat(*inputs)
        jac_expr = ca.jacobian(expr, inputs_cat)

        self.jac_fn = ca.Function("jtimes", [inputs_cat], [jac_expr])

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray | None = None,
        with_sens: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step the dynamics.

        Args:
            x: The current state.
            u: The current input.
            p: The current parameter.
            with_sens: Whether to return the sensitivity.

        Returns:
            The next state.
        """
        x_shape = x.shape

        inputs = [x, u]

        if p is not None:
            inputs.append(p)

        # transpose inputs to match casadi batch format
        inputs = [i.T for i in inputs]

        output = self.dyn_fn(*inputs).full().T.reshape(x_shape)  # type: ignore

        if not with_sens:
            return output

        # compute the jacobian
        sizes = [i.shape[0] for i in inputs]
        splits = np.cumsum(sizes)[:-1]

        inputs_cat = np.concatenate(inputs, axis=0)

        jac = self.jac_fn(inputs_cat).full().T  # type: ignore
        if len(x_shape) > 1:
            jac = jac.reshape(x_shape[0], -1, x_shape[1])
            Sx, Su, _ = np.split(jac, splits, axis=1)
        else:
            Sx, Su, _ = np.split(jac, splits, axis=0)

        return output, Sx, Su

