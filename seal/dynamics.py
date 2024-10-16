import numpy as np
from abc import ABC, abstractmethod

from pathlib import Path
from acados_template import (
    AcadosOcp,
    AcadosSim,
    AcadosSimOpts,
    AcadosSimSolver,
    AcadosOcpOptions,
)
import casadi as ca

from seal.mpc import MPC


def create_dynamics_from_mpc(
    mpc: MPC, export_dir, sens_forw=False, dt=None
) -> "Dynamics":
    """Create a dynamics object from an MPC object."""
    ocp: AcadosOcp = mpc.ocp

    # if there is a discrete dynamics function, use it
    if ocp.model.disc_dyn_expr is not None:
        ca_dyn_fn = ocp.model.disc_dyn_expr
        ca_dyn_fn = ca.Function(
            "disc_dyn",
            [ocp.model.x, ocp.model.u, ocp.model.p],
            [ca_dyn_fn],
            ["x", "u", "p"],
            ["x_next"],
        )
        return CasadiDynamics(ca_dyn_fn)

    # otherwise we create a AcadosSim object
    assert ocp.model.f_expl_expr is not None

    # create sim opts
    ocp_opts = AcadosOcpOptions()
    sim_opts = AcadosSimOpts()

    # assert sim_method_num_stages all equal
    if isinstance(ocp_opts.sim_method_num_stages, np.ndarray):
        assert len(set(ocp_opts.sim_method_num_stages)) == 1
        sim_opts.num_stages = int(ocp_opts.sim_method_num_stages[0])
    else:
        sim_opts.num_stages = ocp_opts.sim_method_num_stages
    if isinstance(ocp_opts.sim_method_num_steps, np.ndarray):
        assert len(set(ocp_opts.sim_method_num_steps)) == 1
        sim_opts.num_steps = int(ocp_opts.sim_method_num_steps[0])
    else:
        sim_opts.num_steps = ocp_opts.sim_method_num_steps
    sim_opts.newton_iter = ocp_opts.sim_method_newton_iter
    sim_opts.integrator_type = ocp_opts.integrator_type

    # create sim
    sim = AcadosSim()
    sim.solver_options = sim_opts
    sim.solver_options.sens_forw = sens_forw
    sim.code_export_directory = str(export_dir.absolute())
    sim.model = ocp.model
    sim.parameter_values = ocp.parameter_values

    if dt is None:
        dt = ocp.solver_options.tf / ocp.dims.N  # type: ignore

    sim.solver_options.T = dt

    return SimDynamics(sim)


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
    def __init__(self, sim: AcadosSim):
        self.sim = sim
        self.sim_solver = None

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
            The next state and the sensitivity if with_sens is True.
        """
        if self.sim_solver is None:
            self.sim_solver = AcadosSimSolver(
                self.sim, str(Path(self.sim.code_export_directory) / "acados_ocp.json")
            )

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
    def __init__(self, ca_dyn_fn: ca.Function):
        self.ca_dyn_fn = ca_dyn_fn

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

        inputs = [x, u]

        if p is not None:
            inputs.append(p)

        if not with_sens:
            return self.ca_dyn_fn(*inputs).full()  # type: ignore

        return self.ca_dyn_fn(*inputs).full()  # type: ignore
