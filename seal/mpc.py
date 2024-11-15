from abc import ABC
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Callable, List, NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)

from seal.util import AcadosFileManager

# def set_discount_factor(ocp_solver: AcadosOcpSolver, discount_factor: float) -> None:
#     for stage in range(0, ocp_solver.acados_ocp.dims.N + 1):  # type:ignore
#         ocp_solver.cost_set(stage, "scaling", discount_factor**stage)


class MPCParameter(NamedTuple):
    """
    A named tuple to store the parameters of the MPC planner.

    Attributes:
        p_global: The part of p_global that should be learned in shape (n_p_global_learnable, ) or (B, n_p_global_learnable).
        p_stagewise: The stagewise parameters in shape
            (N+1, p_stagewise_dim) or (N+1, len(p_stagewise_sparse_idx)) if the next field is set or
            (B, N+1, p_stagewise_dim) or (B, N+1, len(p_stagewise_sparse_idx)) if the next field is set.
            If a multi-phase MPC is used this is a list containing the above arrays for the respective phases.
        p_stagewise_sparse_idx: If not None, stagewise parameters are set in a sparse manner, using these indices.
            The indices are in shape (N+1, n_p_stagewise_sparse_idx) or (B, N+1, n_p_stagewise_sparse_idx).
            If a multi-phase MPC is used this is a list containing the above arrays for the respective phases.
    """

    p_global: np.ndarray | None = None
    p_stagewise: List[np.ndarray] | np.ndarray | None = None
    p_stagewise_sparse_idx: List[np.ndarray] | np.ndarray | None = None


MPCState = AcadosOcpFlattenedIterate


class MPCInput(NamedTuple):
    """
    A named tuple to store the input of the MPC planner.

    Attributes:
        x0: The initial states in shape (B, x_dim) or (x_dim, ).
        u0: The initial actions in shape (B, u_dim) or (u_dim, ).
        parameters: The parameters of the MPC planner.
    """

    x0: np.ndarray
    u0: np.ndarray | None = None
    parameters: MPCParameter | None = None


class MPCOutput(NamedTuple):
    """
    A named tuple to store the solution of the MPC planner.

    Attributes:
        status: The status of the solver.
        u_star: The optimal control actions of the horizon.
        x_star: The optimal states of the horizon.
        Q: The state-action value function.
        V: The value function.
        dvalue_du0: The sensitivity of the value function with respect to the initial action.
        dvalue_dp_global: The sensitivity of the value function with respect to the global parameters.
        du0_dp_global: The sensitivity of the initial action with respect to the global parameters.
        du0_dx0: The sensitivity of the initial action with respect to the initial state.
    """

    status: np.ndarray | None = None
    u_star: np.ndarray | None = None  # (B, N, u_dim) or (N, u_dim)
    x_star: np.ndarray | None = None  # (B, N+1, x_dim)  or (N+1, x_dim)
    Q: np.ndarray | None = None  # (B, ) or (1, )
    V: np.ndarray | None = None  # (B, ) or (1, )
    dvalue_dx0: np.ndarray | None = None  # (B, x_dim) or (x_dim, )
    dvalue_du0: np.ndarray | None = None  # (B, u_dim) or (u_dim, )
    dvalue_dp_global: np.ndarray | None = None  # (B, p_dim) or (p_dim, )
    du0_dp_global: np.ndarray | None = None  # (B, udim, p_dim) or (udim, p_dim)
    du0_dx0: np.ndarray | None = None  # (B, u_dim, x_dim) or (u_dim, x_dim)


def initialize_ocp_solver(
    ocp_solver: AcadosOcpSolver,
    mpc_parameter: MPCParameter | None,
    ocp_iterate: AcadosOcpIterate | AcadosOcpFlattenedIterate | None,
) -> None:
    """Initializes the fields of the OCP solvers with the given values.

    Args:
        mpc_parameter: The parameters to set in the OCP solver.
        acados_ocp_iterate: The iterate of the solver to use as initialization.
    """

    if mpc_parameter is not None:
        if mpc_parameter.p_global is not None:
            ocp_solver.set_p_global_and_precompute_dependencies(mpc_parameter.p_global)

        if mpc_parameter.p_stagewise is not None:
            if mpc_parameter.p_stagewise_sparse_idx is not None:
                for stage, (p, idx) in enumerate(
                    zip(
                        mpc_parameter.p_stagewise,
                        mpc_parameter.p_stagewise_sparse_idx,
                    )
                ):
                    ocp_solver.set_params_sparse(stage, p, idx)
            else:
                for stage, p in enumerate(mpc_parameter.p_stagewise):
                    ocp_solver.set(stage, "p", p)

    if isinstance(ocp_iterate, AcadosOcpIterate):
        ocp_solver.load_iterate_from_obj(ocp_iterate)
    elif isinstance(ocp_iterate, AcadosOcpFlattenedIterate):
        ocp_solver.load_iterate_from_flat_obj(ocp_iterate)


def set_ocp_solver_initial_control_constraints(
    ocp_solver: AcadosOcpSolver, u0: np.ndarray
) -> None:
    """Set the initial control constraints of the OCP solver to the given value."""
    ocp_solver.set(0, "u", u0)
    ocp_solver.constraints_set(0, "lbu", u0)
    ocp_solver.constraints_set(0, "ubu", u0)


def unset_ocp_solver_initial_control_constraints(ocp_solver: AcadosOcpSolver) -> None:
    """Unset the initial control constraints of the OCP solver."""
    ocp_solver.constraints_set(0, "lbu", ocp_solver.acados_ocp.constraints.lbu)  # type: ignore
    ocp_solver.constraints_set(0, "ubu", ocp_solver.acados_ocp.constraints.ubu)  # type: ignore


def set_discount_factor(ocp_solver: AcadosOcpSolver, discount_factor: float) -> None:
    for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):  # type: ignore
        ocp_solver.cost_set(stage, "scaling", discount_factor**stage)


class MPC(ABC):
    """MPC abstract base class."""

    def __init__(
        self,
        ocp: AcadosOcp,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        ocp_solver_backup_fn: Callable | None = None,
        cleanup: bool = True,
    ):
        """
        Initialize the MPC object.

        Args:
            ocp: Optimal control problem.
            discount_factor: Discount factor. If None, acados default cost scaling is used, i.e. dt for intermediate stages, 1 for terminal stage.
            export_directory: Directory to export the generated code.
            export_directory_sensitivity: Directory to export the generated
                code for the sensitivity problem.
            ocp_solver_backup_fn: A function that returns a backup ocp solver iterate to be used in case the solver fails.
            cleanup: Whether to clean up the export directory on exit or
                when the object is deleted.
        """
        self.ocp = ocp

        # setup OCP for sensitivity solver
        self.ocp_sensitivity = deepcopy(ocp)
        self.ocp_sensitivity.translate_cost_to_external_cost()
        self.ocp_sensitivity.solver_options.nlp_solver_type = "SQP"
        self.ocp_sensitivity.solver_options.globalization_fixed_step_length = 0.0
        self.ocp_sensitivity.solver_options.nlp_solver_max_iter = 1
        self.ocp_sensitivity.solver_options.qp_solver_iter_max = 200
        self.ocp_sensitivity.solver_options.tol = self.ocp.solver_options.tol / 1e3
        self.ocp_sensitivity.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp_sensitivity.solver_options.qp_solver_ric_alg = 1
        self.ocp_sensitivity.solver_options.qp_solver_cond_N = (
            self.ocp.solver_options.N_horizon
        )
        self.ocp_sensitivity.solver_options.hessian_approx = "EXACT"
        self.ocp_sensitivity.solver_options.with_solution_sens_wrt_params = True
        self.ocp_sensitivity.solver_options.with_value_sens_wrt_params = True

        # path management
        self.afm = AcadosFileManager(export_directory, cleanup)
        self.afm_sens = AcadosFileManager(export_directory_sensitivity, cleanup)

        self._discount_factor = discount_factor

        self.ocp_solver_backup_fn = ocp_solver_backup_fn

        # constraints and cost functions
        self._h_fn = None
        self._cost_fn = None

    @cached_property
    def ocp_solver(self) -> AcadosOcpSolver:
        solver = self.afm.setup_acados_ocp_solver(self.ocp)

        if self._discount_factor is not None:
            set_discount_factor(solver, self._discount_factor)

        return solver

    @cached_property
    def ocp_sensitivity_solver(self) -> AcadosOcpSolver:
        solver = self.afm_sens.setup_acados_ocp_solver(self.ocp_sensitivity)

        if self._discount_factor is not None:
            set_discount_factor(solver, self._discount_factor)

        return solver

    @property
    def p_global_dim(self) -> int:
        """Return the dimension of p_global."""
        # TODO: Implement this
        raise NotImplementedError()

    @property
    def N(self) -> int:
        return self.ocp.solver_options.N_horizon  # type: ignore

    @property
    def default_p_global(self) -> np.ndarray | None:
        """Return the dimension of p_global."""
        return self.ocp.p_global_values if self.ocp.model.p_global is not None else None

    @property
    def default_p_stagewise(self) -> np.ndarray | None:
        """Return the dimension of p_stagewise."""
        return self.ocp.parameter_values if self.ocp.model.p is not None else None

    def state_value(
        self, state: np.ndarray, p_global: np.ndarray | None, sens: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute the value function for the given state.

        Args:
            state: The state for which to compute the value function.

        Returns:
            The value function and dvalue_dp_global if requested.
        """

        mpc_input = MPCInput(x0=state, parameters=MPCParameter(p_global=p_global))
        mpc_output, _ = self.__call__(mpc_input=mpc_input, dvdp=sens)

        return mpc_output.V, mpc_output.dvalue_dp_global  # type:ignore

    def state_action_value(
        self,
        state: np.ndarray,
        action: np.ndarray,
        p_global: np.ndarray | None,
        sens: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute the state-action value function for the given state and action.

        Args:
            state: The state for which to compute the value function.
            action: The action for which to compute the value function.

        Returns:
            The state-action value function and dQ_dp_global if requested.
        """

        mpc_input = MPCInput(
            x0=state, u0=action, parameters=MPCParameter(p_global=p_global)
        )
        mpc_output, _ = self.__call__(mpc_input=mpc_input, dvdp=sens)

        return mpc_output.Q, mpc_output.dvalue_dp_global  # type:ignore

    def policy(
        self,
        state: np.ndarray,
        p_global: np.ndarray | None,
        sens: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute the policy for the given state.

        Args:
            state: The state for which to compute the policy.
            p_global: The global parameters.
            sens: Whether to compute the sensitivity of the policy with respect to the parameters.
            use_adj_sens: Whether to use adjoint sensitivity.
        Returns:
            The policy and du0_dp_global if requested.
        """

        mpc_input = MPCInput(x0=state, parameters=MPCParameter(p_global=p_global))

        mpc_output, _ = self.__call__(
            mpc_input=mpc_input, dudp=sens, use_adj_sens=use_adj_sens
        )

        return mpc_output.u_star[0], mpc_output.du0_dp_global  # type: ignore

    def __call__(
        self,
        mpc_input: MPCInput,
        mpc_state: list[MPCState] | MPCState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, MPCState | list[MPCState]]:
        """
        Solve the OCP for the given initial state and parameters.

        Args:
            mpc_input: The input of the MPC controller.
            mpc_state: The iterate of the solver to use as initialization.
            dudx: Whether to compute the sensitivity of the action with respect to the state.
            dudp: Whether to compute the sensitivity of the action with respect to the parameters.
            dvdx: Whether to compute the sensitivity of the value function with respect to the state.
            dvdu: Whether to compute the sensitivity of the value function with respect to the action.
            dvdp: Whether to compute the sensitivity of the value function with respect to the parameters.
            use_adj_sens: Whether to use adjoint sensitivity.

        Returns:
            mpc_output: The output of the MPC controller.
            mpc_state: The iterate of the solver.
        """

        if mpc_input.x0.ndim == 1:
            return self._solve(
                mpc_input=mpc_input,
                mpc_state=mpc_state,  # type: ignore
                dudx=dudx,
                dudp=dudp,
                dvdx=dvdx,
                dvdu=dvdu,
                dvdp=dvdp,
                use_adj_sens=use_adj_sens,
            )

        return self._batch_solve(
            mpc_input=mpc_input,
            mpc_state_given=mpc_state,  # type: ignore
            dudx=dudx,
            dudp=dudp,
            dvdx=dvdx,
            dvdu=dvdu,
            dvdp=dvdp,
            use_adj_sens=use_adj_sens,
        )

    def _solve(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, MPCState]:
        # initialize solvers
        if mpc_input is not None:
            initialize_ocp_solver(self.ocp_solver, mpc_input.parameters, mpc_state)
            initialize_ocp_solver(
                self.ocp_sensitivity_solver, mpc_input.parameters, mpc_state
            )

        # set initial control constraints
        if mpc_input.u0 is not None:
            set_ocp_solver_initial_control_constraints(self.ocp_solver, mpc_input.u0)
            set_ocp_solver_initial_control_constraints(
                self.ocp_sensitivity_solver, mpc_input.u0
            )
        elif dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        # solve
        kw = {}

        # TODO: Cover case where we do not want to do a forward evaluation
        kw["u_star"] = self.ocp_solver.solve_for_x0(
            mpc_input.x0, fail_on_nonzero_status=False, print_stats_on_failure=False
        )

        status = self.ocp_solver.status
        kw["status"] = status

        if dudx:
            kw["du0_dx0"] = self.ocp_solver.eval_solution_sensitivity(
                stages=0, with_respect_to="initial_state"
            )[1]

        if dudp or dvdp:
            self.ocp_sensitivity_solver.load_iterate_from_flat_obj(
                self.ocp_solver.store_iterate_to_flat_obj()
            )

            self.ocp_sensitivity_solver.solve_for_x0(
                mpc_input.x0, fail_on_nonzero_status=False, print_stats_on_failure=False
            )

            if dudp:
                if use_adj_sens:
                    kw["du0_dp_global"] = (
                        self.ocp_sensitivity_solver.eval_adjoint_solution_sensitivity(
                            seed_x=[],
                            seed_u=[
                                (
                                    0,
                                    np.eye(self.ocp.dims.nu),  # type:ignore
                                )
                            ],
                            with_respect_to="p_global",
                            sanity_checks=True,
                        )
                    )
                else:
                    kw["du0_dp_global"] = (
                        self.ocp_sensitivity_solver.eval_solution_sensitivity(
                            0, "p_global"
                        )[1]
                    )

            if dvdp:
                kw["dvalue_dp_global"] = (
                    self.ocp_sensitivity_solver.eval_and_get_optimal_value_gradient(
                        "p_global"
                    )
                )

        if dvdx:
            kw["dvalue_dx0"] = self.ocp_solver.eval_and_get_optimal_value_gradient(
                with_respect_to="initial_state"
            )

        # NB: Assumes we are evaluating dQdu0 here
        if dvdu:
            kw["dvalue_du0"] = self.ocp_solver.get(0, "lam")[
                : self.ocp_solver.acados_ocp.dims.nu
            ]

        # unset initial control constraints
        if mpc_input.u0 is not None:
            kw["Q"] = self.ocp_solver.get_cost()
            unset_ocp_solver_initial_control_constraints(self.ocp_solver)
            unset_ocp_solver_initial_control_constraints(self.ocp_sensitivity_solver)
        else:
            kw["V"] = self.ocp_solver.get_cost()

        # get mpc state
        mpc_state = self.ocp_solver.store_iterate_to_flat_obj()

        return MPCOutput(**kw), mpc_state

    def _batch_solve(
        self,
        mpc_input: MPCInput,
        mpc_state_given: list[MPCState] | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, list[MPCState]]:
        # get a single element from the batch
        def get_idx(data, index):
            if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
                elem_type = type(data)
                return elem_type(*(get_idx(elem, index) for elem in data))  # type: ignore

            return None if data is None else data[index]

        batch_size = mpc_input.x0.shape[0]
        outputs = []
        states = []

        for idx in range(batch_size):
            mpc_output, mpc_state = self._solve(
                mpc_input=get_idx(mpc_input, idx),  # type: ignore
                mpc_state=mpc_state_given[idx] if mpc_state_given is not None else None,
                dudx=dudx,
                dudp=dudp,
                dvdx=dvdx,
                dvdu=dvdu,
                dvdp=dvdp,
                use_adj_sens=use_adj_sens,
            )

            outputs.append(mpc_output)
            states.append(mpc_state)

        def collate(key):
            value = getattr(outputs[0], key)
            if value is None:
                return value
            return np.stack([getattr(output, key) for output in outputs])

        mpc_output = MPCOutput(**{key: collate(key) for key in MPCOutput._fields})  # type: ignore

        return mpc_output, states  # type: ignore

    def fetch_param(
        self,
        mpc_param: MPCParameter | None = None,
        stage: int = 0,
    ) -> tuple[None | np.ndarray, None | np.ndarray]:
        """
        Fetch the parameters for the given stage.

        Args:
            mpc_param: The parameters.
            stage: The stage.

        Returns:
            The parameters for the given stage.
        """
        p_global = None
        p_stage = None

        if self.ocp.model.p_global is not None:
            p_global = self.ocp.p_global_values
        if self.ocp.model.p is not None:
            p_stage = self.ocp.parameter_values

        if mpc_param is not None:
            if mpc_param.p_global is not None:
                p_global = mpc_param.p_global

            if mpc_param.p_stagewise is not None:
                if mpc_param.p_stagewise_sparse_idx is None:
                    p_stage = mpc_param.p_stagewise[stage]
                else:
                    p_stage = p_stage.copy()
                    p_stage[mpc_param.p_stagewise_sparse_idx[stage]] = (
                        mpc_param.p_stagewise[stage]
                    )

        return p_global, p_stage

    def stage_cons(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: MPCParameter | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get the value of the stage constraints.

        Args:
            x: State.
            u: Control.
            p: Parameters.

        Returns:
            stage_cons: Stage constraints.
        """
        assert x.ndim == 1 and u.ndim == 1

        def relu(value):
            return value * (value > 0)

        cons = {}

        # state constraints
        if self.ocp.constraints.lbx is not None:
            cons["lbx"] = relu(self.ocp.constraints.lbx - x)
        if self.ocp.constraints.ubx is not None:
            cons["ubx"] = relu(x - self.ocp.constraints.ubx)
        # control constraints
        if self.ocp.constraints.lbu is not None:
            cons["lbu"] = relu(self.ocp.constraints.lbu - u)
        if self.ocp.constraints.ubu is not None:
            cons["ubu"] = relu(u - self.ocp.constraints.ubu)

        # h constraints
        if self.ocp.model.con_h_expr is not None:
            if self._h_fn is None:
                inputs = [self.ocp.model.x, self.ocp.model.u]

                if self.ocp.model.p is not None:
                    inputs.append(self.ocp.model.p)  # type: ignore

                if self.ocp.model.p_global is not None:
                    inputs.append(self.ocp.model.p_global)

                self._h_fn = ca.Function("h", inputs, [self.ocp.model.con_h_expr])

            inputs = [x, u]

            p_global, p_stage = self.fetch_param(p)
            if p_global is not None:
                inputs.append(p_global)

            if p_stage is not None:
                inputs.append(p_stage)

            h = self._h_fn(*inputs)
            cons["lh"] = relu(self.ocp.constraints.lh - h)
            cons["uh"] = relu(h - self.ocp.constraints.uh)

        # Todo (Jasper): Add phi constraints.

        return cons

    def stage_cost(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: MPCParameter | None = None,
    ) -> float:
        """
        Get the value of the stage cost.

        Args:
            x: State.
            u: Control.
            p: Parameters.

        Returns:
            stage_cost: Stage cost.
        """
        assert self.ocp.cost.cost_type == "EXTERNAL"
        assert x.ndim == 1 and u.ndim == 1

        if self._cost_fn is None:
            inputs = [self.ocp.model.x, self.ocp.model.u]

            if self.ocp.model.p is not None:
                inputs.append(self.ocp.model.p)

            if self.ocp.model.p_global is not None:
                inputs.append(self.ocp.model.p_global)

            self._cost_fn = ca.Function(
                "cost", inputs, [self.ocp.model.cost_expr_ext_cost]
            )

        inputs = [x, u]

        p_global, p_stage = self.fetch_param(p)

        if p_global is not None:
            inputs.append(p_global)

        if p_stage is not None:
            inputs.append(p_stage)

        return self._cost_fn(*inputs).full().item()  # type: ignore
