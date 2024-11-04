from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Iterable, NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_ocp_iterate import AcadosOcpIterate

from seal.util import AcadosFileManager


def set_discount_factor(ocp_solver: AcadosOcpSolver, discount_factor: float) -> None:
    for stage in range(0, ocp_solver.acados_ocp.dims.N + 1):  # type:ignore
        ocp_solver.cost_set(stage, "scaling", discount_factor**stage)

class Parameter(NamedTuple):
    """
    It is not possible to set only an incomplete part of p_global: 
    If some information for p_global is given, all information for p_global should be explicitly given.
    
    Parameters:
    p_global_learnable: The part of p_global that should be learned in shape (n_p_global_learnable, ).
    p_global_non_learnable: The part of p_global that should not be learned in shape (n_p_global_non_learnable, ).
    p_stagewise: The stagewise parameters in shape (N+1, p_stagewise_dim) or (N+1, len(p_stagewise_sparse_idx)) if the next field is set.
    p_stagewise_sparse_idx: If not None, stagewise parameters are set in a sparse manner, using these indices. 
    """
    p_global_learnable: np.ndarray | None  # Not yet with batch dim. 
    p_global_non_learnable: np.ndarray | None  # Not yet with batch dim 
    p_stagewise: np.ndarray | None  # Not yet with batch dim
    p_stagewise_sparse_idx: np.ndarray | None  # Not yet with batch dim

class ParameterManager():
    """Class to manage the parameters of the ocp solver, i.e., which parameters are learnable, where the indices are, etc.."""
    
    def __init__(self, global_param_labels_to_idx: list[tuple[str, int, int, str]]):
        """
        Parameters:
        param_labels_to_idx: A list containing for all parameters a tuple containing the parameter label name, the start and stop indices of the parameter in the parameter vector and an identifier, either "global_learnable" or "global_non_learnable".
        global_param_labels_to_learn: A list of the parameter labels where the sensitivities can be computed. Those parameters are always expected to be in order of this list.
        """
        self.global_param_labels_to_info = global_param_labels_to_idx
        self._run_sanity_check_on_global_param_labels_to_info()
        
    @cached_property
    def slice_for_dudp_global_learnable(self) -> list[int]:
        """Slice for transforming the sensitivities du/dp_global to du/dp_global_learnable."""
        return self._calculate_slice_dudp()
    
    @property
    def p_global_dim(self) -> int:
        """Return the dimension of p_global."""
        return self.global_param_labels_to_info[-1][2]
    
    def set_parameters_in_ocp_solver(self, ocp_solvers: list[AcadosOcpSolver], param: Parameter):
        """Set the given parameters in the given ocp_solver. 
        It is not possible to set only an incomplete part of p_global: 
        If some information for p_global is given, all information for p_global should be explicitly given.
        """
        p_global_learnable, p_global_non_learnable, p_stagewise, p_stagewise_sparse_idx = param
        
        if p_global_learnable is not None or p_global_non_learnable is not None:
            if p_global_non_learnable is None:
                p_global_non_learnable = np.array([])
            if p_global_learnable is None:
                p_global_learnable = np.array([])
            p_global = self._merge_learnable_and_not_learnable_to_p_global(p_global_learnable, p_global_non_learnable)
            for ocp_solver in ocp_solvers:
                ocp_solver.set_p_global_and_precompute_dependencies(p_global)
        
        if p_stagewise is not None:
            for ocp_solver in ocp_solvers:
                self._set_p_stagewise(ocp_solver, p_stagewise, p_stagewise_sparse_idx)
            
    
    def _merge_learnable_and_not_learnable_to_p_global(self, parameters_to_learn: np.ndarray, non_learnable_parameters: np.ndarray) -> np.ndarray:
        """
        Merge the learnable and non-learnable parameters to the global parameter vector. Both are expected to be given in the same order as the labels.

        Args:
            parameters_to_learn: Learnable parameters, shape (n_learnable_parameters,).
            non_learnable_parameters: Non-learnable parameters, shape (n_non_learnable_parameters,).

        Returns:
            p_global: A new numpy array of shape (n_global_parameters,) containing the learnable and non-learnable parameters mapped according to the order of labels.
        """
        p_global = np.zeros((parameters_to_learn.shape[0] + non_learnable_parameters.shape[0]))
        if p_global.size < self.p_global_dim:
            raise ValueError("It is not possible to provide only a part of p_global, all information for p_global should be explicitly given.")
        learn_idx = 0
        non_learn_idx = 0
        for i, info in enumerate(self.global_param_labels_to_info):
            _, start, stop, identifier = info
            length = stop - start
            if identifier == "global_learnable":
                p_global[start:stop] = parameters_to_learn[learn_idx:learn_idx + length]
                learn_idx += length
            elif identifier == "global_non_learnable":
                p_global[start:stop] = non_learnable_parameters[non_learn_idx:non_learn_idx + length]
                non_learn_idx += length
            else:
                raise ValueError(f"Unknown identifier {identifier}.")
        if learn_idx != parameters_to_learn.shape[0]:
            raise ValueError("Not all learnable parameters were used. The given learnable parameter array is inconsistent with the global parameter info dict..")
        if non_learn_idx != non_learnable_parameters.shape[0]:
            raise ValueError("Not all non-learnable parameters were used. The given non-learnable parameter array is inconsistent with the global parameter info dict.")
        return p_global
    
    def _run_sanity_check_on_global_param_labels_to_info(self):
        """Check if the global_param_labels_to_info is valid."""
        previous_stop = 0
        for label, start, stop, identifier in self.global_param_labels_to_info:
            if start >= stop:
                raise ValueError(f"Invalid indices {start} and {stop} for parameter {label}.")
            if start != previous_stop:
                raise ValueError(f"Indices for parameter {label} are not continuing where the last label left off.")
            if identifier not in ["global_learnable", "global_non_learnable"]:
                raise ValueError(f"Unknown identifier {identifier}.")
            previous_stop = stop
         
    def _calculate_slice_dudp(self) -> list[int]:
        indices = []
        for info in self.global_param_labels_to_info:
            _, start, stop, identifier = info
            if identifier == "global_learnable":
                indices.extend(list(range(start, stop)))
        return indices
    
    def _get_p_stagewise(self, ocp_solver: AcadosOcpSolver, stages: Iterable[int]) -> np.ndarray:
        """
        Obtain the value of the stagewise parameters in shape (len(stages), p_stagewise_dim) from the solver.
        """
        params = []
        for stage in stages:
            params.append(ocp_solver.get(stage, "p"))
        return np.stack(params, axis=0)

    def _set_p_stagewise(self, ocp_solver: AcadosOcpSolver, p_stagewise: np.ndarray, p_stagewise_sparse_idx: np.ndarray | None) -> None:
        """Set p_stagewise in the respective solver."""
        if p_stagewise_sparse_idx is not None:
            for stage, val in enumerate(p_stagewise):            
                ocp_solver.set_params_sparse(stage, p_stagewise_sparse_idx, p_stagewise)
        else:
            for stage, val in enumerate(p_stagewise):
                ocp_solver.set(stage, "p", val)


@dataclass
class SolverState:
    """Glorified dictionary used for initializing the MPC before solving the problem.
    Every field that is not None will be used for the initialization in the initialize method of the MPC class.
    NOTE: The acados iterate contains fields that will be overwritten by the fields in this class (e.g. x_traj), if provided.
    NOTE: Can be extend by other fields (e.g. duals) if needed.
    
    Parameters:
        acados_iterate: The iterate of the acados solver.
        x_traj: The state trajectory of shape (mpc.N+1, x_dim).
        u_traj: The action trajectory of shape (mpc.N, u_dim).
    """
    acados_ocp_iterate: AcadosOcpIterate | None = None
    x_traj: np.ndarray | None = None
    u_traj: np.ndarray | None = None

class MPC(ABC):
    """
    MPC abstract base class.
    """
    

    def __init__(
        self,
        ocp: AcadosOcp,
        global_param_labels_to_info: list[tuple[str, int, int, str]],
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        cleanup: bool = True,
    ):
        """
        Initialize the MPC object.

        Args:
            ocp: Optimal control problem.
            global_param_labels_to_info: A list containing for all parameters a tuple containing the parameter label name, the start and stop indices of the parameter in the parameter vector and an identifier, either "global_learnable" or "global_non_learnable"
            discount_factor: Discount factor. If None, acados default cost scaling is used, i.e. dt for intermediate stages, 1 for terminal stage.
            export_directory: Directory to export the generated code.
            export_directory_sensitivity: Directory to export the generated
                code for the sensitivity problem.
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
        self.ocp_sensitivity.solver_options.tol = self.ocp.solver_options.tol/1e3
        self.ocp_sensitivity.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp_sensitivity.solver_options.qp_solver_ric_alg = 1
        self.ocp_sensitivity.solver_options.qp_solver_cond_N = self.ocp.solver_options.N_horizon
        self.ocp_sensitivity.solver_options.hessian_approx = "EXACT"
        self.ocp_sensitivity.solver_options.with_solution_sens_wrt_params = True
        self.ocp_sensitivity.solver_options.with_value_sens_wrt_params = True

        # path management
        afm = AcadosFileManager(export_directory, cleanup)
        afm_sens = AcadosFileManager(export_directory_sensitivity, cleanup)

        # setup ocp solvers, we make the creation lazy
        self._ocp_solver_fn = partial(afm.setup_acados_ocp_solver, ocp)
        self._ocp_sensitivity_solver_fn = partial(afm_sens.setup_acados_ocp_solver, self.ocp_sensitivity)
        self._ocp_solver = None
        self._ocp_sensitivity_solver = None
        
        self._parameter_manager = ParameterManager(global_param_labels_to_info)

        self.__parameter_values = np.array([])
        self.__p_global_values = np.array([])
        self.__discount_factor = None

        if discount_factor is not None:
            self.discount_factor = discount_factor # this will overwrite the acados cost scaling

    @property
    def ocp_solver(self) -> AcadosOcpSolver:
        if self._ocp_solver is None:
            self._ocp_solver = self._ocp_solver_fn()
        return self._ocp_solver

    @property
    def ocp_sensitivity_solver(self) -> AcadosOcpSolver:
        if self._ocp_sensitivity_solver is None:
            self._ocp_sensitivity_solver = self._ocp_sensitivity_solver_fn()
        return self._ocp_sensitivity_solver

    @property
    def N(self) -> int:
        return self.ocp_solver.acados_ocp.solver_options.N_horizon  # type: ignore

    @property
    def p_global_dim(self) -> int:
        return self._parameter_manager.p_global_dim

    @property
    def p_global_values(self) -> np.ndarray:
        return self.ocp.p_global_values

    @property
    def discount_factor(self) -> float:
        return self.__discount_factor


    @p_global_values.setter
    def p_global_values(self, p_global_values):
        if isinstance(p_global_values, np.ndarray):
            p_global_values = p_global_values.reshape(-1)
            self.ocp.p_global_values = p_global_values
            self.ocp_sensitivity.p_global_values = p_global_values
            self.ocp_solver.set_p_global_and_precompute_dependencies(p_global_values)
            self.ocp_sensitivity_solver.set_p_global_and_precompute_dependencies(p_global_values)
        else:
            raise Exception("Invalid p_global_values value. " + f"Expected numpy array, got {type(p_global_values)}.")


    @discount_factor.setter
    def discount_factor(self, discount_factor):
        """
        Set the discount factor. This overwrites the scaling of the cost function (control interval length by default).
        """

        if isinstance(discount_factor, float) and discount_factor > 0 and discount_factor <= 1:
            print(f"Setting discount factor to {discount_factor}.")
            self.__discount_factor = discount_factor

            MPC.__set_discount_factor(self.ocp_solver, discount_factor)
            MPC.__set_discount_factor(self.ocp_sensitivity_solver, discount_factor)
        else:
            raise Exception("Invalid discount_factor value. " + f"Expected float in (0, 1], got {discount_factor}.")


    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Solve the optimization problem
        self.status = self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        # action = self.scale_action(action)

        return action


    def q_update(self, x0: np.ndarray, u0: np.ndarray, p: np.ndarray | None = None) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)

        self.ocp_solver.constraints_set(0, "lbu", u0)
        self.ocp_solver.constraints_set(0, "ubu", u0)

        optimal_value, optimal_value_gradient = self.v_update(x0, p)

        # Change bounds back to original
        self.ocp_solver.constraints_set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.constraints_set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        return optimal_value, optimal_value_gradient

    def v_update(self, x0: np.ndarray, p: np.ndarray | None = None) -> int:
        if p is not None:
            self.p_global_values = p

        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Set p_global
        self.ocp_solver.set_p_global_and_precompute_dependencies(self.p_global_values)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        optimal_value = self.ocp_solver.get_cost()

        if status != 0:
            raise RuntimeError(f"Solver failed with status {status}. Exiting.")

        optimal_value_gradient = self.ocp_solver.eval_and_get_optimal_value_gradient(with_respect_to="p_global")

        return optimal_value, optimal_value_gradient

    def pi_update(
        self,
        x0: np.ndarray,
        p: Parameter,
        initialization: SolverState | None = None,
        return_dudp: bool = True,
        return_dudx: bool = False,
    ) -> tuple[np.ndarray, int, tuple[np.ndarray | None, np.ndarray | None]]:
        """Solves the OCP for the initial state x0 and learnable parameters p
        and returns the first action of the horizon, as well as
        the sensitivity of said action with respect to the parameters
        and the status of the solver.

        Parameters:
            x0: Initial state.
            p: Parameter class to set the parameters. For more info see the description of the Parameter class.
            initialization: The MPCInitInfo object to use for initialization. For more info see the description of the MPCInitInfo class.
            return_dudp: Whether to return the sensitivity of the action with respect to the parameters.
            return_dudx: Whether to return the sensitivity of the action with respect to the state.
        Returns:
        A tuple containing in order:
            u: The first action of the (solution) horizon.
            status: The acados status of the solve.
            sensitivities: A tuple with two entries, containing du/dp_learnable in the first entry (or None if not requested) and du/dx in the second entry (or None if not requested).
        """
        self.init_solvers(p, initialization)

        # Solve the optimization problem for initial state x0
        pi = self.ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=True)
        status = self.ocp_solver.get_status()

        # TODO: Should everything below be moved to get_dpi_dp? Or should get_dpi_dp be removed?
        # Transfer the iterate to the sensitivity solver
        iterate = self.ocp_solver.store_iterate_to_flat_obj()
        self.ocp_sensitivity_solver.load_iterate_from_flat_obj(iterate)

        # Call the sensitivity solver to prepare the sensitivity calculations
        self.ocp_sensitivity_solver.solve_for_x0(x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

        # Calculate the policy gradient
        if return_dudp:
            _, dpidp = self.ocp_sensitivity_solver.eval_solution_sensitivity(0, "p_global")
        else:
            dpidp = None

        if return_dudx:
            _, dpidx = self.ocp_sensitivity_solver.eval_solution_sensitivity(0, "initial_state")
        else:
            dpidx = None

        return pi, status, (dpidp, dpidx)  # type:ignore

    def init_solvers(self, p: Parameter | None, initialization: SolverState | None):
        """Initializes the fields of the OCP solvers with the given values.
        Parameters:
            p: The parameters to set in the OCP solver. See description of the Parameter class for more details.
            initialization: See description of the MPCInitInfo class for more details. 
        """
        if p is not None:
            self._parameter_manager.set_parameters_in_ocp_solver([self.ocp_solver, self.ocp_sensitivity_solver], p)
        if initialization is not None:
            if initialization.x_traj is not None:
                for stage, val in enumerate(initialization.x_traj):
                    self.ocp_solver.set(stage, "x", val)
                    self.ocp_sensitivity_solver.set(stage, "x", val)
            if initialization.u_traj is not None:
                values = initialization.u_traj
                for stage in range(self.N):  # Setting u is not possible in the terminal stage.
                    self.ocp_solver.set(stage, "u", values[stage])
                    self.ocp_sensitivity_solver.set(stage, "u", values[stage])
        
    def slice_dudp_global_to_dudp_global_learnable(self, dudp: np.ndarray) -> np.ndarray:
        """Slice the sensitivity du/dp_global to du/dp_global_learnable. Assumes the parameter dimension is the last dimension."""
        return dudp[..., self._parameter_manager.slice_for_dudp_global_learnable]
        
    def get_dV_dp(self) -> float:
        """
        Get the value of the sensitivity of the value function with respect to the parameters.

        Assumes OCP is solved for state and parameters.

        Returns:
            dV_dp: Sensitivity of the value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_Q(self) -> float:
        """
        Get the value of the state-action value function.

        Assumes OCP is solved for state and action.

        Returns:
            Q: State-action value function.
        """
        return self.ocp_solver.get_cost()

    def get_dQ_dp(self) -> float:
        """
        Get the value of the sensitivity of the state-action value function with respect to the parameters.

        Assumes OCP is solved for state, action and parameters.

        Returns:
            dQ_dp: Sensitivity of the state-action value function with respect to the parameters.
        """
        # TODO: Implement this
        pass

    # def get_parameter_labels(self) -> list:
    #     return get_parameter_labels(self.ocp_solver.acados_ocp)

    # def get_state_labels(self) -> list:
    #     return get_state_labels(self.ocp_solver.acados_ocp)

    # def get_input_labels(self) -> list:
    #     return get_input_labels(self.ocp_solver.acados_ocp)

    def update(self, x0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        if status != 0:
            raise RuntimeError(f"Solver failed update with status {status}. Exiting.")

        # test_nlp_sanity(self.nlp)

        return status

    def reset(self, x0: np.ndarray):
        self.ocp_solver.reset()

        for stage in range(self.ocp_solver.acados_ocp.solver_options.N_horizon + 1):
            # self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.lbx_0)
            self.ocp_solver.set(stage, "x", x0)
            self.ocp_sensitivity_solver.set(stage, "x", x0)

    def set(self, stage, field, value):
        if field == "p":
            # TODO: Implement this
            pass

            # p_temp = self.nlp.p.sym(value)
            # This allows to set parameters in the NLP but not in the OCP solver. This can be a problem.
            # for key in p_temp.keys():
            #     self.nlp.set_parameter(key, p_temp[key])

            # if self.ocp_solver.acados_ocp.dims.np > 0:
            #     self.ocp_solver.set(stage, field, p_temp["model"].full().flatten())

            # if self.nlp.vars.val["p", "W_0"].shape[0] > 0:
            #     p_temp = self.nlp.vars.sym(0)
            #     self.ocp_solver.cost_set
            #     # self.nlp.set(stage, field, value)
            #     W_0 = self.nlp.vars.val["p", "W_0"].full()
            #     print("Not implemented")

        else:
            self.ocp_solver.set(stage, field, value)

    # def set_parameter(self, value_, api="new"):
    #     p_temp = self.nlp.p.sym(value_)

    #     if "W_0" in p_temp.keys():
    #         self.nlp.set_parameter("W_0", p_temp["W_0"])
    #         self.ocp_solver.cost_set(
    #             0, "W", self.nlp.get_parameter("W_0").full(), api=api
    #         )

    #     if "W" in p_temp.keys():
    #         self.nlp.set_parameter("W", p_temp["W"])
    #         for stage in range(1, self.ocp_solver.acados_ocp.solver_options.N_horizon):
    #             self.ocp_solver.cost_set(
    #                 stage, "W", self.nlp.get_parameter("W").full(), api=api
    #             )

    #     if "yref_0" in p_temp.keys():
    #         self.nlp.set_parameter("yref_0", p_temp["yref_0"])
    #         self.ocp_solver.cost_set(
    #             0, "yref", self.nlp.get_parameter("yref_0").full().flatten(), api=api
    #         )

    #     if "yref" in p_temp.keys():
    #         self.nlp.set_parameter("yref", p_temp["yref"])
    #         for stage in range(1, self.ocp_solver.acados_ocp.solver_options.N_horizon):
    #             self.ocp_solver.cost_set(
    #                 stage,
    #                 "yref",
    #                 self.nlp.get_parameter("yref").full().flatten(),
    #                 api=api,
    #             )

    #     if self.ocp_solver.acados_ocp.dims.np > 0:
    #         self.nlp.set_parameter("model", p_temp["model"])
    #         for stage in range(self.ocp_solver.acados_ocp.solver_options.N_horizon + 1):
    #             self.ocp_solver.set(stage, "p", p_temp["model"].full().flatten())

    @staticmethod
    def __set_discount_factor(ocp_solver: AcadosOcpSolver, discount_factor: float) -> None:
        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):
            ocp_solver.cost_set(stage, "scaling", discount_factor**stage)


    def get(self, stage, field):
        return self.ocp_solver.get(stage, field)

    # def scale_action(self, action: np.ndarray) -> np.ndarray:
    #     """
    #     Rescale the action from [low, high] to[-1, 1]
    #     (no need for symmetric action space)

    #     : param action: Action to scale
    #     : return: Scaled action
    #     """
    #     low = self.ocp.constraints.lbu
    #     high = self.ocp.constraints.ubu

    #     return 2.0 * ((action - low) / (high - low)) - 1.0

    # def unscale_action(self, action: np.ndarray) -> np.ndarray:
    #     """
    #     Rescale the action from [-1, 1] to[low, high]
    #     (no need for symmetric action space)

    #     : param action: Action to scale
    #     : return: Scaled action
    #     """
    #     low = self.ocp_solver.acados_ocp.constraints.lbu
    #     high = self.ocp_solver.acados_ocp.constraints.ubu

    #     return 0.5 * (high - low) * (action + 1.0) + low

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        # TODO: Implement this or remove. Only need get_dV_dp and get_dQ_dp.

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        # TODO: Implement this

    def get_V(self) -> float:
        """
        Get the value of the value function.

        Assumes OCP is solved for state.

        Returns:
            V: Value function.
        """
        return self.ocp_solver.get_cost()

    def get_pi(self) -> np.ndarray:
        """
        Get the value of the policy.

        Assumes OCP is solved for state.
        """
        return self.ocp_solver.get(0, "u")

    def get_dpi_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the policy with respect to the parameters.

        Assumes OCP is solved for state and parameters.
        """
        raise NotImplementedError("This method is not implemented yet.")


        # TODO: Implement this using ocp_sensitivity_solver

    def stage_cons(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Get the value of the stage constraints.

        Returns:
            stage_cons: Stage constraints.
        """

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

                self._h_fn = ca.Function("h", inputs, [self.ocp.model.con_h_expr])

            inputs = [x, u]
            if self.ocp.model.p:
                # Todo (Jasper): Param update.
                # p = self.fetch_params(mpc_input, stage=0)
                inputs.append(p)  # type: ignore

            h = self._h_fn(*inputs)
            cons["lh"] = relu(self.ocp.constraints.lh - h)
            cons["uh"] = relu(h - self.ocp.constraints.uh)

        # Todo (Jasper): Add phi constraints.

        return cons

    def stage_cost(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Get the value of the stage cost.

        Returns:
            stage_cost: Stage cost.
        """
        # TODO: Implement this
        raise NotImplementedError()
