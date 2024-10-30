import casadi as ca
import numpy as np
import torch
import torch.nn as nn

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from .autograd import AutogradCasadiFunction, DynamicsSimFunction, MPCSolutionFunction

from seal.mpc import MPC


class CasadiExprModule(nn.Module):
    def __init__(self, expr: ca.SX, inputs: list[ca.SX]):
        super().__init__()
        self.expr = expr
        self.inputs = inputs
        self.f = ca.Function("f", inputs, [expr])

        # setup backward AD
        inputs_cat = ca.vertcat(*inputs)
        grad_output = ca.SX.sym("output", expr.shape[0])
        jtimes = ca.jtimes(expr, inputs_cat, grad_output, True)
        self.jtimes = ca.Function("jtimes", [inputs_cat, grad_output], [jtimes])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == len(self.inputs)
        return AutogradCasadiFunction.apply(self.f, self.jtimes,  *inputs)


class AcadosSimModule(nn.Module):
    """A PyTorch module to wrap around AcadosSimSolver."""

    def __init__(self, sim: AcadosSimSolver):
        super().__init__()
        self.sim = sim

    def forward(self, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        return DynamicsSimFunction.apply(self.sim, x, u, p)


class MPCSolutionModule(nn.Module):
    """A pytorch module to represent the implicit policy given by an MPC controller,
    i.e., the first action of the MPC solution.
    Backpropagation works through using the sensitivities of this action with respect to
    the initial state or the parameters of the MPC. Currently only supporting parameters that are global over the horizon
    (contrary to stagewise parameters).

        NOTE: This is uses an mpc for every sample in a batch, so it is not efficient for large batch sizes.

        NOTE: Make sure that you follow the documentation of AcadosOcpSolver.eval_solution_sensitivity,
        or else the gradients used in the backpropagation might be erroneous! In particular,
        the status output can be used to rid yourself from non-converged solutions.

        NOTE: Currently, acados guards calculating sensitivities for parameters in constraints with an Exception.
        Still, it might be that you want to have such parameters,
        and still use the sensitivities of OTHER parameters.
        For this, you can, e.g. in acados github commit 95c341c6,
        disable the Exception in acados_ocp.py, line 800 to still calculate all the sensitivities
        and then only use the sensitivities of the parameters not belonging to constraints
        (because as acados states in eval_solution_sensitivity, they are erroneous).
    """

    def __init__(self, mpc: MPC):
        super().__init__()
        self.mpc = mpc

    def forward(self, x0: torch.Tensor,
                p: torch.Tensor,
                initializations: list[dict[str, np.ndarray]] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x0: The initial state of the MPC, shape (batch_size, xdim).
            p: The parameters of the MPC, shape (batch_size, pdim).
            initializationa: A list of length batch_size, containing maps from the strings of fields (as in AcadosOcpSolver.set()) that should be initialized, to an np array which contains the values for those fields, being of shape (stages_of_that_field, field_dim).
        Returns:
            A tuple of tensors, the first entry containing the first action u[0] of the MPC solution,
            given the initial state and parameters and the second entry containing the acados status
            of that solution (useful for e.g., logging, debugging or cleaning the backward pass from non-converged solutions).

        """
        return MPCSolutionFunction.apply(self.mpc, x0, p, initializations)
