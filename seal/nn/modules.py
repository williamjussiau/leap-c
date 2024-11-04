import math

import casadi as ca
import torch
import torch.nn as nn
from acados_template import AcadosSimSolver

from seal.mpc import MPC, Parameter, SolverState

from .autograd import AutogradCasadiFunction, DynamicsSimFunction, MPCSolutionFunction


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

        NOTE: This solves every sample in the batch sequentially, so it is not efficient for large batch sizes.

        NOTE: Make sure that you follow the documentation of AcadosOcpSolver.eval_solution_sensitivity,
        or else the gradients used in the backpropagation might be erroneous! In particular,
        the status output can be used to rid yourself from non-converged solutions.

        NOTE: Currently, acados guards calculating sensitivities for parameters in constraints with an Exception.
        Still, it might be that you want to have such parameters,
        and still use the sensitivities of OTHER parameters.
        For this, you can, e.g., in acados github commit 95c341c6,
        disable the Exception in acados_ocp.py, line 800 to still calculate all the sensitivities
        and then only use the sensitivities of the parameters not belonging to constraints
        (because as acados states in eval_solution_sensitivity, they are erroneous).
    """

    def __init__(self, mpc: MPC):
        super().__init__()
        self.mpc = mpc

    def forward(self, x0: torch.Tensor,
                p_global_learnable: torch.Tensor,
                p_rests: list[Parameter],
                initializations: list[SolverState] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x0: The initial state of the MPC, shape (batch_size, xdim).
            p_global_learnable: The parameters of the MPC, shape (batch_size, pdim).
            p_rests: A list of length batch_size with the remaining parameter information for the MPC. NOTE that it should not contain p_global_learnable, since this will be overwritten by p!
            initializations: A list of length batch_size which contains the initialization to use in every solve.
        Returns:
            A tuple of tensors, the first entry containing the first action u[0] of the MPC solution,
            given the initial state and parameters, the second entry containing the acados status
            of that solution (useful for e.g., logging, debugging or cleaning the backward pass from non-converged solutions).
        """
        return MPCSolutionFunction.apply(self.mpc, x0, p_global_learnable, p_rests, initializations)


class CleanseAndReducePerSampleLoss(nn.Module):
    """A module that is ment to substitute the last part of a loss,
    taking over the reduction to a scalar and, in particular,
    cleansing the per-sample-loss of the samples whose MPC-part did not converge.
    This basically works by removing all samples that correspond to a status unequal to 0, 
    hence effectively training with a varying batch_size.
    """

    def __init__(self, reduction: str, num_batch_dimensions: int,
                 n_nonconvergences_allowed: int, throw_exception_if_exceeded: bool = False):
        """
        Parameters:
            reduction: Either "mean", "sum" or "none" as in native pytorch modules.
            num_batch_dimensions: Determines how many of the first dimensions should be treated as batch_dimension.
            n_nonconvergences_allowed: The number of nonconvergences allowed in a batch. If this number is exceeded, the loss returned will just be 0 (and hence do nothing when backward is called).
            throw_exception_if_exceeded: If True, an exception will be thrown if the number of nonconvergences is exceeded.
        """
        super().__init__()
        self.reduction = reduction
        self.n_nonconvergences_allowed = n_nonconvergences_allowed
        self.throw_exception_if_exceeded = throw_exception_if_exceeded
        self.num_batch_dimensions = num_batch_dimensions

    def forward(self, per_sample_loss: torch.Tensor, status: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            per_sample_loss: The per_sample_loss.
            status: The status of the MPC solution, of same shape as the per_sample loss for the first how_many_batch_dimensions and afterwards one integer dimension, containing whether the solution converged (0 means converged, all other integers count as not converged).
        """
        if per_sample_loss.shape[:self.num_batch_dimensions] != status.shape[:self.num_batch_dimensions]:
            raise ValueError(
                "The per_sample_loss and status must have the same batch dimensions.")
        if len(status.shape) != self.num_batch_dimensions + 1:
            raise ValueError(
                "The status must have a shape corresponding to the number of batch dimensions, followed by one dimension containing the status.")
        if status.shape[-1] != 1:
            raise ValueError("The last dimension of status must be of size 1, containing the status of each sample.")

        error_mask = status.to(dtype=torch.bool)
        nonconvergent_samples = torch.sum(error_mask)

        if nonconvergent_samples > self.n_nonconvergences_allowed:
            if self.throw_exception_if_exceeded:
                raise ValueError(
                    f"Number of nonconvergences exceeded the allowed number of {self.n_nonconvergences_allowed}.")
            return torch.zeros(1)

        cleansed_loss = per_sample_loss.clone()
        cleansed_loss[error_mask.squeeze()] = 0
        if self.reduction == "mean":
            cleansed_loss = torch.mean(cleansed_loss)
            batch_size = math.prod(per_sample_loss.shape[:self.num_batch_dimensions])
            # We have to adjust the meaned loss by the cleansed samples
            result = cleansed_loss * batch_size / (batch_size - nonconvergent_samples)
        elif self.reduction == "sum":
            result = torch.sum(cleansed_loss)
        elif self.reduction == "none":
            result = cleansed_loss
        else:
            raise ValueError("Reduction must be either 'mean', 'sum' or 'none'.")
        return result
