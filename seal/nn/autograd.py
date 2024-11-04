"""Provides a PyTorch module that wraps a CasADi expression.

TODO (Jasper): Does JIT compilation speed up the module?
"""
import casadi as ca
import numpy as np
import torch
import torch.autograd as autograd
from acados_template import AcadosSimSolver

from seal.mpc import MPC, Parameter, SolverState
from seal.util import tensor_to_numpy


class AutogradCasadiFunction(autograd.Function):
    @staticmethod
    def forward(ctx, f: ca.Function, jtimes: ca.Function, *inputs: torch.Tensor) -> torch.Tensor:
        assert inputs[0].ndim == 2
        device = inputs[0].device
        dtype = inputs[0].dtype

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        inputs = [input.detach().cpu().numpy().T for input in inputs]

        # save the jacobian and inputs for the backward pass
        ctx.inputs = inputs
        ctx.jtimes = jtimes

        result = f(*inputs).full()

        # convert numpy to tensor and transpose
        return torch.tensor(result, device=device, dtype=dtype).T

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype
        grad_output = grad_output.detach().cpu().numpy().T

        # prepare inputs and remember the splits
        inputs = ctx.inputs
        sizes = [input.shape[0] for input in inputs]
        splits = np.cumsum(sizes)[:-1]
        inputs = np.concatenate(inputs, axis=0)

        # compute the jacobian
        grad_input = ctx.jtimes(inputs, grad_output).T
        # split the grads for the individual inputs
        grad_input = np.split(grad_input, splits, axis=1)
        # convert to tensor
        grad_input = [torch.tensor(grad, device=device, dtype=dtype) for grad in grad_input]

        return (None, None, *grad_input)


class DynamicsSimFunction(autograd.Function):
    @staticmethod
    def forward(ctx, sim: AcadosSimSolver, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor | None) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        xdim = x.shape[1]
        udim = u.shape[1]

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        x = x.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        if p is not None:
            p = p.detach().cpu().numpy()

        x_next = np.zeros_like(x)
        Sx = np.zeros((batch_size, xdim, xdim))
        Su = np.zeros((batch_size, xdim, udim))

        for i in range(x.shape[0]):
            # TODO (Jasper): Replace with method simulate!
            sim.set("x", x[i, :])
            sim.set("u", u[i, :])
            if p is not None:
                sim.set("p", p[i, :])
            sim.solve()
            x_next[i, :] = sim.get("x")
            Sx[i, ...] = sim.get("Sx")
            Su[i, ...] = sim.get("Su")

        # save the jacobian for the backward pass
        ctx.Sx = Sx
        ctx.Su = Su

        return torch.tensor(x_next, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype

        # compute gradients of the inputs
        Sx = torch.tensor(ctx.Sx, device=device, dtype=dtype)
        Su = torch.tensor(ctx.Su, device=device, dtype=dtype)
        grad_x = torch.einsum("bkj,bk->bj", Sx, grad_output)
        grad_u = torch.einsum("bkj,bk->bj", Su, grad_output)

        # print(Sx[0])
        # print(grad_x[0])

        return (None, grad_x, grad_u, None)


class MPCSolutionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, mpc: MPC, x0: torch.Tensor, p_global_learnable: torch.Tensor, p_rests: list[Parameter],
                initializations: list[SolverState] | None) -> torch.Tensor:
        device = x0.device
        dtype = x0.dtype
        batch_size = x0.shape[0]
        xdim = x0.shape[1]
        pdim = mpc.p_global_dim
        udim = mpc.ocp.dims.nu

        need_dudx = ctx.needs_input_grad[1]
        need_dudp = ctx.needs_input_grad[2]

        x0 = tensor_to_numpy(x0)
        p_global_learnable = tensor_to_numpy(p_global_learnable)
        p_whole = integrate_p_global_learnable_into_p_rest(p_global_learnable, p_rests)

        u = np.zeros((batch_size, udim))
        dudp_global = np.zeros((batch_size, udim, pdim)) if need_dudp else None
        dudx = np.zeros((batch_size, udim, xdim)) if need_dudx else None
        status = np.zeros(batch_size, dtype=np.int8)

        for i in range(batch_size):
            init = initializations if initializations is None else initializations[i]
            u[i, :], status[i], sens = mpc.pi_update(
                x0=x0[i, :], p=p_whole[i], initialization=init, return_dudp=need_dudp, return_dudx=need_dudx)
            if need_dudp:
                dudp_global[i, :, :] = sens[0]
            if need_dudx:
                dudx[i, :, :] = sens[1]

        ctx.dudp_global_learnable = mpc.slice_dudp_global_to_dudp_global_learnable(dudp_global) if need_dudp else None
        ctx.dudx = dudx

        u = torch.tensor(u, device=device, dtype=dtype)
        status = torch.tensor(status, device=device, dtype=torch.int8)

        ctx.mark_non_differentiable(status)

        return u, status

    @ staticmethod
    @ autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):

        dLdu, _ = grad_outputs

        device = dLdu.device
        dtype = dLdu.dtype

        need_dudx = ctx.needs_input_grad[1]
        need_dudp = ctx.needs_input_grad[2]

        if need_dudx:
            dudx = ctx.dudx
            if dudx is None:
                raise ValueError(
                    "ctx.needs_input_grad wrt. x was not True in forward pass, it is not working as we expected.")
            dudx = torch.tensor(dudx, device=device, dtype=dtype)
            grad_x = torch.einsum("bkj,bk->bj", dudx, dLdu)
        else:
            grad_x = None

        if need_dudp:
            dudp_global_learnable = ctx.dudp_global_learnable
            if dudp_global_learnable is None:
                raise ValueError(
                    "ctx.needs_input_grad wrt. p was not True in forward pass, it is not working as we expected.")
            dudp_global_learnable = torch.tensor(dudp_global_learnable, device=device, dtype=dtype)
            grad_p = torch.einsum("bkj,bk->bj", dudp_global_learnable, dLdu)
        else:
            grad_p = None

        return (None, grad_x, grad_p, None, None)


def integrate_p_global_learnable_into_p_rest(p_learnable: np.ndarray, p_rests: list[Parameter]
                                    ) -> list[Parameter]:
    """NOTE: p_rest should not contain a p_learnable attribute that is None since it will be overwritten!
    """
    batch_size = p_learnable.shape[0]
    if len(p_rests) != batch_size:
        raise ValueError("p_rests must be a list of length equal to the batch size.")
    new_p_rests = []
    if p_rests[0].p_global_learnable is not None: # Simple check if p is already in the p_rests
        raise ValueError("p in initialization would be overwritten!")
    for i, sample_p_rest in enumerate(p_rests):
        new_p_rests.append(Parameter(p_learnable[i, :], sample_p_rest[1], sample_p_rest[2], sample_p_rest[3]))
    return new_p_rests