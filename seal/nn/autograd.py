"""Provides a PyTorch module that wraps a CasADi expression.

TODO (Jasper): Does JIT compilation speed up the module?
"""
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from seal.util import tensor_to_numpy
from seal.mpc import MPC


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
    def forward(ctx, mpc: MPC, x0: torch.Tensor, p: torch.Tensor,
                initializations: list[dict[str, np.ndarray]] | None) -> torch.Tensor:
        device = x0.device
        dtype = x0.dtype
        batch_size = x0.shape[0]
        xdim = x0.shape[1]
        pdim = p.shape[1]
        udim = mpc.ocp.dims.nu

        need_dudx = ctx.needs_input_grad[1]
        need_dudp = ctx.needs_input_grad[2]

        x0 = tensor_to_numpy(x0)
        p = tensor_to_numpy(p)
        initializations = integrate_p_into_initialization(p, initializations, stages=mpc.N+1)

        u = np.zeros((batch_size, udim))
        dudp = np.zeros((batch_size, udim, pdim)) if need_dudp else None
        dudx = np.zeros((batch_size, udim, xdim)) if need_dudx else None
        status = np.zeros(batch_size, dtype=np.int8)

        for i in range(batch_size):
            u[i, :], status[i], sens = mpc.pi_update(
                x0=x0[i, :], initialization=initializations[i], return_dudp=need_dudp, return_dudx=need_dudx)
            if need_dudp:
                dudp[i, :, :] = sens[0]
            if need_dudx:
                dudx[i, :, :] = sens[1]

        ctx.dudp = dudp
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
            dudp = ctx.dudp
            if dudp is None:
                raise ValueError(
                    "ctx.needs_input_grad wrt. p was not True in forward pass, it is not working as we expected.")
            dudp = torch.tensor(dudp, device=device, dtype=dtype)
            grad_p = torch.einsum("bkj,bk->bj", dudp, dLdu)
        else:
            grad_p = None

        return (None, grad_x, grad_p, None, None)


def integrate_p_into_initialization(p: np.ndarray,
                                    initializations: list[dict[str, np.ndarray]] | None,
                                    stages: int) -> list[dict[str, np.ndarray]]:
    """NOTE: If initializations is not None, it will be modified.
    """
    batch_size = p.shape[0]
    if initializations is not None and len(initializations) != batch_size:
        raise ValueError("initializations must be a list of dictionaries with length equal to the batch size.")

    if initializations is not None:
        if "p" in initializations[0].keys():
            raise ValueError("p in initialization would be overwritten!")
        for i, sample_init in initializations:
            sample_init["p"] = np.broadcast_to(p[i, :], shape=(stages, 12))
    else:
        initializations = [{"p": np.broadcast_to(p[i, :], shape=(stages, 12))} for i in range(batch_size)]
    return initializations
