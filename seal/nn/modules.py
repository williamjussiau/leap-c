import casadi as ca
import torch
import torch.nn as nn

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from .autograd import AutogradCasadiFunction, DynamicsSimFunction


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
