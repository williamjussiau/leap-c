import numpy as np
import pytest
import casadi as ca
import torch

from acados_template import AcadosSim, AcadosSimOpts, AcadosModel, AcadosSimSolver
from acados_il.autograd import CasadiExprModule, AcadosSimModule


def test_casadi_function_addition():
    # create a casadi function
    x = ca.SX.sym("x", 1)
    y = ca.SX.sym("y", 1)

    expr = x + 2 * y

    # create a torch module
    module = CasadiExprModule(expr, [x, y])

    # call the module
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])

    result = module(x, y).item()
    assert result == 5

    # call the module with a batch of inputs
    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y = torch.tensor([[3.0], [4.0], [5.0]], requires_grad=True)

    result = module(x, y)
    assert torch.all(result == torch.tensor([[7.0], [10.0], [13.0]]))

    # test backward pass
    result = module(x, y)
    result.backward(torch.ones_like(result))

    assert torch.all(x.grad == torch.tensor([[1.0], [1.0], [1.0]]))
    assert torch.all(y.grad == torch.tensor([[2.0], [2.0], [2.0]]))



def test_casadi_function_vector():
    # create a casadi function
    x = ca.SX.sym("x", 4)
    A = np.diag([1, 2, 3, 4])

    expr = A @ x

    # create a torch module
    module = CasadiExprModule(expr, [x])

    # call the module
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    result = module(x)

    assert torch.all(result == torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    # test backward pass
    result.backward(torch.ones_like(result))

    assert torch.all(x.grad == torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_acados_linear_dynamics(tmp_path):
    model = AcadosModel()
    model.name = "linear_dynamics"

    # states
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u")
    p = ca.SX.sym("p")

    # dynamics
    xdot = ca.vertcat(x[0], x[1])

    # set up the model
    model.f_expl_expr = xdot
    model.x = x
    model.u = u
    model.p = p

    tmp_path.mkdir(parents=True, exist_ok=True)

    sim = AcadosSim()
    sim.solver_options.T = 1.
    sim.solver_options.sens_forw = True
    sim.solver_options.sens_adj = True
    sim.solver_options.sens_hess = True
    sim.solver_options.integrator_type = "ERK"
    sim.code_export_directory = tmp_path 
    sim.model = model
    sim.parameter_values = np.array([0.])

    sim_solver = AcadosSimSolver(sim, tmp_path / "acados_ocp.json")

    # create a torch module
    module = AcadosSimModule(sim_solver)

    x = torch.tensor([[0.0, 0.0]], requires_grad=True)
    u = torch.tensor([[3.0]], requires_grad=True)
    p = torch.tensor([[4.0]], requires_grad=True)

    # call the module
    result = module(x, u, p)
    assert torch.all(result == torch.tensor([[0.0, 0.0]]))

    # test backward pass
    result.backward(torch.ones_like(result))

    # TODO (Jasper): Check this again
    assert torch.all(x.grad == torch.tensor([[2.0, 2.0]]))
    assert torch.all(u.grad == torch.tensor([[0.0]]))


