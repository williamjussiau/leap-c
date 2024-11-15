import numpy as np
import torch

from seal.examples.linear_system import LinearSystemMPC
from seal.mpc import MPCParameter
from seal.nn.modules import CleanseAndReducePerSampleLoss, MPCSolutionModule


def test_MPCSolutionModule_on_LinearSystemMPC(
    learnable_linear_mpc: LinearSystemMPC,
    linear_mpc_test_params: np.ndarray,
    x0: np.ndarray = np.array([0.1, 0.1]),
    batch_size: int = 2,
):
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."
    test_param = linear_mpc_test_params[:batch_size]  # batch_size-many (#p_glob, 10)
    different_params = []
    for i in range(batch_size):
        different_params.append(
            test_param[i][:, i]  # Sort-of-diagonal
        )
    test_param = np.stack(different_params, axis=0)  # (batch_size, #p_glob)

    p_rests = MPCParameter(None, None, None)

    mpc_module = MPCSolutionModule(learnable_linear_mpc)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor([0.5], dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, val, status = mpc_module.forward(
            x0, u0=None, p_global=p, p_stagewise=p_rests, initializations=None
        )
        return u_star, status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, val, status = mpc_module.forward(
            x0, u0=None, p_global=p, p_stagewise=p_rests, initializations=None
        )
        return val, status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        u0.requires_grad = False
        u_star, Q, status = mpc_module.forward(
            x0, u0=u0, p_global=p, p_stagewise=p_rests, initializations=None
        )
        u0.requires_grad = True
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, V, status = mpc_module.forward(
            x0=x0_torch,
            u0=None,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        return u_star, status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, V, status = mpc_module.forward(
            x0=x0_torch,
            u0=None,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        return V, status

    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u0.requires_grad = False
        u_star, Q, status = mpc_module.forward(
            x0=x0_torch,
            u0=u0,
            p_global=p_global,
            p_stagewise=p_rests,
            initializations=None,
        )
        u0.requires_grad = True
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_star, Q, status = mpc_module.forward(
            x0=x0_torch, u0=u0, p_global=p, p_stagewise=p_rests, initializations=None
        )
        assert torch.all(
            torch.isnan(u_star)
        ), "u_star should be nan, since u0 is given."
        return Q, status

    # TODO: Turn on when sensitivities are implemented
    # torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)


def test_CleanseAndReduce():
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=1,
        n_nonconvergences_allowed=2,
        throw_exception_if_exceeded=False,
    )

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    status = torch.tensor([[0], [1], [0], [0]], dtype=torch.int8)
    loss = cleansed_loss(x, status)
    assert loss.item() == 8 / 3

    status = torch.tensor([[0], [0], [0], [0]], dtype=torch.int8)
    loss = cleansed_loss(x, status)
    assert loss.item() == 10 / 4

    status = torch.tensor([[2], [0], [1], [0]], dtype=torch.int8)
    loss = cleansed_loss(x, status)
    assert loss.item() == 6 / 2

    status = torch.tensor([[2], [2], [1], [0]], dtype=torch.int8)
    loss = cleansed_loss(x, status)
    assert loss.item() == 0.0

    status = torch.tensor([[1], [0], [1]], dtype=torch.int8)
    try:
        loss = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True


def test_CleanseAndReduceMultipleBatchAndSampleDims():
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=2,
        n_nonconvergences_allowed=4,
        throw_exception_if_exceeded=False,
    )

    x = torch.ones((3, 3, 3, 3))
    x[0, 0] = 2
    x[0, 1] = 100
    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    loss = cleansed_loss(x, status)
    assert loss.item() == 1.0

    x = torch.ones((3, 3, 3, 3))
    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 4
    status[1, 2] = 5
    loss = cleansed_loss(x, status)
    assert loss.item() == 1.0

    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 1
    status[1, 2] = 2
    status[2, 2] = 2
    loss = cleansed_loss(x, status)
    assert loss.item() == 0.0

    status = torch.zeros((3, 3, 3, 1), dtype=torch.int8)
    try:
        loss = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True

    status = torch.zeros((3, 1), dtype=torch.int8)
    try:
        loss = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True

    status = torch.ones((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 1
    status[1, 2] = 2
    status[2, 2] = 2
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=2,
        n_nonconvergences_allowed=4,
        throw_exception_if_exceeded=True,
    )
    try:
        loss = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True
