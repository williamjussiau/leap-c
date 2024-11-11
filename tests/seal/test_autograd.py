from functools import partial

import numpy as np
import torch
from examples.test_linear_system import set_up_mpc, set_up_test_parameters

from seal.mpc import MPCParameter
from seal.nn.modules import CleanseAndReducePerSampleLoss, MPCSolutionModule


# def test_MPCSolutionModule_on_LinearSystemMPC(
#     generate_code: bool = False,
#     build_code: bool = False,
#     json_file_prefix: str = "acados_ocp_linear_system",
#     x0: np.ndarray = np.array([0.1, 0.1]),
#     varying_param_label: str = "A_0",
#     batch_size: int = 1,
# ):
#     # We only vary A_0 here but still mark all parameters as learnable parameters such that the whole gradient wrt. p_global is checked.
#     mpc = set_up_mpc(
#         generate_code,
#         build_code,
#         json_file_prefix,
#         learnable_params=["A", "B", "b", "f", "V_0"],
#     )
# 
#     test_param = set_up_test_parameters(
#         mpc, batch_size, varying_param_label=varying_param_label
#     ).T  # (batch_size, 12)
# 
#     p_rests = [MPCParameter(None, None, None, None) for i in range(batch_size)]
# 
#     mpc_module = MPCSolutionModule(mpc)
#     x0 = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
#     x0 = torch.tile(x0, (batch_size, 1))
#     p = torch.tensor(test_param, dtype=torch.float64)
# 
#     x0.requires_grad = True
#     p.requires_grad = True
# 
#     fix_p_rests = partial(mpc_module.forward, p_rests=p_rests)
# 
#     torch.autograd.gradcheck(
#         fix_p_rests, (x0, p), atol=1e-2, eps=1e-4, raise_exception=True
#     )


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


if __name__ == "__main__":
    test_MPCSolutionModule_on_LinearSystemMPC()
