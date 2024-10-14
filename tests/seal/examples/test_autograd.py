import numpy as np
import torch

from seal.nn.modules import MPCSolutionModule

from test_linear_system import set_up_mpc, set_up_test_parameters


def test_MPCSolutionModule_on_LinearSystemMPC(generate_code: bool = False,
                                              build_code: bool = False,
                                              json_file_prefix: str = "acados_ocp_linear_system",
                                              x0: np.ndarray = np.array([0.1, 0.1]),
                                              varying_param_label: str = "A_0",
                                              batch_size: int = 1,
                                              ):
    mpc = set_up_mpc(generate_code, build_code, json_file_prefix)

    test_param = set_up_test_parameters(mpc, batch_size, varying_param_label=varying_param_label).T
    mpc_module = MPCSolutionModule(mpc)
    x0 = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
    x0 = torch.tile(x0, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)

    x0.requires_grad = True
    p.requires_grad = True

    assert torch.autograd.gradcheck(mpc_module.forward, (x0, p), atol=1e-3, eps=1e-4)


if __name__ == "__main__":
    test_MPCSolutionModule_on_LinearSystemMPC(build_code=True, generate_code=True, batch_size=2)
