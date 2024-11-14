import numpy as np
import numpy.testing as npt
from seal.mpc import MPC
import pytest

import matplotlib.pyplot as plt


def find_param_index_and_increment(test_param):
    parameter_increment = test_param[:, 1] - test_param[:, 0]

    # Find index of nonzero element in parameter_increment
    parameter_index = np.where(parameter_increment != 0)[0][0]

    return parameter_index, parameter_increment


def compare_acados_value_gradients_to_finite_differences(
    test_param, values, value_gradient_acados, plot: bool = True
):
    # Assumes a constant parameter increment
    parameter_index, parameter_increment = find_param_index_and_increment(test_param)

    value_gradient_finite_differences = np.gradient(
        values, parameter_increment[parameter_index]
    )

    absolute_difference = np.abs(
        np.gradient(values, parameter_increment[parameter_index])
        - value_gradient_acados[:, parameter_index]
    )

    if plot:
        reconstructed_values = np.cumsum(value_gradient_acados @ parameter_increment)
        reconstructed_values += values[0] - reconstructed_values[0]

        relative_difference = absolute_difference / np.abs(
            value_gradient_acados[:, parameter_index]
        )

        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(values, label="original values")
        plt.plot(reconstructed_values, label="reconstructed values")
        plt.ylabel("value")
        plt.grid()
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(
            value_gradient_finite_differences,
            label="value gradient via finite differences",
        )
        plt.plot(
            value_gradient_acados[:, parameter_index], label="value gradient via acados"
        )
        plt.ylabel("value gradient")
        plt.grid()
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(absolute_difference)
        plt.ylabel("absolute difference")
        plt.grid()
        plt.subplot(4, 1, 4)
        plt.plot(relative_difference)
        plt.ylabel("relative difference")
        plt.grid()
        plt.show()

    return absolute_difference


def run_test_state_value_for_varying_parameters(
    mpc: MPC, x0, test_param, plot: bool = False
):
    np_test = test_param.shape[1]

    # parameter_index, _ = find_param_index_and_increment(test_param)

    # Evaluate value and value_gradient using acados
    value = []
    value_gradient = []
    for i in range(np_test):
        v_i, dvdp_i = mpc.state_value(state=x0, p_global=test_param[:, i], sens=True)
        value.append(v_i)
        value_gradient.append(dvdp_i)
    value = np.array(value)
    value_gradient = np.array(value_gradient)

    # Evaluate v and dvdp using finite differences and compare
    try:
        absolute_difference = compare_acados_value_gradients_to_finite_differences(
            test_param, value, value_gradient, plot=plot
        )
    except Exception as e:
        print(e)

    return absolute_difference


def run_test_state_action_value_for_varying_parameters(
    mpc: MPC,
    x0: np.ndarray,
    u0: np.ndarray,
    test_param: np.ndarray,
    plot: bool = False,
):
    # Evaluate q and dqdp using acados
    value = []
    value_gradient = []
    for i in range(test_param.shape[1]):
        q_i, dqdp_i = mpc.state_action_value(
            state=x0, action=u0, p_global=test_param[:, i], sens=True
        )
        value.append(q_i)
        value_gradient.append(dqdp_i)
    value = np.array(value)
    value_gradient = np.array(value_gradient)

    # Evaluate value and value_gradient using finite differences and compare
    absolute_difference = compare_acados_value_gradients_to_finite_differences(
        test_param, value, value_gradient, plot=plot
    )

    return absolute_difference


def run_test_policy_for_varying_parameters(
    mpc: MPC,
    x0,
    test_param,
    use_adj_sens: bool = False,
    plot: bool = False,
) -> np.ndarray:
    # Evaluate v and dvdp using acados
    np_test = test_param.shape[1]

    parameter_index, parameter_increment = find_param_index_and_increment(test_param)

    policy = []
    policy_gradient = []

    for i in range(np_test):
        # p = MPCParameter(
        #     p_global_learnable=None,
        #     p_global_non_learnable=test_param[:, i],
        #     p_stagewise=None,
        #     p_stagewise_sparse_idx=None,
        # )
        pi_i, sens = mpc.policy(
            state=x0, p_global=test_param[:, i], sens=True, use_adj_sens=use_adj_sens
        )
        dpidp_i = sens[0]
        policy.append(pi_i)
        policy_gradient.append(dpidp_i)

    policy = np.array(policy)
    policy_gradient = np.array(policy_gradient)

    policy_gradient_acados = policy_gradient[:, parameter_index]

    # Evaluate pi and dpidp using finite differences and compare
    # Assumes a constant parameter increment
    dp = parameter_increment[parameter_index]
    policy_gradient_finite_differences = np.gradient(policy, dp, axis=0)

    absolute_difference = np.abs(
        policy_gradient_finite_differences - policy_gradient_acados
    )

    if plot:
        reconstructed_policy = np.cumsum(policy_gradient_acados * dp, axis=0)
        reconstructed_policy += policy[0] - reconstructed_policy[0]

        # Avoid division by zero when policy_gradient_acados is zero

        relative_difference = np.zeros_like(absolute_difference)
        if mpc.ocp_solver.acados_ocp.dims.nu == 1:
            relative_difference = absolute_difference / np.abs(policy_gradient_acados)
        else:
            for i in range(np_test):
                for j in range(mpc.ocp_solver.acados_ocp.dims.nu):
                    if np.abs(policy_gradient_acados[i, j]) > 1e-10:
                        relative_difference[i, j] = absolute_difference[i, j] / np.abs(
                            policy_gradient_acados[i, j]
                        )

        if mpc.ocp_solver.acados_ocp.dims.nu == 1:
            fig, ax = plt.subplots(
                4, mpc.ocp_solver.acados_ocp.dims.nu, figsize=(10, 20)
            )
            for i in range(mpc.ocp_solver.acados_ocp.dims.nu):
                ax[0].plot(policy, label="policy")
                ax[0].plot(
                    reconstructed_policy,
                    label="reconstructed policy from policy gradients",
                )
                ax[0].set_ylabel("policy")
                ax[0].legend()
                ax[1].plot(policy_gradient_acados, label="policy gradient via acados")
                ax[1].plot(
                    policy_gradient_finite_differences,
                    label="policy gradient via finite differences",
                )
                ax[1].set_ylabel("policy gradient")
                ax[1].legend()
                ax[2].plot(absolute_difference, label="absolute difference")
                ax[2].set_ylabel("absolute difference")
                ax[3].plot(relative_difference, label="relative difference")
                ax[3].set_ylabel("relative difference")
                for j in range(4):
                    ax[j].grid()
            plt.show()
        else:
            fig, ax = plt.subplots(
                4, mpc.ocp_solver.acados_ocp.dims.nu, figsize=(10, 20)
            )
            for i in range(mpc.ocp_solver.acados_ocp.dims.nu):
                ax[0, i].plot(policy[:, i], label="policy")
                ax[0, i].plot(
                    reconstructed_policy[:, i],
                    label="reconstructed policy from policy gradients",
                )
                ax[1, i].plot(
                    policy_gradient_acados[:, i], label="policy gradient via acados"
                )
                ax[1, i].plot(
                    policy_gradient_finite_differences[:, i],
                    label="policy gradient via finite differences",
                )
                ax[2, i].plot(absolute_difference[:, i], label="absolute difference")
                ax[3, i].plot(relative_difference[:, i], label="relative difference")
                for j in range(4):
                    ax[j, i].legend()
                    ax[j, i].grid()
            plt.legend()
            plt.show()

    return absolute_difference


def test_stage_cost(linear_mpc: MPC):
    x = np.array([0.0, 0.0])
    u = np.array([0.0])

    stage_cost = linear_mpc.stage_cost(x, u)

    assert stage_cost == 0.0


def test_stage_cons(linear_mpc: MPC):
    x = np.array([2.0, 1.0])
    u = np.array([0.0])

    stage_cons = linear_mpc.stage_cons(x, u)

    npt.assert_array_equal(stage_cons["ubx"], np.array([1.0, 0.0]))


def test_policy_gradient_via_adjoint_sensitivity(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params,
):
    for test_param in linear_mpc_test_params:
        absolute_difference = run_test_policy_for_varying_parameters(
            mpc=learnable_linear_mpc,
            x0=np.array([0.1, 0.1]),
            test_param=test_param,
            use_adj_sens=True,
            plot=False,
        )

        assert np.median(absolute_difference) <= 1e-1


def test_policy(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params: list[np.ndarray],
    x0=np.array([0.1, 0.1]),
):
    for i, test_param in enumerate(linear_mpc_test_params):
        absolute_difference = run_test_policy_for_varying_parameters(
            mpc=learnable_linear_mpc,
            x0=x0,
            test_param=test_param,
            use_adj_sens=False,
            plot=False,
        )

        assert np.median(absolute_difference) <= 1e-1


def test_state_value(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params: list[np.ndarray],
    x0: np.ndarray = np.array([0.1, 0.1]),
):
    for test_param in linear_mpc_test_params:
        absolute_difference = run_test_state_value_for_varying_parameters(
            mpc=learnable_linear_mpc, x0=x0, test_param=test_param, plot=False
        )

        assert np.median(absolute_difference) <= 1e-1


def test_state_action_value(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params: list[np.ndarray],
    x0: np.ndarray = np.array([0.1, 0.1]),
    u0: np.ndarray = np.array([0.0]),
):
    for test_param in linear_mpc_test_params:
        absolute_difference = run_test_state_action_value_for_varying_parameters(
            mpc=learnable_linear_mpc,
            x0=x0,
            u0=u0,
            test_param=test_param,
            plot=False,
        )

        assert np.median(absolute_difference) <= 1e-1


def test_closed_loop(
    learnable_linear_mpc: MPC,
    x0: np.ndarray = np.array([0.5, 0.5]),
):
    x = [x0]
    u = []

    p_global = learnable_linear_mpc.ocp.p_global_values

    for _ in range(100):
        u.append(learnable_linear_mpc.policy(x[-1], p_global=p_global)[0])
        x.append(learnable_linear_mpc.ocp_solver.get(1, "x"))
        assert learnable_linear_mpc.ocp_solver.get_status() == 0

    x = np.array(x)
    u = np.array(u)

    assert (
        np.median(x[-10:, 0]) <= 1e-1
        and np.median(x[-10:, 1]) <= 1e-1
        and np.median(u[-10:]) <= 1e-1
    )


if __name__ == "__main__":
    pytest.main([__file__])
