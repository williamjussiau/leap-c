from seal.examples.pendulum_on_cart import PendulumOnCartMPC
import numpy as np

import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver
from typing import Tuple


def plot_cart_pole_solution(
    ocp_solver: AcadosOcpSolver,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    k = np.arange(0, ocp_solver.N + 1)
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

    fig, axs = plt.subplots(5, 1)
    labels = ["x", "theta", "dx", "dtheta"]

    for i in range(4):
        axs[i].step(k, x[:, i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid()

    axs[4].step(k[:-1], u)
    axs[4].set_ylabel("F")
    axs[4].set_xlabel("k")
    axs[4].grid()

    return fig, axs


def test_solution(
    mpc: PendulumOnCartMPC = PendulumOnCartMPC(
        learnable_params=["M", "m", "g", "l", "Q", "R"]
    ),
):
    ocp_solver = mpc.ocp_solver
    ocp_solver.solve_for_x0(np.array([0.0, np.pi, 0.0, 0.0]))

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_cart_pole_solution(ocp_solver)


def main():
    test_solution()
    plt.show()


if __name__ == "__main__":
    main()
