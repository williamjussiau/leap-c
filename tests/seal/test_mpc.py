from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd

from seal.examples.linear_system import LinearSystemMPC
from seal.mpc import MPC, MPCInput, MPCOutput, MPCParameter
from seal.util import SX_to_labels


def compute_diff_solve_batch_solve(results, mpc):
    keys = ["state_value", "state_action_value", "policy"]
    fields = ["value", "gradient"]

    diff_solve_batch_solve = {
        key: {field: np.zeros((mpc.n_batch, mpc.p_global_dim)) for field in fields}
        for key in ["state_value", "state_action_value", "policy"]
    }

    for key in keys:
        for field in fields:
            diff_solve_batch_solve[key][field] = (
                results["_solve"][key][field] - results["_batch_solve"][key][field]
            )

    return diff_solve_batch_solve


def print_median_table(
    diff_solve_batch_solve: dict,
    param_names: list[str],
    keys: list[str] = ["state_value", "state_action_value", "policy"],
    fields: list[str] = ["value", "gradient"],
) -> None:
    """
    Print a formatted table of median absolute differences with parameter names as headers.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
        param_names (list[str]): List of parameter names
        keys (list[str]): List of keys to analyze
        fields (list[str]): List of fields to analyze
    """
    # Import pandas for nice table formatting

    pd.set_option("display.float_format", lambda x: "{:.2e}".format(x))

    # Create rows for the table
    rows = []
    for key in keys:
        for field in fields:
            data = diff_solve_batch_solve[key][field]

            # Check for NaN values
            if np.isnan(data).any():
                print(f"\nWarning: NaN values found in {key} - {field}")

            # Calculate median absolute differences
            median_values = np.median(np.abs(data), axis=0)

            # Create row dictionary
            row = {"Component": f"{key} - {field}"}
            row.update(
                {param: value for param, value in zip(param_names, median_values)}
            )
            rows.append(row)

    # Create DataFrame and display
    df = pd.DataFrame(rows)
    df = df.set_index("Component")

    print("\nMedian Absolute Differences:")
    print(df)

    # Additional statistics
    print("\nSummary:")
    print(f"Largest median difference: {df.max().max():.2e}")
    print(f"Smallest median difference: {df.min().min():.2e}")

    # Identify parameters with largest differences
    max_param = df.max().idxmax()
    print(f"Parameter with largest median difference: {max_param}")


def plot_solve_differences(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    keys: list[str] = ["state_value", "state_action_value", "policy"],
    fields: list[str] = ["value", "gradient"],
    fig_size: tuple = (15, 10),
    scale="semilog",
) -> plt.Figure:
    """
    Create a grid of plots showing the differences between solve and batch_solve results.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
        keys (list[str]): List of keys to plot
        fields (list[str]): List of fields to plot
        fig_size (tuple): Figure size in inches

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    diff_solve_batch_solve = compute_diff_solve_batch_solve(results, mpc)

    # Print summary statistics
    # analyze_nan_ratios(diff_solve_batch_solve)
    # Create subplot grid
    n_rows = len(keys)
    n_cols = len(fields)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    # If only one row or column, make axes 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot each combination of key and field
    for i, key in enumerate(keys):
        for j, field in enumerate(fields):
            ax = axes[i, j]
            data = diff_solve_batch_solve[key][field]

            # Check if data includes nan
            if np.isnan(data).any():
                print(f"NaN values found in {key} - {field}")

            # Create parameter names based on dimension

            # print(f"Plotting {key} - {field} differences:")
            # print("median absolute difference:")
            param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)
            # print(np.median(data, axis=0))

            # Plot using the provided function
            plot_batch_params(
                batch_data=data,
                param_names=param_names,
                title=f"{key} - {field}",
                ylabel="Difference"
                if j == 0
                else None,  # Only add ylabel for first column
                alpha=0.5,
                ax=ax,
                scale=scale,
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Give a super title to the fig
    fig.suptitle("Differences between _solve and _batch_solve results", fontsize=16)
    return fig


def print_difference_statistics(diff_solve_batch_solve: dict) -> None:
    """
    Print summary statistics for the differences.

    Args:
        diff_solve_batch_solve (dict): Nested dictionary containing differences
    """
    for key in diff_solve_batch_solve:
        print(f"\n{key}:")
        for field in diff_solve_batch_solve[key]:
            data = diff_solve_batch_solve[key][field]
            abs_data = np.abs(data)
            print(f"  {field}:")
            print(f"    Max absolute difference: {np.max(abs_data):.2e}")
            print(f"    Mean absolute difference: {np.mean(abs_data):.2e}")
            print(f"    Median absolute difference: {np.median(abs_data):.2e}")


def plot_gradient_comparison_grid(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    n_width: int = 4,
) -> plt.Figure:
    """
    Plot gradient comparisons in a grid of subplots.

    Args:
        results (dict): Dictionary containing gradient results
        mpc: MPC object containing model information
        n_width (int): Number of subplots horizontally
    """
    n_params = mpc.p_global_dim

    # Calculate number of rows needed
    n_rows = int(np.ceil(n_params / n_width))

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_width, figsize=(5 * n_width, 4 * n_rows))

    # Flatten axes for easier indexing if there are multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_width > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create each subplot
    for i_param in range(n_params):
        ax = axes[i_param]

        # Plot gradients
        ax.plot(
            results["_batch_solve"]["state_value"]["gradient"][:, i_param],
            label="gradient",
        )
        ax.plot(
            results["_batch_solve"]["state_value"]["gradient_fd"][:, i_param],
            label="gradient_fd",
        )

        # Add labels and title
        ax.set_title(SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)[i_param])
        ax.set_ylabel("gradient")
        ax.set_xlabel("batch index")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def plot_batch_params(
    batch_data: np.ndarray,
    param_names: list[str] = None,
    title: str = None,
    ylabel: str = None,
    alpha: float = 0.5,
    ax: plt.Axes = None,
    scale: str = "semilog",
) -> plt.Axes:
    """
    Create a plot showing values as dots arranged vertically for each parameter index.

    Args:
        batch_data (np.ndarray): Array of shape (n_batch, n_param) containing the data to plot
        param_names (list, optional): List of parameter names for x-axis labels
        title (str, optional): Plot title
        ylabel (str, optional): Label for y-axis
        alpha (float, optional): Transparency of dots
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used.
    """
    if ax is None:
        ax = plt.gca()

    n_batch, n_param = batch_data.shape

    # For each parameter
    for param_idx in range(n_param):
        # Get all batch values for this parameter
        param_values = batch_data[:, param_idx]

        # Create slight random x-offset for better visibility when values overlap
        x = np.random.normal(param_idx, 0.02, size=n_batch)

        # Calculate and plot median
        median_value = np.median(param_values)
        if scale == "semilog":
            ax.semilogy(x, param_values, "o", alpha=alpha, markersize=6)
            ax.semilogy(
                [param_idx - 0.2, param_idx + 0.2],  # x coordinates for horizontal line
                [median_value, median_value],  # y coordinates for horizontal line
                color="red",
                linewidth=2,
                solid_capstyle="butt",
            )
        else:
            ax.plot(x, param_values, "o", alpha=alpha, markersize=6)
            ax.plot(
                [param_idx - 0.2, param_idx + 0.2],  # x coordinates for horizontal line
                [median_value, median_value],  # y coordinates for horizontal line
                color="red",
                linewidth=2,
                solid_capstyle="butt",
            )

    # Set x-axis ticks and labels
    if param_names is not None:
        ax.set_xticks(range(n_param))
        ax.set_xticklabels(param_names, rotation=45)

    # Add labels and title
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    return ax


def plot_gradient_diff_comparison(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
    figsize: Tuple[float, float] = (15, 8),
    scale="semilog",
) -> plt.Figure:
    """
    Plot gradient differences for different methods and keys in a 3x2 grid.

    Args:
        results (dict): Dictionary containing gradient results
        mpc: MPC object containing model information
        figsize (tuple): Figure size (width, height)
    """
    # Create figure with 3x2 grid
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Get parameter names once
    param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)

    # Keys for columns
    keys = ["state_value", "state_action_value", "policy"]

    # Methods for rows
    methods = ["_solve", "_batch_solve"]

    # Create plots
    for col, key in enumerate(keys):
        for row, method in enumerate(methods):
            plot_batch_params(
                results[method][key]["gradient_diff"],
                param_names=param_names,
                ylabel="gradient difference"
                if col == 0
                else "",  # Only show ylabel on leftmost plots
                title=f"{key} via {method}".replace("_", " "),
                ax=axes[row, col],
                scale=scale,
            )

            # Add horizontal line at y=0 for reference
            axes[row, col].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig


def mpc_outputs_assert_allclose(
    mpc_output: MPCOutput, mpc_output2: MPCOutput, test_u_star: bool
):
    allclose = True
    for fld in mpc_output._fields:
        val1 = getattr(mpc_output, fld)
        val2 = getattr(mpc_output2, fld)
        if fld == "u0" and not test_u_star:
            continue  # Testing u_star when different u0 were given makes no sense
        if isinstance(val1, np.ndarray):
            tolerance = (
                1e-5 if not fld.startswith("d") else 1e-3
            )  # 1e-3 is probably close enough for gradients, at least thats also what we do in pytorch.gradcheck wrt the numerical gradient
            assert np.allclose(
                val1, val2, atol=tolerance
            ), f"Field {fld} not close, maximal difference is {np.abs(val1 - val2).max()}"
        elif isinstance(val1, type(None)):
            assert val1 == val2
        else:
            raise NotImplementedError(
                "Only np.ndarray fields known. Did new fields get added to MPCOutput?"
            )
    return allclose


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


def test_statelessness(
    x0: np.ndarray = np.array([0.5, 0.5]), u0: np.ndarray = np.array([0.5])
):
    # Create MPC with some stateless and some global parameters
    lin_mpc = LinearSystemMPC(learnable_params=["A", "B", "Q", "R", "f"])
    mpc_input_standard = MPCInput(x0=x0, u0=u0)
    solution_standard, _ = lin_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    p_global = lin_mpc.default_p_global
    assert p_global is not None
    p_global = p_global + np.ones(p_global.shape[0]) * 0.01
    p_stagewise = lin_mpc.default_p_stagewise
    assert p_stagewise is not None
    p_stagewise = p_stagewise + np.ones(p_stagewise.shape[0]) * 0.01
    assert (
        len(p_stagewise.shape) == 1
    ), f"I assumed this would be flat, but shape is {p_stagewise.shape}"
    p_stagewise = np.tile(p_stagewise, (lin_mpc.N + 1, 1))
    params = MPCParameter(p_global, p_stagewise)
    x0_different = x0 - 0.01
    u0_different = u0 - 0.01
    mpc_input_different = MPCInput(x0=x0_different, u0=u0_different, parameters=params)
    solution_different, _ = lin_mpc(
        mpc_input=mpc_input_different, dudp=True, dvdp=True, dudx=True
    )
    # Use this as proxy to verify the different solution is different enough
    assert not np.allclose(
        solution_standard.Q,  # type:ignore
        solution_different.Q,  # type:ignore
    )
    solution_supposedly_standard, _ = lin_mpc(
        mpc_input=mpc_input_standard, dudp=True, dvdp=True, dudx=True
    )
    mpc_outputs_assert_allclose(
        solution_standard, solution_supposedly_standard, test_u_star=True
    )


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


def initialize_results(mpc: MPC) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Initialize the results dictionary structure."""
    results = {}
    for method in ["_solve", "_batch_solve"]:
        results[method] = {}
        for key in ["policy", "state_value", "state_action_value"]:
            results[method][key] = {
                "value": np.zeros(shape=(mpc.n_batch, mpc.p_global_dim)),
                "gradient": np.zeros(shape=(mpc.n_batch, mpc.p_global_dim)),
                "gradient_fd": np.zeros(shape=(mpc.n_batch, mpc.p_global_dim)),
                "gradient_diff": np.zeros(shape=(mpc.n_batch, mpc.p_global_dim)),
            }
    return results


def run_batch_solve(
    mpc: MPC,
    x0: np.ndarray,
    u0: np.ndarray,
    p_global: np.ndarray,
    i_param: int,
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
):
    """Execute batch solve computations for all value types."""
    computations = {
        "state_value": lambda: mpc.state_value(state=x0, p_global=p_global, sens=True),
        "state_action_value": lambda: mpc.state_action_value(
            state=x0, action=u0, p_global=p_global, sens=True
        ),
        "policy": lambda: mpc.policy(
            state=x0, p_global=p_global, sens=True, use_adj_sens=True
        ),
    }

    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    for key, compute_func in computations.items():
        batch_value, batch_gradient = compute_func()
        results["_batch_solve"][key]["value"][:, i_param] = (
            batch_value if key != "policy" else batch_value
        ).flatten()
        results["_batch_solve"][key]["gradient"][:, i_param] = (
            batch_gradient[:, i_param]
            if key != "policy"
            else batch_gradient[:, 0, i_param]
        )


def run_individual_solve(
    mpc: MPC,
    x0: np.ndarray,
    u0: np.ndarray,
    p_global: np.ndarray,
    i_param: int,
    k: int,
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
):
    """Execute individual solve computations for all value types."""
    computations = {
        "state_value": lambda: mpc.state_value(
            state=x0[k, :], p_global=p_global[k, :], sens=True
        ),
        "state_action_value": lambda: mpc.state_action_value(
            state=x0[k, :], action=u0[k, :], p_global=p_global[k, :], sens=True
        ),
        "policy": lambda: mpc.policy(
            state=x0[k, :], p_global=p_global[k, :], sens=True, use_adj_sens=True
        ),
    }

    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    for key, compute_func in computations.items():
        solve_value, solve_gradient = compute_func()
        results["_solve"][key]["value"][k, i_param] = solve_value
        results["_solve"][key]["gradient"][k, i_param] = (
            solve_gradient[i_param]
            if key != "policy"
            else solve_gradient.reshape(-1)[i_param]
        )


def compute_finite_differences(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    p_global: np.ndarray,
    i_param: int,
):
    """Compute finite differences and gradient differences."""
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value", "policy"]:
            results[method][key]["gradient_fd"][:, i_param] = np.gradient(
                results[method][key]["value"][:, i_param],
                p_global[:, i_param],
                edge_order=2,
            )

            results[method][key]["gradient_diff"][:, i_param] = np.abs(
                results[method][key]["gradient_fd"][:, i_param]
                - results[method][key]["gradient"][:, i_param]
            )


def verify_shapes(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mpc: MPC,
):
    # TODO: This assumes nu = 1 for the policy. Modify to cover the case where nu > 1
    """Verify the shapes of all result arrays."""
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value", "policy"]:
            assert results[method][key]["value"].shape[0] == mpc.n_batch
            assert results[method][key]["gradient"].shape == (
                mpc.n_batch,
                mpc.p_global_dim,
            )
            assert results[method][key]["gradient_fd"].shape == (
                mpc.n_batch,
                mpc.p_global_dim,
            )


def test_mpc_solve_and_batch_solve_on_batch_p_global(
    learnable_linear_mpc: MPC,
    linear_mpc_test_params,
    plot: bool = False,
) -> None:
    mpc = learnable_linear_mpc

    test_x_init = 0.1 * np.ones(
        (
            linear_mpc_test_params.shape[0],
            mpc.ocp.dims.nx,
            linear_mpc_test_params.shape[2],
        )
    )

    test_u_init = 0.2 * np.ones(
        (
            linear_mpc_test_params.shape[0],
            mpc.ocp.dims.nu,
            linear_mpc_test_params.shape[2],
        )
    )

    # Initialize results structure
    results = initialize_results(mpc)

    # Main computation loop
    for i_param in range(linear_mpc_test_params.shape[2]):
        p_global = linear_mpc_test_params[:, :, i_param]
        x_init = test_x_init[:, :, i_param]
        u_init = test_u_init[:, :, i_param]
        print(SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)[i_param])

        # Run batch and individual solves
        run_batch_solve(mpc, x_init, u_init, p_global, i_param, results)
        for k in range(mpc.n_batch):
            run_individual_solve(mpc, x_init, u_init, p_global, i_param, k, results)

        # Compute finite differences
        compute_finite_differences(results, p_global, i_param)

    # Verify results
    verify_shapes(results, mpc)

    tol = {"state_value": 1e-3, "state_action_value": 1e-3, "policy": 1e-2}
    for method in ["_solve", "_batch_solve"]:
        for key in ["state_value", "state_action_value"]:
            assert (
                np.median(
                    np.abs(
                        results[method]["state_value"]["gradient"]
                        - results[method]["state_value"]["gradient_fd"]
                    )
                )
                < tol[key]
            ), f"Method {method} key {key} gradient not close enough, maximal difference is {np.abs(results[method][key]['gradient'] - results[method][key]['gradient_fd']).max()}"

    diff_solve_batch_solve = compute_diff_solve_batch_solve(results, mpc)

    # Stack diff_solve_batch_solve into a single array and compute the norm
    diff_solve_batch_solve_norm = np.linalg.norm(
        np.stack(
            [
                diff_solve_batch_solve[key][field]
                for key in diff_solve_batch_solve
                for field in diff_solve_batch_solve[key]
            ]
        )
    )

    assert (
        diff_solve_batch_solve_norm == 0.0
    ), f"Norm difference between _solve and _batch_solve should be 0, but is: {diff_solve_batch_solve_norm}"

    for key in diff_solve_batch_solve.keys():
        for field in diff_solve_batch_solve[key].keys():
            assert (
                np.median(np.abs(diff_solve_batch_solve[key][field])) < 1e-5
            ), f"Method {method} key {key} gradient not close enough, maximal difference is {np.abs(results[method][key]['gradient'] - results[method][key]['gradient_fd']).max()}"

    print_difference_statistics(diff_solve_batch_solve)
    keys = ["state_value", "state_action_value", "policy"]
    fields = ["value", "gradient"]
    for i, key in enumerate(keys):
        for j, field in enumerate(fields):
            data = diff_solve_batch_solve[key][field]

            # Check if data includes nan
            assert not np.isnan(data).any(), f"NaN values found in {key} - {field}"

            # Create parameter names based on dimension
            param_names = SX_to_labels(mpc.ocp_solver.acados_ocp.model.p_global)
            print_median_table(diff_solve_batch_solve, param_names, keys, fields)

    # Plot results if requested
    if plot:
        _ = plot_gradient_diff_comparison(results, mpc, scale="semilog")
        _ = plot_gradient_comparison_grid(results, mpc)
        _ = plot_solve_differences(results, mpc, scale="semilog")
        plt.show()


def linear_mpc_test_params(
    learnable_linear_mpc: MPC,
    n_batch: int,
    width: float,
) -> np.ndarray:
    """Set up test parameters for the linear system MPC.

    Args:
        learnable_linear_mpc (MPC): Linear system MPC with learnable parameters.
        n_batch (int): Number of test parameters to generate.

    Returns:
    --------
    np.ndarray (n_batch, n_p_global, n_p_global): Test parameters for the linear system MPC where n_p_global is the number of
    global parameters. Varying parameters are for on column each along the (:, :, i) axis.
    """

    mean = learnable_linear_mpc.ocp_solver.acados_ocp.p_global_values

    width = np.array([width * p if np.abs(p) > 1e-6 else width for p in mean])

    # repeat mean into an array with shape (n_batch, n_param)
    params = np.tile(mean, (n_batch, 1))

    # repeat mean along a third axis
    params = np.repeat(params[:, :, np.newaxis], mean.shape[0], axis=2)

    for i, idx in enumerate(np.arange(len(mean))):
        params[:, idx, i] = np.linspace(
            mean[idx] - width[idx], mean[idx] + width[idx], n_batch
        )

    return params


if __name__ == "__main__":
    n_batch = 32
    width = 0.05
    mpc = LinearSystemMPC(learnable_params=["A", "B", "Q", "R", "b"], n_batch=n_batch)

    linear_mpc_test_params = linear_mpc_test_params(mpc, n_batch, width)

    test_mpc_solve_and_batch_solve_on_batch_p_global(
        mpc, linear_mpc_test_params, plot=False
    )

    # pytest.main([__file__])
