import numpy as np
from scipy.special import erfcx
import os
from generate_dataset import calculate_Cu_Cl, generate_iv_data_a_fixed_trained_model
from Optimised_BS import Delta_Z_ht, C_ht, C_ht_default
from train_save import *
from datetime import datetime
import matplotlib.pyplot as plt

Householder_EPSILON = 1e-18


def Householder_iter(A, C_0, B_0, n, mode="advanced"):
    """
    Improves the estimate of B using n Householder iterations.
    Supports vectorized inputs for A, C_0, and B_0.

    Args:
        A: Parameter A (scalar or array)
        C_0: Target C values (scalar or array)
        B_0: Initial guesses for B (scalar or array)
        n: Number of iterations
        mode: "advanced" uses C_ht, "default" uses C_ht_default
    """
    # Ensure inputs are numpy arrays and broadcastable
    A = np.asarray(A, dtype=np.float64)
    C_0 = np.asarray(C_0, dtype=np.float64)
    B_curr = np.asarray(B_0, dtype=np.float64)

    # Broadcast to a common shape to allow vectorized operations across mixed input types
    shape = np.broadcast_shapes(A.shape, C_0.shape, B_curr.shape)
    A = np.broadcast_to(A, shape)
    C_0 = np.broadcast_to(C_0, shape)
    B_curr = B_curr.copy() if B_curr.shape == shape else np.broadcast_to(B_curr, shape).copy()

    # Pre-calculate C_u and C_l once for the batch based on A
    C_u, C_l = calculate_Cu_Cl(A, mode=mode)

    # Track the history of iterations: [B_0, B_1, ..., B_n]
    history = [B_curr.copy()]

    # Mathematical Constants
    SQRT_2PI_INV = 1.0 / np.sqrt(2 * np.pi)
    LN_2PI_HALF = 0.5 * np.log(2 * np.pi)

    for _ in range(n):
        B = B_curr
        h = A / B
        t = B / 2.0
        u = -h + t
        v = -(h + t)

        # Householder 3rd order involves derivatives of u w.r.t B
        u_prime = A / (B ** 2) + 0.5
        u_double_prime = -2.0 * A / (B ** 3)

        # Base components for h2 and h3
        C21 = -u * u_prime
        C31 = (u * u_prime) ** 2 - u * u_double_prime - (u_prime ** 2)

        # Containers for the iteration parameters
        nu = np.zeros_like(B)
        h2 = np.zeros_like(B)
        h3 = np.zeros_like(B)

        # --- Boolean Masks for Branching ---
        case1_mask = C_0 < C_l
        case2_mask = C_0 > C_u
        case3_mask = ~(case1_mask | case2_mask)

        # --- Case 1: C_0 < C_l (Lower Tail) ---
        if np.any(case1_mask):
            h_c1 = h[case1_mask]
            t_c1 = t[case1_mask]
            dz = Delta_Z_ht(h_c1, t_c1)

            L = np.log(dz) - LN_2PI_HALF - 0.5 * (h_c1 - t_c1) ** 2
            C01 = dz
            log_C0 = np.log(C_0[case1_mask])

            nu[case1_mask] = ((log_C0 - L) / log_C0) * C01 * L
            h2[case1_mask] = C21[case1_mask] - (1.0 + 2.0 / L) / C01
            h3[case1_mask] = (C31[case1_mask] +
                              (2.0 + (6.0 / L) * (1.0 + 1.0 / L)) / (C01 ** 2) -
                              3.0 * (1.0 + 2.0 / L) * (C21[case1_mask] / C01))

        # --- Case 2: C_0 > C_u (Upper Tail) ---
        if np.any(case2_mask):
            u_c2 = u[case2_mask]
            v_c2 = v[case2_mask]

            # Using erfcx for numerical stability (scaled complementary error function)
            u_s = u_c2 / np.sqrt(2.0)
            v_s = -v_c2 / np.sqrt(2.0)
            e_sum = erfcx(u_s) + erfcx(v_s)

            Ltilde = -np.log(2.0) - 0.5 * (u_c2 ** 2) + np.log(e_sum)
            Ctilde01 = np.sqrt(np.pi / 2.0) * e_sum

            nu[case2_mask] = (Ltilde - np.log(1.0 - C_0[case2_mask])) * Ctilde01
            h2[case2_mask] = C21[case2_mask] + 1.0 / Ctilde01
            h3[case2_mask] = C31[case2_mask] + 3.0 * (C21[case2_mask] / Ctilde01) + 2.0 / (Ctilde01 ** 2)

        # --- Case 3: Remaining Values (Interior) ---
        if np.any(case3_mask):
            h_c3 = h[case3_mask]
            t_c3 = t[case3_mask]
            u_c3 = u[case3_mask]

            # Use the specified function based on the mode
            if mode == "advanced":
                C_B = C_ht(h_c3, t_c3)
            else:
                C_B = C_ht_default(h_c3, t_c3)

            phi_u = SQRT_2PI_INV * np.exp(-0.5 * u_c3 ** 2)

            nu[case3_mask] = -(C_B - C_0[case3_mask]) / phi_u
            h2[case3_mask] = C21[case3_mask]
            h3[case3_mask] = C31[case3_mask]

        # --- Householder Update Formula ---
        num = 1.0 + 0.5 * nu * h2
        den = 1.0 + nu * (h2 + (1.0 / 6.0) * nu * h3)

        B_curr = B + nu * (num / den)
        history.append(B_curr.copy())

    return history


def benchmark_model_refinement(
        A_fixed: float,
        B_min: float,
        B_max: float,
        num_points: int,
        num_iters: int = 3,
        mode: str = "advanced"
) -> Dict[str, float]:
    """
    Evaluates the performance of NN models after Householder refinement.
    Computes efficiency metrics using log(max(|B_n/B_true - 1|, TRAIN_EPSILON)).
    Outputs metrics to screen and prints a corresponding structured LaTeX table.
    """

    # 1. Load models
    trained_models = {}
    for name, ModelClass, config in MODEL_CONFIGS:
        try:
            trained_models[name] = load_model(name, ModelClass, config)
        except Exception:
            continue

    if not trained_models:
        print("[ERROR] No models available for benchmarking.")
        return {}

    # 2. Generate Ground Truth and Initial Guesses
    data = generate_iv_data_a_fixed_trained_model(
        A_fixed=A_fixed,
        B_min=B_min,
        B_max=B_max,
        num_points=num_points,
        mode=mode,
        trained_models=trained_models
    )

    B_true = data["B_true"]
    C_target = data["C_true"]
    A_val = data["A_fixed"]
    model_results = data["model_predictions"]

    efficiency_metrics = {}

    print(f"\n--- Benchmarking Results (A={A_val:.2f}, Iters={num_iters}) ---")

    # 3. Process each model
    for model_name, B_guess in model_results.items():
        try:
            # Apply Householder Iterations
            history = Householder_iter(
                A=A_val,
                C_0=C_target,
                B_0=B_guess,
                n=num_iters,
                mode=mode
            )

            B_final = history[-1]

            # 1) Change definition of error to log(max(|B_i/B_true - 1|, TRAIN_EPSILON))
            relative_errors = np.abs(B_final / B_true - 1.0)
            log_errors = np.log(np.maximum(relative_errors, Householder_EPSILON))

            avg_log_error = np.mean(log_errors)
            std_log_error = np.std(log_errors)
            max_log_error = np.max(log_errors)

            efficiency_metrics[model_name] = {
                "avg_log_error": avg_log_error,
                "std_log_error": std_log_error,
                "max_log_error": max_log_error
            }

            print(f"Model: {model_name:20} | Avg Log Error: {avg_log_error:+.6e}")

        except Exception as e:
            print(f"[ERROR] Failed refinement for {model_name}: {e}")

    # 2) Output a table in LaTeX giving all efficiency metrics per model (avg, std and max)
    if efficiency_metrics:
        print("\n% --- LaTeX Table Generated for Efficiency Metrics ---")
        print("\\begin{table}[htbp]")
        print("  \\centering")
        print(
            f"  \\caption{{Householder Refinement Efficiency Metrics ($A={A_val:.2f}$, $\\text{{Iterations}}={num_iters}$)}}")
        print("  \\begin{tabular}{lccc}")
        print("    \\hline")
        print(
            "    \\textbf{Model Architecture} & \\textbf{Avg Log Error} & \\textbf{Std Log Error} & \\textbf{Max Log Error} \\\\")
        print("    \\hline")
        for m_name, metrics in efficiency_metrics.items():
            # Escape strings for latex compatibility if underscore is present
            escaped_name = m_name.replace("_", "\\_")
            print(
                f"    {escaped_name:<30} & {metrics['avg_log_error']:+.6e} & {metrics['std_log_error']:.6e} & {metrics['max_log_error']:+.6e} \\\\")
        print("    \\hline")
        print("  \\end{tabular}")
        print("\\end{table}\n")

    return efficiency_metrics


def plot_iteration_convergence(
        A_fixed: float,
        B_min: float,
        B_max: float,
        num_points: int,
        num_iters: int = 3,
        mode: str = "advanced",
        save_dir: str = "Results_yb/convergence_Householder"
):
    """
    Generates convergence plots using log(max(|B_i/B_true - 1|, TRAIN_EPSILON)).
    Produces individual plots and a aligned side-by-side single comparison plot using uniform scaling.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trained_models = {}
    for name, ModelClass, config in MODEL_CONFIGS:
        try:
            model = load_model(name, ModelClass, config)
            if model is not None:
                trained_models[name] = model
        except Exception as e:
            print(f"[WARNING] Could not load {name}: {e}")

    if not trained_models:
        print("[ERROR] No models loaded. Plotting aborted.")
        return

    data = generate_iv_data_a_fixed_trained_model(
        A_fixed=A_fixed,
        B_min=B_min,
        B_max=B_max,
        num_points=num_points,
        mode=mode,
        trained_models=trained_models
    )

    B_true = data["B_true"]
    C_target = data["C_true"]
    model_predictions = data["model_predictions"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Storage dictionary to gather computed plot histories for aligned subplot execution
    all_model_histories = {}

    # 1) Calculate errors across all items to identify universal bounds for scaling alignment
    global_min_err = 0.0
    global_max_err = -np.inf

    for model_name, B_0 in model_predictions.items():
        history = Householder_iter(
            A=A_fixed,
            C_0=C_target,
            B_0=B_0,
            n=num_iters,
            mode=mode
        )

        processed_history = []
        for B_i in history:
            # 1) Change definition of error to log(max(|B_i/B_true - 1|, TRAIN_EPSILON))
            relative_errors = np.abs(B_i / B_true - 1.0)
            log_errors = np.log(np.maximum(relative_errors, Householder_EPSILON))
            processed_history.append(log_errors)

            global_max_err = max(global_max_err, np.max(log_errors))
            global_min_err = min(global_min_err, np.min(log_errors))

        all_model_histories[model_name] = processed_history

    # Floor slightly lower to leave visual breathing room at grid baseline
    global_min_err = np.floor(global_min_err) - 0.5
    global_max_err = np.ceil(global_max_err) + 0.5

    # Generate individual models plots using standard layout
    for model_name, history_errors in all_model_histories.items():
        plt.figure(figsize=(12, 8))
        colors = plt.cm.inferno(np.linspace(0.1, 0.8, len(history_errors)))

        for i, log_err_vector in enumerate(history_errors):
            label = f"Iteration {i}" if i > 0 else "Initial NN Guess (B_0)"
            plt.plot(B_true, log_err_vector, label=label, color=colors[i], lw=1.5, alpha=0.9)

        plt.title(f"Refinement Convergence: {model_name}\nHouseholder {mode} (A={A_fixed})", fontsize=14)
        plt.xlabel(r"True Value of $B$", fontsize=12)
        plt.ylabel(r"$\log(\max(|B_i / B_{true} - 1|, \epsilon))$", fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(loc='lower right', frameon=True, fontsize=10)
        plt.ylim(global_min_err, global_max_err)
        plt.tight_layout()

        filename = f"log_convergence_{model_name}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

    # 3) Produce a single plot with all the plots lined up one next to the other (Uniform scale)
    num_models = len(all_model_histories)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5), sharey=True)

    # Ensure axes handles iterable processing gracefully if only evaluating single models
    if num_models == 1:
        axes = [axes]

    for idx, (model_name, history_errors) in enumerate(all_model_histories.items()):
        ax = axes[idx]
        colors = plt.cm.inferno(np.linspace(0.1, 0.8, len(history_errors)))

        for i, log_err_vector in enumerate(history_errors):
            label = f"Iteration {i}" if i > 0 else "NN initial guess (B_0)"
            ax.plot(B_true, log_err_vector, color=colors[i], lw=1.2, alpha=0.85, label=label)

        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel(r"True $B$", fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_ylim(global_min_err, global_max_err)

        if idx == 0:
            ax.set_ylabel(r"$\log(\max(|B_i / B_{true} - 1|, \epsilon))$", fontsize=11)
        if idx == num_models - 1:
            ax.legend(loc='lower right', frameon=True, fontsize=8)

    # plt.suptitle(f"Multi-Model Comparison Alignment (Householder Mode: {mode} | A={A_fixed})", fontsize=12, y=1.02)
    plt.tight_layout()

    combined_filepath = os.path.join(save_dir, f"log_convergence_COMBINED_LINED_{timestamp}.png")
    plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved uniform shared-scale side-by-side comparison matrix to {combined_filepath}")


if __name__ == "__main__":
    results = benchmark_model_refinement(
        A_fixed=1.5,
        B_min=1e-7,
        B_max=1.22,
        num_points=500,
        num_iters=2
    )

    # Settings for a high-resolution check
    plot_iteration_convergence(
        A_fixed=1.5,
        B_min=1e-7,
        B_max=1.22,
        num_points=500,
        num_iters=2
    )