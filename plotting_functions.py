import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import os

def plot_loss_evolution(
        all_loss_histories: Dict[str, List[List[float]]],
        num_epochs: int,
        loss_function: str,
        output_path: str
):
    """
    Plots the mean training loss evolution across N runs with standard deviation band.
    """
    plt.figure(figsize=(12, 6))
    epochs = np.arange(1, num_epochs + 1)

    for model_name, histories in all_loss_histories.items():
        if not histories:
            continue

        # Convert list of lists (runs) to a numpy array for easy calculation
        loss_array = np.array(histories)

        # Calculate mean and standard deviation across the runs (axis=0)
        mean_loss = np.mean(loss_array, axis=0)
        std_loss = np.std(loss_array, axis=0)

        # Plot mean loss
        plt.plot(epochs, mean_loss, label=model_name)

        # Plot shaded area for standard deviation (mean +/- std)
        plt.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.15
        )

    plt.title(f'Mean Training Loss Evolution ({loss_function}) across {len(histories)} Runs')
    plt.xlabel('Epoch')
    plt.ylabel(f'Mean {loss_function} (with $\pm$ Std. Dev.)')
    plt.yscale('log')  # Use log scale for y-axis for better visibility of convergence
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    print(f"\n[SUCCESS] Loss evolution plot saved to: {output_path}")


def plot_iv_guess_comparison(
    A_fixed: float,
    B_values: np.ndarray,
    C_values: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    plot_path: str,
    C_l: float,          # C corresponding to B_l
    C_u: float,          # C corresponding to B_u
    B_l_point: float,    # B_l value
    B_u_point: float,    # B_u value
    true_C_mode: str = "Analytical C_ht",
    title_suffix: str = ""
) -> None:
    """
    Plots the true inverse map B_inv: C -> B against the predictions of trained models
    for a fixed value of A, including boundary visualizations.

    Args:
        A_fixed: The fixed value of A used in the analysis.
        B_values: The sampled B values (y-axis, true targets).
        C_values: The true C values corresponding to B_values (x-axis).
        model_predictions: Dictionary mapping model name (str) to predicted B values (np.ndarray).
        plot_path: Full file path to save the plot (e.g., 'Plots/inverse_analysis.png').
        C_l, C_u: The C values corresponding to the lower and upper boundaries B_l, B_u.
        B_l_point, B_u_point: The analytical B_l and B_u values.
        true_C_mode: Label for the true C curve (e.g., 'C_ht (Advanced)').
        title_suffix: Additional text to append to the plot title.
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    # 0. Define Plotting Boundaries (for filling regions)

    # Use the minimum/maximum C values from the valid data for X range
    B_min_data = np.min(B_values)
    B_max_data = np.max(B_values)

    # Set x and y limits slightly outside the data to show the boundary points clearly
    ax.set_xlim(0, 1.05)
    ax.set_ylim(B_min_data * 0.9, B_max_data * 1.1)

    # --- Area Coloring (e, f, g) ---
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    # e) Area between C=0 and C=C_l (Lower boundary region)
    ax.axvspan(0, C_l, facecolor='lightblue', alpha=0.5)

    # f) Area between C=C_u and C=1 (Upper boundary region)
    ax.axvspan(C_u, 1, facecolor='lightcoral', alpha=0.5)

    # g) Area between C=C_l and C=C_u (Interior region)
    ax.axvspan(C_l, C_u, facecolor='lightgreen', alpha=0.5)

    # 1. Plot the True Inverse Map (Solid Line)
    # The true map is C -> B. C is on the x-axis, B is on the y-axis.
    ax.plot(C_values, B_values, 'k-', linewidth=3, label=f'True Map ({true_C_mode})')

    # 2. Plot Model Predictions (Dotted Lines)
    for model_name, B_pred in model_predictions.items():
        # Predictions are B_pred, which should be close to B_values (y-axis).
        # We plot C_values (x-axis) vs B_pred (y-axis)
        ax.plot(C_values, B_pred, '--', linewidth=2, label=f'Model: {model_name}')

    # --- Boundary Visualizations (a, b, c, d) ---

    # B_l and B_u line segment endpoints on the true map (C_l, B_l) and (C_u, B_u)
    # NOTE: Since the true map B_values are sampled, we find the closest points

    # Assuming C_l, C_u, B_l_point, B_u_point are calculated externally

    # c) Vertical dotted lines from x-axis (C_l and C_u) to true map
    ax.plot([C_l, C_l], [y_min, B_l_point], ':', color='gray', linewidth=1, zorder=3)
    ax.plot([C_u, C_u], [y_min, B_u_point], ':', color='gray', linewidth=1, zorder=3)

    # d) Horizontal dotted lines from y-axis (B_l and B_u) to true map
    # Note: x_min is 0.
    ax.plot([x_min, C_l], [B_l_point, B_l_point], ':', color='gray', linewidth=1, zorder=3)
    ax.plot([x_min, C_u], [B_u_point, B_u_point], ':', color='gray', linewidth=1, zorder=3)

    # --- Text Labels on Axes (No Markers) ---

    # Label C_l on the x-axis (at y_min)
    ax.text(C_l, y_min, r'$C_l$',
            ha='center', va='top',
            fontsize=12)

    # Label C_u on the x-axis (at y_min)
    ax.text(C_u, y_min, r'$C_u$',
            ha='center', va='top',
            fontsize=12)

    # Label B_l on the y-axis (at x_min=0)
    ax.text(x_min, B_l_point, r'$B_l$',
            ha='right', va='center',
            fontsize=12)

    # Label B_u on the y-axis (at x_min=0)
    ax.text(x_min, B_u_point, r'$B_u$',
            ha='right', va='center',
            fontsize=12)

    # --- Formatting ---
    ax.set_title(f'Inverse Map Comparison: C \u2192 B at Fixed A = {A_fixed:.4f} {title_suffix}', fontsize=14)
    ax.set_xlabel('C (Input for Inverse Map)', fontsize=12)
    ax.set_ylabel('B (Implied Volatility / Output)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Create directory if needed and save the plot
    try:
        os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"\n[SUCCESS] Inverse map comparison plot saved to: {plot_path}")
    except Exception as e:
        print(f"\n[ERROR] Could not save plot to {plot_path}. Error: {e}")