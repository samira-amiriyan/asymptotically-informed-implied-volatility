import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import verified pipeline assets
from Optimised_BS import C_ht, C_ht_default
from generate_dataset import construct_NN_inputs_1to1, DS_EPSILON
from train_tools import device, DTYPE
from train_save import MODEL_CONFIGS, load_model

# Import Householder method
from Householder import Householder_iter, Householder_EPSILON

# Enforce uniform platform default tensor tracking types
torch.set_default_dtype(DTYPE)


class HeatmapConfigHouseholder:
    """Configuration class for heatmap generation after Householder refinement."""

    def __init__(
            self,
            A_min: float = 0.5,
            A_max: float = 15.0,
            B_min: float = 0.0,
            B_max: float = 7.0,
            n_A: int = 200,
            n_B: int = 200,
            num_iters: int = 3,
            loss_type: str = "log_rel_err",  # "MSE", "MSRE", or "log_rel_err"
            c_ht_mode: str = "advanced",
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: str = "plasma",
            output_dir: str = "Results_yb/heatmaps_householder",
            figsize: Tuple[int, int] = (7, 5),
            dpi: int = 150,
            mask_unreachable: bool = True,
            use_log_scale: bool = True,
            layout: str = "grid"  # "grid" or "horizontal"
    ):
        # Grid boundaries
        self.A_min: float = A_min
        self.A_max: float = A_max
        self.B_min: float = B_min
        self.B_max: float = B_max

        # Grid resolution
        self.n_A: int = n_A
        self.n_B: int = n_B

        # Householder iterations
        self.num_iters: int = num_iters

        # Loss function type
        self.loss_type: str = loss_type

        # C_ht computation mode
        self.c_ht_mode: str = c_ht_mode

        # Color scaling
        self.vmin: Optional[float] = vmin
        self.vmax: Optional[float] = vmax
        self.use_log_scale: bool = use_log_scale

        # Matplotlib settings
        self.cmap: str = cmap
        self.output_dir: str = output_dir
        self.figsize: Tuple[int, int] = figsize
        self.dpi: int = dpi

        # Option to mask unreachable regions
        self.mask_unreachable: bool = mask_unreachable

        # Layout option
        self.layout: str = layout


def build_direct_grid_householder(cfg: HeatmapConfigHouseholder) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (A, B) evaluation grid directly for Householder refinement.

    Returns:
        A_grid: 2D array of A values (shape: n_B × n_A)
        B_grid: 2D array of B values (shape: n_B × n_A)
        C_grid: 2D array of C values computed directly from (A, B)
        valid_mask: Boolean mask of physically valid points (0 < C < 1)
        A_vals: 1D array of A axis values (length n_A)
        B_vals: 1D array of B axis values (length n_B)
        loss_mask: Boolean mask for valid loss computation points
    """
    # Create A and B grid axes
    A_vals = np.linspace(cfg.A_min, cfg.A_max, cfg.n_A, dtype=np.float64)

    # Offset B_min by DS_EPSILON to avoid B = 0
    B_vals = np.linspace(cfg.B_min + DS_EPSILON, cfg.B_max, cfg.n_B, dtype=np.float64)

    # Create Cartesian grid (rows = B, cols = A)
    A_grid, B_grid = np.meshgrid(A_vals, B_vals)

    # Compute h and t parameters
    h = A_grid / B_grid
    t = B_grid / 2.0

    # Compute C directly from (A, B) pairs
    if cfg.c_ht_mode == "advanced":
        C_grid = C_ht(h.ravel(), t.ravel()).reshape(A_grid.shape)
    else:
        C_grid = C_ht_default(h.ravel(), t.ravel()).reshape(A_grid.shape)

    # Filter physically valid points (0 < C < 1)
    valid_mask = (C_grid > DS_EPSILON) & (C_grid < 1.0 - DS_EPSILON)

    # Loss mask is the same as valid mask
    loss_mask = valid_mask.copy()

    return A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask


def compute_householder_loss(
        B_pred_initial: np.ndarray,
        B_true: np.ndarray,
        A_flat: np.ndarray,
        C_flat: np.ndarray,
        num_iters: int,
        loss_type: str,
        mode: str = "advanced"
) -> np.ndarray:
    """
    Apply Householder refinement and compute loss.

    Args:
        B_pred_initial: Initial predictions from NN
        B_true: True B values
        A_flat: Flat array of A values
        C_flat: Flat array of C values
        num_iters: Number of Householder iterations
        loss_type: Type of loss to compute
        mode: C_ht computation mode

    Returns:
        Array of loss values after refinement
    """
    # Apply Householder iterations
    history = Householder_iter(
        A=A_flat,
        C_0=C_flat,
        B_0=B_pred_initial,
        n=num_iters,
        mode=mode
    )

    B_final = history[-1]  # After n iterations

    # Compute loss based on type
    if loss_type == "MSE":
        return np.abs(B_final - B_true)
    elif loss_type == "MSRE":
        safe_target = np.maximum(np.abs(B_true), DS_EPSILON)
        return np.abs(B_final - B_true) / safe_target
    elif loss_type == "log_rel_err":
        relative_errors = np.abs(B_final / B_true - 1.0)
        return np.log(np.maximum(relative_errors, Householder_EPSILON))
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")


def load_all_models_householder(cfg: HeatmapConfigHouseholder,
                                model_names: Optional[List[str]] = None) -> Dict[str, nn.Module]:
    """Load all models from MODEL_CONFIGS."""
    loaded_models = {}

    for name, ModelClass, model_cfg in MODEL_CONFIGS:
        if model_names is not None and name not in model_names:
            continue

        try:
            model = load_model(name, ModelClass, model_cfg)
            model.eval()
            loaded_models[name] = model
            print(f"  [OK]      {name}")
        except FileNotFoundError:
            print(f"  [SKIP]    {name} – checkpoint not found")
        except Exception as exc:
            print(f"  [ERROR]   {name} – {exc}")

    return loaded_models


def plot_heatmaps_householder_grid(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigHouseholder,
        norm,
        scale_label: str
) -> plt.Figure:
    """Generate heatmaps in a grid layout."""
    n_models = len(loss_maps)

    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (name, loss_image) in enumerate(loss_maps.items()):
        ax = axes[idx // ncols][idx % ncols]

        im = ax.imshow(
            loss_image,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
            interpolation='nearest'
        )
        ax.set_facecolor('#dcdcdc')
        ax.set_xlabel("$A$", fontsize=10)
        ax.set_ylabel("$B$", fontsize=10)
        ax.set_title(name, fontsize=9)

        # Add contour to show valid region boundary
        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        # Add text annotation for mean loss
        valid_losses = loss_image[loss_mask]
        mean_loss = np.mean(valid_losses)
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.2e}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide surplus axes
    for spare in range(n_models, nrows * ncols):
        axes[spare // ncols][spare % ncols].set_visible(False)

    # Shared color bar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=11)

    fig.suptitle(
        f"After {cfg.num_iters} Householder Iterations\n{cfg.loss_type} in (A,B) space",
        fontsize=12, y=1.01
    )

    return fig


def plot_heatmaps_householder_horizontal(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigHouseholder,
        norm,
        scale_label: str
) -> plt.Figure:
    """Generate heatmaps in a horizontal row layout."""
    n_models = len(loss_maps)

    fig_width = 5 * n_models + 1.5
    fig_height = 5

    fig, axes = plt.subplots(1, n_models,
                             figsize=(fig_width, fig_height),
                             squeeze=False)
    axes = axes[0]

    for idx, (name, loss_image) in enumerate(loss_maps.items()):
        ax = axes[idx]

        im = ax.imshow(
            loss_image,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
            interpolation='nearest'
        )
        ax.set_facecolor('#dcdcdc')
        ax.set_xlabel("$A$", fontsize=10)
        if idx == 0:
            ax.set_ylabel("$B$", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_title(name, fontsize=9, wrap=True)

        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        # Add mean loss text
        valid_losses = loss_image[loss_mask]
        mean_loss = np.mean(valid_losses)
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.2e}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Shared color bar on the far right
    fig.subplots_adjust(right=0.92, left=0.06, wspace=0.3)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=11)

    fig.suptitle(
        f"After {cfg.num_iters} Householder Iterations\n{cfg.loss_type} in (A,B) space",
        fontsize=12, y=1.02
    )

    return fig


def plot_heatmaps_householder(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigHouseholder
) -> None:
    """Generate individual and combined heatmaps after Householder refinement."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    n_models = len(loss_maps)

    # Determine uniform color limits across all models
    all_losses = np.concatenate([loss[loss_mask] for loss in loss_maps.values()])
    all_losses = all_losses[np.isfinite(all_losses)]

    if len(all_losses) == 0:
        print("[ERROR] No valid loss values found for color scaling")
        return

    if cfg.vmin is None:
        if cfg.loss_type == "log_rel_err":
            vmin = float(np.percentile(all_losses, 1))
        else:
            vmin = float(np.percentile(all_losses, 1)) if np.min(all_losses) > 0 else 1e-12
    else:
        vmin = cfg.vmin

    if cfg.vmax is None:
        vmax = float(np.percentile(all_losses, 99))
    else:
        vmax = cfg.vmax

    # Guard against degenerate color limits
    if vmin <= 0 and cfg.loss_type != "log_rel_err":
        positive_losses = all_losses[all_losses > 0]
        vmin = float(np.min(positive_losses)) if len(positive_losses) > 0 else 1e-12
    if vmax <= vmin:
        vmax = vmin * 1e4

    # Choose normalization
    if cfg.use_log_scale and cfg.loss_type != "log_rel_err":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        scale_label = "log scale"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        scale_label = "linear scale"

    print(f"\n      Loss type: {cfg.loss_type}")
    print(f"      Colour scale: [{vmin:.3e}, {vmax:.3e}] ({scale_label})")
    print(f"      Layout: {cfg.layout}")
    print(f"      Householder iterations: {cfg.num_iters}")

    # ------------------------------------------------------------------
    # 1. Individual heatmaps
    # ------------------------------------------------------------------
    print("\n[4/5] Generating individual heatmaps...")

    for name, loss_image in loss_maps.items():
        fig, ax = plt.subplots(figsize=cfg.figsize)

        im = ax.imshow(
            loss_image,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
            interpolation='nearest'
        )

        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        ax.set_facecolor('#dcdcdc')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=10)

        ax.set_xlabel("$A$", fontsize=12)
        ax.set_ylabel("$B$", fontsize=12)

        note_text = "Grey region: (A,B) where C is outside (0,1)"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom', color='gray', style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_title(f"{name}\n{cfg.loss_type} after {cfg.num_iters} Householder iterations", fontsize=11)

        # Summary statistics
        valid_losses = loss_image[loss_mask]
        mean_loss = np.mean(valid_losses)
        median_loss = np.median(valid_losses)
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.2e}\nMedian: {median_loss:.2e}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fname = os.path.join(
            cfg.output_dir,
            f"heatmap_householder_{name}_{cfg.loss_type}_iter{cfg.num_iters}_{timestamp}.png"
        )
        fig.tight_layout()
        fig.savefig(fname, dpi=cfg.dpi)
        plt.close(fig)
        print(f"      Saved: {fname}")

    # ------------------------------------------------------------------
    # 2. Combined figure
    # ------------------------------------------------------------------
    print("\n[5/5] Generating combined figure...")

    if cfg.layout == "horizontal":
        fig = plot_heatmaps_householder_horizontal(
            loss_maps, A_vals, B_vals, loss_mask, cfg, norm, scale_label
        )
        layout_suffix = "horizontal"
    else:
        fig = plot_heatmaps_householder_grid(
            loss_maps, A_vals, B_vals, loss_mask, cfg, norm, scale_label
        )
        layout_suffix = "grid"

    combined_fname = os.path.join(
        cfg.output_dir,
        f"heatmap_householder_COMBINED_{cfg.loss_type}_iter{cfg.num_iters}_{layout_suffix}_{timestamp}.png"
    )
    fig.savefig(combined_fname, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"      Combined figure saved: {combined_fname}")


def main_householder(config: Optional[HeatmapConfigHouseholder] = None,
                     model_names: Optional[List[str]] = None,
                     iterations_list: Optional[List[int]] = None) -> None:
    """
    Main entry point for Householder refinement heatmap generation.

    Args:
        config: Configuration object
        model_names: Optional list of specific models to evaluate
        iterations_list: Optional list of iteration counts to test (e.g., [0,1,2,3])
                         If None, only uses config.num_iters
    """
    if config is None:
        config = HeatmapConfigHouseholder()

    if iterations_list is None:
        iterations_list = [config.num_iters]

    print("=" * 60)
    print("Neural Network Performance Heatmap Generator - After Householder Refinement")
    print("=" * 60)
    print(f"Grid: {config.n_A} × {config.n_B} = {config.n_A * config.n_B} points")
    print(f"Loss function: {config.loss_type}")
    print(f"C_ht mode: {config.c_ht_mode}")
    print(f"Color scale: {'Logarithmic' if config.use_log_scale else 'Linear'}")
    print(f"Layout: {config.layout}")
    print(f"Householder iterations to test: {iterations_list}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 1: Build direct (A, B) evaluation grid
    # ------------------------------------------------------------------
    print("\n[1/4] Building evaluation grid (A, B) → C ...")
    A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask = build_direct_grid_householder(config)

    # Extract valid points for NN input
    A_flat = A_grid[valid_mask]
    B_flat = B_grid[valid_mask]
    C_flat = C_grid[valid_mask]

    n_valid = len(A_flat)
    total_points = config.n_A * config.n_B
    print(f"      Total grid points: {total_points}")
    print(f"      Valid points: {n_valid} ({100 * n_valid / total_points:.1f}%)")

    if n_valid == 0:
        print("[ERROR] No valid points found. Check parameter ranges.")
        return

    # ------------------------------------------------------------------
    # Step 2: Build NN input features
    # ------------------------------------------------------------------
    print("\n[2/4] Building NN input features (7-dimensional)...")
    X_np = construct_NN_inputs_1to1(A_flat, C_flat, mode=config.c_ht_mode)
    X_torch = torch.tensor(X_np, dtype=DTYPE).to(device)
    print(f"      Input shape: {X_np.shape}")

    # ------------------------------------------------------------------
    # Step 3: Load trained models and get initial predictions
    # ------------------------------------------------------------------
    print("\n[3/4] Loading trained models...")
    models = load_all_models_householder(config, model_names)

    if not models:
        print("[ERROR] No models could be loaded – aborting.")
        return

    # Get initial predictions from all models
    initial_predictions = {}
    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            B_pred_torch = model(X_torch).squeeze().cpu()
        initial_predictions[name] = B_pred_torch.numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Step 4: Generate heatmaps for each iteration count
    # ------------------------------------------------------------------
    print("\n[4/4] Generating heatmaps for each iteration count...")

    for num_iters in iterations_list:
        print(f"\n{'=' * 50}")
        print(f"Processing {num_iters} Householder iterations")
        print(f"{'=' * 50}")

        # Compute losses after refinement
        loss_images = {}
        for name, B_pred_initial in initial_predictions.items():
            print(f"  Computing for {name}...")

            pointwise_losses = compute_householder_loss(
                B_pred_initial=B_pred_initial,
                B_true=B_flat,
                A_flat=A_flat,
                C_flat=C_flat,
                num_iters=num_iters,
                loss_type=config.loss_type,
                mode=config.c_ht_mode
            )

            # Map back to 2D grid
            loss_image = np.full((config.n_B, config.n_A), np.nan, dtype=np.float64)
            loss_image[valid_mask] = pointwise_losses
            loss_images[name] = loss_image

            mean_loss = np.mean(pointwise_losses)
            print(f"    {name}: mean {config.loss_type} = {mean_loss:.6e}")

        # Create a temporary config with current iteration count for plotting
        temp_config = HeatmapConfigHouseholder(
            A_min=config.A_min, A_max=config.A_max,
            B_min=config.B_min, B_max=config.B_max,
            n_A=config.n_A, n_B=config.n_B,
            num_iters=num_iters,
            loss_type=config.loss_type,
            c_ht_mode=config.c_ht_mode,
            vmin=config.vmin, vmax=config.vmax,
            cmap=config.cmap,
            output_dir=config.output_dir,
            figsize=config.figsize,
            dpi=config.dpi,
            mask_unreachable=config.mask_unreachable,
            use_log_scale=config.use_log_scale,
            layout=config.layout
        )

        plot_heatmaps_householder(loss_images, A_vals, B_vals, loss_mask, temp_config)

    print("\n" + "=" * 60)
    print("Heatmap generation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    # Configuration for Householder refinement heatmaps
    config = HeatmapConfigHouseholder(
        A_min=0.0,
        A_max=15.0,
        B_min=1e-5,
        B_max=7.07,
        n_A=1000,
        n_B=1000,
        num_iters=2,  # Default, can be overridden by iterations_list
        loss_type="log_rel_err",  # "MSE", "MSRE", or "log_rel_err"
        c_ht_mode="advanced",
        output_dir="Results_yb/heatmaps_householder",
        mask_unreachable=True,
        use_log_scale=True,  # For MSE/MSRE; log_rel_err uses linear scale
        layout="horizontal"  # "grid" or "horizontal"
    )

    # Test multiple iteration counts (including iteration 0 = initial NN guess)
    # iterations_to_test = [2]

    # Run the main function
    main_householder(
        config=config,
        model_names=None,  # Evaluate all available models
    )
    # main_householder(
    #     config=config,
    #     model_names=None,  # Evaluate all available models
    #     iterations_list=iterations_to_test
    # )