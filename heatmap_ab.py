import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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

# Enforce uniform platform default tensor tracking types
torch.set_default_dtype(DTYPE)


class HeatmapConfigAB:
    """Configuration class for heatmap generation in (A, B) space."""

    def __init__(
            self,
            A_min: float = 0.5,
            A_max: float = 15.0,
            B_min: float = 0.0,
            B_max: float = 7.0,
            n_A: int = 500,
            n_B: int = 500,
            loss_type: str = "MSRE",
            c_ht_mode: str = "advanced",
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: str = "plasma",
            output_dir: str = "Results_yb/heatmaps_AB",
            figsize: Tuple[int, int] = (7, 5),
            dpi: int = 150,
            mask_unreachable: bool = True,
            use_log_scale: bool = True,
            layout: str = "grid"  # "grid" or "horizontal"
    ):
        # Grid boundaries (A, B) - direct computation
        self.A_min: float = A_min
        self.A_max: float = A_max
        self.B_min: float = B_min
        self.B_max: float = B_max

        # Grid resolution
        self.n_A: int = n_A
        self.n_B: int = n_B

        # Loss function: "MSE" or "MSRE"
        self.loss_type: str = loss_type

        # C_ht computation mode: "advanced" or "default"
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

        # Option to mask unreachable regions (where C is out of bounds)
        self.mask_unreachable: bool = mask_unreachable

        # Layout option: "grid" (default) or "horizontal"
        self.layout: str = layout


def build_direct_grid_ab(cfg: HeatmapConfigAB) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (A, B) evaluation grid directly.

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

    # Compute C directly from (A, B) pairs - NO INTERPOLATION
    if cfg.c_ht_mode == "advanced":
        C_grid = C_ht(h.ravel(), t.ravel()).reshape(A_grid.shape)
    else:
        C_grid = C_ht_default(h.ravel(), t.ravel()).reshape(A_grid.shape)

    # Filter physically valid points (0 < C < 1)
    valid_mask = (C_grid > DS_EPSILON) & (C_grid < 1.0 - DS_EPSILON)

    # Loss mask is the same as valid mask for AB grid
    loss_mask = valid_mask.copy()

    return A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask


def pointwise_loss_ab(pred: np.ndarray, target: np.ndarray, loss_type: str) -> np.ndarray:
    """Compute element-wise loss between prediction and target arrays."""
    diff = pred - target

    if loss_type == "MSE":
        return np.abs(diff)  # |B_pred - B_true|
    elif loss_type == "MSRE":
        safe_target = np.maximum(np.abs(target), DS_EPSILON)
        return np.abs(diff / safe_target)  # |(B_pred - B_true) / B_true|
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose 'MSE' or 'MSRE'.")


def load_all_models_ab(cfg: HeatmapConfigAB, model_names: Optional[List[str]] = None) -> Dict[str, nn.Module]:
    """Load every model listed in MODEL_CONFIGS, skipping missing checkpoints."""
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


def plot_heatmaps_grid_ab(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigAB,
        norm,
        scale_label: str,
        B_min: float,
        B_max: float
) -> plt.Figure:
    """
    Generate heatmaps in a grid layout (original behavior) for (A,B) space.
    """
    n_models = len(loss_maps)
    n_A = len(A_vals)
    n_B = len(B_vals)

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

        # Add contour to show valid region boundary if masking is enabled
        if cfg.mask_unreachable:
            # Create binary mask for contour
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

    # Hide surplus axes
    for spare in range(n_models, nrows * ncols):
        axes[spare // ncols][spare % ncols].set_visible(False)

    # Shared color bar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Pointwise {cfg.loss_type} ({scale_label})", fontsize=11)

    fig.suptitle(
        f"{cfg.loss_type} heatmaps in (A,B) space \n",
        fontsize=11, y=1.01
    )

    return fig


def plot_heatmaps_horizontal_ab(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigAB,
        norm,
        scale_label: str,
        B_min: float,
        B_max: float
) -> plt.Figure:
    """
    Generate heatmaps in a horizontal row layout with legend on the far right for (A,B) space.
    """
    n_models = len(loss_maps)
    n_A = len(A_vals)
    n_B = len(B_vals)

    # Calculate figure size
    fig_width = 5 * n_models + 1.5
    fig_height = 5

    fig, axes = plt.subplots(1, n_models,
                             figsize=(fig_width, fig_height),
                             squeeze=False)
    axes = axes[0]  # Flatten to 1D

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

        # Add contour to show valid region boundary
        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

    # Add a single colorbar on the far right
    fig.subplots_adjust(right=0.92, left=0.06, wspace=0.3)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Pointwise {cfg.loss_type} ({scale_label})", fontsize=11)

    fig.suptitle(
        f"{cfg.loss_type} heatmaps in (A,B) space\n",
        fontsize=11, y=1.02
    )

    return fig


def plot_heatmaps_ab(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigAB
) -> None:
    """
    Generate individual and combined heatmaps in (A,B) space.
    Layout can be either "grid" (default) or "horizontal".
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    n_models = len(loss_maps)

    # Determine uniform color limits across all models
    all_losses = np.concatenate([loss[loss_mask] for loss in loss_maps.values()])
    all_losses = all_losses[np.isfinite(all_losses) & (all_losses > 0)]

    if len(all_losses) == 0:
        print("[ERROR] No valid loss values found for color scaling")
        return

    if cfg.vmin is None:
        vmin = float(np.percentile(all_losses, 1))
    else:
        vmin = cfg.vmin

    if cfg.vmax is None:
        vmax = float(np.percentile(all_losses, 99))
    else:
        vmax = cfg.vmax

    # Guard against degenerate color limits
    if vmin <= 0:
        positive_losses = all_losses[all_losses > 0]
        vmin = float(np.min(positive_losses)) if len(positive_losses) > 0 else 1e-12
    if vmax <= vmin:
        vmax = vmin * 1e4

    # Choose normalization based on user preference
    if cfg.use_log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        scale_label = "log scale"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        scale_label = "linear scale"

    print(f"\n      Colour scale ({cfg.loss_type}):  [{vmin:.3e},  {vmax:.3e}]  ({scale_label})")
    print(f"      Layout: {cfg.layout}")
    print(f"      A range: [{cfg.A_min:.2f}, {cfg.A_max:.2f}]")
    print(f"      B range: [{cfg.B_min:.2f}, {cfg.B_max:.2f}]")

    n_A = len(A_vals)
    n_B = len(B_vals)

    # ------------------------------------------------------------------
    # 1. Individual heatmaps (always saved separately)
    # ------------------------------------------------------------------
    print("\n[4/5] Generating individual heatmaps...")

    for name, loss_image in loss_maps.items():
        fig, ax = plt.subplots(figsize=cfg.figsize)

        # Plot the heatmap
        im = ax.imshow(
            loss_image,
            origin="lower",
            aspect="auto",
            norm=norm,
            cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
            interpolation='nearest'
        )

        # Add contour lines to show the valid region boundary
        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5],
                       extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        ax.set_facecolor('#dcdcdc')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"Pointwise {cfg.loss_type} ({scale_label})", fontsize=10)

        ax.set_xlabel("$A$", fontsize=12)
        ax.set_ylabel("$B$", fontsize=12)

        # Add explanatory note about grey region
        note_text = "Grey region: (A,B) pairs where C is outside (0,1)"
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom', color='gray', style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_title(f"{name}\n({cfg.loss_type} heatmap in (A,B) space, {cfg.c_ht_mode} mode)", fontsize=11)

        # Add summary statistics
        valid_losses = loss_image[loss_mask]
        mean_loss = np.mean(valid_losses)
        median_loss = np.median(valid_losses)
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.2e}\nMedian: {median_loss:.2e}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fname = os.path.join(
            cfg.output_dir,
            f"heatmap_AB_{name}_{cfg.loss_type}_{timestamp}.png"
        )
        fig.tight_layout()
        fig.savefig(fname, dpi=cfg.dpi)
        plt.close(fig)
        print(f"      Saved: {fname}")

    # ------------------------------------------------------------------
    # 2. Combined figure (layout depends on cfg.layout)
    # ------------------------------------------------------------------
    print("\n[5/5] Generating combined figure...")

    if cfg.layout == "horizontal":
        fig = plot_heatmaps_horizontal_ab(
            loss_maps, A_vals, B_vals, loss_mask,
            cfg, norm, scale_label, cfg.B_min, cfg.B_max
        )
        layout_suffix = "horizontal"
    else:  # Default to grid layout
        fig = plot_heatmaps_grid_ab(
            loss_maps, A_vals, B_vals, loss_mask,
            cfg, norm, scale_label, cfg.B_min, cfg.B_max
        )
        layout_suffix = "grid"

    combined_fname = os.path.join(
        cfg.output_dir,
        f"heatmap_AB_COMBINED_{cfg.loss_type}_{layout_suffix}_{timestamp}.png"
    )
    fig.savefig(combined_fname, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"      Combined figure saved: {combined_fname}")


def main_ab(config: Optional[HeatmapConfigAB] = None, model_names: Optional[List[str]] = None) -> None:
    """Main entry point for (A,B) space heatmap generation."""
    if config is None:
        config = HeatmapConfigAB()

    print("=" * 60)
    print("Neural Network Performance Heatmap Generator - (A,B) Space")
    print("=" * 60)
    print(f"Grid: {config.n_A} × {config.n_B} = {config.n_A * config.n_B} points")
    print(f"Loss function: {config.loss_type}")
    print(f"C_ht mode: {config.c_ht_mode}")
    print(f"Color scale: {'Logarithmic' if config.use_log_scale else 'Linear'}")
    print(f"Layout: {config.layout}")
    print(f"Mask unreachable regions: {config.mask_unreachable}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 1: Build direct (A, B) evaluation grid
    # ------------------------------------------------------------------
    print("\n[1/5] Building evaluation grid (A, B) → C ...")
    A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask = build_direct_grid_ab(config)

    # Extract valid points for NN input
    A_flat = A_grid[valid_mask]
    B_flat = B_grid[valid_mask]
    C_flat = C_grid[valid_mask]

    n_valid = len(A_flat)
    total_points = config.n_A * config.n_B
    print(f"      Total grid points: {total_points}")
    print(f"      Valid points: {n_valid} ({100 * n_valid / total_points:.1f}%)")
    print(f"      NOTE: Grey regions represent ({100 * (total_points - n_valid) / total_points:.1f}%) of (A,B) space")
    print(f"            where C is outside the physically valid range (0,1)")

    if n_valid == 0:
        print("[ERROR] No valid points found. Check parameter ranges.")
        return

    # ------------------------------------------------------------------
    # Step 2: Build NN input features
    # ------------------------------------------------------------------
    print("\n[2/5] Building NN input features (7-dimensional)...")
    X_np = construct_NN_inputs_1to1(A_flat, C_flat, mode=config.c_ht_mode)
    X_torch = torch.tensor(X_np, dtype=DTYPE).to(device)
    print(f"      Input shape: {X_np.shape}")

    # ------------------------------------------------------------------
    # Step 3: Load trained models
    # ------------------------------------------------------------------
    print("\n[3/5] Loading trained models...")
    models = load_all_models_ab(config, model_names)

    if not models:
        print("[ERROR] No models could be loaded – aborting.")
        return

    # ------------------------------------------------------------------
    # Step 4: Compute predictions and losses
    # ------------------------------------------------------------------
    print("\n[4/5] Computing model predictions and losses...")

    # Pre-allocate loss images for each model
    loss_images = {}
    for name in models.keys():
        loss_images[name] = np.full((config.n_B, config.n_A), np.nan, dtype=np.float64)

    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            B_pred_torch = model(X_torch).squeeze().cpu()
        B_pred_np = B_pred_torch.numpy().astype(np.float64)

        # Compute pointwise loss
        pointwise_losses = pointwise_loss_ab(B_pred_np, B_flat, config.loss_type)

        # Map back to 2D grid
        loss_images[name][valid_mask] = pointwise_losses

        # Print summary statistics
        mean_loss = np.mean(pointwise_losses)
        print(f"      {name}: mean {config.loss_type} = {mean_loss:.6f}")

    # ------------------------------------------------------------------
    # Step 5: Generate heatmaps
    # ------------------------------------------------------------------
    plot_heatmaps_ab(loss_images, A_vals, B_vals, loss_mask, config)

    print("\n" + "=" * 60)
    print("Heatmap generation completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    # Example 1: Grid layout (default)
    config_grid = HeatmapConfigAB(
        A_min=0.0,
        A_max=3.0,
        B_min=1e-7,
        B_max=1.22,
        n_A=1000,
        n_B=1000,
        loss_type="MSRE",
        c_ht_mode="advanced",
        output_dir="Results_yb/heatmaps",
        mask_unreachable=True,
        use_log_scale=True,
        layout="horizontal"  # Traditional "grid" layout OR "horizontal"
    )

    model_names = None  # Evaluate all available models

    # Choose which config to use
    main_ab(config_grid, model_names)
    # main_ab(config_horizontal, model_names)