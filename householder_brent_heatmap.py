import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from scipy.optimize import brentq

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
            brent_max_iter: int = 100,  # Added maximum iterations configuration for Brent
            include_brent: bool = True,  # Flag to toggle Brent method heatmap
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

        # Iterations control
        self.num_iters: int = num_iters
        self.brent_max_iter: int = brent_max_iter
        self.include_brent: bool = include_brent

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
    """Build the (A, B) evaluation grid directly."""
    A_vals = np.linspace(cfg.A_min, cfg.A_max, cfg.n_A, dtype=np.float64)
    B_vals = np.linspace(cfg.B_min + DS_EPSILON, cfg.B_max, cfg.n_B, dtype=np.float64)

    A_grid, B_grid = np.meshgrid(A_vals, B_vals)

    h = A_grid / B_grid
    t = B_grid / 2.0

    if cfg.c_ht_mode == "advanced":
        C_grid = C_ht(h.ravel(), t.ravel()).reshape(A_grid.shape)
    else:
        C_grid = C_ht_default(h.ravel(), t.ravel()).reshape(A_grid.shape)

    valid_mask = (C_grid > DS_EPSILON) & (C_grid < 1.0 - DS_EPSILON)
    loss_mask = valid_mask.copy()

    return A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask


def compute_loss_metric(B_final: np.ndarray, B_true: np.ndarray, loss_type: str) -> np.ndarray:
    """Compute structural loss comparison vector/matrix."""
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


def run_brent_solver(A_flat: np.ndarray, C_flat: np.ndarray, cfg: HeatmapConfigHouseholder) -> np.ndarray:
    """Runs Brent's root-finding method pointwise across the valid flattened arrays."""
    B_brent = np.zeros_like(A_flat)

    # Objective function: find B such that C_ht(A/B, B/2) - C_target = 0
    def objective(b, a, c_target):
        h_val = a / b
        t_val = b / 2.0
        c_calc = C_ht(h_val, t_val) if cfg.c_ht_mode == "advanced" else C_ht_default(h_val, t_val)
        return c_calc - c_target

    # Bracket limits safely avoiding zero boundaries
    b_low = max(cfg.B_min, DS_EPSILON)
    b_high = cfg.B_max

    for i in range(len(A_flat)):
        a = A_flat[i]
        c_target = C_flat[i]
        try:
            # Solve using Scipy's BrentQ variant up to max internal steps configured
            res = brentq(objective, b_low, b_high, args=(a, c_target), maxiter=cfg.brent_max_iter)
            B_brent[i] = res
        except Exception:
            # Fallback handling in case of rare bracketing sign failures
            B_brent[i] = np.nan

    return B_brent


def compute_householder_loss(
        B_pred_initial: np.ndarray,
        B_true: np.ndarray,
        A_flat: np.ndarray,
        C_flat: np.ndarray,
        num_iters: int,
        loss_type: str,
        mode: str = "advanced"
) -> np.ndarray:
    """Apply Householder refinement and compute loss."""
    history = Householder_iter(
        A=A_flat,
        C_0=C_flat,
        B_0=B_pred_initial,
        n=num_iters,
        mode=mode
    )
    B_final = history[-1]
    return compute_loss_metric(B_final, B_true, loss_type)


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (name, loss_image) in enumerate(loss_maps.items()):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(
            loss_image, origin="lower", aspect="auto", norm=norm, cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]], interpolation='nearest'
        )
        ax.set_facecolor('#dcdcdc')
        ax.set_xlabel("$A$", fontsize=10)
        ax.set_ylabel("$B$", fontsize=10)
        ax.set_title(name, fontsize=9)

        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5], extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        valid_losses = loss_image[loss_mask]
        valid_losses = valid_losses[np.isfinite(valid_losses)]
        mean_loss = np.mean(valid_losses) if len(valid_losses) > 0 else np.nan
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.3e}", transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for spare in range(n_models, nrows * ncols):
        axes[spare // ncols][spare % ncols].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=11)
    fig.suptitle(f"Heatmap of logarithms of relative errors in (A,B) space", fontsize=12, y=1.01)
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

    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]

    for idx, (name, loss_image) in enumerate(loss_maps.items()):
        ax = axes[idx]
        im = ax.imshow(
            loss_image, origin="lower", aspect="auto", norm=norm, cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]], interpolation='nearest'
        )
        ax.set_facecolor('#dcdcdc')
        ax.set_xlabel("$A$", fontsize=10)
        ax.set_ylabel("$B$", fontsize=10) if idx == 0 else ax.set_ylabel("")
        ax.set_title(name, fontsize=9, wrap=True)

        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5], extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        valid_losses = loss_image[loss_mask]
        valid_losses = valid_losses[np.isfinite(valid_losses)]
        mean_loss = np.mean(valid_losses) if len(valid_losses) > 0 else np.nan
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.3e}", transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.subplots_adjust(right=0.92, left=0.06, wspace=0.3)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cfg.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=11)
    fig.suptitle(f"Heatmap of logarithms of relative errors in (A,B) space", fontsize=12, y=1.02)
    return fig


def plot_heatmaps_householder(
        loss_maps: Dict[str, np.ndarray],
        A_vals: np.ndarray,
        B_vals: np.ndarray,
        loss_mask: np.ndarray,
        cfg: HeatmapConfigHouseholder
) -> None:
    """Generate individual and combined figures."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    # Filter out valid color scale bounds
    all_losses = np.concatenate([loss[loss_mask] for loss in loss_maps.values()])
    all_losses = all_losses[np.isfinite(all_losses)]

    if len(all_losses) == 0:
        print("[ERROR] No valid loss values found for color scaling")
        return

    if cfg.vmin is None:
        vmin = float(np.percentile(all_losses, 1)) if cfg.loss_type == "log_rel_err" else max(
            float(np.percentile(all_losses, 1)), 1e-12)
    else:
        vmin = cfg.vmin

    if cfg.vmax is None:
        vmax = float(np.percentile(all_losses, 99))
    else:
        vmax = cfg.vmax

    if vmin <= 0 and cfg.loss_type != "log_rel_err":
        positive_losses = all_losses[all_losses > 0]
        vmin = float(np.min(positive_losses)) if len(positive_losses) > 0 else 1e-12
    if vmax <= vmin:
        vmax = vmin * 1e4

    if cfg.use_log_scale and cfg.loss_type != "log_rel_err":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        scale_label = "log scale"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        scale_label = "linear scale"

    print(f"\n      Loss type: {cfg.loss_type}")
    print(f"      Colour scale: [{vmin:.3e}, {vmax:.3e}] ({scale_label})")

    # 1. Individual heatmaps
    print("\nGenerating individual heatmaps...")
    for name, loss_image in loss_maps.items():
        fig, ax = plt.subplots(figsize=cfg.figsize)
        im = ax.imshow(
            loss_image, origin="lower", aspect="auto", norm=norm, cmap=cfg.cmap,
            extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]], interpolation='nearest'
        )

        if cfg.mask_unreachable:
            mask_for_contour = loss_mask.astype(float)
            ax.contour(mask_for_contour, levels=[0.5], extent=[A_vals[0], A_vals[-1], B_vals[0], B_vals[-1]],
                       colors='white', linewidths=0.5, alpha=0.5, linestyles='dashed')

        ax.set_facecolor('#dcdcdc')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f"{cfg.loss_type} ({scale_label})", fontsize=10)
        ax.set_xlabel("$A$", fontsize=12)
        ax.set_ylabel("$B$", fontsize=12)

        valid_losses = loss_image[loss_mask]
        valid_losses = valid_losses[np.isfinite(valid_losses)]
        mean_loss = np.mean(valid_losses) if len(valid_losses) > 0 else np.nan
        median_loss = np.median(valid_losses) if len(valid_losses) > 0 else np.nan
        ax.text(0.05, 0.95, f"Mean: {mean_loss:.3e}\nMedian: {median_loss:.2e}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f"{name}\nLoss assessment matrix visualization", fontsize=11)
        fname = os.path.join(cfg.output_dir, f"heatmap_{name}_{cfg.loss_type}_{timestamp}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=cfg.dpi)
        plt.close(fig)

    # 2. Combined matrix figure
    print("\nGenerating combined matrix figure...")
    if cfg.layout == "horizontal":
        fig = plot_heatmaps_householder_horizontal(loss_maps, A_vals, B_vals, loss_mask, cfg, norm, scale_label)
    else:
        fig = plot_heatmaps_householder_grid(loss_maps, A_vals, B_vals, loss_mask, cfg, norm, scale_label)

    combined_fname = os.path.join(cfg.output_dir, f"heatmap_COMBINED_{cfg.loss_type}_{cfg.layout}_{timestamp}.png")
    fig.savefig(combined_fname, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"      Combined figure saved: {combined_fname}")


def main_householder(config: Optional[HeatmapConfigHouseholder] = None,
                     model_names: Optional[List[str]] = None,
                     iterations_list: Optional[List[int]] = None) -> None:
    """Main execution orchestrator."""
    if config is None:
        config = HeatmapConfigHouseholder()
    if iterations_list is None:
        iterations_list = [config.num_iters]

    print("=" * 60)
    print("Performance Heatmap Generator with Brent and Householder Variants")
    print("=" * 60)

    # Step 1: Build base evaluation canvas
    A_grid, B_grid, C_grid, valid_mask, A_vals, B_vals, loss_mask = build_direct_grid_householder(config)
    A_flat = A_grid[valid_mask]
    B_flat = B_grid[valid_mask]
    C_flat = C_grid[valid_mask]

    if len(A_flat) == 0:
        print("[ERROR] Boundaries produced an empty projection space.")
        return

    # Step 2: Extract baseline Brent benchmark if toggled
    brent_loss_image = None
    if config.include_brent:
        print("\nRunning baseline Brent Optimization solver across grid...")
        B_brent_pred = run_brent_solver(A_flat, C_flat, config)
        brent_losses = compute_loss_metric(B_brent_pred, B_flat, config.loss_type)

        brent_loss_image = np.full((config.n_B, config.n_A), np.nan, dtype=np.float64)
        brent_loss_image[valid_mask] = brent_losses

    # Step 3: Load NN structures
    models = load_all_models_householder(config, model_names)
    X_np = construct_NN_inputs_1to1(A_flat, C_flat, mode=config.c_ht_mode)
    X_torch = torch.tensor(X_np, dtype=DTYPE).to(device)

    initial_predictions = {}
    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            initial_predictions[name] = model(X_torch).squeeze().cpu().numpy().astype(np.float64)

    # Step 4: Iterative evaluations loop
    for num_iters in iterations_list:
        print(f"\nProcessing configuration for refinement depth: {num_iters}")

        # Using a Dict to preserve sorting layout order
        loss_images = {}

        # Inject Brent as the outermost key to place it at the front (leftmost column)
        if brent_loss_image is not None:
            loss_images["Brent Method"] = brent_loss_image

        for name, B_pred_initial in initial_predictions.items():
            pointwise_losses = compute_householder_loss(
                B_pred_initial=B_pred_initial, B_true=B_flat, A_flat=A_flat, C_flat=C_flat,
                num_iters=num_iters, loss_type=config.loss_type, mode=config.c_ht_mode
            )
            loss_image = np.full((config.n_B, config.n_A), np.nan, dtype=np.float64)
            loss_image[valid_mask] = pointwise_losses
            loss_images[name] = loss_image

        temp_config = HeatmapConfigHouseholder(
            A_min=config.A_min, A_max=config.A_max, B_min=config.B_min, B_max=config.B_max,
            n_A=config.n_A, n_B=config.n_B, num_iters=num_iters, loss_type=config.loss_type,
            c_ht_mode=config.c_ht_mode, vmin=config.vmin, vmax=config.vmax, cmap=config.cmap,
            output_dir=config.output_dir, figsize=config.figsize, dpi=config.dpi,
            mask_unreachable=config.mask_unreachable, use_log_scale=config.use_log_scale, layout=config.layout
        )

        plot_heatmaps_householder(loss_images, A_vals, B_vals, loss_mask, temp_config)


if __name__ == '__main__':
    config = HeatmapConfigHouseholder(
        A_min=0.0,
        A_max=3.0,
        B_min=1e-7,
        B_max=1.22,
        n_A=200,
        n_B=200,
        num_iters=2,
        brent_max_iter=50,  # Brent root convergence limits
        include_brent=True,  # Keeps Brent active at the start of the plotting maps
        loss_type="log_rel_err",
        c_ht_mode="advanced",
        output_dir="Results_yb/heatmaps_householder",
        mask_unreachable=True,
        use_log_scale=True,
        layout="horizontal"
    )

    main_householder(config=config, model_names=None)