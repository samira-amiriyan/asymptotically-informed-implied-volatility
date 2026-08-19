import pandas as pd
import os
import sys
import torch
from typing import Dict, Any


# Import the required C_ht functions
from Optimised_BS import *

# --- PREAMBLE & DEVICE CONFIGURATION & GLOBAL NUMERICAL STABILITY CONSTANT---
if torch.cuda.is_available():
    device = torch.device("cuda")
    DTYPE = torch.float32  # Fast execution on NVIDIA
else:
    device = torch.device("cpu")
    DTYPE = torch.float64  # Maximum fallback precision on CPU

torch.set_default_dtype(DTYPE)

pd.set_option('display.precision', 18)

# --- GLOBAL NUMERICAL CONFIGURATION ---
# These constants control the switching logic and depth of series expansions in C_ht.
h_max_DEFAULT = 35.0
t_max_DEFAULT = 0.1
N_DEFAULT = 16
N1_DEFAULT = 17
N2_DEFAULT = 20
tau_DEFAULT = 0.1

# Grid density for dataset generation (n_A rows x n_B columns)
n_A_DEFAULT = 500
n_B_DEFAULT = 500

# File system settings
dir_name_DEFAULT = "datasets"

# Numerical floor (epsilon) to prevent log(0) or division by zero in high-precision math
DS_EPSILON = 1e-50

# Pre-computed mathematical constants in float64 (Double Precision)
SQRT_2 = np.sqrt(2.0, dtype=np.float64)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0, dtype=np.float64)
SQRT_2_PI = np.sqrt(2.0 * np.pi, dtype=np.float64)


def _calculate_auxiliary_features(A, C, mode="advanced"):
    """
        Computes  features from the primary variables A and C.

        Inputs:
            A: np.ndarray (Dimension: N) - Parameter A values
            C: np.ndarray (Dimension: N)
            mode: str - "advanced" for stabilized expansions, "default" for textbook formulas

        Returns:
            tuple (y, log_A, log_y, z_u, z_l) - All arrays are of dimension N.
    """
    # y = 1/C - 1, Shape: (N,)
    y = 1.0 / C - 1.0

    # log features
    log_A = np.log(np.maximum(A, DS_EPSILON))
    log_y = np.log(np.maximum(y, DS_EPSILON))

    # Boundary calculations, shape (N,).
    B_u_val = B_u(A)
    B_l_val = B_u_val - SQRT_2_PI

    # z_u calculation (upper boundary feature)
    h_u = A / B_u_val
    t_u = B_u_val / 2.0
    C_ht_u = C_ht_default(h_u, t_u) if mode == "default" else C_ht(h_u, t_u, h_max=h_max_DEFAULT, t_max=t_max_DEFAULT, N=N_DEFAULT,
            N_1=N1_DEFAULT, N_2=N2_DEFAULT, tau=tau_DEFAULT)

    z_u_raw = (C_ht_u / np.maximum(C, DS_EPSILON)) - 1.0
    z_u = np.maximum(z_u_raw, DS_EPSILON)

    # z_l calculation (lower boundary feature)
    h_l = A / np.maximum(B_l_val, DS_EPSILON)
    t_l = B_l_val / 2.0
    C_ht_l = C_ht_default(h_l, t_l) if mode == "default" else C_ht(h_l, t_l, h_max=h_max_DEFAULT, t_max=t_max_DEFAULT, N=N_DEFAULT,
            N_1=N1_DEFAULT, N_2=N2_DEFAULT, tau=tau_DEFAULT)

    numerator = C - C_ht_l
    z_l_raw = numerator / np.maximum(1.0 - C, DS_EPSILON)
    z_l = np.maximum(z_l_raw, DS_EPSILON)

    return y, log_A, log_y, z_u, z_l

# --- DATASET CONSTRUCTION ---
def construct_dataset_ABC(A_min=0.0, A_max=64.0, B_min=0.0, B_max=10.0, n_A=n_A_DEFAULT, n_B=n_B_DEFAULT, mode="advanced") -> str:
    """
        Generates a full synthetic dataset by sweeping across a grid of (A, B).

        Dimensions:
            Output dataframe rows = n_A * n_B (before filtering invalid values).
            The final Parquet file contains 8 columns: [A, C, y, log_A, log_y, z_u, z_l, B].
    """
    # 1. Define A and B grid points
    A_values = np.linspace(A_min, A_max, n_A, dtype=np.float64)
    B_offset = (B_max - B_min) / (n_B * 2)
    B_values = np.linspace(B_min + B_offset, B_max, n_B, dtype=np.float64)

    # 2. Create the 2D Cartesian grid and flatten into 1D vectors for processing
    A_grid, B_grid = np.meshgrid(A_values, B_values)
    A_flat = A_grid.flatten()
    B_flat = B_grid.flatten()

    # 3. Calculate h and t for the grid (A, B)
    h = A_flat / B_flat
    t = B_flat / 2.0

    # 4. Calculate C using the specified mode
    if mode == "default":
        print(f"Constructing dataset using C_ht_default (Default Formula)...")
        C_flat = C_ht_default(h, t)
    elif mode == "advanced":
        print(f"Constructing dataset using C_ht (Advanced Approximation)...")
        # Explicitly pass the default parameters for the series approximations
        C_flat = C_ht(
            h, t,
            h_max=h_max_DEFAULT,
            t_max=t_max_DEFAULT,
            N=N_DEFAULT,
            N_1=N1_DEFAULT,
            N_2=N2_DEFAULT,
            tau=tau_DEFAULT)
    else:
        raise ValueError(f"Mode must be 'default' or 'advanced', got '{mode}'")

    # 5. The physical map requires 0 < C <= 1. Filter out invalid values
    valid_mask = (C_flat > DS_EPSILON) & (C_flat <= 1.0 - DS_EPSILON)

    A_final = A_flat[valid_mask]
    B_final = B_flat[valid_mask]
    C_final = C_flat[valid_mask]

    # 6. Calculate required auxiliary input variables for the neural network
    y, log_A, log_y, z_u, z_l = _calculate_auxiliary_features(A_final, C_final, mode)


    # 8. Create DataFrame and define file path
    data = pd.DataFrame({
        'A': A_final,
        'C': C_final,
        'y': y,
        'log_A': log_A,
        'log_y': log_y,
        'z_u': z_u,
        'z_l': z_l,
        'B': B_final
    })

    # File management: generate descriptive filename
    dir_name = dir_name_DEFAULT
    os.makedirs(dir_name, exist_ok=True)

    file_name = (
        f"dataset_AC_to_B_{mode}_{int(A_min)}_{int(A_max)}_{int(n_A)}_{int(B_min)}_{int(B_max)}_{int(n_B)}.parquet"
    )
    file_path = os.path.join(dir_name, file_name)

    # 9. Save the dataset
    data.to_parquet(file_path, index=False)
    print(f"Successfully generated dataset with {len(data)} rows.")
    print(f"The learning function is (A, C, y, log_A, log_y, z_u, z_l) -> B (7 inputs).")
    print(f"Saved to: {file_path}")

    return file_path

def sample_and_save_excel(parquet_file_path: str, sample_frac=0.1):
    """
    Loads a parquet file, randomly selects a fraction of the data (default 10%),
    and saves it to an Excel file for inspection.
    """
    print("---")
    try:
        # 1. Load the Parquet dataset
        print(f"Loading dataset from: {parquet_file_path}")
        data = pd.read_parquet(parquet_file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {parquet_file_path}. Cannot sample.")
        return

    # 2. Randomly select the sample
    sample_size = int(len(data) * sample_frac)
    sample_data = data.sample(n=sample_size, random_state=42)

    # 3. Define the Excel file name
    base_name = os.path.basename(parquet_file_path)
    excel_file_name = base_name.replace(".parquet", "_10pct_sample.xlsx")
    excel_file_path = os.path.join(os.path.dirname(parquet_file_path), excel_file_name)

    # 4. Save to Excel
    sample_data.to_excel(excel_file_path, index=False)

    print(f"Successfully created a {sample_frac * 100:.0f}% random sample with {len(sample_data)} rows.")
    print(f"Saved sample to: {excel_file_path}")
    print("---")


def construct_NN_inputs(A_values, C_values, mode="advanced"):
    """
        Creates a full input matrix for Neural Network evaluation.

        Inputs:
            A_values: Scalar or Array (Size M)
            C_values: Scalar or Array (Size K)

        Returns input matrix [A, C, y, log_A, log_y, z_u, z_l]:
            np.ndarray (Dimension: M*K by 7) - Cartesian product of all inputs.
    """
    # Ensure inputs are at least 1D arrays
    A_arr = np.atleast_1d(A_values).astype(np.float64)
    C_arr = np.atleast_1d(C_values).astype(np.float64)

    # meshgrid creates the Cartesian product (all combinations)
    # indexing='ij' ensures A varies across rows, C across columns
    A_grid, C_grid = np.meshgrid(A_arr, C_arr, indexing='ij')

    A_flat = A_grid.flatten()
    C_flat = C_grid.flatten()

    y, log_A, log_y, z_u, z_l = _calculate_auxiliary_features(A_flat, C_flat, mode)

    # Stack into final NN input format (N, 7)
    return np.column_stack([A_flat, C_flat, y, log_A, log_y, z_u, z_l])


def construct_NN_inputs_1to1(A_vals, C_vals, mode="advanced"):
    """
        A simplified version of input construction for paired (A, C) points.

        Inputs:
            A_vals: np.ndarray (Size N)
            C_vals: np.ndarray (Size N)

        Returns:
            np.ndarray (Dimension: N by 7)
    """
    A = np.atleast_1d(A_vals)
    C = np.atleast_1d(C_vals)

    y, log_A, log_y, z_u, z_l = _calculate_auxiliary_features(A, C, mode)
    return np.column_stack([A, C, y, log_A, log_y, z_u, z_l])


def calculate_zu_zl(A: np.ndarray, C: np.ndarray, mode: str = "advanced") -> tuple[np.ndarray, np.ndarray]:
    """
        Standalone calculator for z_u and z_l features.

        Inputs:
            A, C: np.ndarrays of same shape.
        Returns:
            (z_u, z_l) tuple of np.ndarrays.
    """
    A_final = A.astype(np.float64)
    C_final = C.astype(np.float64)

    # Calculate B_u and B_l for A_final
    B_u_final = B_u(A_final)
    B_l_final = B_l(A_final)

    # --- Calculate z_u ---
    h_u = A_final / B_u_final
    t_u = B_u_final / 2.0

    C_ht_u = C_ht(h_u, t_u) if mode == "advanced" else C_ht_default(h_u, t_u)

    # z_u = max(C_{ht}(h_u,t_u)/C - 1, 0) + EPSILON
    z_u_raw = (C_ht_u / C_final) - 1.0
    z_u_final = np.maximum(z_u_raw, DS_EPSILON)

    # --- Calculate z_l ---
    h_l = A_final / np.maximum(B_l_final, DS_EPSILON)
    t_l = B_l_final / 2.0

    C_ht_l = C_ht(h_l, t_l) if mode == "advanced" else C_ht_default(h_l, t_l)

    # z_l = max((C - C_{ht}(h_l,t_l))/(1-C), 0) + EPSILON
    numerator = C_final - C_ht_l
    z_l_raw = numerator / np.maximum(1.0 - C, DS_EPSILON) # Added EPSILON for stability
    z_l_final = np.maximum(z_l_raw, DS_EPSILON)

    return z_u_final, z_l_final


# --- MAIN ANALYSIS FUNCTION ---
def generate_iv_data_a_fixed_trained_model(
        A_fixed: float,
        B_min: float,
        B_max: float,
        num_points: int,
        mode: str = "advanced",
        trained_models: Dict[str, torch.nn.Module] = None
) -> Dict[str, Any]:
    """
    Computes true IV map data and model predictions.
    Returns a dictionary containing all arrays and boundary scalars.
    """
    # 1. Sample True Map Data (B -> C)
    B_values_np = np.linspace(B_min + DS_EPSILON, B_max, num_points)
    A_array = np.full_like(B_values_np, A_fixed)

    h = A_array / B_values_np
    t = B_values_np / 2.0

    if mode == "advanced":
        C_values_np = C_ht(h, t)
    else:
        C_values_np = C_ht_default(h, t)

    # Filter valid range
    valid_mask = (C_values_np > DS_EPSILON) & (C_values_np <= 1.0 - DS_EPSILON)
    B_final = B_values_np[valid_mask]
    C_final = C_values_np[valid_mask]
    A_final = A_array[valid_mask]

    # 2. Compute Boundaries for specific A
    B_u_val = B_u(np.array([A_fixed]))[0]
    B_l_val = B_l(np.array([A_fixed]))[0]

    # Boundary C values
    h_u, t_u = A_fixed / B_u_val, B_u_val / 2.0
    h_l, t_l = A_fixed / B_l_val, B_l_val / 2.0

    if mode == "advanced":
        C_u = C_ht(np.array([h_u]), np.array([t_u]))[0]
        C_l = C_ht(np.array([h_l]), np.array([t_l]))[0]
    else:
        C_u = C_ht_default(np.array([h_u]), np.array([t_u]))[0]
        C_l = C_ht_default(np.array([h_l]), np.array([t_l]))[0]

    # 3. Model Predictions using construct_NN_inputs
    X_numpy = construct_NN_inputs_1to1(A_final, C_final, mode=mode)
    X_torch = torch.tensor(X_numpy, dtype=DTYPE).to(device)

    model_preds = {}
    if trained_models:
        for name, model in trained_models.items():
            model.eval()
            with torch.no_grad():
                pred = model(X_torch).squeeze().cpu().numpy()
                model_preds[name] = pred

    return {
        "A_fixed": A_fixed,
        "B_true": B_final,
        "C_true": C_final,
        "model_predictions": model_preds,
        "boundaries": {
            "B_u": B_u_val, "B_l": B_l_val,
            "C_u": C_u, "C_l": C_l
        },
        "mode": mode
    }


if __name__ == '__main__':
    # --- Example Usage for Dataset Generation ---
    # Range configuration for Grid
    A_min_val, A_max_val = 0.0, 16.0
    B_min_val, B_max_val = 1e-5, 7.07

    # Generate the dataset using the 'advanced' (approximate) mode
    advanced_path = construct_dataset_ABC(
        A_min=A_min_val, A_max=A_max_val,
        B_min=B_min_val, B_max=B_max_val,
        n_A=n_A_DEFAULT, n_B=n_B_DEFAULT, mode="advanced"
    )

    # Sample and save to Excel for the 'advanced' dataset
    sample_and_save_excel(advanced_path)