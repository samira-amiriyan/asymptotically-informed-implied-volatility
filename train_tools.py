import torch
import torch.nn as nn
import pandas as pd
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader


# --- PREAMBLE & DEVICE CONFIGURATION & GLOBAL NUMERICAL STABILITY CONSTANT---
# The value of EPSILON is used by default to prevent division by zero or
# issues with log operations on zero or near-zero inputs.
if torch.cuda.is_available():
    device = torch.device("cuda")
    DTYPE = torch.float32  # Fast execution on NVIDIA
    TRAIN_EPSILON = 1e-7
else:
    device = torch.device("cpu")
    DTYPE = torch.float64  # Maximum fallback precision on CPU
    TRAIN_EPSILON = 1e-18

torch.set_default_dtype(DTYPE)

# --- CUSTOM LOSS FUNCTIONS ---
class MSRELoss(nn.Module):
    """Mean Squared Relative Error Loss."""
    def __init__(self, epsilon: float = TRAIN_EPSILON):
        super().__init__()
        # Store epsilon as a buffer to handle device movement automatically (cpu/gpu)
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=DTYPE))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates MSRE: Mean((pred - target) / max(target, epsilon))^2
        """
        # Ensure target is at least epsilon to avoid division by zero
        safe_target = torch.clamp(target, min=self.epsilon)

        # Calculate relative error
        relative_error = (pred - target) / safe_target

        # Return Mean Squared Relative Error
        return torch.mean(relative_error ** 2)


class MixedLoss(nn.Module):
    """
    Mixed Loss: 0.25 * MSE + 0.75 * MSRE.
    Balances absolute error (MSE) and relative error (MSRE).
    """

    def __init__(self, epsilon: float = TRAIN_EPSILON):
        super().__init__()
        # Store epsilon as a buffer to handle device movement automatically (cpu/gpu)
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=DTYPE))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate the raw difference once for efficiency
        diff = pred - target

        # 1. MSE component: Mean Squared Error
        # Standard absolute error contribution
        mse = torch.mean(diff ** 2)

        # 2. MSRE component: Mean Squared Relative Error
        # Use torch.clamp to enforce max(target, epsilon) in the denominator
        # This prevents division by zero without shifting large target values
        safe_target = torch.clamp(target, min=self.epsilon)
        relative_error = diff / safe_target
        msre = torch.mean(relative_error ** 2)

        # Mixed Loss = 0.25 * MSE + 0.75 * MSRE
        # Heavily weighted toward relative error to handle various scales
        return 0.25 * mse + 0.75 * msre


# --- TRAINING AND EVALUATION FUNCTIONS ---

#--- Standard Full-Batch Gradient Descent.---
def train_model(model: nn.Module, X_train: torch.Tensor, B_train: torch.Tensor, epochs: int, lr: float, criterion: nn.Module) -> Tuple[nn.Module, List[float]]:
    """
    Trains the given model and returns the model and the loss history.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_history: List[float] = []

    for _ in range(epochs):
        B_pred = model(X_train)
        loss = criterion(B_pred, B_train)

        # Record loss for the plot
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss_history

#--- Mini-batch SGD training with full-batch loss recording. ---
def train_model_sgd(
        model: nn.Module,
        X_train: torch.Tensor,
        B_train: torch.Tensor,
        epochs: int,
        lr: float,
        criterion: nn.Module,
        batch_size: int = 512
) -> Tuple[nn.Module, List[float]]:
    """
    Trains the model using Mini-batch SGD, but calculates and records
    the loss on the entire training set at the end of each epoch.
    """
    # Setup data loading for the training updates
    dataset = TensorDataset(X_train, B_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: List[float] = []

    for epoch in range(epochs):
        # --- Update Phase (SGD) ---
        model.train()
        for batch_X, batch_B in loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_B)
            loss.backward()
            optimizer.step()

        # --- Recording Phase (Full Dataset Loss) ---
        # We switch to eval mode to ensure consistent behavior (e.g., BatchNorm/Dropout)
        model.eval()
        with torch.no_grad():
            # Calculate full loss in batches to avoid OOM crashes
            # and ensure the "Full Dataset Loss" is recorded correctly.
            eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            total_loss = 0.0
            for eval_X, eval_B in eval_loader:
                preds = model(eval_X)
                # We multiply by batch size to get weighted average later
                l = criterion(preds, eval_B)
                total_loss += l.item() * eval_X.size(0)

            avg_loss = total_loss / len(dataset)
            loss_history.append(avg_loss)

    return model, loss_history

#--- SGD with variable Learning Rate based on VALIDATION set performance ---
def train_model_sgd_var(
        model: nn.Module,
        X_train: torch.Tensor,
        B_train: torch.Tensor,
        X_val: torch.Tensor,
        B_val: torch.Tensor,
        epochs: int,
        lr: float,
        criterion: nn.Module,
        batch_size: int = 512,
        patience: int = 5,
        factor: float = 0.25
) -> Tuple[nn.Module, List[float]]:
    """
    Trains the model using Mini-batch SGD with an adaptive learning rate scheduler.
    The learning rate is reduced when the validation loss hits a plateau or
    becomes excessively noisy.
    """
    # Setup data loading
    dataset = TensorDataset(X_train, B_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler monitors Validation Loss
    # Initialize the Scheduler
    # mode='min' because we want to minimize loss
    # factor=0.5 reduces LR by half when triggered
    # patience=5 waits for 5 epochs of no improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        threshold=1e-2,  # Minimum change to qualify as an improvement
    )

    loss_history: List[float] = []

    # Setup data loading for evaluation (to compute train loss via mini-batches)
    eval_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # --- Update Phase (SGD) ---
        model.train()
        for batch_X, batch_B in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_B)
            loss.backward()
            optimizer.step()

        # --- Recording & Adaptive Step Phase ---
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            # 1. Compute Training Loss using mini-batches
            for eval_X, eval_B in eval_train_loader:
                preds = model(eval_X)
                l = criterion(preds, eval_B)
                total_loss += l.item() * eval_X.size(0)

            avg_train_loss = total_loss / len(dataset)
            loss_history.append(avg_train_loss)

            # 2. Compute Validation Loss to trigger LR reduction
            # Using the full validation set directly
            # Validation loss triggers LR reduction
            val_loss = criterion(model(X_val), B_val).item()
            # Update the scheduler based on the average epoch loss
            # This will automatically adjust the LR inside the optimizer
            scheduler.step(val_loss)

    return model, loss_history

# --- DATA HANDLING & EVALUATION ---
def calculate_terminal_metrics(model: nn.Module, X: torch.Tensor, B: torch.Tensor, msre_scorer: MSRELoss) -> Dict[str, float]:
    """
    Computes MSE, MSRE, Max Absolute Diff, Max Relative Diff.
    """
    model.eval()
    with torch.no_grad():
        B_pred = model(X)

        # Absolute Difference for MaxAbsDiff
        abs_diff = torch.abs(B_pred - B)

        safe_B = torch.clamp(B, min=msre_scorer.epsilon)
        rel_diff = abs_diff / safe_B

        return {
            "MSE": torch.mean(abs_diff ** 2).item(),
            "MSRE": msre_scorer(B_pred, B).item(),
            "MaxAbsDiff": torch.max(abs_diff).item(),
            "MaxRelDiff": torch.max(rel_diff).item(),
        }

# --- DATA HANDLING FUNCTIONS  ---

def load_data_and_split(file_path: str, test_size: float, val_size: float = 0.0) -> tuple[torch.Tensor, ...]:
    """
    Loads data once and splits it into train, validation and test sets using the provided seed.
    """
    # Use a static attribute to load the dataframe only once
    if not hasattr(load_data_and_split, 'df_loaded'):
        try:
            load_data_and_split.df_loaded = pd.read_parquet(file_path)
            print(f"\n[INFO] Data loaded successfully from: {file_path}")
            print(f"[INFO] Total samples: {load_data_and_split.df_loaded.shape[0]}")
        except FileNotFoundError:
            raise FileNotFoundError(f"FATAL ERROR: Dataset file not found at '{file_path}'.")
        except Exception as e:
            raise Exception(f"FATAL ERROR: Could not load data from '{file_path}'. Error: {e}")

    df = load_data_and_split.df_loaded

    # Input features: (A, C, y, log_A, log_y, z_u, z_l) -> 7 features
    X = torch.tensor(df[['A', 'C', 'y', 'log_A', 'log_y', 'z_u', 'z_l']].values, dtype=DTYPE)
    # Target: B
    B = torch.tensor(df['B'].values, dtype=DTYPE).unsqueeze(1)

    # Consistent Train/Test Split using the provided seed
    # UNCOMMENT IF NEEDED
    #torch.manual_seed(seed)

    num_samples = X.shape[0]
    indices = torch.randperm(num_samples)

    # Calculate split sizes
    n_test = int(num_samples * test_size)
    n_val = int(num_samples * val_size)
    n_train = num_samples - n_test - n_val

    # Slice the indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Handle case where val_size is 0.0
    if n_val == 0:
        X_val = torch.empty(0, X.shape[1])
        B_val = torch.empty(0, 1)
    else:
        X_val, B_val = X[val_idx], B[val_idx]

    return X[train_idx], B[train_idx], X_val, B_val, X[test_idx], B[test_idx]


def get_model_characteristic_name(name: str, N_TEST: int, N_F_TEST: int, N_G_TEST: int) -> str:
    """
    Generates a descriptive model name including hyperparameters based on naming patterns.

    Refined Logic:
    - "Sig" in name -> appends N_g
    - "Simple" in name -> appends N (using N_TEST)
    - Default -> appends N_f and N_g
    """
    if "Sig" in name:
        return f"{name}_N_g{N_G_TEST}"

    if "Simple" in name:
        return f"{name}_N{N_TEST}"

    return f"{name}_N_f{N_F_TEST}_N_g{N_G_TEST}"