import torch
import torch.nn as nn
from typing import List, Tuple, Type, Any
import os
import copy

import sys

# Add project paths
sys.path.append(".")

from PolyAC import *
from PolyC import *
from Simple import *
from HardAC import *
from Aspt import *
from Gauss import *

from train_tools import *

# Enforce the system-wide default data type chosen for this platform
torch.set_default_dtype(DTYPE)

# --- Configuration & Global Parameter Preamble ---
FILE_PATH = "datasets/..."
MODEL_DIR = "Trained_Models/Dataset2"
N_EPOCHS = 150
LR = 1e-3
# Dataset Splits
TEST_SIZE = 0.2
VAL_SIZE = 0.15

N_RUNS = 10  # Number of training attempts per model configuration

# SEED = 42 # Fixed seed for final, deterministic training
N_TEST = 128  # Hidden layer size for Simple models
N_F_TEST = 5  # Number of terms in f0/f1 summation for Inv models
N_G_TEST = 64  # Hidden layer size for g-MLP in Poly models

# Target learning function method
train_model_method = train_model_sgd_var #train_model, train_model_sgd, or train_model_sgd_var

# Specify the chosen models, their configurations, and the required loss function
MODEL_CONFIGS: List[Tuple[str, Type[nn.Module], Dict[str, Any]]] = [
    # Name MUST be unique and reflects the model/loss choice
    ("SimpleExp_MSE", SimpleExp, {"N": N_TEST, "loss_name": "MSE"}),
    ("PolyACInvGenInter_MSE", PolyACInvGenInter, {"N_f": N_F_TEST, "N_g": N_G_TEST, "loss_name": "MSE"}),
    ("PolyCInvGenFree_MSE", PolyCInvGenFree, {"N_f": N_F_TEST, "N_g": N_G_TEST, "loss_name": "MSE"}),
    ("GaussACInvGenInter_MSE", GaussACInvGenInter, {"N_f": N_F_TEST, "N_g": N_G_TEST, "loss_name": "MSE"}),
    ("HardACInvExpFree_MSE", HardACInvExpFree, {"N_f": N_F_TEST, "N_g": N_G_TEST, "loss_name": "MSE"}),
]

# --- LOSS FUNCTION MAPPING ---
CRITERION_MAP: Dict[str, Type[nn.Module]] = {
    "MSE": nn.MSELoss,
    "MSRE": MSRELoss,
    "MixedLoss": MixedLoss
}


def save_model(model: nn.Module, model_name: str, config: Dict[str, Any], best_score: float) -> str:
    """Saves the model's state dictionary, configuration, and best recorded loss."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    file_path = os.path.join(MODEL_DIR, f"{model_name}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_name': model_name,
        'best_test_loss': best_score
    }, file_path)
    print(f"[SUCCESS] Best model for {model_name} (Loss: {best_score:.6f}) saved to: {file_path}")
    return file_path

def load_model(model_name: str, ModelClass: Type[nn.Module], config: Dict[str, Any]) -> nn.Module:
    """Loads a model from a saved checkpoint."""
    file_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if not os.path.exists(file_path):
        # This is expected behavior if model_manager.py hasn't been run yet
        raise FileNotFoundError(f"Saved model file not found for {model_name}")

    # FIX: Added weights_only=True to suppress the FutureWarning and improve security
    checkpoint = torch.load(file_path, weights_only=True)

    # We must explicitly initialize the model using the class and configuration
    model = ModelClass(**{k: v for k, v in config.items() if k not in ['loss_name']})
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] Successfully loaded model: {model_name}")
    return model

def evaluate_loss(model: nn.Module, X: torch.Tensor, B: torch.Tensor, criterion: nn.Module) -> float:
    """Evaluates the model on the test set and returns the loss."""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        loss = criterion(predictions, B)
    return loss.item()


def fx_model_training_saving(
        data_file_path: str,
        models_to_run: List[Tuple[str, Type[nn.Module], Dict[str, Any]]],
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        epochs: int = N_EPOCHS,
        lr: float = LR,
        n_runs: int = N_RUNS
) -> Dict[str, nn.Module]:
    """
    Trains specified models multiple times.
    Keeps the iteration that performs best on the test set (the 20% holdout).
    """

    print("--- Starting One-Time Model Training and Saving ---")

    # Split for training/val/test
    data_tensors = load_data_and_split(data_file_path, test_size, val_size)

    # Parallel comparison feature: Send all data split tensors straight to device
    X_train, B_train, X_val, B_val, X_test, B_test = [t.to(device) for t in data_tensors]

    final_best_models: Dict[str, nn.Module] = {}

    for name, ModelClass, kwargs in models_to_run:
        loss_name = kwargs.get('loss_name', 'MSE')
        if loss_name not in CRITERION_MAP:
            print(f"[ERROR] Unknown loss function '{loss_name}'. Skipping {name}.")
            continue

        criterion_class = CRITERION_MAP[loss_name]
        criterion = criterion_class()

        best_val_loss = float('inf')
        best_model_state = None

        print(f"\n>> Optimizing {name} (using {loss_name})...")

        for run_idx in range(n_runs):
            # Instantiate model (exclude loss_name from init)
            model_config = {k: v for k, v in kwargs.items() if k != 'loss_name'}
            model = ModelClass(**model_config)

            # Check if using dynamic step size variant (sgd_var)
            if train_model_method.__name__ == 'train_model_sgd_var':
                trained_model, _ = train_model_method(
                    model, X_train, B_train, X_val, B_val, epochs, lr, criterion
                )
            else:
                trained_model, _ = train_model_method(
                    model, X_train, B_train, epochs, lr, criterion
                )

            # Evaluate on the validation set
            current_val_loss = evaluate_loss(trained_model, X_val, B_val, criterion)

            print(f"   Run {run_idx + 1}/{n_runs} - Validation Loss: {current_val_loss:.6f}")

            # Keep track of the best performing version
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = copy.deepcopy(trained_model.state_dict())

        # Load the best state into a new instance to return/save
        if best_model_state is not None:
            final_model = ModelClass(**model_config)
            final_model.load_state_dict(best_model_state)

            # Calculate an unbiased final metric score using untouched Test split data
            final_test_loss = evaluate_loss(final_model, X_test, B_test, criterion)
            print(f" -> Best Run Selected! Final Unbiased Test Loss: {final_test_loss:.6f}")

            save_model(final_model, name, kwargs, final_test_loss)
            final_best_models[name] = final_model
        else:
            print(f"[ERROR] No valid model state found for {name}")

    return final_best_models

if __name__ == '__main__':
    fx_model_training_saving(FILE_PATH, MODEL_CONFIGS)