import pandas as pd
import torch
from typing import Type, Dict, Any, List, Tuple
from datetime import datetime
import time
import re
import os
import sys


# --- PREAMBLE & DEVICE CONFIGURATION ---
# Detect the best available hardware accelerator
from Final_05_train_tools import *
# Enforce the system-wide default data type chosen for this platform
torch.set_default_dtype(DTYPE)

print(f"--- Logic initialized on device: {device} using precision: {DTYPE} ---")


# Import custom architectures and tools
from F_260428_PolyAC import *
from F_260512_Gauss import *
from F_260513_PolyC import *
from F_260519_Simple import *
from F_260519_HardAC import *
from F_260519_Aspt import *
from yb_251030_plotting_functions import plot_loss_evolution

# --- MAIN EXECUTION BLOCK ---
total_start_time = time.time()

if __name__ == '__main__':
    print("--- Neural Network Comparison for Implied Volatility Computation---")

    # --- EXPERIMENT CONFIGURATION ---
    FILE_PATH = "datasets/dataset_AC_to_B_advanced_0_16_500_0_7_500.parquet"
    NUM_EXPERIMENTS = 6    # <-- Run the comparison this many times
    OUTPUT_DIR = "Results_yb"

    # --- Hyperparameters ---
    N_TEST = 128  # Hidden layer size for Simple models
    N_F_TEST = 5  # Number of terms in f0/f1 summation for Inv models
    N_G_TEST = 32  # Hidden layer size for g-MLP in Poly models
    N_EPOCHS = 150  # Number of training iterations (OR HIGHER (200) FOR GD)
    LR = 1e-3  # Learning rate
    TEST_SIZE = 0.2

    # --- LOSS FUNCTION SELECTION ---
    # Options: 'MSE', 'MSRE', 'MixedLoss'
    LOSS_FUNCTION = 'MSRE'
    train_model_method = train_model_sgd_var # train_model, train_model_sgd, train_model_sgd_var
    # Set VALIDATION_SIZE to 0 for standard GD/SGD, or > 0 for train_model_sgd_var
    # The data loader will handle the creation of the 3 sets automatically.
    VALIDATION_SIZE = 0.15 if 'sgd_var' in train_model_method.__name__ else 0.0

    # Map string names to actual Loss Class instances
    loss_map = {
        'MSE': nn.MSELoss().to(device),
        'MSRE': MSRELoss().to(device),
        'MixedLoss': MixedLoss().to(device),
    }

    try:
        CRITERION = loss_map[LOSS_FUNCTION]
    except KeyError:
        CRITERION = loss_map['MSE'].to(device)
        LOSS_FUNCTION = 'MSE'

    # Initialize separate scorers for consistent evaluation, regardless of CRITERION
    MSRE_SCORER = MSRELoss()

    # --- Models to Train ---
    # defining a list whose elements are tuples of the form: (name,model class,parameter dictionary)
    models_to_run: list[tuple[str, Type[nn.Module], dict]] = [
        #("SimpleGen", SimpleGen, {"N": N_TEST}),
        ("PolyACInvExpInter", PolyACInvExpInter, {"N_f": N_F_TEST, "N_g": N_G_TEST}),
        ("PolyACepsInvExpInter", PolyACepsInvExpInter, {"N_f": N_F_TEST, "N_g": N_G_TEST}),
        ("PolyACInvExpFree", PolyACInvExpFree, {"N_f": N_F_TEST, "N_g": N_G_TEST}),
        ("PolyACepsInvExpFree", PolyACepsInvExpFree, {"N_f": N_F_TEST, "N_g": N_G_TEST}),
    ]

    # --- DATA LOADING ---
    # We load the data once globally. The tool load_data_and_split is assumed to return
    # exactly 6 tensors: (X_train, B_train, X_val, B_val, X_test, B_test).
    print(f"Loading data from {os.path.basename(FILE_PATH)}...")

    data_tensors = load_data_and_split(FILE_PATH, TEST_SIZE, VALIDATION_SIZE)

    # Move all tensors to the detected GPU/Device immediately to keep the loop fast.
    X_train, B_train, X_val, B_val, X_test, B_test = [t.to(device) for t in data_tensors]


    # Initialize structure to hold all terminal results and all loss histories
    # A list whose elements will be dictionaries of the form: {metric name -> float value}
    all_results: List[Dict[str, float]] = []
    # Initialize a dictionary to store lists of loss histories per model name
    # For each tuple (name,model_class,params) in models_to_run, create a dictionary entry name -> []
    # Dict[str, List[List[float]]]. So: Keys: model names (strings) & Values: lists of loss histories [l_1, ..., L_N],
    # where each l_i is a list of losses itself
    all_loss_histories: Dict[str, List[List[float]]] = {name: [] for name, _, _ in models_to_run}

    print(f"Starting {NUM_EXPERIMENTS} runs using **{LOSS_FUNCTION}** loss...")

    # --- N-Experiment Loop ---
    for run_index in range(NUM_EXPERIMENTS):
        run_results = {}

        # Train and evaluate all models for the current run
        for name, ModelClass, kwargs in models_to_run:
            # Re-initialize the model for a fresh training start: Initialize and move to GPU
            model = ModelClass(**kwargs).to(device)

            # Train Model and retrieve loss history
            # We pass the appropriate sets based on method name
            if train_model_method.__name__ == 'train_model_sgd_var':
                trained_model, loss_history = train_model_method(
                    model, X_train, B_train, X_val, B_val, N_EPOCHS, LR, CRITERION
                )
            else:
                trained_model, loss_history = train_model_method(
                    model, X_train, B_train, N_EPOCHS, LR, CRITERION
                )

            # Store loss history for plotting later
            all_loss_histories[name].append(loss_history)

            # Evaluate Model on Train and Test sets
            train_metrics = calculate_terminal_metrics(trained_model, X_train, B_train, MSRE_SCORER)
            test_metrics = calculate_terminal_metrics(trained_model, X_test, B_test, MSRE_SCORER)

            # Store results for this run
            # Store all 4 metrics for train and test
            run_results[f"{name}_Train_MSE"] = train_metrics["MSE"]
            run_results[f"{name}_Train_MSRE"] = train_metrics["MSRE"]
            run_results[f"{name}_Train_MaxAbsDiff"] = train_metrics["MaxAbsDiff"]
            run_results[f"{name}_Train_MaxRelDiff"] = train_metrics["MaxRelDiff"]

            run_results[f"{name}_Test_MSE"] = test_metrics["MSE"]
            run_results[f"{name}_Test_MSRE"] = test_metrics["MSRE"]
            run_results[f"{name}_Test_MaxAbsDiff"] = test_metrics["MaxAbsDiff"]
            run_results[f"{name}_Test_MaxRelDiff"] = test_metrics["MaxRelDiff"]

        all_results.append(run_results)
        print(f"-> Completed Run {run_index + 1}/{NUM_EXPERIMENTS}")

    # --- POST-PROCESSING & OUTPUT ---
    now = datetime.now()

    datetime_str = now.strftime("%y%m%d_%H%M")
    loss_name = LOSS_FUNCTION.upper()
    base_filename = os.path.basename(FILE_PATH)

    match = re.search(r'to_B_(.*)\.parquet', base_filename)
    if match:
        file_characteristics = match.group(1).replace('/', '_')
    else:
        file_characteristics = "unknown_params"

    output_base_name = f"{datetime_str}_{loss_name}_{file_characteristics}"
    output_filename_excel = f"results_{output_base_name}.xlsx"
    output_filename_plot = f"loss_evolution_{output_base_name}.png"

    # 3. Create directory
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"\n[ERROR] Could not create output directory. Error: {e}")

    # --- Calculate and Print Averages ---

    # Convert results list to DataFrame for easy calculation of means
    df_results = pd.DataFrame(all_results)
    mean_results = df_results.mean().to_dict()

    # --- Prepare Final DataFrame for Excel and Printing ---
    final_data = []

    # Define Column Headers/Keys (10 metrics total)
    columns = [
        ("Train MSE", "_Train_MSE"),
        ("Train MSRE", "_Train_MSRE"),
        ("Train Max Abs Diff", "_Train_MaxAbsDiff"),
        ("Train Max Rel Diff", "_Train_MaxRelDiff"),

        ("Test MSE", "_Test_MSE"),
        ("Test MSRE", "_Test_MSRE"),
        ("Test Max Abs Diff", "_Test_MaxAbsDiff"),
        ("Test Max Rel Diff", "_Test_MaxRelDiff"),
    ]

    for name, _, _ in models_to_run:
        # Get the descriptive model name
        model_char_name = get_model_characteristic_name(name, N_TEST, N_F_TEST, N_G_TEST)

        row = {"Architecture": model_char_name}

        for col_name, key_suffix in columns:
            avg_value = mean_results[f"{name}{key_suffix}"]
            row[col_name] = avg_value

        final_data.append(row)

    # Final DataFrame with descriptive names as index
    df_final = pd.DataFrame(final_data).set_index("Architecture")

    # --- Printing Results ---
    HEADER_WIDTH = 250
    print("\n" + "="*HEADER_WIDTH)
    print("--- ARCHITECTURE COMPARISON: AVERAGE TERMINAL METRICS ---")
    print(f"Dataset: {os.path.basename(FILE_PATH)}, Runs: {NUM_EXPERIMENTS}, Training Loss: {LOSS_FUNCTION}")
    print("="*HEADER_WIDTH)

    # Print Header Row
    header_row = f"{'Model':<30}"
    for col_name, _ in columns:
        header_row += f"| {col_name:^20} "
    print(header_row)
    print("-" * HEADER_WIDTH)

    # Print Data Rows from the final DataFrame
    for model_name, row in df_final.iterrows():
        data_row = f"{model_name:<30}"
        for col_name, _ in columns:
            avg_value = row[col_name]
            data_row += f"| {avg_value:.6e} "
        print(data_row)

    print("="*HEADER_WIDTH)

    # Save Excel
    try:
        full_path_excel = os.path.join(OUTPUT_DIR, output_filename_excel)
        df_final.to_excel(full_path_excel, sheet_name='Average Metrics (N Runs)')
        print(f"\n[SUCCESS] Results saved to Excel file: {full_path_excel}")
    except Exception as e:
        print(f"\n[ERROR] Could not save results to Excel. Error: {e}")

    # --- Generate and Save Plot ---
    full_path_plot = os.path.join(OUTPUT_DIR, output_filename_plot)
    plot_loss_evolution(all_loss_histories, N_EPOCHS, LOSS_FUNCTION, full_path_plot)
    
total_end_time = time.time()
total_runtime = total_end_time - total_start_time

print("\nTotal runtime of all experiments: {:.2f} seconds".format(total_runtime))
print("Total runtime: {:.2f} minutes".format(total_runtime / 60))