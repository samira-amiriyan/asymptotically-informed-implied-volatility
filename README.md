# ASYMPTOTICALLY-INFORMED NEURAL NETWORKS FOR BLACK-SCHOLES IMPLIED VOLATILITY COMPUTATION

**Samira Amiriyan and Youness Boutaib**

## Overview

This repository contains the code and computational resources associated with the paper: "ASYMPTOTICALLY-INFORMED NEURAL NETWORKS FOR BLACK-SCHOLES IMPLIED VOLATILITY COMPUTATION"

The purpose of this repository is to provide the code required to reproduce the numerical experiments, figures, and tables presented in the paper.

The paper investigates the computation of implied volatility using neural networks, incorporating information from different asymptotic regimes into the learning framework.

---

### Figure 1A:
Run the **'3regions.py'** script.
Before running the code, at the beginning of the code, set the value of the fixed parameter (A) to the desired value. In the paper, we use (A=3).
After setting (A), run the script to generate the figure.

### Figure 1B:
Run the **'3regions_inverse.py'** script.
Before running the code, at the beginning of the code, set the value of the fixed parameter (A) to the desired value. In the paper, we use (A=3).
After setting (A), run the script to generate the figure.

### Figure 2:
Run the **'3regions_asymptotic.py'** script.
Before running the code, set the desired range and number of grid points for (A) in 
```python
A_vals = np.linspace(... , ... , ...)
```
In the paper, we use (A \in [0.001,100]) with 500 grid points.
Then run the script to generate figure.

## Dataset Generation

To generate the datasets, run the **'generate_dataset.py'** script.
Before running the code, near the beginning of the script, the number of grid points in the (A)- and (B)-directions can be specified using
```python
n_A_DEFAULT = ...
n_B_DEFAULT = ...
```

and at the end of the main block, inside the 'if __name__ == '__main__':', the ranges of (A) and (B) can be specified by

```python
A_min_val, A_max_val = ...
B_min_val, B_max_val = ...
```
Two computation modes are available:
* `mode="default"` uses the direct Black--Scholes formulation.
* `mode="advanced"` uses the numerical approximations and asymptotic expansions described in the paper for improved numerical stability in the relevant regimes.

The two corresponding dataset-generation blocks are located at the end of the script, inside the same 'if __name__ == '__main__':' block. The script also contains optional commands for saving a random sample of the generated dataset as an Excel file for inspection.
For the default mode, uncomment 
```python
sample_and_save_excel(default_path)
```
and for the advanced mode, uncomment 
```python
sample_and_save_excel(advanced_path)
```
Make sure that the sampling command corresponds to the dataset-generation mode that is currently active.

Then run the script. The generated dataset is automatically saved in
```text
datasets/
```
in Parquet format.

---


### Figures 3, 8, 12 & Tables 2, 4, 6:
Run **'train_eval.py'**.
Before running the script:

1. Open **'train_tools.py'** and adjust the parameters related to `train_model_sgd` or `train_model_sgd_var` if required. The rest of this file should remain unchanged.

2. In **'train_eval.py'**, inside the 'if __name__ == '__main__':' set the the datasets generated previously:
```python
FILE_PATH = "datasets/..."
```

3. Immediately below the same section, set the number of repetitions and the main hyperparameters as desired:

```python
NUM_EXPERIMENTS = ...
N_TEST = ...
N_F_TEST = ...
N_G_TEST = ...
N_EPOCHS = ...
LR = ...
```

4. Under the '--- LOSS FUNCTION SELECTION ---' section, select the loss function (Available options are: MSE, MSRE, MixedLoss):

```python
LOSS_FUNCTION = '...'
```
and below that select the training method (Available options are: train_model, train_model_sgd, train_model_sgd_var):
```python
train_model_method = ...
```

5. In the 'Models to Train' section, uncomment the model or models that you want to run.

Then run 'train_eval.py'.
The resulting Excel file containing the evaluation metrics (Tables) and the loss-evolution Figures are saved in:

```text
Results_yb/
```

### Figures 4, 9, 13:
Before generating Figures, the required models must first be trained and saved using **'train_save.py'**. In **'train_save.py'** under  '--- Configuration & Global Parameter Preamble ---' section:

1. Specify the dataset:

```python
FILE_PATH = "datasets/..."
```

2. Immediately below the same section, set the training parameters, if required:

```python
N_EPOCHS = ...
LR = ...
TEST_SIZE = ...
VAL_SIZE = ...
N_RUNS = ...

N_TEST = ...
N_F_TEST = ...
N_G_TEST = ...
```

3. Below them, select the training method with the available options train_model, train_model_sgd, train_model_sgd_var:

```python
train_model_method = ...
```

4. In `MODEL_CONFIGS`, activate the desired architectures and specify the corresponding loss function (`MSE`, `MSRE`).

Then run it. 
The best trained model from the specified runs is saved in the directory defined by:

```python
MODEL_DIR = "Trained_Models/..."
```

These saved models are then used to generate Figures.


Then, in **'heatmap_ab.py'**, at the end of the main block, inside the 'if __name__ == '__main__':', specify the desired heatmap settings `A_min`, `A_max`, `B_min`, and `B_max` determine the plotting region, `n_A` and `n_B` determine the grid resolution. The error measure (MSE or MSRE) can be selected through `loss_type`, and `layout` can be set to either `"grid"` or `"horizontal"`. The color scale of the heatmaps can be selected using `use_log_scale`:
```python
use_log_scale = True #or False
```
Finally, run **'heatmap_ab.py'**. The generated heatmaps are saved in the directory specified by `output_dir`.


### Figures 5, 10, 14 & Tables 3, 5, 7:
First train and save the desired model architectures following the instructions provided in the **Figure 4** section above.

Then, in the main block of the **'Householder.py'** script, at the end of the main block, inside the 'if __name__ == '__main__':', set the desired parameters `A_fixed` specifies the fixed value of (A), `B_min` and `B_max` specify the range of (B), `num_points` determines the number of evaluation points, `num_iters` specifies the number of Householder iterations in:

```python
results = benchmark_model_refinement(
...
)

plot_iteration_convergence(
...
)
```
Use the same parameter values in both functions and then run the script.

The LaTeX code for Tables are printed in the terminal, while the Figures are saved in:

```text
Results_yb/convergence_Householder/
```

### Figure 6, 7, 11, 15:
First train and save the desired model architectures following the instructions provided in the **Figure 4** section above.

Then, in the main block of the **'householder_brent_heatmap.py'** script, at the end of the main block, inside the 'if __name__ == '__main__':', set the desired configuration `A_min`, `A_max`, `B_min`, and `B_max` specify the evaluation region, `n_A` and `n_B` determine the grid resolution, `num_iters` specifies the number of Householder iterations, `brent_max_iter` specifies the maximum number of Brent iterations, `loss_type` can be set to `"MSE"`, `"MSRE"`,  `layout` can be `"grid"` or `"horizontal"` in:

```python
config = HeatmapConfigHouseholder(
    ...
)
```
Please note To generate Figure 6, set `include_brent=False`. For Figures 7, 11, and 15, set `include_brent=True`.

Then run the script. The resulting heatmaps are saved in:

```text
Results_yb/heatmaps_householder/
```

### Figures 16 and 17

Both figures are generated using the **'plot_compare_BS.py'** script containing the functions `plot_C_small_t(...)` and `plot_C_h(...)`. These functions compare the expanded approximation `C_ht` with the direct formulation `C_ht_default` through the logarithmic ratio.

The required plotting function and its parameters can be selected at the end of the script.

To generate Figure 16, at the end of the script, uncomment the required `plot_C_small_t(...)` call(s) and specify:

* `t`: the fixed value of (t);
* `h_min`: the lower bound of the (h)-interval;
* `h_max`: the upper bound of the (h)-interval.

For example,
```python
plot_C_small_t(t=0.01, h_min=0, h_max=10)
```
generates the fixed-(t) plot for (t=0.01) and (h\in[0,10]).

When generating Figure 17, the `plot_C_h(...)` calls at the end of the script can be commented out if the fixed-(h) plots are not required.

To generate Figure 17, use the function, at the end of the script, uncomment the required `plot_C_h(...)` call(s) and specify:

* `h`: the fixed value of (h);
* `t_min`: the lower bound of the (t)-interval;
* `t_max`: the upper bound of the (t)-interval.

For example,
```python
plot_C_h(h=0.1, t_min=0.001, t_max=0.2)
```
generates the fixed-(h) plot for (h=0.1) and (t\in[0.001,0.2]).

When generating Figure 18, the `plot_C_small_t(...)` calls can be commented out if the fixed-(t) plots are not required.



---




