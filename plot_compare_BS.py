import numpy as np
import matplotlib.pyplot as plt
from Optimised_BS import C_ht, C_ht_default


# Default constants that MUST be used when calling C_ht
h_max_DEFAULT = 35.0
t_max_DEFAULT = 0.1
N_DEFAULT = 16
N1_DEFAULT = 17
N2_DEFAULT = 20
tau_DEFAULT = 0.1
# Default epsilon for avoiding numerical errors
EPSILON_DEFAULT = 1e-50


def plot_C_small_t(t, h_min=0, h_max=35, epsilon=EPSILON_DEFAULT):
    """
    Generates two sequential plots comparing the approximation C_ht(h,t) with the
    closed-form C_ht_default(h,t) for a fixed, small t.

    The plots display one after the other:
    1. A comparison plot of the two functions across the h range.
    2. A log-ratio plot showing log(C_ht_default(h,t) / C_ht(h,t)).

    Args:
        t (float): The fixed value of t to use for the plots.
        h_min (float): Minimum value of h to plot.
        h_max (float): Maximum value of h to plot.
    """

    # Generate h values for plotting
    h_values = np.linspace(h_min, h_max, 500)

    # Calculate C_ht (Approximation) and C_ht_default (True Value)
    C_approx = C_ht(h_values, t,
                    h_max = h_max_DEFAULT,
                    t_max = t_max_DEFAULT,
                    N = N_DEFAULT,
                    N_1 = N1_DEFAULT,
                    N_2 = N2_DEFAULT,
                    tau = tau_DEFAULT)
    C_true = C_ht_default(h_values, t)

    # Calculate the ratio
    # ratio = np.abs(C_true / np.maximum(C_approx, epsilon))
    #
    # # Calculate the log of the ratio
    # log_ratio = np.log(np.maximum(ratio, epsilon))

    ratio = C_true / C_approx

    # Calculate the log of the ratio
    log_ratio = np.log(ratio)

    # ------------------------------------------------------------------
    # # 1. First Plot: Direct Comparison
    # # ------------------------------------------------------------------
    # fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    # fig1.suptitle(f'Plot 1 (Fixed t): Direct Comparison (t = {t:.4f})', fontsize=16)
    #
    # ax1.plot(h_values, C_true, label=r'Exact ($C_{ht}^{default}$)', color='blue', linewidth=2)
    # ax1.plot(h_values, C_approx, label=r'Approximate ($C_{ht}$)', color='red', linestyle='--', alpha=0.7)
    #
    # ax1.set_title(r'$C_{ht}(h, t)$ vs. $h$')
    # ax1.set_xlabel(r'$h$')
    # ax1.set_ylabel(r'$C_{ht}(h, t)$ Value')
    # ax1.legend()
    # ax1.grid(True, linestyle=':', alpha=0.6)
    #
    # plt.show()  # Show the first plot

    # ------------------------------------------------------------------
    # 2. Second Plot: Log Ratio (Sequential Plot)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    # fig2.suptitle(r'Log Error: $\log \left( \frac{C_{default}(h, t)}{C_{Taylor}(h, t)} \right)$'+ f', for fixed t = {t:.4f}', fontsize=16)

    ax2.plot(h_values, log_ratio, label=r'$\log(\text{Ratio})$', color='darkorange', linewidth=1)

    # ax2.set_title(r'Log Error: $\log \left( \frac{C_{default}(h, t)}{C_{Taylor}(h, t)} \right)$')
    ax2.set_xlabel(r'$h$')
    ax2.set_ylabel(r'$\log \left( \frac{C_{default}(h, t)}{C_{expanded}(h, t)} \right)$')

    # Set tight y-limits around 0 to highlight tiny errors
    #ax2.set_ylim(-1e-4, 1e-4)

    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()  # Show the second plot


def plot_C_h(h, t_min=0, t_max=10, epsilon=EPSILON_DEFAULT):
    """
    Generates two sequential plots comparing the approximation C_ht(h,t) with the
    closed-form C_ht_default(h,t) for a fixed h.

    The plots display one after the other:
    1. A comparison plot of the two functions across the t range.
    2. A log-ratio plot showing log10(C_ht_default(h,t) / C_ht(h,t)).

    Args:
        h (float): The fixed value of h to use for the plots.
        t_min (float): Minimum value of t to plot.
        t_max (float): Maximum value of t to plot.
    """

    # Generate t values for plotting
    t_values = np.linspace(t_min, t_max, 500)

    # Calculate C_ht (Approximation) and C_ht_default (True Value)
    C_approx = C_ht(h, t_values,
                    h_max = h_max_DEFAULT,
                    t_max = t_max_DEFAULT,
                    N = N_DEFAULT,
                    N_1 = N1_DEFAULT,
                    N_2 = N2_DEFAULT,
                    tau = tau_DEFAULT)
    C_true = C_ht_default(h, t_values)

    # Calculate the ratio
    # ratio = np.abs(C_true / np.maximum(C_approx, epsilon))
    #
    # # Calculate the log of the ratio
    # log_ratio = np.log(np.maximum(ratio, epsilon))
    ratio = C_true / C_approx

    # Calculate the log of the ratio
    log_ratio = np.log(ratio)

    # # ------------------------------------------------------------------
    # # 1. First Plot: Direct Comparison
    # # ------------------------------------------------------------------
    # fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    # fig1.suptitle(f'Plot 1 (Fixed h): Direct Comparison (h = {h:.4f})', fontsize=16)
    #
    # ax1.plot(t_values, C_true, label=r'Exact ($C_{ht}^{default}$)', color='blue', linewidth=2)
    # ax1.plot(t_values, C_approx, label=r'Approximate ($C_{ht}$)', color='red', linestyle='--', alpha=0.7)
    #
    # ax1.set_title(r'$C_{ht}(h, t)$ vs. $t$')
    # ax1.set_xlabel(r'$t$')
    # ax1.set_ylabel(r'$C_{ht}(h, t)$ Value')
    # ax1.legend()
    # ax1.grid(True, linestyle=':', alpha=0.6)
    #
    # #plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()  # Show the first plot

    # ------------------------------------------------------------------
    # 2. Second Plot: Log Ratio (Sequential Plot)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    # fig2.suptitle(r'Log Error: $\log \left( \frac{C_{default}(h, t)}{C_{Taylor}(h, t)} \right)$'+ f', for fixed h = {h:.4f})', fontsize=16)

    ax2.plot(t_values, log_ratio, label=r'$\log(\text{Ratio})$', color='darkorange', linewidth=1)

    # ax2.set_title(r'Log Error: $\log_{10} \left( \frac{C_{ht}^{default}(h, t)}{C_{ht}(h, t)} \right)$')
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\log \left( \frac{C_{default}(h, t)}{C_{expanded}(h, t)} \right)$')

    # Set tight y-limits around 0
    #ax2.set_ylim(-1e-4, 1e-4)

    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.show()  # Show the second plot


# Example Usage
# plot_C_small_t(0.001, 0, 10)
# plot_C_small_t(0.001, 0, 10)
# plot_C_small_t(0.01, 0, 10)
# plot_C_small_t(0.1, 0, 10)
# plot_C_small_t(1, 0, 10)
# plot_C_small_t(10, 0, 10)


plot_C_h(h=0.01, t_min=0.001, t_max=0.2)
plot_C_h(h=0.1, t_min=0.001, t_max=0.2)
plot_C_h(h=1, t_min=0.001, t_max=0.2)
plot_C_h(h=10, t_min=0.001, t_max=0.2)



