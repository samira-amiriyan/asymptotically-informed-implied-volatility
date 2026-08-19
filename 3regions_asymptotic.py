# Re-import libraries after code execution environment reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt, pi

# Redefine the function C(A, B)
def C_func(A, B):
    return norm.cdf(-A / B + B / 2) - norm.cdf(-A / B - B / 2) * np.exp(A)

# Define A values
A_vals = np.linspace(0.001, 100, 500)

# Compute Bu, Bl for each A and the corresponding C values
sqrt_2A = np.sqrt(2 * A_vals)
phi_val = norm.cdf(-sqrt_2A)
exp_A = np.exp(A_vals)

Bu_vals = sqrt_2A + sqrt(pi / 2) + np.sqrt(2 * pi) * phi_val * exp_A
Bl_vals = sqrt_2A - sqrt(pi / 2) + np.sqrt(2 * pi) * phi_val * exp_A

Cu_vals = C_func(A_vals, Bu_vals)
Cl_vals = C_func(A_vals, Bl_vals)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Curve 1: C = 1
ax.axhline(1, color='black', linestyle='dotted', label=r"$C=1$")

# Curve 2: C(A, Bu)
ax.plot(A_vals, Cu_vals, color='red', label=r"$C(A,B_{\mathrm{u}}(A))$")

# Curve 3: C(A, Bl)
ax.plot(A_vals, Cl_vals, color='blue', label=r"$C(A,B_{\mathrm{l}}(A))$")

# Fill regions
ax.fill_between(A_vals, Cu_vals, 1, where=(Cu_vals < 1), color='lightcoral', alpha=0.5)
ax.fill_between(A_vals, Cl_vals, Cu_vals, where=(Cl_vals < Cu_vals), color='lightgreen', alpha=0.5)
ax.fill_between(A_vals, 0, Cl_vals, where=(Cl_vals > 0), color='lightblue', alpha=0.5)

# Labels and legend
ax.set_xlabel(r"$A$")
ax.set_ylabel(r"$C$")
ax.legend(loc='lower right')
ax.grid(True)

plt.tight_layout()
plt.show()
