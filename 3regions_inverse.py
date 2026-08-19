import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt, pi, exp

# Fixed parameter A
A = 3.0


# Function C(A, B) - used to generate coordinates
def C_func(A, B):
    B = np.array(B)
    # Black-Scholes terms
    term1 = norm.cdf(-A / B + B / 2)
    term2 = norm.cdf(-A / B - B / 2)
    return term1 - term2 * np.exp(A)


# Special values calculations
Bc = sqrt(2 * A)
Cc = 0.5 - norm.cdf(-sqrt(2 * A)) * exp(A)
Bu = Bc + sqrt(pi / 2) + sqrt(2 * pi) * norm.cdf(-sqrt(2 * A)) * exp(A)
Cu = C_func(A, Bu)
Bl = Bc - sqrt(pi / 2) + sqrt(2 * pi) * norm.cdf(-sqrt(2 * A)) * exp(A)
Cl = C_func(A, Bl)

# Range of B values to generate the curve (inverted)
# We go from very small B to slightly past Bu
B_curve_vals = np.linspace(0.05, Bu + 2, 1000)
C_curve_vals = C_func(A, B_curve_vals)

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

# The "Inverse" curve is simply B(C), so C is on the X-axis, B on the Y-axis
ax.plot(C_curve_vals, B_curve_vals, label=r"$B_{\mathrm{inv}}(A,C)$", color='black', linewidth=2)

# Special points coordinates: (C_val, B_val)
special_points = [
    (Cc, Bc, r"$C_{\mathrm{c}}$", r"$B_{\mathrm{c}}$"),
    (Cu, Bu, r"$C_{\mathrm{u}}$", r"$B_{\mathrm{u}}$"),
    (Cl, Bl, r"$C_{\mathrm{l}}$", r"$B_{\mathrm{l}}$")
]

# Vertical and horizontal dashed lines for special points
for C_val, B_val, C_label, B_label in special_points:
    # Vertical lines to C-axis
    ax.vlines(C_val, 0, B_val, color='gray', linestyle='--', alpha=0.7)
    # Horizontal lines to B-axis
    ax.hlines(B_val, 0, C_val, color='gray', linestyle='--', alpha=0.7)

    # Text labels
    ax.text(C_val, -0.1, C_label, ha='center', va='top', fontsize=12)
    ax.text(-0.02, B_val, B_label, va='center', ha='right', fontsize=12)

# Colored regions along the C-axis
# Region 1: [0, Cl] - Blue
ax.axvspan(0, Cl, facecolor='lightblue', alpha=0.4)
# Region 2: [Cl, Cu] - Green
ax.axvspan(Cl, Cu, facecolor='lightgreen', alpha=0.4)
# Region 3: [Cu, 1] - Red
ax.axvspan(Cu, 1, facecolor='lightcoral', alpha=0.4)

# Formatting
ax.set_xlabel(r"$C$", fontsize=14)
ax.set_ylabel(r"$B$", fontsize=14)

# Ensure the plot looks at the relevant range
ax.set_xlim(0, 1.0)
ax.set_ylim(0, max(B_curve_vals))

ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()