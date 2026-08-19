import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt, pi, exp

# Fixed parameter A
A = 3.0

# Function C(A, B)
def C(A, B):
    B = np.array(B)
    term1 = norm.cdf(-A / B + B / 2)
    term2 = norm.cdf(-A / B - B / 2)
    return term1 - term2 * np.exp(A)

# Special values
Bc = sqrt(2 * A)
Cc = 0.5 - norm.cdf(-sqrt(2 * A)) * exp(A)
Bu = Bc + sqrt(pi / 2) + sqrt(2 * pi) * norm.cdf(-sqrt(2 * A)) * exp(A)
Cu = C(A, Bu)
Bl = Bc - sqrt(pi / 2) + sqrt(2 * pi) * norm.cdf(-sqrt(2 * A)) * exp(A)
Cl = C(A, Bl)

# Tangent at Bc
def tangent(B):
    return (B - Bc) / sqrt(2 * pi) + Cc

# Range of B values
B_vals = np.linspace(0.1, Bu + 2, 500)
C_vals = C(A, B_vals)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(B_vals, C_vals, label=r"$C(A,B)$", color='blue')

# Horizontal line at C = 1
ax.axhline(1, color='black', linestyle='dotted')

# Vertical and horizontal dashed lines for Bc, Bu, Bl
special_points = [(Bc, Cc, r"$B_{\mathrm{c}}$", r"$C_{\mathrm{c}}$"),
                  (Bu, Cu, r"$B_{\mathrm{u}}$", r"$C_{\mathrm{u}}$"),
                  (Bl, Cl, r"$B_{\mathrm{l}}$", r"$C_{\mathrm{l}}$")]

for B_val, C_val, B_label, C_label in special_points:
    ax.axvline(B_val, ymax=(C_val - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]), color='gray', linestyle='--')
    ax.axhline(C_val, xmax=(B_val - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]), color='gray', linestyle='--')
    ax.text(B_val, ax.get_ylim()[0] - 0.05, B_label, ha='center', va='top', fontsize=12)
    ax.text(ax.get_xlim()[0] - 0.1, C_val, C_label, va='center', ha='right', fontsize=12)

# Plot the tangent line
B_tangent = np.linspace(Bl, Bu, 300)
C_tangent = tangent(B_tangent)
ax.plot(B_tangent, C_tangent, color='orange', linestyle='--')

# Hashing color regions
ax.axhspan(0, Cl, facecolor='lightblue', alpha=0.5)
ax.axhspan(Cl, Cu, facecolor='lightgreen', alpha=0.5)
ax.axhspan(Cu, 1, facecolor='lightcoral', alpha=0.5)


# Labels and legend
ax.set_xlabel(r"$B$")
ax.set_ylabel(r"$C(A,B)$")
ax.legend(loc='lower right')
ax.grid(True)

plt.tight_layout()
plt.show()
