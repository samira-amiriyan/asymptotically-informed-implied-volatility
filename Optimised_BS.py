import numpy as np
from scipy.stats import norm
from scipy.special import factorial, comb, erfc, erfcx

from typing import Union


# --- GLOBAL CONSTANTS ---
# Controls the precision of approximations. Higher values increase accuracy but cost more compute.
N_DEFAULT = 16      # Order for the Power Series expansion
N_1_DEFAULT = 17    # Primary order for the Asymptotic expansion
N_2_DEFAULT = 20    # Secondary order for the Asymptotic expansion

# Thresholds that define the "Regimes." These decide when to switch from
# standard formulas to Taylor or Asymptotic expansions to avoid numerical errors.
h_max_DEFAULT=35.0
t_max_DEFAULT=0.1
tau_DEFAULT=0.1

# Minimum value used to prevent DivisionByZero or Log(0) errors.
BS_GLOBAL_EPSILON = 1e-50

# Float64 limits for np.exp to prevent 'inf' (overflow) or 0.0 (underflow) prematurely.
SAFE_EXP_ARG_MIN = -700.0
SAFE_EXP_ARG_MAX = 700.0


# Pre-computed square root constants for cleaner formulas and faster execution.
SQRT_2 = np.sqrt(2.0, dtype=np.float64)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0, dtype=np.float64)
SQRT_2_PI = np.sqrt(2.0 * np.pi, dtype=np.float64)


def Phi(x: np.ndarray) -> np.ndarray:
    """
    Standard Gaussian Cumulative Distribution Function (CDF).
    Uses scipy.special.erfc for significantly improved numerical stability
    and accuracy, especially in the small tails (large negative x values).
    """
    x = x.astype(np.float64)
    return 0.5 * erfc(-x / SQRT_2)

# --- BOUNDARY B FUNCTIONS ---
def B_u(A: np.ndarray) -> np.ndarray:
    """
    Calculates the upper boundary B_u(A) = sqrt(2*A) + sqrt(pi/2) + sqrt(2*pi)*Phi(-sqrt(2*A))*e^A.
    """
    sqrt_2A = SQRT_2 * np.sqrt(A.astype(np.float64))

    return sqrt_2A + SQRT_PI_OVER_2 + SQRT_2_PI* Phi(-sqrt_2A) * np.exp(A.astype(np.float64))

def B_l(A: np.ndarray) -> np.ndarray:
    """
    Calculates the lower boundary B_l(A) = B_u(A) - sqrt(2*pi).
    """
    return B_u(A) - SQRT_2_PI

def calculate_Z(h: Union[float, np.ndarray], N: int) -> np.ndarray:
    """
        Generates (inductively) a sequence of values (Mills ratio and derivatives)
        Z_0, Z_1, ..., Z_N for a given input h.

        We start with Z[0] using erfcx (the scaled complementary error function),
        which is specifically designed to handle large inputs without overflowing.
    """
    h_arr = np.atleast_1d(h).astype(np.float64)
    # Z has shape (N+1, number_of_inputs)
    Z = np.zeros((N + 1, h_arr.size), dtype=np.float64)

    # Z[0](h) = sqrt(pi/2) * erfcx(h/sqrt(2))
    Z[0] = np.sqrt(np.pi / 2.0) * erfcx(h_arr / np.sqrt(2.0))

    # Higher orders are computed iteratively: Z_{n+1} = h*Z_n + n*Z_{n-1}
    if N >= 1:
        Z[1] = -1.0 + h_arr * Z[0]
        for n in range(1, N):
            Z[n + 1] = h_arr * Z[n] + n * Z[n - 1]

    return Z

def precompute_constants(N_1: int, N_2: int):
    """
    Precompute the coefficients used in the asymptotic expansion.

    These coefficients depend only on the truncation orders and can
    therefore be generated once and reused throughout the program.

    The coefficients are
        c_{k,n}
        = (-1)^n (2n-1)!! binom(2n+k+1, 2n),

    where only even values of k are used.

    Parameters
    ----------
    N_1 : int
        Maximum outer index n.
    N_2 : int
        Maximum odd power 2p+1.

    Returns
    -------
    ndarray
        Two-dimensional lookup table of coefficients.
    """
    max_n = N_1
    max_k = N_2

    # Calculate the odd double factorial (2n-1)!!
    double_factorial = np.ones(max_n + 1, dtype=np.float64)
    for n in range(1, max_n + 1):
        double_factorial[n] = double_factorial[n - 1] * (2 * n - 1)

    # Pre-compute constant_k,n for k=2p (even) and all n
    # The array is sized up to max_k (N_2) rows, but only even-indexed rows are used.
    constants = np.zeros((max_k + 1, max_n + 1), dtype=np.float64)

    for n in range(max_n + 1):
        # We loop over p_idx = 2p+1, from 1 up to N_2 (inclusive, stepping by 2).
        # This correctly restricts the calculation based on the summation bounds.
        for p_idx in range(1, max_k + 1, 2):
            k = p_idx - 1  # k is the even index (2p)

            # {2n+k+1 \choose 2n} = {2n+p_idx \choose 2n}
            # Use comb for high precision.
            comb_term = comb(2 * n + p_idx, 2 * n, exact=False)

            df = double_factorial[n]
            sign = (-1.0) ** n

            constants[k, n] = sign * df * comb_term

    return constants

# Pre-calculate the constants using the default N_1 and N_2 values
_CONSTANTS_PRECOMPUTED = precompute_constants(N_1=N_1_DEFAULT, N_2=N_2_DEFAULT)


def Delta_Z_ht(h, t, h_max=h_max_DEFAULT, t_max=t_max_DEFAULT, N=N_DEFAULT, N_1=N_1_DEFAULT,
               N_2=N_2_DEFAULT, tau=tau_DEFAULT) -> np.ndarray:
    """
    Computes the equivalent of Z(h-t) - Z(h+t) using three regime-based approximations.
    """
    h_b, t_b = np.broadcast_arrays(np.atleast_1d(h).astype(np.float64),
                                   np.atleast_1d(t).astype(np.float64))
    delta_z = np.zeros_like(h_b)

    # Logic to determine which mathematical approximation to use for each data point
    mask1 = (t_b < t_max) & (h_b < h_max)
    mask2 = (h_b >= h_max) & (t_b < tau * h_b)
    mask3 = ~(mask1 | mask2)

    # Regime 1: Power Series
    if np.any(mask1):
        h1, t1 = h_b[mask1], t_b[mask1]
        Z_vecs = calculate_Z(h1, N)
        series = np.zeros_like(h1)
        for k in range(1, N + 1, 2):
            series += Z_vecs[k] * (t1 ** k / factorial(k))
        delta_z[mask1] = -2.0 * series

    # Regime 2: Asymptotic Expansion
    if np.any(mask2):
        h2, t2 = h_b[mask2], t_b[mask2]
        asymp_sum = np.zeros_like(h2)
        t_over_h = t2 / h2
        for n in range(N_1 + 1):
            sum_p = np.zeros_like(h2)
            for p_idx in range(1, N_2 + 1, 2):
                sum_p += _CONSTANTS_PRECOMPUTED[p_idx - 1, n] * (t_over_h ** p_idx)
            asymp_sum += (1.0 / (h2 ** (2 * n + 1))) * sum_p
        delta_z[mask2] = 2.0 * asymp_sum

    # Regime 3: Direct Difference (Safe Zone)
    if np.any(mask3):
        h3, t3 = h_b[mask3], t_b[mask3]
        z_vals_low = calculate_Z(h3 - t3, 0)[0]
        z_vals_high = calculate_Z(h3 + t3, 0)[0]
        delta_z[mask3] = z_vals_low - z_vals_high
    return delta_z


def C_ht(h, t, h_max=h_max_DEFAULT, t_max=t_max_DEFAULT, N=N_DEFAULT, N_1=N_1_DEFAULT,
         N_2=N_2_DEFAULT, tau=tau_DEFAULT, epsilon=BS_GLOBAL_EPSILON) -> Union[float, np.ndarray]:
    """
        This function wraps the Delta_Z logic and applies the necessary Gaussian
        weighting factors. It handles the potential overflow of exponents by
        switching to log-space when (h-t) is large.
        """
    # Calculate Delta_Z
    dz = Delta_Z_ht(h, t, h_max, t_max, N, N_1, N_2, tau)

    h_b, t_b = np.broadcast_arrays(np.atleast_1d(h).astype(np.float64),
                                   np.atleast_1d(t).astype(np.float64))

    diff = h_b - t_b
    result = np.zeros_like(h_b)

    # High precision split: a) Direct calculation, b) Log-stable calculation
    mask_low = diff <= 35.0
    mask_high = ~mask_low

    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    exp_factor = -0.5 * (diff ** 2)

    # Standard scale for smaller inputs
    if np.any(mask_low):
        result[mask_low] = inv_sqrt_2pi * np.exp(exp_factor[mask_low]) * dz[mask_low]

    # Log-stabilized scale for large inputs to prevent 0.0 * inf results
    if np.any(mask_high):
        # Prevent log(0) using the specified epsilon
        log_dz = np.log(np.maximum(dz[mask_high], epsilon))
        result[mask_high] = inv_sqrt_2pi * np.exp(exp_factor[mask_high] + log_dz)

    # Return scalar if inputs were scalars
    if np.isscalar(h) and np.isscalar(t):
        return result.item()
    return result


def C_ht_default(h, t) -> np.ndarray:
    """
        The naive, standard closed-form for C(h, t).

        Formula: Phi(t - h) - Phi(-t - h) * exp(2 * h * t)

    Args:
        h: A non-negative scalar or numpy array of non-negative values.
        t: A non-negative scalar or numpy array of non-negative values.

    Returns:
        A numpy array containing the computed C_ht values.
    """
    # Use float64 for all calculations
    h_arr = np.atleast_1d(h).astype(np.float64)
    t_arr = np.atleast_1d(t).astype(np.float64)

    # Ensure h and t are broadcastable
    try:
        h_b, t_b = np.broadcast_arrays(h_arr, t_arr)
    except ValueError as e:
        raise ValueError(f"Input arrays h and t must be broadcastable: {e}")

    # Calculate the terms:
    # exp(2*h*t): The exponent is calculated first to avoid intermediate underflow issues
    exp_arg = 2.0 * h_b * t_b
    # Clip the argument to prevent overflow and stabilize multiplication ---
    safe_exp_arg = np.clip(exp_arg, None, SAFE_EXP_ARG_MAX)
    exp_term = np.exp(safe_exp_arg)

    # Compute using standard scipy Normal CDF
    result = norm.cdf(-h_b + t_b) - norm.cdf(-h_b - t_b) * exp_term

    # If the original input h/t was a scalar, return a scalar
    if np.isscalar(h) and np.isscalar(t):
        return result.item()
    else:
        # Return the array
        return result


def calculate_Cu_Cl(A: np.ndarray, mode: str = "advanced") -> tuple[np.ndarray, np.ndarray]:
    """
        Compute the critical boundary values Cu and Cl.

        Args:
            A: The input parameter array.
            mode: "advanced" uses the stabilized C_ht.
                  "default" uses the simple BS formula.
    """
    A_final = A.astype(np.float64)

    # Calculate B_u and B_l for A_final
    B_u_final = B_u(A_final)
    B_l_final = B_l(A_final)

    # Map Boundaries to h and t parameters for Cu
    h_u = A_final / B_u_final
    t_u = B_u_final / 2.0

    C_ht_u = C_ht(h_u, t_u) if mode == "advanced" else C_ht_default(h_u, t_u)

    # Map Boundaries to h and t parameters for Cl
    h_l = A_final / np.maximum(B_l_final, BS_GLOBAL_EPSILON)
    t_l = B_l_final / 2.0

    C_ht_l = C_ht(h_l, t_l) if mode == "advanced" else C_ht_default(h_l, t_l)

    return C_ht_u, C_ht_l