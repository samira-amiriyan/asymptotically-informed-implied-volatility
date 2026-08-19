import torch
import torch.nn as nn
from torch.special import ndtri
import numpy as np
from typing import Tuple


# --- PREAMBLE & DEVICE CONFIGURATION & GLOBAL NUMERICAL STABILITY CONSTANT---
# The value of EPSILON is used by default to prevent division by zero or
# issues with log operations on zero or near-zero inputs.
#if torch.cuda.is_available():
 #   device = torch.device("cuda")
  #  DTYPE = torch.float32  # Fast execution on NVIDIA
   # NN_GLOBAL_EPSILON = 1e-7
#else:
 #   device = torch.device("cpu")
  #  DTYPE = torch.float64  # Maximum fallback precision on CPU
   # NN_GLOBAL_EPSILON = 1e-18

device = torch.device("cpu")
DTYPE = torch.float64
NN_GLOBAL_EPSILON = 1e-18


def PhiInverse(p: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the Standard Gaussian CDF (Phi^-1).
    Uses torch.erfinv for high precision.
    """

    # Dynamic upper bound configuration matching structural floating point precision limits
    max_bound = torch.tensor(1.0, dtype=p.dtype, device=p.device) - epsilon
    p_clipped = torch.clamp(p, min=epsilon, max=max_bound)

    erfinv_arg = 2.0 * p_clipped - 1.0
    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=p.dtype, device=p.device))
    return sqrt_2 * torch.erfinv(erfinv_arg)


class PolyCInvAsptDefInter(nn.Module):
    """
    Asymptotic neural network architecture: PolyCInvAsptDefInter.
    It combines learned switching functions (f0, f1) and learned inner function (g2)
    with explicitly defined asymptotic functions (g0, g1) derived from theory.

    Input: (A, C, y, log_A, log_y, z_u, z_l) -> uses all 7 features
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Store epsilon as a non-trainable parameter for use in forward pass
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Free Parameters (positive exponents) ---
        # aa_1=exp(a_1), dd_1=exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (negative exponents) ---
        # aa_0=exp(a_0), dd_0=-exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- g2(A, C) MLP Network (Input: A, C) ---
        # The output layer is linear, as specified by the user.
        self.g2_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 1, dtype=DTYPE)
        )

    def _calculate_g1(self, A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates the upper asymptotic term g1(A,C) using the formula:
        g1(A,C) = -2 * Phi^-1( (1-C)/2 * exp(-A/2) )
        """
        p = (1.0 - C) / 2.0 * torch.exp(-A / 2.0)

        # Pass 'p' directly to PhiInverse, which handles internal clipping.
        return -2.0 * PhiInverse(p, epsilon=self.epsilon_val)

    def _calculate_g0(self, A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        term_base = C / torch.clamp(2.0 * torch.pi * A, min=self.epsilon_val)
        base_clipped = torch.clamp(term_base, min=self.epsilon_val)
        inner_term = torch.pow(base_clipped, 1.0 / 3.0)
        exp_term = torch.exp(-A / 6.0)

        sqrt_3 = torch.sqrt(torch.tensor(3.0, dtype=A.dtype, device=A.device))
        p = sqrt_3 * exp_term * inner_term

        # Target boundary for structural positivity criteria
        max_bound = torch.tensor(0.5, dtype=A.dtype, device=A.device) - self.epsilon_val
        p_safe = torch.clamp(p, min=self.epsilon_val, max=max_bound)

        phi_inv = PhiInverse(p_safe, epsilon=self.epsilon_val)
        phi_inv_safe = torch.clamp(phi_inv.abs(), min=self.epsilon_val) * phi_inv.sign()

        return -(A / sqrt_3) / phi_inv_safe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        # Extract required inputs
        # Native slicing grabs a contiguous 2D block with zero memory duplication overhead
        A_C = x[:, 0:2]  # Natively produces the perfect shape: (batch_size, 2)

        # If you still need individual A, C, and y column vectors further down for your calculations:
        A_col = A_C[:, 0:1]  # Shape: (batch_size, 1)
        C_col = A_C[:, 1:2]  # Shape: (batch_size, 1)
        y_col = x[:, 2:3]  # Shape: (batch_size, 1) (Cleaner replacement for x[:, 2].unsqueeze(-1))

        # --- 1. Compute f1(A, y) ---
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)  # positive exponent

        term_y_f1 = torch.pow(y_col, dd1)

        f1_terms = aa1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        aa0 = torch.exp(self.a0)
        dd0_neg = -torch.exp(self.d0)  # negative exponent

        # Use epsilon for stability when y is near zero
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0_neg)

        f0_terms = aa0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute g2(A, C) (Learned) ---
        # The output is linear, as specified, without exponentiation
        g2 = self.g2_net(A_C)

        # --- 4. Compute g1(A, C) and g0(A, C) (Defined) ---
        g1 = self._calculate_g1(A_col, C_col)
        g0 = self._calculate_g0(A_col, C_col)

        # --- 5. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyCInvAsptAdvInter(nn.Module):
    """
    Advanced Asymptotic neural network architecture: PolyCInvAsptAdvInter.

    Difference from DefInter:
    - The MLP outputs a 3D vector: (g00, g11, g2).
    - g1 and g0 are computed using learned parameters g11 and g00 inside
      theoretical asymptotic structures.
    - Constraints are applied to Phi^-1 arguments to ensure positive volatility outcomes.
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = 1e-18):
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters ---
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Parameters ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- MLP Network (Input: A, C) -> Output: (g00, g11, g2) ---
        self.mlp_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)  # 3-dim output
        )

    def _calculate_asymptotics(self, A: torch.Tensor, g00: torch.Tensor, g11: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Calculates g1 and g0 using the learned g11 and g00.
        Applies constraints to guarantee positivity.
        """
        max_bound = torch.tensor(0.5, dtype=A.dtype, device=A.device) - self.epsilon_val

        # g1 calculation
        p1 = g11 * torch.exp(-A / 2.0)
        p1_safe = torch.clamp(p1, min=self.epsilon_val, max=max_bound)
        g1 = -2.0 * PhiInverse(p1_safe, epsilon=self.epsilon_val)

        # g0 calculation
        denom_g0 = torch.clamp(2.0 * torch.pi * A, min=self.epsilon_val)
        g00_safe = torch.clamp(g00, min=self.epsilon_val)

        inner_term = torch.pow(g00_safe / denom_g0, 1.0 / 3.0)
        exp_term = torch.exp(-A / 6.0)

        sqrt_3 = torch.sqrt(torch.tensor(3.0, dtype=A.dtype, device=A.device))
        p0 = sqrt_3 * exp_term * inner_term

        p0_safe = torch.clamp(p0, min=self.epsilon_val, max=max_bound)
        phi_inv0 = PhiInverse(p0_safe, epsilon=self.epsilon_val)
        g0 = -(A / sqrt_3) / phi_inv0

        return g0, g1

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Extraction
        # Native slicing grabs a contiguous 2D block with zero memory duplication overhead
        A_C = x[:, 0:2]  # Natively produces the perfect shape: (batch_size, 2)

        # If you still need individual A, C, and y column vectors further down for your calculations:
        A = A_C[:, 0:1]  # Shape: (batch_size, 1)
        y = x[:, 2:3]  # Shape: (batch_size, 1) (Cleaner replacement for x[:, 2].unsqueeze(-1))

        # 1. Compute Switching Functions f1 and f0
        # f1
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)
        sum_f1 = torch.sum(aa1 * torch.pow(y, dd1), dim=1, keepdim=True)
        f1 = 1.0 / (1.0 + sum_f1)

        # f0
        aa0 = torch.exp(self.a0)
        dd0_neg = -torch.exp(self.d0)
        sum_f0 = torch.sum(aa0 * torch.pow(y + self.epsilon_val, dd0_neg), dim=1, keepdim=True)
        f0 = 1.0 / (1.0 + sum_f0)

        # 2. Forward pass through MLP for learned components
        mlp_out = self.mlp_net(A_C)
        g00 = mlp_out[:, 0:1]
        g11 = mlp_out[:, 1:2]
        g2 = mlp_out[:, 2:3]

        # 3. Compute structural asymptotics
        g0, g1 = self._calculate_asymptotics(A, g00, g11)

        # 4. Final Combination
        # B = f0*g0 + f1*g1 + (1 - f0 - f1)*g2
        B_pred = f0 * g0 + f1 * g1 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyACInvAsptAdvInter(nn.Module):
    """
    Asymptotic neural network architecture: PolyACInvAsptAdvInter.
    It combines non-linear polynomial-exponential terms (f0, f1) with an MLP-based
    component (g0, g1, g2) to form asymptotic terms (k0, k1).

    The model learns the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    B = f1 * k1 + f0 * k0 + (1 - f0 - f1) * g2

    where k1 and k0 involve the inverse Gaussian CDF (ndtri).
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvAsptAdvInter model.

        Args:
            N_f (int): Number of terms in the f0 and f1 summations.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g

        # Store epsilon as a non-trainable parameter for use in forward pass
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # Precompute constants needed for k0 calculation
        self.sqrt_3 = torch.tensor(np.sqrt(3.0), dtype=DTYPE)
        self.two_pi = torch.tensor(2.0 * np.pi, dtype=DTYPE)

        # --- f1(A, y) Free Parameters (N_f terms) ---
        # aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms) ---
        # aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Output: 3 dimensions (Linear output is activated in forward pass)
        self.g_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)  # Linear output: g00, g10, g20
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                              containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs (only A, C, y are used in the core calculation)
        # Native slicing grabs a contiguous 2D block with zero memory duplication overhead
        A_C = x[:, 0:2]  # Natively produces the perfect shape: (batch_size, 2)

        # If you still need individual A, C, and y column vectors further down for your calculations:
        A = A_C[:, 0:1]  # Shape: (batch_size, 1)
        y = x[:, 2:3]  # Shape: (batch_size, 1) (Cleaner replacement for x[:, 2].unsqueeze(-1))

        # Robust clipping boundaries for ndtri (Phi^-1) input
        max_bound = torch.tensor(1.0, dtype=x.dtype, device=x.device) - self.epsilon_val

        # --- 1. Compute f1(A, y) ---
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        term_A_f1 = torch.pow(A + cc1, bb1)
        term_y_f1 = torch.pow(y, dd1)

        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Negative exponent
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)  # Negative exponent

        term_A_f0 = torch.pow(A + cc0, bb0)

        # Apply epsilon to y to handle y=0 for negative exponent (1/y^d)
        term_y_f0 = torch.pow(y + self.epsilon_val, dd0)

        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---

        # Get the linear output (g00, g10, g20)
        g_linear_out = self.g_net(A_C)

        # Apply the specified activations: g0 = sigmoid(g00), g1 = sigmoid(g10), g2 = g20
        g0 = torch.sigmoid(g_linear_out[:, 0].unsqueeze(-1))
        g1 = torch.sigmoid(g_linear_out[:, 1].unsqueeze(-1))
        g2 = g_linear_out[:, 2].unsqueeze(-1)

        # --- 4. Compute k1(A, C) ---
        # k1 = -2 * Phi^{-1}( g1 * e^{-A/2} )
        arg_k1_clipped = torch.clamp(g1 * torch.exp(-A / 2.0), min=self.epsilon_val, max=max_bound)
        k1 = -2.0 * ndtri(arg_k1_clipped)

        # --- 5. Compute k0(A, C) ---
        # k0 = - (A / sqrt(3)) * 1 / Phi^{-1}(sqrt(3) * e^{-A/6} * (g0 / (2*pi*A))^(1/3))

        # Denominator of the cubic root: 2*pi*A. Clamp A to ensure stability.
        A_safe = torch.maximum(A, self.epsilon_val)

        # Term inside Phi^{-1}
        # (g0 / (2*pi*A))^(1/3)
        term_cubic_root = torch.pow(torch.clamp(g0, min=self.epsilon_val) / (self.two_pi * A_safe), 1.0 / 3.0)

        # Argument of Phi^{-1}: arg_k0 = sqrt(3) * e^{-A/6} * term_cubic_root
        arg_k0_clipped = torch.clamp(self.sqrt_3 * torch.exp(-A / 6.0) * term_cubic_root, min=self.epsilon_val,
                                     max=max_bound)
        k0 = (-A / self.sqrt_3) / ndtri(arg_k0_clipped)

        # --- 6. Final Output Combination ---
        # B = f1*k1 + f0*k0 + (1 - f0 - f1)*g2
        B_pred = f1 * k1 + f0 * k0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyACInvAsptDefInter(nn.Module):
    """
    Asymptotic neural network architecture
    It combines learned (A,C) polynomial switching functions (f0, f1) and learned inner function (g2)
    with explicitly defined asymptotic functions (g0, g1) derived from theory.

    Input: (A, C, y, log_A, log_y) (others ignored)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvAsptDefInter model.

        Args:
            N_f (int): Number of terms in the f0 and f1 summations.
            N_g (int): Number of neurons in the hidden layers of the g_2-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
            This is used when computing (1/c-1 + esp)^{-d} in f_0
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Store epsilon as a non-trainable parameter for use in forward pass
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Free Parameters (positive exponents) ---
        # aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (negative exponents) ---
        # aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- g2(A, C) MLP Network (Input: A, C) ---
        # The output layer is linear, as specified by the user.
        self.g2_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 1, dtype=DTYPE)
        )

    def _calculate_g1(self, A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates the upper asymptotic term g1(A,C) using the formula:
        g1(A,C) = -2 * Phi^-1( (1-C)/2 * exp(-A/2) )
        """
        p = (1.0 - C) / 2.0 * torch.exp(-A / 2.0)
        return -2.0 * PhiInverse(p, epsilon=self.epsilon_val)

    def _calculate_g0(self, A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates the lower asymptotic term g0(A,C) using the formula:
        g0(A,C) = - (A / sqrt(3)) / Phi^-1( sqrt(3) * exp(-A/6) * (C / (2*pi*A))^(1/3) )
        """
        term_base = C / (2.0 * torch.pi * A + self.epsilon_val)
        base_clipped = torch.clamp(term_base, min=self.epsilon_val)
        p = torch.sqrt(torch.tensor(3.0, dtype=A.dtype, device=A.device)) * torch.exp(-A / 6.0) * torch.pow(
            base_clipped, 1.0 / 3.0)

        phi_inv = PhiInverse(p, epsilon=self.epsilon_val)
        phi_inv_safe = phi_inv + torch.sign(phi_inv) * self.epsilon_val
        return -(A / torch.sqrt(torch.tensor(3.0, dtype=A.dtype, device=A.device))) / phi_inv_safe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        # Extract required inputs
        # Native slicing grabs a contiguous 2D block with zero memory duplication overhead
        A_C = x[:, 0:2]  # Natively produces the perfect shape: (batch_size, 2)

        # If you still need individual A, C, and y column vectors further down for your calculations:
        A_col = A_C[:, 0:1]  # Shape: (batch_size, 1)
        C_col = A_C[:, 1:2]  # Shape: (batch_size, 1)
        y_col = x[:, 2:3]  # Shape: (batch_size, 1) (Cleaner replacement for x[:, 2].unsqueeze(-1))


        # --- 1. Compute f1(A, y) ---
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)  # positive exponent
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)  # positive exponent

        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        aa0 = torch.exp(self.a0)
        bb0_neg = -torch.exp(self.b0)  # negative exponent
        cc0 = torch.exp(self.c0)
        dd0_neg = -torch.exp(self.d0)  # negative exponent

        term_A_f0 = torch.pow(A_col + cc0, bb0_neg)

        # Use epsilon for stability when y is near zero
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0_neg)

        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute g2(A, C) (Learned) ---
        # The output is linear, as specified, without exponentiation
        g2 = self.g2_net(A_C)

        # --- 4. Compute g1(A, C) and g0(A, C) (Defined) ---
        g1 = self._calculate_g1(A_col, C_col)
        g0 = self._calculate_g0(A_col, C_col)

        # --- 5. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred  # Ensure final output type
