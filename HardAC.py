import torch
import torch.nn as nn

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

class HardACInvGenInter(nn.Module):
    """
    Asymptotic neural network architecture using boundary terms (z_u, z_l)
    for the switching functions f0 and f1.

    Input: (A, C, y, log_A, log_y, z_u, z_l) -> uses all 7 features
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        super().__init__()
        self.N_f = N_f
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(z_u) Free Parameters ---
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(z_l) Free Parameters ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network (Input: A, C) ---
        self.g_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        A = x[:, 0].unsqueeze(-1)
        C = x[:, 1].unsqueeze(-1)
        z_u = x[:, 5].unsqueeze(-1)  # z_u (for f1 input)
        z_l = x[:, 6].unsqueeze(-1)  # z_l (for f0 input)

        # --- 1. Compute f1(z_u) ---
        # f1 = 1/(1+sum{i=1}^N_f aa_{1,i}*(z_u)^{dd_{1,i}})
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # Use epsilon for numerical stability when raising z_u to a power, as z_u can be zero.
        z_u_pow_dd1 = torch.pow(z_u, dd1)
        f1_sum = torch.sum(aa1 * z_u_pow_dd1, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(z_l) ---
        # f0 = 1/(1+sum{i=1}^N_f aa_{0,i}*(z_l)^{dd_{0,i}})
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = torch.exp(self.d0).unsqueeze(0)  # dd0 are used as exponents (positive)

        # Use epsilon for numerical stability when raising z_l to a power, as z_l can be zero.
        z_l_pow_dd0 = torch.pow(z_l, dd0)
        f0_sum = torch.sum(aa0 * z_l_pow_dd0, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)
        g1 = g[:, 1].unsqueeze(-1)
        g2 = g[:, 2].unsqueeze(-1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2
        return B_pred


class HardACInvExpInter(nn.Module):
    """
    Asymptotic neural network architecture using boundary terms (z_u, z_l)
    for the switching functions f0 and f1.

    Input: (A, C, y, log_A, log_y, z_u, z_l) -> uses all 7 features
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        super().__init__()
        self.N_f = N_f
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(z_u) Free Parameters ---
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(z_l) Free Parameters ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network (Input: A, C) ---
        self.g_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        A = x[:, 0].unsqueeze(-1)
        C = x[:, 1].unsqueeze(-1)
        z_u = x[:, 5].unsqueeze(-1)  # z_u (for f1 input)
        z_l = x[:, 6].unsqueeze(-1)  # z_l (for f0 input)

        # --- 1. Compute f1(z_u) ---
        # f1 = 1/(1+sum{i=1}^N_f aa_{1,i}*(z_u)^{dd_{1,i}})
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # Use epsilon for numerical stability when raising z_u to a power, as z_u can be zero.
        z_u_pow_dd1 = torch.pow(z_u, dd1)
        f1_sum = torch.sum(aa1 * z_u_pow_dd1, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(z_l) ---
        # f0 = 1/(1+sum{i=1}^N_f aa_{0,i}*(z_l)^{dd_{0,i}})
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = torch.exp(self.d0).unsqueeze(0)  # dd0 are used as exponents (positive)

        # Use epsilon for numerical stability when raising z_l to a power, as z_l can be zero.
        z_l_pow_dd0 = torch.pow(z_l, dd0)
        f0_sum = torch.sum(aa0 * z_l_pow_dd0, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)
        g_linear_out = self.g_net(A_C)
        g = torch.exp(g_linear_out)  # Exponentiate the result to ensure positive values (g0, g1, g2)
        g0 = g[:, 0].unsqueeze(-1)
        g1 = g[:, 1].unsqueeze(-1)
        g2 = g[:, 2].unsqueeze(-1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2
        return B_pred


class HardACInvExpFree(nn.Module):
    """
    Asymptotic neural network architecture using boundary terms (z_u, z_l)
    for the switching functions f0 and f1.

    Input: (A, C, y, log_A, log_y, z_u, z_l) -> uses all 7 features
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        super().__init__()
        self.N_f = N_f
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(z_u) Free Parameters ---
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(z_l) Free Parameters ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network (Input: A, C) ---
        self.g_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        A = x[:, 0].unsqueeze(-1)
        C = x[:, 1].unsqueeze(-1)
        z_u = x[:, 5].unsqueeze(-1)  # z_u (for f1 input)
        z_l = x[:, 6].unsqueeze(-1)  # z_l (for f0 input)

        # --- 1. Compute f1(z_u) ---
        # f1 = 1/(1+sum{i=1}^N_f aa_{1,i}*(z_u)^{dd_{1,i}})
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # Use epsilon for numerical stability when raising z_u to a power, as z_u can be zero.
        z_u_pow_dd1 = torch.pow(z_u, dd1)
        f1_sum = torch.sum(aa1 * z_u_pow_dd1, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(z_l) ---
        # f0 = 1/(1+sum{i=1}^N_f aa_{0,i}*(z_l)^{dd_{0,i}})
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = torch.exp(self.d0).unsqueeze(0)  # dd0 are used as exponents (positive)

        # Use epsilon for numerical stability when raising z_l to a power, as z_l can be zero.
        z_l_pow_dd0 = torch.pow(z_l , dd0)
        f0_sum = torch.sum(aa0 * z_l_pow_dd0, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)
        g_linear_out = self.g_net(A_C)
        g = torch.exp(g_linear_out)  # Exponentiate the result to ensure positive values (g0, g1, g2)
        g0 = g[:, 0].unsqueeze(-1)
        g1 = g[:, 1].unsqueeze(-1)
        g2 = g[:, 2].unsqueeze(-1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 +  g2
        return B_pred


class HardACInvGenFree(nn.Module):
    """
    Asymptotic neural network architecture using boundary terms (z_u, z_l)
    for the switching functions f0 and f1.

    Input: (A, C, y, log_A, log_y, z_u, z_l) -> uses all 7 features
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        super().__init__()
        self.N_f = N_f
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(z_u) Free Parameters ---
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(z_l) Free Parameters ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network (Input: A, C) ---
        self.g_net = nn.Sequential(
            nn.Linear(2, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, N_g, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N_g, 3, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (A, C, y, log_A, log_y, z_u, z_l)

        A = x[:, 0].unsqueeze(-1)
        C = x[:, 1].unsqueeze(-1)
        z_u = x[:, 5].unsqueeze(-1)  # z_u (for f1 input)
        z_l = x[:, 6].unsqueeze(-1)  # z_l (for f0 input)

        # --- 1. Compute f1(z_u) ---
        # f1 = 1/(1+sum{i=1}^N_f aa_{1,i}*(z_u)^{dd_{1,i}})
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # Use epsilon for numerical stability when raising z_u to a power, as z_u can be zero.
        z_u_pow_dd1 = torch.pow(z_u , dd1)
        f1_sum = torch.sum(aa1 * z_u_pow_dd1, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(z_l) ---
        # f0 = 1/(1+sum{i=1}^N_f aa_{0,i}*(z_l)^{dd_{0,i}})
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = torch.exp(self.d0).unsqueeze(0)  # dd0 are used as exponents (positive)

        # Use epsilon for numerical stability when raising z_l to a power, as z_l can be zero.
        z_l_pow_dd0 = torch.pow(z_l , dd0)
        f0_sum = torch.sum(aa0 * z_l_pow_dd0, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)
        g1 = g[:, 1].unsqueeze(-1)
        g2 = g[:, 2].unsqueeze(-1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2
        return B_pred