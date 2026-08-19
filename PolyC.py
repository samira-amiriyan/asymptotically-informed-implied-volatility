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
torch.set_default_dtype(DTYPE)

class PolyCInvExpInter(nn.Module):
    """
    Asymptotic neural network architecture:
    It combines non-linear C polynomial-exponential terms (f0, f1) with an exponentiated standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y) (others ignored)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCInvExpInter model.

        Args:
            N_f (int): Number of terms in the sum for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
            This is used when computing (1/c-1 + esp)^{-d} in f_0
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i})
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i})
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        y = x[:, 2].unsqueeze(-1)  # y (for f-net input)

        # --- 1. Compute f1(y) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i}) (positive exponents)
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # y_col shape (B, 1). Using torch.pow(B, 1)**(1, N_f) -> (B, N_f)
        y_pow_dd1 = torch.pow(y, dd1)

        # f1_terms shape (B, N_f)
        f1_terms = aa1 * y_pow_dd1

        # Sum over the N_f dimension to get (B, 1)
        f1_sum = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(y) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i}) (negative exponents)
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = -torch.exp(self.d0).unsqueeze(0)  # Shape (1, N_f)

        # Using epsilon_val is safer for numerically sensitive power calculations with negative exponents
        # though the synthetic data ensures y > 0.
        y_pow_dd0 = torch.pow(y + self.epsilon_val, dd0)

        # f0_terms shape (B, N_f)
        f0_terms = aa0 * y_pow_dd0

        # Sum over the N_f dimension to get (B, 1)
        f0_sum = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyCInvExpFree(nn.Module):
    """
    Architecture using Inverse Exponential terms (1 / (1 + sum(aa * y**dd)))
    for f0 and f1, which depend only on y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCInvExpFree model.

        Args:
            N_f (int): Number of terms in the sum for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
                This is used when computing (1/c-1 + esp)^{-d} in f_0
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i})
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i})
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        y = x[:, 2].unsqueeze(-1)  # y (for f-net input)

        # --- 1. Compute f1(y) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i}) (positive exponents)
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # y_col shape (B, 1). Using torch.pow(B, 1)**(1, N_f) -> (B, N_f)
        y_pow_dd1 = torch.pow(y, dd1)

        # f1_terms shape (B, N_f)
        f1_terms = aa1 * y_pow_dd1

        # Sum over the N_f dimension to get (B, 1)
        f1_sum = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(y) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i}) (negative exponents)
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = -torch.exp(self.d0).unsqueeze(0)  # Shape (1, N_f)

        # Using epsilon_val is safer for numerically sensitive power calculations with negative exponents
        # though the synthetic data ensures y > 0.
        y_pow_dd0 = torch.pow(y + self.epsilon_val, dd0)

        # f0_terms shape (B, N_f)
        f0_terms = aa0 * y_pow_dd0

        # Sum over the N_f dimension to get (B, 1)
        f0_sum = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred


class PolyCInvGenInter(nn.Module):
    """
    Architecture using Inverse Exponential terms (1 / (1 + sum(aa * y**dd)))
    for f0 and f1, which depend only on y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCInvExpInter model.

        Args:
            N_f (int): Number of terms in the sum for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i})
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i})
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        y = x[:, 2].unsqueeze(-1)  # y (for f-net input)

        # --- 1. Compute f1(y) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i}) (positive exponents)
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # y_col shape (B, 1). Using torch.pow(B, 1)**(1, N_f) -> (B, N_f)
        y_pow_dd1 = torch.pow(y, dd1)

        # f1_terms shape (B, N_f)
        f1_terms = aa1 * y_pow_dd1

        # Sum over the N_f dimension to get (B, 1)
        f1_sum = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(y) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i}) (negative exponents)
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = -torch.exp(self.d0).unsqueeze(0)  # Shape (1, N_f)

        # Using epsilon_val is safer for numerically sensitive power calculations with negative exponents
        # though the synthetic data ensures y > 0.
        y_pow_dd0 = torch.pow(y + self.epsilon_val, dd0)

        # f0_terms shape (B, N_f)
        f0_terms = aa0 * y_pow_dd0

        # Sum over the N_f dimension to get (B, 1)
        f0_sum = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyCInvGenFree(nn.Module):
    """
    Architecture using Inverse Exponential terms (1 / (1 + sum(aa * y**dd)))
    for f0 and f1, which depend only on y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCInvExpInter model.

        Args:
            N_f (int): Number of terms in the sum for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i})
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(y) Free Parameters (N_f terms, 2*N_f total) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i})
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        y = x[:, 2].unsqueeze(-1)  # y (for f-net input)

        # --- 1. Compute f1(y) ---
        # aa_{1,i} = exp(a_{1,i}), dd_{1,i} = exp(d_{1,i}) (positive exponents)
        aa1 = torch.exp(self.a1).unsqueeze(0)  # Shape (1, N_f)
        dd1 = torch.exp(self.d1).unsqueeze(0)  # Shape (1, N_f)

        # y_col shape (B, 1). Using torch.pow(B, 1)**(1, N_f) -> (B, N_f)
        y_pow_dd1 = torch.pow(y, dd1)

        # f1_terms shape (B, N_f)
        f1_terms = aa1 * y_pow_dd1

        # Sum over the N_f dimension to get (B, 1)
        f1_sum = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + f1_sum)  # Shape (B, 1)

        # --- 2. Compute f0(y) ---
        # aa_{0,i} = exp(a_{0,i}), dd_{0,i} = -exp(d_{0,i}) (negative exponents)
        aa0 = torch.exp(self.a0).unsqueeze(0)  # Shape (1, N_f)
        dd0 = -torch.exp(self.d0).unsqueeze(0)  # Shape (1, N_f)

        # Using epsilon_val is safer for numerically sensitive power calculations with negative exponents
        # though the synthetic data ensures y > 0.
        y_pow_dd0 = torch.pow(y + self.epsilon_val, dd0)

        # f0_terms shape (B, N_f)
        f0_terms = aa0 * y_pow_dd0

        # Sum over the N_f dimension to get (B, 1)
        f0_sum = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + f0_sum)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred


class PolyCSigExpInter(nn.Module):
    """
    Architecture using Sigmoid activation on log_y terms
    for f0 and f1, which depend only on log_y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_y) Free Parameters ---
        # t1 = a1 - dd1 * log_y, where dd1 = exp(d1)
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- f0(log_y) Free Parameters ---
        # t0 = a0 + dd0 * log_y, where dd0 = exp(d0)
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_y) ---
        # dd1 = exp(d1)
        dd1 = torch.exp(self.d1)  # Shape (1, 1)
        # t1 = a1 - dd1 * log_y
        t1 = self.a1 - dd1 * log_y
        # f1 = sigmoid(t1)
        f1 = torch.sigmoid(t1)  # Shape (B, 1)

        # --- 2. Compute f0(log_y) ---
        # dd0 = exp(d0)
        dd0 = torch.exp(self.d0)  # Shape (1, 1)
        # t0 = a0 + dd0 * log_y
        t0 = self.a0 + dd0 * log_y
        # f0 = sigmoid(t0)
        f0 = torch.sigmoid(t0)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyCSigExpFree(nn.Module):
    """
    Architecture using Sigmoid activation on log_y terms
    for f0 and f1, which depend only on log_y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_y) Free Parameters ---
        # t1 = a1 - dd1 * log_y, where dd1 = exp(d1)
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- f0(log_y) Free Parameters ---
        # t0 = a0 + dd0 * log_y, where dd0 = exp(d0)
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_y) ---
        # dd1 = exp(d1)
        dd1 = torch.exp(self.d1)  # Shape (1, 1)
        # t1 = a1 - dd1 * log_y
        t1 = self.a1 - dd1 * log_y
        # f1 = sigmoid(t1)
        f1 = torch.sigmoid(t1)  # Shape (B, 1)

        # --- 2. Compute f0(log_y) ---
        # dd0 = exp(d0)
        dd0 = torch.exp(self.d0)  # Shape (1, 1)
        # t0 = a0 + dd0 * log_y
        t0 = self.a0 + dd0 * log_y
        # f0 = sigmoid(t0)
        f0 = torch.sigmoid(t0)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred


class PolyCSigGenInter(nn.Module):
    """
    Architecture using Sigmoid activation on log_y terms
    for f0 and f1, which depend only on log_y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_y) Free Parameters ---
        # t1 = a1 - dd1 * log_y, where dd1 = exp(d1)
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- f0(log_y) Free Parameters ---
        # t0 = a0 + dd0 * log_y, where dd0 = exp(d0)
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_y) ---
        # dd1 = exp(d1)
        dd1 = torch.exp(self.d1)  # Shape (1, 1)
        # t1 = a1 - dd1 * log_y
        t1 = self.a1 - dd1 * log_y
        # f1 = sigmoid(t1)
        f1 = torch.sigmoid(t1)  # Shape (B, 1)

        # --- 2. Compute f0(log_y) ---
        # dd0 = exp(d0)
        dd0 = torch.exp(self.d0)  # Shape (1, 1)
        # t0 = a0 + dd0 * log_y
        t0 = self.a0 + dd0 * log_y
        # f0 = sigmoid(t0)
        f0 = torch.sigmoid(t0)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class PolyCSigGenFree(nn.Module):
    """
    Architecture using Sigmoid activation on log_y terms
    for f0 and f1, which depend only on log_y.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyCSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_y) Free Parameters ---
        # t1 = a1 - dd1 * log_y, where dd1 = exp(d1)
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- f0(log_y) Free Parameters ---
        # t0 = a0 + dd0 * log_y, where dd0 = exp(d0)
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
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
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs, unsqueeze(-1) for broadcasting
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_y) ---
        # dd1 = exp(d1)
        dd1 = torch.exp(self.d1)  # Shape (1, 1)
        # t1 = a1 - dd1 * log_y
        t1 = self.a1 - dd1 * log_y
        # f1 = sigmoid(t1)
        f1 = torch.sigmoid(t1)  # Shape (B, 1)

        # --- 2. Compute f0(log_y) ---
        # dd0 = exp(d0)
        dd0 = torch.exp(self.d0)  # Shape (1, 1)
        # t0 = a0 + dd0 * log_y
        t0 = self.a0 + dd0 * log_y
        # f0 = sigmoid(t0)
        f0 = torch.sigmoid(t0)  # Shape (B, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (B, 2)

        # Get the linear output (B, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (B, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (B, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (B, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred