import torch
import torch.nn as nn

# --- PREAMBLE & DEVICE CONFIGURATION ---
#if torch.cuda.is_available():
 #   device = torch.device("cuda")
  #  DTYPE = torch.float32  # Fast execution on NVIDIA
#else:
 #   device = torch.device("cpu")
  #  DTYPE = torch.float64  # Maximum fallback precision on CPU

device = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# --- GLOBAL NUMERICAL STABILITY CONSTANT ---
# This value is used by default to prevent division by zero or
# issues with log operations on zero or near-zero inputs.
NN_GLOBAL_EPSILON = 1e-18


class GaussACInvExpInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + cc_{0,i})^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), cc0=exp(c0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + cc0)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + cc0, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussACInvGenInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + cc_{0,i})^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = MLP(A, C)
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), cc0=exp(c0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + cc0)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + cc0, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussACInvExpFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + cc_{0,i})^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), cc0=exp(c0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + cc0)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + cc0, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 +  g2

        return B_pred


class GaussACInvGenFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + cc_{0,i})^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = MLP(A, C)
    - Output B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), cc0=exp(c0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + cc0)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + cc0, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 +  g2

        return B_pred


class GaussACepsInvExpInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + epsilon)^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + epsilon)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + self.epsilon_val, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussACepsInvExpFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + epsilon)^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + epsilon)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + self.epsilon_val, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred


class GaussACepsInvGenInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + epsilon)^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = MLP(A, C)
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + epsilon)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + self.epsilon_val, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussACepsInvGenFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i} * (A + cc_{1,i})^{bb_{1,i}} * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * (A + epsilon)^{bb_{0,i}} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), bb1=exp(b1), cc1=exp(c1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * (A + cc1)**bb1 * y**dd1))
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        # Base (A + cc1) is guaranteed positive because cc1=exp(c1)
        term_f1 = aa1 * torch.pow(A + cc1, bb1) * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0), bb0=-exp(b0), dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * (A + epsilon)**bb0 * y**dd0))
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(A + self.epsilon_val, bb0) * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred


class GaussCInvExpInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i}  * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i}  * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * y**dd1))
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        term_f1 = aa1 * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0),  , dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * y**dd0))
        aa0 = torch.exp(self.a0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussCInvExpFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i}  * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i} * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = exp(MLP(A, C))
    - Output B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * y**dd1))
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        term_f1 = aa1 * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0),  , dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * y**dd0))
        aa0 = torch.exp(self.a0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g_linear = self.g_net(A_C)
        g = torch.exp(g_linear)  # Ensure positivity

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred


class GaussCInvGenInter(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i}  * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i}  * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = MLP(A, C)
    - Output B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * y**dd1))
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        term_f1 = aa1 * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0),  , dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * y**dd0))
        aa0 = torch.exp(self.a0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred


class GaussCInvGenFree(nn.Module):
    """
    Gaussian-based Asymptotic Neural Network Architecture.

    This model implements the map: (A, C, y, log_A, log_y, z_u, z_l) -> B

    Structure:
    - f1(A, y) = exp(- sum_{i=1}^{N_f} aa_{1,i}  * y^{dd_{1,i}})
    - f0(A, y) = exp(- sum_{i=1}^{N_f} aa_{0,i}  * y^{dd_{0,i}})
    - (g0, g1, g2)(A, C) = MLP(A, C)
    - Output B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the GaussACInvExpInter model with float64 precision.

        Args:
            N_f (int): Number of terms in the exponential summations for f0 and f1.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability.
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Using float64 is critical for precision and cross-platform consistency (M1/Intel/Windows)
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Parameters (Summation inside exp) ---
        # Learned as free parameters, transformed to required signs via exp in forward pass
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- f0(A, y) Parameters (Summation inside exp) ---
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.01)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # 3 Hidden layers with ReLU, followed by a linear layer and exponentiation
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
        Performs the forward pass with high precision (float64).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 7)
                             containing (A, C, y, log_A, log_y, z_u, z_l).

        Returns:
            torch.Tensor: Predicted B value of shape (batch_size, 1).
        """

        # Extract features
        A = x[:, 0].unsqueeze(-1)  # (batch_size, 1)
        C = x[:, 1].unsqueeze(-1)  # (batch_size, 1)
        y = x[:, 2].unsqueeze(-1)  # (batch_size, 1)

        # --- 1. Compute f1(A, y) ---
        # aa1=exp(a1), dd1=exp(d1)
        # Resulting f1 = exp(- sum(aa1 * y**dd1))
        aa1 = torch.exp(self.a1)
        dd1 = torch.exp(self.d1)

        # Power operations with epsilon for y to prevent log(0) issues in gradients
        term_f1 = aa1 * torch.pow(y + self.epsilon_val, dd1)
        f1 = torch.exp(-torch.sum(term_f1, dim=1, keepdim=True))

        # --- 2. Compute f0(A, y) ---
        # aa0=exp(a0),  , dd0=-exp(d0)
        # Resulting f0 = exp(- sum(aa0 * y**dd0))
        aa0 = torch.exp(self.a0)
        dd0 = -torch.exp(self.d0)

        # Power operations with epsilon for y since dd0 is negative
        term_f0 = aa0 * torch.pow(y + self.epsilon_val, dd0)
        f0 = torch.exp(-torch.sum(term_f0, dim=1, keepdim=True))

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        A_C = torch.cat([A, C], dim=1)  # (batch_size, 2)
        g = self.g_net(A_C)

        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        g2 = g[:, 2:3]

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 +  g2

        return B_pred