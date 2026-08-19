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

class PolyACInvExpInter(nn.Module):
    """
    Asymptotic neural network architecture:
    It combines non-linear (A,C) polynomial-exponential terms (f0, f1) with a standard exponentiated MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y) (others ignored)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

        Args:
            N_f (int): Number of terms in the f0 and f1 summations.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability
            This is used when computing (1/c-1 + esp)^{-d} in f_0
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Store epsilon as a non-trainable parameter for use in forward pass
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + cc0) is always positive.
        term_A_f0 = torch.pow(A_col + cc0, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACepsInvExpInter(nn.Module):
    """
    Asymptotic neural network architecture:
    It combines non-linear (A,C) polynomial-exponential terms (f0, f1) with a standard exponentiated MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y) (others ignored)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

        Args:
            N_f (int): Number of terms in the f0 and f1 summations.
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability
            This is used when computing (1/c-1 + esp)^{-d} in f_0
        """
        super().__init__()

        self.N_f = N_f
        self.N_g = N_g
        # Store epsilon as a non-trainable parameter for use in forward pass
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0),  dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + eps) is always positive.
        term_A_f0 = torch.pow(A_col + self.epsilon_val, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACInvExpFree(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + cc0) is always positive.
        term_A_f0 = torch.pow(A_col + cc0, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred

class PolyACepsInvExpFree(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base A_col is always positive.
        term_A_f0 = torch.pow(A_col + self.epsilon_val, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred

class PolyACInvGenInter(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + cc0) is always positive.
        term_A_f0 = torch.pow(A_col + cc0, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACepsInvGenInter(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + eps) is always positive.
        term_A_f0 = torch.pow(A_col + self.epsilon_val, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACInvGenFree(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        cc0 = torch.exp(self.c0)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + cc0) is always positive.
        term_A_f0 = torch.pow(A_col + cc0, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred

class PolyACepsInvGenFree(nn.Module):
    """
    A non-standard neural network architecture for high-precision learning.
    It combines non-linear polynomial-exponential terms (f0, f1) with a standard MLP (g0, g1, g2).

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_f: int, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACInvExpInter model.

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

        # --- f1(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_1 = exp(a_1), bb_1 = exp(b_1), cc_1 = exp(c_1), dd_1 = exp(d_1)
        self.a1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.c1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d1 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- f0(A, y) Free Parameters (N_f terms, 4*N_f total) ---
        # aa_0 = exp(a_0), bb_0 = -exp(b_0), cc_0 = exp(c_0), dd_0 = -exp(d_0)
        self.a0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)
        self.d0 = nn.Parameter(torch.randn(N_f, dtype=DTYPE) * 0.1)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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
        Performs the forward pass of the network, calculating f0, f1, and g0, g1, g2.

        NOTE: This version uses torch.pow() for direct computation of the polynomial-exponential terms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract required inputs
        A = x[:, 0]  # A
        C = x[:, 1]  # C
        y = x[:, 2]  # y

        # Prepare batch dimensions for broadcast: (batch_size, 1)
        A_col = A.unsqueeze(-1)
        y_col = y.unsqueeze(-1)

        # --- 1. Compute f1(A, y) ---
        # Parameters: aa_1=exp(a_1), bb_1=exp(b_1), cc_1=exp(c_1), dd_1=exp(d_1)
        aa1 = torch.exp(self.a1)
        bb1 = torch.exp(self.b1)
        cc1 = torch.exp(self.c1)
        dd1 = torch.exp(self.d1)

        # Calculate terms for f1 using direct torch.pow()
        # Exponents (bb1, dd1) are positive, so this is numerically robust.
        term_A_f1 = torch.pow(A_col + cc1, bb1)
        term_y_f1 = torch.pow(y_col, dd1)

        # Summation: 1 / (1 + sum(aa1 * term_A_f1 * term_y_f1))
        f1_terms = aa1 * term_A_f1 * term_y_f1
        sum_f1 = torch.sum(f1_terms, dim=1).unsqueeze(-1)
        f1 = 1.0 / (1.0 + sum_f1)

        # --- 2. Compute f0(A, y) ---
        # Parameters: aa_0=exp(a_0), bb_0=-exp(b_0), cc_0=exp(c_0), dd_0=-exp(d_0)
        aa0 = torch.exp(self.a0)
        bb0 = -torch.exp(self.b0)  # Note the negative sign (negative exponent)
        dd0 = -torch.exp(self.d0)  # Note the negative sign (negative exponent)

        # Calculate terms for f0 using direct torch.pow()
        # Exponent bb0 is negative, but base (A_col + cc0) is always positive.
        term_A_f0 = torch.pow(A_col + self.epsilon_val, bb0)

        # Exponent dd0 is negative. We use self.epsilon_val on y to ensure numerical
        # stability if y_col is very close to zero, preventing explosion to infinity.
        term_y_f0 = torch.pow(y_col + self.epsilon_val, dd0)

        # Summation: 1 / (1 + sum(aa0 * term_A_f0 * term_y_f0))
        f0_terms = aa0 * term_A_f0 * term_y_f0
        sum_f0 = torch.sum(f0_terms, dim=1).unsqueeze(-1)
        f0 = 1.0 / (1.0 + sum_f0)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.stack([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred

class PolyACSigExpInter(nn.Module):
    """
    Architecture using Sigmoid activation on exponential-linear terms
    for f0 and f1, which take log_A and log_y as input.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_A, log_y) Free Parameters (3 total) ---
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_1 = exp(b_1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_1 = exp(d_1)

        # --- f0(log_A, log_y) Free Parameters (3 total) ---
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_0 = exp(b_0)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_0 = exp(d_0)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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

        # Extract required inputs
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_A = x[:, 3].unsqueeze(-1)  # log_A (for f-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_A, log_y) using Sigmoid ---
        # Parameters: bb_1 = exp(b_1), dd_1 = exp(d_1)
        bb1 = torch.exp(self.b1)
        dd1 = torch.exp(self.d1)

        # Linear combination: t_1 = a_1 - bb_1 * log_A - dd_1 * log_y
        t1 = self.a1 - bb1 * log_A - dd1 * log_y

        # Sigmoid activation: f_1 = sigmoid(t_1)
        f1 = torch.sigmoid(t1)  # Shape (batch_size, 1)

        # --- 2. Compute f0(log_A, log_y) using Sigmoid ---
        # Parameters: bb_0 = exp(b_0), dd_0 = exp(d_0)
        bb0 = torch.exp(self.b0)
        dd0 = torch.exp(self.d0)

        # Linear combination: t_0 = a_0 + bb_0 * log_A + dd_0 * log_y
        t0 = self.a0 + bb0 * log_A + dd0 * log_y

        # Sigmoid activation: f_0 = sigmoid(t_0)
        f0 = torch.sigmoid(t0)  # Shape (batch_size, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACSigExpFree(nn.Module):
    """
    Model 2: New architecture using Sigmoid activation on exponential-linear terms
    for f0 and f1, which take log_A and log_y as input.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 - f0*g0 + g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACSigExpFree model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        if N_g <= 0:
            raise ValueError("N_g must be a positive integer.")

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_A, log_y) Free Parameters (3 total) ---
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_1 = exp(b_1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_1 = exp(d_1)

        # --- f0(log_A, log_y) Free Parameters (3 total) ---
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_0 = exp(b_0)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_0 = exp(d_0)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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

        # Extract required inputs
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_A = x[:, 3].unsqueeze(-1)  # log_A (for f-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_A, log_y) using Sigmoid ---
        # Parameters: bb_1 = exp(b_1), dd_1 = exp(d_1)
        bb1 = torch.exp(self.b1)
        dd1 = torch.exp(self.d1)

        # Linear combination: t_1 = a_1 - bb_1 * log_A - dd_1 * log_y
        t1 = self.a1 - bb1 * log_A - dd1 * log_y

        # Sigmoid activation: f_1 = sigmoid(t_1)
        f1 = torch.sigmoid(t1)  # Shape (batch_size, 1)

        # --- 2. Compute f0(log_A, log_y) using Sigmoid ---
        # Parameters: bb_0 = exp(b_0), dd_0 = exp(d_0)
        bb0 = torch.exp(self.b0)
        dd0 = torch.exp(self.d0)

        # Linear combination: t_0 = a_0 + bb_0 * log_A + dd_0 * log_y
        t0 = self.a0 + bb0 * log_A + dd0 * log_y

        # Sigmoid activation: f_0 = sigmoid(t_0)
        f0 = torch.sigmoid(t0)  # Shape (batch_size, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g_linear_out = self.g_net(A_C)

        # Exponentiate the result to ensure positive values (g0, g1, g2)
        g = torch.exp(g_linear_out)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 - f0*g0 + g2
        B_pred = f1 * g1 - f0 * g0 + g2

        return B_pred

class PolyACSigGenInter(nn.Module):
    """
    Model 2: New architecture using Sigmoid activation on exponential-linear terms
    for f0 and f1, which take log_A and log_y as input.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACSigExpInter model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        if N_g <= 0:
            raise ValueError("N_g must be a positive integer.")

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_A, log_y) Free Parameters (3 total) ---
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_1 = exp(b_1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_1 = exp(d_1)

        # --- f0(log_A, log_y) Free Parameters (3 total) ---
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_0 = exp(b_0)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_0 = exp(d_0)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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

        # Extract required inputs
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_A = x[:, 3].unsqueeze(-1)  # log_A (for f-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_A, log_y) using Sigmoid ---
        # Parameters: bb_1 = exp(b_1), dd_1 = exp(d_1)
        bb1 = torch.exp(self.b1)
        dd1 = torch.exp(self.d1)

        # Linear combination: t_1 = a_1 - bb_1 * log_A - dd_1 * log_y
        t1 = self.a1 - bb1 * log_A - dd1 * log_y

        # Sigmoid activation: f_1 = sigmoid(t_1)
        f1 = torch.sigmoid(t1)  # Shape (batch_size, 1)

        # --- 2. Compute f0(log_A, log_y) using Sigmoid ---
        # Parameters: bb_0 = exp(b_0), dd_0 = exp(d_0)
        bb0 = torch.exp(self.b0)
        dd0 = torch.exp(self.d0)

        # Linear combination: t_0 = a_0 + bb_0 * log_A + dd_0 * log_y
        t0 = self.a0 + bb0 * log_A + dd0 * log_y

        # Sigmoid activation: f_0 = sigmoid(t_0)
        f0 = torch.sigmoid(t0)  # Shape (batch_size, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + (1.0 - f0 - f1) * g2

        return B_pred

class PolyACSigGenFree(nn.Module):
    """
    Model 2: New architecture using Sigmoid activation on exponential-linear terms
    for f0 and f1, which take log_A and log_y as input.

    Input: (A, C, y, log_A, log_y)
    Output: B = f1*g1 + f0*g0 + g2
    """

    def __init__(self, N_g: int, epsilon_val: float = NN_GLOBAL_EPSILON):
        """
        Initializes the PolyACSigExpFree model.

        Args:
            N_g (int): Number of neurons in the hidden layers of the g-network.
            epsilon_val (float): Small constant for numerical stability (default 1e-18).
        """
        super().__init__()

        self.N_g = N_g
        self.register_buffer('epsilon_val', torch.tensor(epsilon_val, dtype=DTYPE))

        # --- f1(log_A, log_y) Free Parameters (3 total) ---
        self.a1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_1 = exp(b_1)
        self.d1 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_1 = exp(d_1)

        # --- f0(log_A, log_y) Free Parameters (3 total) ---
        self.a0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for bb_0 = exp(b_0)
        self.d0 = nn.Parameter(torch.randn(1, dtype=DTYPE) * 0.1)  # Used for dd_0 = exp(d_0)

        # --- (g0, g1, g2) MLP Network ---
        # Input: (A, C) -> 2 dimensions
        # Hidden: 3 layers, N_g neurons, ReLU activation
        # Output: 3 dimensions (Linear output is exponentiated in forward pass)
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

        # Extract required inputs
        A = x[:, 0].unsqueeze(-1)  # A (for g-net input)
        C = x[:, 1].unsqueeze(-1)  # C (for g-net input)
        log_A = x[:, 3].unsqueeze(-1)  # log_A (for f-net input)
        log_y = x[:, 4].unsqueeze(-1)  # log_y (for f-net input)

        # --- 1. Compute f1(log_A, log_y) using Sigmoid ---
        # Parameters: bb_1 = exp(b_1), dd_1 = exp(d_1)
        bb1 = torch.exp(self.b1)
        dd1 = torch.exp(self.d1)

        # Linear combination: t_1 = a_1 - bb_1 * log_A - dd_1 * log_y
        t1 = self.a1 - bb1 * log_A - dd1 * log_y

        # Sigmoid activation: f_1 = sigmoid(t_1)
        f1 = torch.sigmoid(t1)  # Shape (batch_size, 1)

        # --- 2. Compute f0(log_A, log_y) using Sigmoid ---
        # Parameters: bb_0 = exp(b_0), dd_0 = exp(d_0)
        bb0 = torch.exp(self.b0)
        dd0 = torch.exp(self.d0)

        # Linear combination: t_0 = a_0 + bb_0 * log_A + dd_0 * log_y
        t0 = self.a0 + bb0 * log_A + dd0 * log_y

        # Sigmoid activation: f_0 = sigmoid(t_0)
        f0 = torch.sigmoid(t0)  # Shape (batch_size, 1)

        # --- 3. Compute (g0, g1, g2)(A, C) ---
        # Input to g-net is (A, C)
        A_C = torch.cat([A, C], dim=1)  # Shape (batch_size, 2)

        # Get the linear output (batch_size, 3)
        g = self.g_net(A_C)

        g0 = g[:, 0].unsqueeze(-1)  # Shape (batch_size, 1)
        g1 = g[:, 1].unsqueeze(-1)  # Shape (batch_size, 1)
        g2 = g[:, 2].unsqueeze(-1)  # Shape (batch_size, 1)

        # --- 4. Final Output Combination ---
        # B = f1*g1 + f0*g0 + (1 - f0 - f1)*g2
        B_pred = f1 * g1 + f0 * g0 + g2

        return B_pred
