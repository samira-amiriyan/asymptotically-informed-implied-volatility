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
NN_GLOBAL_EPSILON = 1e-18
torch.set_default_dtype(DTYPE)


class SimpleExp(nn.Module):
    """
    Model 5: Simple MLP with Exponential output.
    Input: (A, C). Output: exp(Linear(last_hidden_layer))
    """

    def __init__(self, N: int):
        """
        Initializes the SimpleExp model.

        Args:
            N (int): Number of neurons in the hidden layers.
            epsilon_val (float): Small constant (not used here but kept for consistency).
        """
        super().__init__()

        # The input is (A, C) -> 2 dimensions
        self.mlp = nn.Sequential(
            nn.Linear(2, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, 1, dtype=DTYPE)  # Final linear layer outputting 1 dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract only (A, C) which are the first two columns
        A_C = x[:, 0:2]  # Shape (B, 2)

        # Pass through the MLP to get the linear output
        linear_out = self.mlp(A_C)

        # Apply the exponential map to get the final output
        B_pred = torch.exp(linear_out)

        return B_pred


class SimpleGen(nn.Module):
    """
    Model 6: Simple MLP with General Linear output.
    Input: (A, C). Output: Linear(last_hidden_layer)
    """

    def __init__(self, N: int):
        """
        Initializes the SimpleGen model.

        Args:
            N (int): Number of neurons in the hidden layers.
            epsilon_val (float): Small constant (not used here but kept for consistency).
        """
        super().__init__()

        # The input is (A, C) -> 2 dimensions
        self.mlp = nn.Sequential(
            nn.Linear(2, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, N, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(N, 1, dtype=DTYPE)  # Final linear layer outputting 1 dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5)
                              containing (A, C, y, log_A, log_y).

        Returns:
            torch.Tensor: The predicted B value of shape (batch_size, 1).
        """

        # Extract only (A, C) which are the first two columns
        A_C = x[:, 0:2]  # Shape (B, 2)

        # Pass through the MLP to get the final linear output
        B_pred = self.mlp(A_C)

        return B_pred
