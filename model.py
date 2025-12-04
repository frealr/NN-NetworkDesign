import torch
import torch.nn as nn
from layers import GPLayer

class ConstrainedOptNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, u, R, p, B, t):
        """
        Neural Network for solving the constrained optimization problem.
        
        Args:
            input_dim (int): Dimension of input (can be dummy or problem params).
            hidden_dim (int): Hidden layer size.
            u (Tensor): Particular solution for equality constraints.
            R (Tensor): Null space basis for equality constraints.
            p (Tensor): Interior point for inequality constraints (in w-space).
            B (Tensor): Transformed inequality matrix.
            t (Tensor): Transformed inequality thresholds.
        """
        super().__init__()
        
        self.register_buffer('u', u)
        self.register_buffer('R', R)
        
        dim_w = R.shape[1]
        
        # Simple MLP to generate r and s
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output: r (dim_w) + s (1)
            nn.Linear(hidden_dim, dim_w + 1)
        )
        
        # Constraint Layer
        self.gp_layer = GPLayer(p, B, t)
        
    def forward(self, x_in):
        """
        Args:
            x_in: Input tensor.
        Returns:
            x_out: Feasible solution in original space.
        """
        out = self.net(x_in)
        
        # Split into r and s
        # Assuming last dimension is split
        r = out[:, :-1]
        s = out[:, -1:]
        
        # Get feasible w
        w = self.gp_layer(r, s)
        
        # Recover x = R w + u
        # w is (batch, dim_w)
        # R is (n_vars, dim_w) -> need transpose for matmul if w is on right, 
        # or w @ R.T
        
        # x = (R @ w.T).T + u
        # x = w @ R.T + u
        x_out = w @ self.R.T + self.u.unsqueeze(0)
        
        return x_out
