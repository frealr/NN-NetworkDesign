import torch
import torch.nn as nn

class GPLayer(nn.Module):
    def __init__(self, p, B, t):
        """
        Initializes the Gradient Projection-like Layer.
        
        Args:
            p (Tensor): Interior point of the polytope (dim_w).
            B (Tensor): Matrix defining inequalities B w <= t (n_ineq, dim_w).
            t (Tensor): Thresholds for inequalities (n_ineq).
        """
        super().__init__()
        # Register p, B, t as buffers so they are part of the state_dict but not trained
        self.register_buffer('p', p)
        self.register_buffer('B', B)
        self.register_buffer('t', t)
        
    def forward(self, r, s):
        """
        Forward pass to compute feasible w.
        
        Args:
            r (Tensor): Direction vector (batch_size, dim_w).
            s (Tensor): Scalar parameter (batch_size, 1).
            
        Returns:
            w (Tensor): Feasible point (batch_size, dim_w).
        """
        # Ensure inputs match buffer devices/types
        # p: (dim_w) -> (1, dim_w)
        p = self.p.unsqueeze(0)
        
        # 1. Compute distances to boundaries along ray r
        # We want to find alpha such that B(p + alpha * r) <= t
        # alpha * B r <= t - B p
        
        # Numerator: t - B p (n_ineq)
        # Since p is interior, t - B p > 0
        numer = self.t - (self.B @ self.p) # Shape: (n_ineq)
        numer = numer.unsqueeze(0) # Shape: (1, n_ineq)
        
        # Denominator: B r (batch_size, n_ineq)
        denom = r @ self.B.T # Shape: (batch_size, n_ineq)
        
        # 2. Calculate alpha candidates
        # alpha_i = numer_i / denom_i
        # We only care about boundaries we are approaching, i.e., denom_i > 0.
        # If denom_i <= 0, the ray is moving away or parallel, so distance is infinity.
        
        # To handle this vectorized, we can set alpha to infinity where denom <= 0
        # Add a small epsilon to avoid division by zero if needed, though mask handles it.
        
        mask = denom > 1e-6
        
        # Initialize alphas with a large number (infinity)
        alphas = torch.full_like(denom, float('inf'))
        
        # Compute valid alphas
        # We use where to safely divide
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))
        valid_alphas = numer / safe_denom
        
        alphas = torch.where(mask, valid_alphas, alphas)
        
        # 3. Compute alpha_max = min_i(alpha_i)
        # This is the distance to the closest boundary in direction r
        alpha_max, _ = torch.min(alphas, dim=1, keepdim=True) # (batch_size, 1)
        
        # Handle infinite alpha_max (unbounded directions)
        # Replace inf with a large number
        alpha_max = torch.clamp(alpha_max, max=1e5)
        
        # 4. Compute actual step size alpha
        # alpha = sigmoid(s) * alpha_max
        alpha = torch.sigmoid(s) * alpha_max
        
        # 5. Compute w
        # w = p + alpha * r
        w = p + alpha * r
        
        return w
