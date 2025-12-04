import torch
import torch.optim as optim
import numpy as np
import cvxpy as cp
from utils import get_equality_solution_and_null_basis, find_interior_point
from model import ConstrainedOptNet

from problem_data import create_problem_data

def solve_custom_nn(data=None):
    if data is None:
        # Default data if none provided
        t = np.array([1, 2, 1, 1.5, 2, 1.5])
        u = np.array([100, 0.9, 1.1, 1.3,100, 1.4, 1, 1.2,100])
        data = create_problem_data(t, u)

    n = data['n']
    a = data['a']
    entr_coefs = data['entr_coefs']
    c_vec = data['c']
    uext = data['uext']
    A = data['A']
    b_vec = data['b']

    # Convert to Tensors
    A_tensor = torch.tensor(A, dtype=torch.float32)
    b_tensor = torch.tensor(b_vec, dtype=torch.float32)
    entr_coefs_tensor = torch.tensor(entr_coefs, dtype=torch.float32)
    c_tensor = torch.tensor(c_vec, dtype=torch.float32)
    uext_tensor = torch.tensor(uext, dtype=torch.float32)

    # --- 2. Preprocessing Constraints ---
    # Equality: A x = b
    # Inequality: x >= 0  =>  -I x <= 0
    
    # --- 2. Preprocessing Constraints (Decoupled) ---
    print("Preprocessing constraints (Decoupled)...")
    from decoupled_basis import get_decoupled_u_R
    u_part, R = get_decoupled_u_R(n, a)
    
    # Inequality constraints: x >= 0
    # R w + u >= 0  =>  -R w <= u
    # So B = -R, t = u
    B = -R
    t = u_part
    
    print("Finding interior point...")
    from utils import find_interior_point
    p = find_interior_point(B, t)
    
    # --- 3. Model Setup ---
    class DirectParamModel(torch.nn.Module):
        def __init__(self, u, R, p, B, t):
            super().__init__()
            self.register_buffer('u', u)
            
            # Convert R to sparse tensor
            # R is (n_vars, dim_w)
            indices = torch.nonzero(R).t()
            values = R[indices[0], indices[1]]
            self.R_sparse = torch.sparse_coo_tensor(indices, values, R.shape)
            
            # Keep dense R for GPLayer initialization only (if needed inside, but GPLayer takes dense)
            # Actually GPLayer needs dense R for internal calculations usually, let's check.
            # ConstrainedOptNet uses R for projection. If we want to optimize THAT too, it's bigger work.
            # For now, let's optimize the reconstruction step which is the heavy one in forward.
            
            # Initialize GPLayer (it likely expects dense R for now)
            self.gp_layer = ConstrainedOptNet(1, 1, u, R, p, B, t).gp_layer
            
            dim_w = R.shape[1]
            self.r = torch.nn.Parameter(torch.randn(1, dim_w) * 0.1)
            self.s = torch.nn.Parameter(torch.zeros(1, 1)) 
            
        def forward(self, _):
            w = self.gp_layer(self.r, self.s)
            
            # Sparse Reconstruction: x = R w + u
            # w is (batch, dim_w) -> (1, 30)
            # R is (n_vars, dim_w) -> (72, 30)
            # We want x (batch, n_vars)
            # x = (R @ w.T).T
            
            # Ensure R_sparse is on same device as w
            if self.R_sparse.device != w.device:
                self.R_sparse = self.R_sparse.to(w.device)
                
            x_out = torch.sparse.mm(self.R_sparse, w.t()).t() + self.u.unsqueeze(0)
            return x_out

    model = DirectParamModel(u_part, R, p, B, t)

    
    # Print initialization stats
    print(f"u_part stats: min={u_part.min():.4f}, max={u_part.max():.4f}, mean={u_part.mean():.4f}")
    print(f"R stats: min={R.min():.4f}, max={R.max():.4f}, mean={R.mean():.4f}")
    
    # Use Adam for potentially better convergence on this landscape
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Remove CyclicLR for Adam, use simple StepLR or ReduceLROnPlateau if needed, or just constant
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    # --- 4. Training Loop ---
    print("Starting training...")
    n_epochs = 100000 
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        x_pred = model(None).squeeze()
        
        # Custom Objective
        y = x_pred * entr_coefs_tensor
        mask = entr_coefs_tensor > 0
        y_active = y[mask]
        y_active = torch.clamp(y_active, min=1e-16)
        
        entropy_term = torch.sum(y_active * torch.log(y_active) + y_active)
        linear_term = torch.dot(x_pred, c_tensor + uext_tensor)
        
        loss = entropy_term + linear_term
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6e}")

    # --- 5. Validation ---
    print("\nValidation...")
    model.eval()
    with torch.no_grad():
        x_final = model(None).squeeze()
        
        y = x_final * entr_coefs_tensor
        mask = entr_coefs_tensor > 0
        y_active = y[mask]
        y_active = torch.clamp(y_active, min=1e-16)
        entropy_term = torch.sum(y_active * torch.log(y_active) + y_active)
        linear_term = torch.dot(x_final, c_tensor + uext_tensor)
        final_loss = entropy_term + linear_term
        
    print(f"NN Final Loss: {final_loss.item():.6f}")
    
    # Check Constraints
    max_eq_viol = torch.max(torch.abs(A_tensor @ x_final - b_tensor))
    max_ineq_viol = torch.max(-x_final) # Should be <= 0
    
    print(f"Max Equality Violation: {max_eq_viol.item():.6e}")
    print(f"Max Inequality Violation: {max_ineq_viol.item():.6e} (Should be <= 0)")
    
    # Compare with CVXPY (Hardcoded result from previous run or re-run)
    # CVXPY Optimal Value: ~13.1715
    print(f"Target CVXPY Value: 13.171467")
    print(f"Difference: {100*abs(abs(final_loss.item() - 13.17)/13.17):.6f}")

    print(f"sol final : {x_final}")
    
    from visualization import visualize_results
    visualize_results(x_final, n, a, "NN_Projection")
    
    return x_final, final_loss.item()

if __name__ == "__main__":
    solve_custom_nn()
