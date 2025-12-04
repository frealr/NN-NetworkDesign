import torch
import torch.optim as optim
import numpy as np
import cvxpy as cp
from utils import get_equality_solution_and_null_basis, find_interior_point
from model import ConstrainedOptNet

def solve_custom_nn():
    # --- 1. Problem Data Setup (from user's code) ---
    n = 3
    a = 6

    entr_coefs = np.concatenate([np.ones(2 * n**2), np.zeros(a * n**2)])

    t_vec = np.array([1, 2, 1, 1.5, 2, 1.5])
    c_vec = np.concatenate([np.zeros(2 * n**2), np.tile(t_vec, n**2)])

    u_vec = np.array([100, 2, 4, 2,100, 2, 4, 2,100])
    uext = np.concatenate([np.zeros(n**2), u_vec, np.zeros(a * n**2)])

    # --- Zero out coefficients for o=d ---
    for o in range(n):
        d = o
        # Block 1: f(o,d)
        idx1 = n*o + d
        entr_coefs[idx1] = 0
        c_vec[idx1] = 0
        uext[idx1] = 0
        
        # Block 2: fext(o,d)
        idx2 = n**2 + n*o + d
        entr_coefs[idx2] = 0
        c_vec[idx2] = 0
        uext[idx2] = 0
        
        # Block 3: f(i,j,o,d)
        start_idx3 = 2*n**2 + (n*o + d)*a
        for k in range(a):
            idx3 = start_idx3 + k
            entr_coefs[idx3] = 0
            c_vec[idx3] = 0
            uext[idx3] = 0

    Ain = np.array([
        [0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0]
    ])

    Aout = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ])

    A_list = []
    total_length = 2*n**2 + n**2*a

    for o in range(n):
      for d in range(n):
        cons = np.zeros(2*n**2 + n**2*a)
        cons[n*o+d] = 1
        cons[n**2 + n*o+d] = 1
        A_list.append(cons)
    A_aux = np.array(A_list)
    n_rows_A_c1 = A_aux.shape[0]
    b_vec = np.ones((n_rows_A_c1, 1))

    for o in range(n):          # o = 0..n-1
        for d in range(n):      # d = 0..n-1
            if o != d:
                for i in range(n):   # i = 0..n-1
                    current_length = 2*n**2 + a*n*o + d*a
                    cons = np.concatenate([
                        np.zeros(2*n**2),
                        np.zeros(n*a*(o) + a*d),
                        Aout[i] - Ain[i],
                        np.zeros(total_length - current_length - a)
                    ])
                    # Ajustes en función de i
                    if i == o:
                        cons[n*o + d] = -1
                    if i == d:
                        cons[n*o + d] = 1
                    A_list.append(cons)

    # --- New Constraints: f(o,d) = 0 and f(i,j,o,d) = 0 if o == d ---
    for o in range(n):
        # 1. f(o,o) = 0 (Block 1)
        cons = np.zeros(total_length)
        cons[n*o + o] = 1
        A_list.append(cons)
        
        # 2. f(i,j,o,o) = 0 (Block 3)
        start_idx = 2*n**2 + a*(n*o + o)
        for k in range(a):
            cons = np.zeros(total_length)
            cons[start_idx + k] = 1
            A_list.append(cons)

    A = np.array(A_list)
    n_rows_A = A.shape[0]
    b_vec = b_vec.flatten()
    b_vec = np.concatenate([b_vec, np.zeros(n_rows_A - n_rows_A_c1)])
    b_vec = b_vec.flatten()

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
