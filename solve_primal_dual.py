import torch
import torch.optim as optim
import numpy as np
import cvxpy as cp
from utils import get_equality_solution_and_null_basis

def solve_primal_dual():
    # --- 1. Problem Data Setup (Identical to solve_custom_nn.py) ---
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

    for o in range(n):
        for d in range(n):
            if o != d:
                for i in range(n):
                    current_length = 2*n**2 + a*n*o + d*a
                    cons = np.concatenate([
                        np.zeros(2*n**2),
                        np.zeros(n*a*(o) + a*d),
                        Aout[i] - Ain[i],
                        np.zeros(total_length - current_length - a)
                    ])
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

    # --- 2. Preprocessing (Decoupled) ---
    print("Preprocessing constraints (Decoupled)...")
    from decoupled_basis import get_decoupled_u_R
    u_part, R = get_decoupled_u_R(n, a)
    
    # Convert R to sparse tensor
    indices = torch.nonzero(R).t()
    values = R[indices[0], indices[1]]
    R_sparse = torch.sparse_coo_tensor(indices, values, R.shape)
    
    dim_w = R.shape[1]
    n_vars = u_part.shape[0]
    
    # --- 3. Primal-Dual Variables ---
    # Initialize w from interior point p to start feasible
    from utils import find_interior_point
    # 2. Transform Inequality Constraints: x >= 0 => -R w <= u_part
    B = -R
    t = u_part
    print("Finding interior point for initialization...")
    p = find_interior_point(B, t)
    
    # Primal variable w (unconstrained)
    w = torch.nn.Parameter(p.clone().detach())
    
    # Dual variable lambda for x >= 0 (constrained >= 0)
    # We want to enforce x >= 0, so g(x) = -x <= 0.
    # Lagrangian L = f(x) + lambda^T (-x) = f(x) - lambda^T x
    # lambda >= 0
    lambda_ineq = torch.zeros(n_vars, dtype=torch.float32)
    
    # --- 4. Optimization Loop (Augmented Lagrangian) ---
    # Hyperparameters
    n_outer_iter = 100
    n_inner_iter = 200
    rho = 10.0 # Initial penalty parameter
    rho_max = 1e6
    rho_step = 1.2
    
    print("Starting Augmented Lagrangian optimization...")
    
    for outer in range(n_outer_iter):
        
        # --- Primal Update (Minimize Augmented Lagrangian w.r.t w) ---
        optimizer = optim.Adam([w], lr=1e-3)
        
        def closure():
            optimizer.zero_grad()
            
            # Recover x using Sparse MM
            if R_sparse.device != w.device:
                R_sparse_local = R_sparse.to(w.device)
            else:
                R_sparse_local = R_sparse
                
            x = torch.sparse.mm(R_sparse_local, w.unsqueeze(1)).squeeze() + u_part
            
            # Objective Function f(x)
            y = x * entr_coefs_tensor
            mask = entr_coefs_tensor > 0
            y_active = y[mask]
            
            # Safe Entropy
            epsilon = 1e-8
            y_safe = torch.clamp(y_active, min=epsilon)
            entropy_term = torch.sum(y_safe * torch.log(y_safe) + y_safe)
            
            linear_term = torch.dot(x, c_tensor + uext_tensor)
            f_x = entropy_term + linear_term
            
            # Augmented Lagrangian Terms for x >= 0
            # Constraint: g(x) = -x <= 0
            # ALM term: (rho/2) * || max(0, -x + lambda/rho) ||^2 - (1/(2*rho)) * ||lambda||^2
            # But standard practice often separates the linear dual term and quadratic penalty
            # L_aug = f(x) + lambda^T (-x) + (rho/2) || max(0, -x) ||^2
            # Actually, the "Multiplier Method" update is cleaner:
            # L(x, lambda) = f(x) + (rho/2) * sum( max(0, -x + lambda/rho)^2 )
            
            # Let's use the standard form:
            # violation = -x
            # term = max(0, violation + lambda/rho)
            # loss = f(x) + (rho/2) * sum(term^2)
            
            violation = -x
            aug_term = torch.sum(torch.square(torch.relu(violation + lambda_ineq / rho)))
            loss = f_x + (rho / 2.0) * aug_term
            
            loss.backward()
            return loss

        # Run inner optimization
        for inner in range(n_inner_iter):
             optimizer.step(closure)
        
        # --- Dual & Penalty Update ---
        with torch.no_grad():
            x_curr = torch.sparse.mm(R_sparse, w.unsqueeze(1)).squeeze() + u_part
            
            # Update Lambda
            # lambda_new = max(0, lambda + rho * (-x))
            lambda_ineq = torch.relu(lambda_ineq + rho * (-x_curr))
            
            # Update Rho (increase penalty to force feasibility)
            rho = min(rho * rho_step, rho_max)
            
            # Metrics
            y = x_curr * entr_coefs_tensor
            mask = entr_coefs_tensor > 0
            y_active = y[mask]
            y_safe = torch.clamp(y_active, min=1e-16)
            entropy_term = torch.sum(y_safe * torch.log(y_safe) + y_safe)
            linear_term = torch.dot(x_curr, c_tensor + uext_tensor)
            primal_val = (entropy_term + linear_term).item()
            
            min_x = x_curr.min().item()
            max_viol = torch.max(-x_curr).item()
            
            if outer % 5 == 0:
                print(f"Iter {outer}: Obj = {primal_val:.4f}, Min x = {min_x:.6e}, Max Viol = {max_viol:.6e}, Rho = {rho:.1f}")
            
            # Check convergence
            if max_viol < 1e-6 and outer > 20:
                print("Converged!")
                break

    # --- 5. Final Results ---
    x_final = (torch.sparse.mm(R_sparse, w.unsqueeze(1)).squeeze() + u_part).detach()
    
    # Final cleanup: clamp small negative values to 0 for reporting if they are negligible
    x_final = torch.relu(x_final) # Force non-negative for final output if very close
    
    y = x_final * entr_coefs_tensor
    mask = entr_coefs_tensor > 0
    y_active = y[mask]
    y_safe = torch.clamp(y_active, min=1e-16)
    entropy_term = torch.sum(y_safe * torch.log(y_safe) + y_safe)
    linear_term = torch.dot(x_final, c_tensor + uext_tensor)
    final_loss = entropy_term + linear_term
    
    print(f"\nFinal Results:")
    print(f"Objective Value: {final_loss.item():.6f}")
    print(f"Target CVXPY Value: 13.17")
    print(f"Difference: {100*abs(abs(final_loss.item() - 13.17)/13.17):.6f}")
    print(f"Min x: {x_final.min().item():.6e}")
    print(f"Max Equality Violation: {torch.max(torch.abs(A_tensor @ x_final - b_tensor)).item():.6e}")
    print(f"sol final : {x_final}")
    
    from visualization import visualize_results
    visualize_results(x_final, n, a, "Primal_Dual_Direct")
    
    return x_final, final_loss.item()

if __name__ == "__main__":
    solve_primal_dual()
