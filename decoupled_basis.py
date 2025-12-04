import torch
import numpy as np
import scipy.linalg

def get_decoupled_u_R(n, a):
    """
    Computes the particular solution u and null space basis R 
    by exploiting the block-diagonal structure of the constraints per (o,d) pair.
    
    Returns:
        u (Tensor): Global particular solution (n_vars).
        R (Tensor): Global null space basis (n_vars, total_null_dim).
    """
    # Problem constants
    # Variables per (o,d): f(o,d), fext(o,d), and 'a' arc flows.
    # Total vars per pair = 2 + a
    vars_per_pair = 2 + a
    total_vars = (2 * n**2) + (a * n**2)
    
    # Global containers
    u_global = np.zeros(total_vars)
    R_cols = [] # Will store columns of R
    
    # Topology matrices (same as in cvxpy_problem.py)
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
    
    # Iterate over each (o,d) pair
    for o in range(n):
        for d in range(n):
            # --- 1. Identify Global Indices for this pair ---
            # f(o,d) is at index n*o + d
            idx_f = n*o + d
            # fext(o,d) is at index n^2 + n*o + d
            idx_fext = n**2 + n*o + d
            # Arc flows start at 2*n^2 + (n*o + d)*a
            start_arc = 2*n**2 + (n*o + d)*a
            indices_arcs = list(range(start_arc, start_arc + a))
            
            # Local variable mapping:
            # 0: f(o,d)
            # 1: fext(o,d)
            # 2..2+a-1: arc flows
            
            # --- 2. Build Local Constraints A_loc * x_loc = b_loc ---
            A_loc_list = []
            b_loc_list = []
            
            # Constraint 1: Demand (f + fext = 1)
            # 1 * x[0] + 1 * x[1] = 1
            row = np.zeros(vars_per_pair)
            row[0] = 1
            row[1] = 1
            A_loc_list.append(row)
            b_loc_list.append(1)
            
            # Constraint 2: Flow Conservation
            if o != d:
                for i in range(n):
                    row = np.zeros(vars_per_pair)
                    # Arc coefficients
                    row[2:] = Aout[i] - Ain[i]
                    
                    # f(o,d) coefficient
                    if i == o:
                        row[0] = -1
                    elif i == d:
                        row[0] = 1
                    
                    A_loc_list.append(row)
                    b_loc_list.append(0)
            
            # Constraint 3: Zero Constraints (if o == d)
            if o == d:
                # f(o,o) = 0
                row = np.zeros(vars_per_pair)
                row[0] = 1
                A_loc_list.append(row)
                b_loc_list.append(0)
                
                # f_arcs(o,o) = 0
                for k in range(a):
                    row = np.zeros(vars_per_pair)
                    row[2+k] = 1
                    A_loc_list.append(row)
                    b_loc_list.append(0)
            
            # Convert to numpy
            A_loc = np.array(A_loc_list)
            b_loc = np.array(b_loc_list)
            
            # --- 3. Compute Local u and R ---
            # Use lstsq for u
            u_loc, residuals, rank, s = scipy.linalg.lstsq(A_loc, b_loc)
            
            # Use SVD for R
            U, S, Vh = scipy.linalg.svd(A_loc)
            tol = 1e-10 * np.max(S) if len(S) > 0 else 1e-10
            effective_rank = np.sum(S > tol)
            R_loc = Vh[effective_rank:, :].T # (vars_per_pair, null_dim)
            
            # --- 4. Place into Global Structures ---
            # Map local u to global u
            u_global[idx_f] = u_loc[0]
            u_global[idx_fext] = u_loc[1]
            u_global[indices_arcs] = u_loc[2:]
            
            # Map local R to global R
            # Each column of R_loc becomes a column in global R
            # The global column will be sparse (zeros everywhere except at local indices)
            num_local_basis = R_loc.shape[1]
            for k in range(num_local_basis):
                col_global = np.zeros(total_vars)
                col_global[idx_f] = R_loc[0, k]
                col_global[idx_fext] = R_loc[1, k]
                col_global[indices_arcs] = R_loc[2:, k]
                R_cols.append(col_global)
                
    # Stack R columns
    if len(R_cols) > 0:
        R_global = np.stack(R_cols, axis=1)
    else:
        R_global = np.zeros((total_vars, 0))
        
    return torch.tensor(u_global, dtype=torch.float32), torch.tensor(R_global, dtype=torch.float32)

if __name__ == "__main__":
    # Test the function
    u, R = get_decoupled_u_R(3, 6)
    print(f"Global u shape: {u.shape}")
    print(f"Global R shape: {R.shape}")
