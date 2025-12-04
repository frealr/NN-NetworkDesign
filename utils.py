import torch
import numpy as np
import scipy.linalg
from scipy.optimize import linprog

def generate_random_qp(n_vars, n_eq, n_ineq, seed=42):
    """
    Generates a random Quadratic Programming problem:
    min 1/2 x^T Q x + c^T x
    s.t. A x <= b
         E x = d
    
    Q is positive semi-definite.
    Ensures the problem is feasible.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate a random feasible point x0
    x0 = np.random.randn(n_vars)
    
    # Generate Equality Constraints E x = d
    # E should be full rank for simplicity in this demo, though SVD handles rank deficient too.
    E = np.random.randn(n_eq, n_vars)
    d = E @ x0
    
    # Generate Inequality Constraints A x <= b
    A = np.random.randn(n_ineq, n_vars)
    # Make sure x0 is strictly interior for A x <= b initially to guarantee feasibility
    # b = A x0 + positive_slack
    slack = np.random.uniform(0.1, 1.0, size=n_ineq)
    b = A @ x0 + slack
    
    # Generate Objective
    # Q = M^T M to ensure PSD
    M = np.random.randn(n_vars, n_vars)
    Q = M.T @ M
    c = np.random.randn(n_vars)
    
    return (
        torch.tensor(Q, dtype=torch.float32),
        torch.tensor(c, dtype=torch.float32),
        torch.tensor(A, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        torch.tensor(E, dtype=torch.float32),
        torch.tensor(d, dtype=torch.float32)
    )

def get_equality_solution_and_null_basis(E, d):
    """
    Finds particular solution u and null space basis R such that:
    {x | E x = d} = {R w + u | w in R^k}
    
    Args:
        E: (n_eq, n_vars) matrix
        d: (n_eq) vector
        
    Returns:
        u: (n_vars) particular solution
        R: (n_vars, null_dim) null space basis
    """
    E_np = E.detach().cpu().numpy()
    d_np = d.detach().cpu().numpy()
    
    # 1. Particular solution u using Least Squares (min norm solution)
    u_np, residuals, rank, s = scipy.linalg.lstsq(E_np, d_np)
    
    # 2. Null space basis R using SVD
    # E = U S V^T
    # The rows of V^T corresponding to singular values ~ 0 form the null space basis.
    # Or columns of V corresponding to small singular values.
    U, S, Vh = scipy.linalg.svd(E_np)
    
    # Tolerance for zero singular values
    tol = 1e-10 * np.max(S)
    null_mask = S < tol
    # If E is wide (n_eq < n_vars), there are (n_vars - n_eq) null dimensions automatically 
    # if E is full rank. SVD returns min(n_eq, n_vars) singular values.
    # Vh is (n_vars, n_vars).
    # The null space corresponds to the last (n_vars - rank) rows of Vh.
    
    effective_rank = np.sum(S > tol)
    # The basis vectors are the rows of Vh from effective_rank onwards
    # (Since Vh is V^T, the rows of Vh are the columns of V)
    R_np = Vh[effective_rank:, :].T
    
    u = torch.tensor(u_np, dtype=torch.float32)
    R = torch.tensor(R_np, dtype=torch.float32)
    
    return u, R

def find_interior_point(B, t):
    """
    Finds a strictly interior point p such that B p < t.
    Uses Chebyshev center formulation via Linear Programming.
    
    Maximize r
    s.t. B_i p + ||B_i|| r <= t_i
    
    Variables for LP: [p_1, ..., p_k, r]
    """
    B_np = B.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    
    n_constraints, n_vars = B_np.shape
    
    # Norms of rows of B
    row_norms = np.linalg.norm(B_np, axis=1)
    
    # Construct LP matrices
    # We want to maximize r, so minimize -r.
    # Objective c_lp = [0, ..., 0, -1]
    c_lp = np.zeros(n_vars + 1)
    c_lp[-1] = -1
    
    # Constraints: B p + ||B_i|| r <= t
    # [B, Norms] @ [p; r] <= t
    A_ub = np.hstack([B_np, row_norms.reshape(-1, 1)])
    # Add small tolerance to t to prevent numerical infeasibility for active constraints
    b_ub = t_np + 1e-5
    
    # Bounds: p is bounded to avoid unboundedness, r >= 0 (strictly > 0 for interior)
    # We use smaller bounds for p to keep initialization reasonable.
    bounds = [(-10, 10)] * n_vars + [(1e-6, None)] # Enforce strictly positive radius
    
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        raise ValueError(f"Could not find interior point. Status: {res.status}, Message: {res.message}")
        
    p_np = res.x[:-1]
    radius = res.x[-1]
    
    print(f"Found interior point with radius: {radius}")
    
    return torch.tensor(p_np, dtype=torch.float32)
