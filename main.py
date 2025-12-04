import torch
import torch.optim as optim
import numpy as np
import cvxpy as cp
from utils import generate_random_qp, get_equality_solution_and_null_basis, find_interior_point
from model import ConstrainedOptNet

def solve_with_cvxpy(Q, c, A, b, E, d):
    """
    Solves the QP using CVXPY for ground truth.
    """
    n = Q.shape[0]
    x = cp.Variable(n)
    
    Q_np = Q.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    E_np = E.detach().cpu().numpy()
    d_np = d.detach().cpu().numpy()
    
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q_np) + c_np.T @ x)
    constraints = [
        A_np @ x <= b_np,
        E_np @ x == d_np
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return x.value, prob.value

def train():
    # 1. Setup Problem
    print("Generating random QP problem...")
    n_vars = 20
    n_eq = 5
    n_ineq = 10
    
    Q, c, A, b, E, d = generate_random_qp(n_vars, n_eq, n_ineq, seed=123)
    
    # Debug: Check if x0 (implied by generation) is feasible
    # We need to recover x0 or just trust the generator. 
    # Let's trust generator but verify B w < t existence.
    
    # 2. Preprocessing
    print("Preprocessing constraints...")
    # Equality constraints: x = R w + u
    u, R = get_equality_solution_and_null_basis(E, d)
    
    # Transform inequalities: A(Rw + u) <= b  =>  (AR)w <= b - Au
    B = A @ R
    t = b - A @ u
    
    # Check if we can find a feasible w0 from the problem generation logic?
    # The generator created x0. Let's assume we don't have it, but we know it exists.
    # If linprog fails, we might need to relax bounds or check numerics.
    
    # Find interior point p for B w <= t
    p = find_interior_point(B, t)
    
    # 3. Model Initialization
    input_dim = 10 # Dummy input dimension
    hidden_dim = 64
    
    model = ConstrainedOptNet(input_dim, hidden_dim, u, R, p, B, t)
    # Use Adam with CosineAnnealingLR for smooth convergence
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=1e-6)
    
    # 4. Training Loop
    print("Starting training...")
    n_epochs = 50000
    
    dummy_input = torch.randn(1, input_dim)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        x_pred = model(dummy_input)
        
        # Loss: 1/2 x^T Q x + c^T x
        term1 = 0.5 * torch.sum((x_pred @ Q) * x_pred, dim=1)
        term2 = x_pred @ c
        
        loss = (term1 + term2).mean()
        
        loss.backward()
        # Optional: Gradient clipping to prevent divergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6e}")
            
    # 5. Validation
    print("\nValidation...")
    model.eval()
    with torch.no_grad():
        x_final = model(dummy_input).squeeze()
        final_loss = 0.5 * (x_final @ Q @ x_final) + c @ x_final
        
    print(f"NN Final Loss: {final_loss.item():.6f}")
    
    # Check Constraints
    max_eq_viol = torch.max(torch.abs(E @ x_final - d))
    max_ineq_viol = torch.max((A @ x_final - b))
    
    print(f"Max Equality Violation: {max_eq_viol.item():.6e}")
    print(f"Max Inequality Violation: {max_ineq_viol.item():.6e} (Should be <= 0)")
    
    # Compare with CVXPY
    print("\nSolving with CVXPY for ground truth...")
    try:
        x_cvx, val_cvx = solve_with_cvxpy(Q, c, A, b, E, d)
        print(f"CVXPY Optimal Value: {val_cvx:.6f}")
        
        diff_norm = np.linalg.norm(x_final.numpy() - x_cvx)
        print(f"Euclidean distance between NN and CVXPY solution: {diff_norm:.6e}")
        
        # Percentage difference in objective value
        obj_diff_percent = abs(final_loss.item() - val_cvx) / abs(val_cvx) * 100
        print(f"Objective Value Difference: {obj_diff_percent:.4f}%")
        
    except Exception as e:
        print(f"CVXPY failed or not installed properly: {e}")

if __name__ == "__main__":
    train()
