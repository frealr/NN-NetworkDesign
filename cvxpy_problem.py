import cvxpy as cp
import torch
import numpy as np
from scipy.linalg import null_space
from problem_data import create_problem_data

def solve_cvxpy(data=None):
    if data is None:
        # Default data if none provided
        t = np.array([1, 2, 1, 1.5, 2, 1.5])
        u = np.array([100, 0.9, 1.1, 1.3,100, 1.4, 1, 1.2,100])
        data = create_problem_data(t, u)

    n = data['n']
    a = data['a']
    entr_coefs = data['entr_coefs']
    c = data['c']
    uext = data['uext']
    A = data['A']
    b = data['b']

    # === Variable ===
    x = cp.Variable(2*(n**2) + a*(n**2))

    # === Restricciones ===
    constraints = [
        A @ x == b.flatten(),
        x >= 0
    ]

    # === Función objetivo ===
    objective_expr = (
        - cp.sum(cp.entr(cp.multiply(entr_coefs, x)) - cp.multiply(entr_coefs, x))
        + c @ x + uext @ x
    )

    # === Problema ===
    prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    try:
        res = prob.solve(verbose=False)
    except cp.error.SolverError:
        # Fallback to SCS if default fails (though usually default is fine)
        res = prob.solve(solver=cp.SCS, verbose=False)

    # print("status:", prob.status)
    # print("objective:", prob.value)
    # print("x =", x.value)
    
    # Handle inf objective if x is valid but objective calc fails
    if prob.value == float('inf') or prob.value == float('-inf'):
        # Manually calc objective
        x_val = torch.tensor(x.value, dtype=torch.float32)
        y = x_val * torch.tensor(entr_coefs, dtype=torch.float32)
        y = torch.clamp(y, min=1e-16)
        entropy_term = torch.sum(y * torch.log(y) + y)
        linear_term = torch.dot(x_val, torch.tensor(c, dtype=torch.float32) + torch.tensor(uext, dtype=torch.float32))
        obj_val = (entropy_term + linear_term).item()
        print(f"Recalculated Objective: {obj_val}")
        return torch.tensor(x.value, dtype=torch.float32), obj_val

    return torch.tensor(x.value, dtype=torch.float32), prob.value

if __name__ == "__main__":
    x_cvx, obj_val = solve_cvxpy()
    print(f"Objective: {obj_val}")
    
    # For visualization, we need n and a, which are in the default data
    t = np.array([1, 2, 1, 1.5, 2, 1.5])
    u = np.array([100, 0.9, 1.1, 1.3,100, 1.4, 1, 1.2,100])
    data = create_problem_data(t, u)
    
    from visualization import visualize_results
    visualize_results(x_cvx, data['n'], data['a'], "CVXPY")

