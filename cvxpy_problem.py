import cvxpy as cp
import torch
import numpy as np
from scipy.linalg import null_space

# --- Datos ---
n = 3
a = 6

entr_coefs = np.concatenate([np.ones(2 * n**2), np.zeros(a * n**2)])

t = np.array([1, 2, 1, 1.5, 2, 1.5])
c = np.concatenate([np.zeros(2 * n**2), np.tile(t, n**2)])

u = np.array([100, 2, 4, 2,100, 2, 4, 2,100])
uext = np.concatenate([np.zeros(n**2), u, np.zeros(a * n**2)])

# --- Zero out coefficients for o=d ---
for o in range(n):
    d = o
    # Block 1: f(o,d)
    idx1 = n*o + d
    entr_coefs[idx1] = 0
    c[idx1] = 0
    uext[idx1] = 0
    
    # Block 2: fext(o,d)
    idx2 = n**2 + n*o + d
    entr_coefs[idx2] = 0
    c[idx2] = 0
    uext[idx2] = 0
    
    # Block 3: f(i,j,o,d)
    start_idx3 = 2*n**2 + (n*o + d)*a
    for k in range(a):
        idx3 = start_idx3 + k
        entr_coefs[idx3] = 0
        c[idx3] = 0
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

# 1. Demand Constraints
for o in range(n):
  for d in range(n):
    cons = np.zeros(2*n**2 + n**2*a)
    cons[n*o+d] = 1
    cons[n**2 + n*o+d] = 1
    A_list.append(cons)
    
A_aux = np.array(A_list)
n_rows_A_c1 = A_aux.shape[0]
b = np.ones((n_rows_A_c1, 1))


# 2. Flow Conservation Constraints
for o in range(n):          # o = 0..n-1
    for d in range(n):      # d = 0..n-1
        if o != d:
            for i in range(n):   # i = 0..n-1

                current_length = 2*n**2 + a*n*o + d*a

                cons = np.concatenate([
                    np.zeros(2*n**2),
                    np.zeros(n*a*(o) + a*d),
                    Aout[i] - Ain[i],                     # i ya es 0-based
                    np.zeros(total_length - current_length - a)
                ])

                # Ajustes en función de i
                if i == o:
                    cons[n*o + d] = -1

                if i == d:
                    cons[n*o + d] = 1

                # Ahora sí: cada cons es una fila nueva de A
                A_list.append(cons)

# 3. New Constraints: f(o,d) = 0 and f(i,j,o,d) = 0 if o == d
for o in range(n):
    # f(o,o) = 0 (Block 1)
    cons = np.zeros(total_length)
    cons[n*o + o] = 1
    A_list.append(cons)
    
    # f(i,j,o,o) = 0 (Block 3)
    start_idx = 2*n**2 + a*(n*o + o)
    for k in range(a):
        cons = np.zeros(total_length)
        cons[start_idx + k] = 1
        A_list.append(cons)

A = np.array(A_list)
n_rows_A = A.shape[0]
b = b.flatten()
# Pad b with zeros for all constraints added after the first block
b = np.concatenate([b, np.zeros(n_rows_A - n_rows_A_c1)])
b = b.flatten()

print(f"DEBUG: A shape: {A.shape}")
print(f"DEBUG: b shape: {b.shape}")

epsilon = 1e-3

# Solve for particular solution (initial check)
x_var = cp.Variable(A.shape[1])
objective = cp.Minimize(cp.sum_squares(A@x_var - b.flatten()))
constraints = [x_var >= epsilon, x_var <= 1-epsilon]

problem = cp.Problem(objective, constraints)
problem.solve()

x_particular = x_var.value

def solve_cvxpy():
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

    print("status:", prob.status)
    print("objective:", prob.value)
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

    null_space_basis = null_space(A)

    print("null_space_basis shape:", null_space_basis.shape)
    
    from visualization import visualize_results
    # n=3, a=6 are defined globally in the file
    visualize_results(x_cvx, n, a, "CVXPY")

