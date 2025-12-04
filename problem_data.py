import numpy as np
import torch

def create_problem_data(t_vec, u_vec, n=3, a=6):
    """
    Generates the problem data (matrices and vectors) based on input t_vec and u_vec.
    
    Args:
        t_vec (np.array): Vector of size 6 (for n=3, a=6 case).
        u_vec (np.array): Vector of size 9 (for n=3 case).
        n (int): Number of nodes.
        a (int): Number of arcs per pair.
        
    Returns:
        dict: Dictionary containing all problem data.
    """
    
    entr_coefs = np.concatenate([np.ones(2 * n**2), np.zeros(a * n**2)])
    
    # c_vec construction
    # Original: c = np.concatenate([np.zeros(2 * n**2), np.tile(t, n**2)])
    c_vec = np.concatenate([np.zeros(2 * n**2), np.tile(t_vec, n**2)])
    
    # uext construction
    # Original: uext = np.concatenate([np.zeros(n**2), u, np.zeros(a * n**2)])
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

    # 1. Demand Constraints
    for o in range(n):
      for d in range(n):
        cons = np.zeros(2*n**2 + n**2*a)
        cons[n*o+d] = 1
        cons[n**2 + n*o+d] = 1
        A_list.append(cons)
        
    A_aux = np.array(A_list)
    n_rows_A_c1 = A_aux.shape[0]
    b_vec = np.ones((n_rows_A_c1, 1))

    # 2. Flow Conservation Constraints
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
    b_vec = b_vec.flatten()
    b_vec = np.concatenate([b_vec, np.zeros(n_rows_A - n_rows_A_c1)])
    b_vec = b_vec.flatten()
    
    return {
        'n': n,
        'a': a,
        'entr_coefs': entr_coefs,
        'c': c_vec,
        'uext': uext,
        'A': A,
        'b': b_vec
    }
