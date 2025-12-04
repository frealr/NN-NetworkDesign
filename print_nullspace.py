import torch
import numpy as np
import pandas as pd
from utils import get_equality_solution_and_null_basis

def print_nullspace():
    print("--- Computing and Printing Null Space Vectors (R) ---")
    n = 3
    a = 6
    
    # 1. Reconstruct A matrix
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

    # Demand Constraints
    for o in range(n):
      for d in range(n):
        cons = np.zeros(total_length)
        cons[n*o+d] = 1
        cons[n**2 + n*o+d] = 1
        A_list.append(cons)
        
    # Flow Conservation
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
                    if i == o: cons[n*o + d] = -1
                    if i == d: cons[n*o + d] = 1
                    A_list.append(cons)

    # Zero Constraints
    for o in range(n):
        cons = np.zeros(total_length)
        cons[n*o + o] = 1
        A_list.append(cons)
        start_idx = 2*n**2 + a*(n*o + o)
        for k in range(a):
            cons = np.zeros(total_length)
            cons[start_idx + k] = 1
            A_list.append(cons)

    A = np.array(A_list)
    b = np.zeros(A.shape[0])
    
    A_tensor = torch.tensor(A, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    
    # 2. Compute Null Space Basis R
    _, R = get_equality_solution_and_null_basis(A_tensor, b_tensor)
    R_np = R.numpy()
    
    print(f"R shape: {R_np.shape}")
    
    # 3. Print R
    # Set pandas options to display all columns/rows if needed
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    df = pd.DataFrame(R_np)
    df.columns = [f"w_{i}" for i in range(R_np.shape[1])]
    
    # Create row labels
    row_labels = []
    # f(o,d)
    for o in range(n):
        for d in range(n):
            row_labels.append(f"f({o},{d})")
    # fext(o,d)
    for o in range(n):
        for d in range(n):
            row_labels.append(f"fext({o},{d})")
    # fijod
    for o in range(n):
        for d in range(n):
            for k in range(a):
                row_labels.append(f"arc{k}_({o},{d})")
                
    df.index = row_labels
    
    print("\nNull Space Basis Vectors (Columns are basis vectors w_i):")
    print(df.iloc[:20, :10]) # Print a snippet
    
    print("\nSaving full matrix to 'nullspace_matrix.csv'...")
    df.to_csv("nullspace_matrix.csv")
    print("Done.")

if __name__ == "__main__":
    print_nullspace()
