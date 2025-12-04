import torch
import numpy as np
from decoupled_basis import get_decoupled_u_R

def verify_sparse_implementation():
    print("--- Verifying Sparse Matrix Multiplication ---")
    n = 3
    a = 6
    
    # 1. Get Dense R
    print("Computing decoupled basis (Dense)...")
    u, R_dense = get_decoupled_u_R(n, a)
    print(f"R_dense shape: {R_dense.shape}")
    
    # 2. Convert to Sparse
    print("Converting to Sparse Tensor...")
    # PyTorch sparse tensors are usually indices + values
    # We want to use it for: x = w @ R.T  <=>  x = (R @ w.T).T
    # It's often easier to store R as sparse and do: x = (R_sparse @ w_col).T
    
    indices = torch.nonzero(R_dense).t()
    values = R_dense[indices[0], indices[1]]
    shape = R_dense.shape
    
    R_sparse = torch.sparse_coo_tensor(indices, values, shape)
    
    # 3. Generate random w
    dim_w = R_dense.shape[1]
    batch_size = 5
    w = torch.randn(batch_size, dim_w)
    
    # 4. Compare Operations
    print(f"Testing with batch size {batch_size}...")
    
    # Dense Operation: w @ R.T
    # (batch, dim_w) @ (dim_w, n_vars) -> (batch, n_vars)
    x_dense = w @ R_dense.T
    
    # Sparse Operation
    # PyTorch sparse matmul (spmm) is usually: Sparse @ Dense
    # We want: w @ R.T
    # This is equivalent to: (R @ w.T).T
    # R is (n_vars, dim_w), w.T is (dim_w, batch)
    # Result is (n_vars, batch). Transpose to get (batch, n_vars)
    
    x_sparse_t = torch.sparse.mm(R_sparse, w.t())
    x_sparse = x_sparse_t.t()
    
    # 5. Check Difference
    diff = torch.abs(x_dense - x_sparse)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax Difference: {max_diff:.6e}")
    print(f"Mean Difference: {mean_diff:.6e}")
    
    if max_diff < 1e-6:
        print("\nSUCCESS: Sparse multiplication matches dense multiplication.")
    else:
        print("\nFAILURE: Significant difference detected.")

if __name__ == "__main__":
    verify_sparse_implementation()
