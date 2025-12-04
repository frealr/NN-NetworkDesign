import torch
import numpy as np
from solve_custom_nn import solve_custom_nn
from solve_primal_dual import solve_primal_dual
from solve_nn_primal_dual import solve_nn_primal_dual

def relative_distance_obj(val1, val2):
    """Calculates percentage difference relative to the first value."""
    return abs(val1 - val2) / abs(val1) * 100

from cvxpy_problem import solve_cvxpy
from visualization import visualize_results

def normalized_absolute_distance(vec1, vec2):
    """
    Calculates sum of absolute differences divided by the sum of the reference vector elements.
    Metric: sum(|v1 - v2|) / sum(|v1|)
    """
    # Handle nan in vec1 or vec2
    if torch.isnan(vec1).any() or torch.isnan(vec2).any():
        return float('nan')
        
    abs_diff = torch.sum(torch.abs(vec1 - vec2))
    sum_val = torch.sum(torch.abs(vec1))
    return (abs_diff / sum_val).item()

def main():
    n = 3
    a = 6
    
    print("--- Running CVXPY (Ground Truth) ---")
    x_cvx, obj_cvx = solve_cvxpy()
    visualize_results(x_cvx, n, a, "CVXPY")
    
    print("\n--- Running Approach 1: NN Projection (solve_custom_nn) ---")
    x1, obj1 = solve_custom_nn()
    
    print("\n--- Running Approach 2: Primal-Dual Direct (solve_primal_dual) ---")
    x2, obj2 = solve_primal_dual()
    
    print("\n--- Running Approach 3: Primal-Dual NN Loop (solve_nn_primal_dual) ---")
    x3, obj3 = solve_nn_primal_dual()
    
    print("\n" + "="*50)
    print("COMPARISON REPORT")
    print("="*50)
    
    approaches = [
        ("CVXPY", x_cvx, obj_cvx),
        ("NN Projection", x1, obj1),
        ("Primal-Dual Direct", x2, obj2),
        ("Primal-Dual NN Loop", x3, obj3)
    ]
    
    # 1. Objective Value Comparison
    print("\nObjective Values:")
    for name, _, obj in approaches:
        print(f"{name}: {obj:.4f}")
        
    print("\nObjective Value Distances (%):")
    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            name1, _, val1 = approaches[i]
            name2, _, val2 = approaches[j]
            dist = relative_distance_obj(val1, val2)
            print(f"{name1} vs {name2}: {dist:.4f}%")

    # 2. Solution Vector Comparison
    print("\nSolution Vector Distances (Sum(|diff|) / Sum(|ref|)):")
    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            name1, vec1, _ = approaches[i]
            name2, vec2, _ = approaches[j]
            dist = normalized_absolute_distance(vec1, vec2)
            print(f"{name1} vs {name2}: {dist:.4f}")

if __name__ == "__main__":
    main()
