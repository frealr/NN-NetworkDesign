import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results(x, n, a, method_name):
    """
    Visualizes the results of the optimization.
    
    Args:
        x (Tensor or np.array): Solution vector.
        n (int): Number of nodes.
        a (int): Number of arcs per pair (or total arcs in graph?). 
                 Based on problem, it seems 'a' is number of arcs in the graph, 
                 and we have 'a' variables for each (o,d).
        method_name (str): Name of the method for file naming.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    print(f"\n--- Visualization for {method_name} ---")
    
    # 1. Extract and Plot f(o,d) Matrix
    # f(o,d) are the first n*n variables
    f_od_flat = x[:n*n]
    f_od_matrix = f_od_flat.reshape(n, n)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(f_od_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Flow Matrix f(o,d) - {method_name}")
    plt.xlabel("Destination (d)")
    plt.ylabel("Origin (o)")
    
    filename = f"flow_matrix_{method_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Saved flow matrix heatmap to {filename}")
    plt.close()
    
    # 2. Show Active f(i,j,o,d) Arcs
    # f(i,j,o,d) starts at index 2*n^2
    # Structure: For each o, for each d, there are 'a' arc variables.
    offset = 2 * n**2
    
    print("\nActive Arcs per (o,d) pair (threshold > 1e-3):")
    for o in range(n):
        for d in range(n):
            # Skip if o == d (should be 0 flow)
            if o == d:
                continue
                
            # Get flow for this pair
            pair_flow = f_od_matrix[o, d]
            if pair_flow < 1e-3:
                continue
                
            print(f"\nPair (o={o}, d={d}) | Total Flow: {pair_flow:.4f}")
            
            # Extract arc flows for this pair
            # Index range: offset + (n*o + d)*a  to  offset + (n*o + d)*a + a
            start_idx = offset + (n * o + d) * a
            end_idx = start_idx + a
            arc_flows = x[start_idx:end_idx]
            
            # Print active arcs
            active_indices = np.where(arc_flows > 1e-3)[0]
            if len(active_indices) > 0:
                for arc_idx in active_indices:
                    print(f"  - Arc {arc_idx}: {arc_flows[arc_idx]:.4f}")
            else:
                print("  (No active arcs found > 1e-3)")

    # 3. Arc-Centric View: For each Arc, show flows from OD pairs
    print("\nFlows per Arc (threshold > 1e-3):")
    for k in range(a):
        print(f"\nArc {k}:")
        total_arc_flow = 0.0
        active_ods = []
        
        for o in range(n):
            for d in range(n):
                if o == d: continue
                
                # Index of arc k for pair (o,d)
                idx = offset + (n * o + d) * a + k
                flow = x[idx]
                
                if flow > 1e-3:
                    active_ods.append((o, d, flow))
                    total_arc_flow += flow
        
        if active_ods:
            print(f"  Total Flow: {total_arc_flow:.4f}")
            for o, d, flow in active_ods:
                print(f"  - OD ({o},{d}): {flow:.4f}")
        else:
            print("  (No active flow)")

if __name__ == "__main__":
    # Create dummy data for testing
    print("Running visualization.py in standalone mode with dummy data...")
    n = 3
    a = 6
    total_vars = 2*n**2 + a*n**2
    x_dummy = torch.rand(total_vars)
    # Make some flows significant
    x_dummy[:n*n] = torch.rand(n*n) * 10
    
    visualize_results(x_dummy, n, a, "Dummy_Test")
