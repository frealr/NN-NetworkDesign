import numpy as np
import torch
from problem_data import create_problem_data
from cvxpy_problem import solve_cvxpy
from solve_custom_nn import solve_custom_nn
from solve_primal_dual import solve_primal_dual
from solve_nn_primal_dual import solve_nn_primal_dual
import time

def run_batch_experiments(n_runs=5):
    print(f"Starting batch experiments with {n_runs} runs...")
    print("Parameters: t_vec and u_vec in [0.5, 2.5], diagonal u_vec = 100")
    
    # Lists to store metrics for statistics
    results = []
    gaps_nn = []
    gaps_pd = []
    gaps_pd_nn = []
    
    times_cvxpy = []
    times_nn = []
    times_pd = []
    times_pd_nn = []

    for i in range(n_runs):
        print(f"\n{'='*20} Run {i+1}/{n_runs} {'='*20}")
        
        # Generate random parameters
        # t_vec: size 6
        t_vec = np.random.uniform(0.5, 2.5, size=6)
        
        # u_vec: size 9
        u_vec = np.random.uniform(0.5, 2.5, size=9)
        # Set diagonals to 100 (indices 0, 4, 8 for n=3)
        u_vec[0] = 100
        u_vec[4] = 100
        u_vec[8] = 100
        
        print(f"t_vec: {t_vec}")
        print(f"u_vec: {u_vec}")
        
        # Create problem data
        data = create_problem_data(t_vec, u_vec)
        
        run_stats = {
            'run_id': i,
            't_vec': t_vec,
            'u_vec': u_vec
        }
        
        # 1. CVXPY
        print("\n--- CVXPY ---")
        try:
            start_time = time.time()
            x_cvx, obj_cvx = solve_cvxpy(data)
            duration = time.time() - start_time
            run_stats['cvxpy_obj'] = obj_cvx
            run_stats['cvxpy_time'] = duration
            times_cvxpy.append(duration)
            print(f"CVXPY Obj: {obj_cvx:.4f} (Time: {duration:.2f}s)")
        except Exception as e:
            print(f"CVXPY Failed: {e}")
            run_stats['cvxpy_obj'] = float('nan')
            run_stats['cvxpy_time'] = 0
            obj_cvx = float('nan')

        # 2. NN Projection
        print("\n--- NN Projection ---")
        try:
            start_time = time.time()
            x_nn, obj_nn = solve_custom_nn(data)
            duration = time.time() - start_time
            run_stats['nn_proj_obj'] = obj_nn
            run_stats['nn_proj_time'] = duration
            times_nn.append(duration)
            print(f"NN Proj Obj: {obj_nn:.4f} (Time: {duration:.2f}s)")
        except Exception as e:
            print(f"NN Projection Failed: {e}")
            run_stats['nn_proj_obj'] = float('nan')
            run_stats['nn_proj_time'] = 0

        # 3. Primal-Dual Direct
        print("\n--- Primal-Dual Direct ---")
        try:
            start_time = time.time()
            x_pd, obj_pd = solve_primal_dual(data)
            duration = time.time() - start_time
            run_stats['pd_direct_obj'] = obj_pd
            run_stats['pd_direct_time'] = duration
            times_pd.append(duration)
            print(f"PD Direct Obj: {obj_pd:.4f} (Time: {duration:.2f}s)")
        except Exception as e:
            print(f"Primal-Dual Direct Failed: {e}")
            run_stats['pd_direct_obj'] = float('nan')
            run_stats['pd_direct_time'] = 0

        # 4. Primal-Dual NN Loop
        print("\n--- Primal-Dual NN Loop ---")
        try:
            start_time = time.time()
            x_pd_nn, obj_pd_nn = solve_nn_primal_dual(data)
            duration = time.time() - start_time
            run_stats['pd_nn_obj'] = obj_pd_nn
            run_stats['pd_nn_time'] = duration
            times_pd_nn.append(duration)
            print(f"PD NN Loop Obj: {obj_pd_nn:.4f} (Time: {duration:.2f}s)")
        except Exception as e:
            print(f"Primal-Dual NN Loop Failed: {e}")
            run_stats['pd_nn_obj'] = float('nan')
            run_stats['pd_nn_time'] = 0
            
        results.append(run_stats)
        
        # Calculate gaps if CVXPY succeeded
        if not np.isnan(obj_cvx):
            gap_nn = abs(run_stats['nn_proj_obj'] - obj_cvx) / abs(obj_cvx) * 100
            gap_pd = abs(run_stats['pd_direct_obj'] - obj_cvx) / abs(obj_cvx) * 100
            gap_pd_nn = abs(run_stats['pd_nn_obj'] - obj_cvx) / abs(obj_cvx) * 100
            
            gaps_nn.append(gap_nn)
            gaps_pd.append(gap_pd)
            gaps_pd_nn.append(gap_pd_nn)
            
            print(f"\nGaps vs CVXPY:")
            print(f"NN Proj: {gap_nn:.4f}%")
            print(f"PD Direct: {gap_pd:.4f}%")
            print(f"PD NN Loop: {gap_pd_nn:.4f}%")

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Run':<5} | {'CVXPY Obj':<10} | {'NN Obj':<10} | {'PD Obj':<10} | {'PD NN Obj':<10} | {'CVXPY(s)':<8} | {'NN(s)':<8} | {'PD(s)':<8} | {'PD NN(s)':<8}")
    print("-" * 100)
    
    for res in results:
        print(f"{res['run_id']:<5} | {res['cvxpy_obj']:<10.4f} | {res['nn_proj_obj']:<10.4f} | {res['pd_direct_obj']:<10.4f} | {res['pd_nn_obj']:<10.4f} | {res['cvxpy_time']:<8.2f} | {res['nn_proj_time']:<8.2f} | {res['pd_direct_time']:<8.2f} | {res['pd_nn_time']:<8.2f}")

    print("\n" + "="*80)
    print("STATISTICS (Mean +/- Std Dev)")
    print("="*80)
    
    # Helper to print stats
    def print_stat(name, values, unit=""):
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{name:<25}: {mean_val:.4f} +/- {std_val:.4f} {unit}")
        else:
            print(f"{name:<25}: N/A")

    print("--- Objective Value Gaps vs CVXPY ---")
    print_stat("NN Projection Gap", gaps_nn, "%")
    print_stat("Primal-Dual Direct Gap", gaps_pd, "%")
    print_stat("Primal-Dual NN Loop Gap", gaps_pd_nn, "%")
    
    print("\n--- Execution Times ---")
    print_stat("CVXPY Time", times_cvxpy, "s")
    print_stat("NN Projection Time", times_nn, "s")
    print_stat("Primal-Dual Direct Time", times_pd, "s")
    print_stat("Primal-Dual NN Loop Time", times_pd_nn, "s")

if __name__ == "__main__":
    run_batch_experiments()
