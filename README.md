# Constrained Optimization with Neural Networks for Network Design

This repository implements and compares different approaches for solving constrained optimization problems, specifically focusing on network flow design with entropy and linear costs. The goal is to explore how Neural Networks can be used to solve optimization problems with hard constraints (equality and inequality) and compare them against classical convex optimization solvers.

## Overview

The project solves a network flow problem where the objective is to minimize a cost function composed of:
1.  **Entropy Cost**: $x \log(x)$
2.  **Linear Cost**: $c^T x$

Subject to:
*   **Equality Constraints**: $Ax = b$ (Flow conservation and demand satisfaction)
*   **Inequality Constraints**: $x \ge 0$ (Non-negative flows)

## Implemented Methods

We compare four different approaches:

### 1. CVXPY (Benchmark)
*   **File**: `cvxpy_problem.py`
*   **Description**: Uses the `cvxpy` library to solve the convex optimization problem. This serves as the **ground truth** for accuracy and performance comparisons.

### 2. NN Projection (Feasible-Path)
*   **File**: `solve_custom_nn.py`
*   **Description**: A Deep Learning approach that guarantees feasibility by construction.
*   **Mechanism**: It uses a null-space decomposition ($x = Rw + u_{part}$) to satisfy equality constraints and a custom projection layer to strictly satisfy inequality constraints ($x \ge 0$).

### 3. Primal-Dual Direct
*   **File**: `solve_primal_dual.py`
*   **Description**: A direct numerical optimization method using the **Augmented Lagrangian** method.
*   **Mechanism**:
    *   **Primal Variable**: Optimizes a static tensor `w` directly.
    *   **Equality Constraints**: Satisfied by construction using the null-space method ($x = Rw + u_{part}$).
    *   **Inequality Constraints**: Handled via the Augmented Lagrangian method.
    *   **Multiplier Update**: The Lagrange multipliers $\lambda$ are updated using the dual ascent step:
        $$ \lambda_{k+1} = \max(0, \lambda_k + \rho \cdot (-x_k)) $$
        where $\rho$ is the penalty parameter which is also increased iteratively.

### 4. Primal-Dual NN Loop
*   **File**: `solve_nn_primal_dual.py`
*   **Description**: Similar to the Primal-Dual Direct method but parameterizes the solution using a **Neural Network**.
*   **Mechanism**:
    *   **Primal Variable**: The output of a Neural Network (`PrimalModel`) which predicts $x$ (or $w$).
    *   **Optimization**: The network weights are trained to minimize the Augmented Lagrangian.
    *   **Multiplier Update**: Uses the same explicit update rule for $\lambda$ as the Direct method:
        $$ \lambda_{k+1} = \max(0, \lambda_k + \rho \cdot (-x_k)) $$
    *   **Key Difference**: While the Direct method optimizes a specific instance's values, this approach learns a function (the neural network) to generate the solution. This architecture allows for **amortized optimization**, where the network could be conditioned on problem parameters (like $t$ or $u$) to solve a family of problems without retraining.

## Installation

1.  Clone the repository.
2.  Create a virtual environment (optional but recommended).
3.  Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Batch Experiments
To run a battery of experiments comparing all methods across multiple random problem instances:

```bash
python batch_experiments.py
```

This script will:
1.  Generate random problem instances (varying `t` and `u` parameters).
2.  Run all four solvers on each instance.
3.  Report the objective values, execution times, and the gap relative to the CVXPY solution.
4.  Display a final summary table with statistics.

### Running Individual Solvers
You can also run each solver individually to see detailed optimization logs and visualizations:

```bash
python cvxpy_problem.py
python solve_custom_nn.py
python solve_primal_dual.py
python solve_nn_primal_dual.py
```

## Project Structure

*   `problem_data.py`: Centralized logic for generating problem data (matrices $A, b$, vectors $c$, etc.).
*   `batch_experiments.py`: Main script for running comparative experiments.
*   `decoupled_basis.py`: Helper for computing the null-space basis $R$ and particular solution $u_{part}$.
*   `utils.py`: Utility functions for finding interior points and other helpers.
*   `visualization.py`: Functions for plotting flow matrices and results.
