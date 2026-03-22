from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def replicator_rhs(
    t: float,
    x: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Replicator dynamics ODE.

    dx_i/dt = x_i * [(Ax)_i - x^T * A * x]

    x: strategy frequency vector (sums to 1)
    A: payoff matrix (n x n)

    Strategies that perform above average grow; below average shrink.
    """
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total < 1e-9:
        return np.zeros_like(x)
    x = x / total  # renormalize

    Ax = A @ x
    avg_fitness = float(x @ Ax)
    dx = x * (Ax - avg_fitness)
    return dx


def step_replicator(
    frequencies: np.ndarray,
    payoff_matrix: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Solve replicator dynamics for one timestep.

    Args:
        frequencies: current strategy frequency vector (sums to 1)
        payoff_matrix: n x n payoff matrix
        dt: timestep

    Returns:
        Updated frequency vector (normalized to sum to 1).
    """
    n = len(frequencies)
    if n == 0:
        return frequencies

    x0 = np.maximum(frequencies, 1e-6)
    x0 = x0 / x0.sum()

    sol = solve_ivp(
        replicator_rhs,
        [0, dt],
        x0,
        args=(payoff_matrix,),
        method="RK45",
        max_step=0.25,
    )

    if sol.success and sol.y.shape[1] > 0:
        result = np.maximum(sol.y[:, -1], 0.0)
    else:
        dx = replicator_rhs(0, x0, payoff_matrix)
        result = np.maximum(x0 + dx * dt, 0.0)

    total = result.sum()
    if total > 1e-9:
        result = result / total
    else:
        result = np.ones(n) / n

    return result


def build_strategy_payoff_matrix(
    strategy_performance: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Build payoff matrix from observed strategy performance.

    Simple model: strategies compete for emphasis. A strategy's payoff
    against another is proportional to its performance relative to the other.

    Returns:
        (payoff_matrix, strategy_names)
    """
    names = list(strategy_performance.keys())
    n = len(names)
    if n == 0:
        return np.array([[]]), names

    perf = np.array([strategy_performance[name] for name in names])
    perf = np.maximum(perf, 0.0)

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = perf[i]
            else:
                # payoff of strategy i when facing j:
                # higher if i outperforms j
                A[i, j] = perf[i] - 0.5 * perf[j]

    return A, names
