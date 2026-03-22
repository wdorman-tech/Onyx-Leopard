from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


def lotka_volterra_rhs(
    t: float,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """N-species Lotka-Volterra competition ODE.

    dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j) / K_i)

    Args:
        y: population vector (N_1, ..., N_n)
        r: growth rate vector
        K: carrying capacity vector
        alpha: competition matrix (n x n), alpha_ii = 1
    """
    N = np.maximum(y, 0.0)
    interaction = alpha @ N
    growth = r * N * (1.0 - interaction / np.maximum(K, 1e-6))
    return growth


def compute_competition_matrix(
    n_company: int,
    n_competitors: int,
    competitor_market_shares: list[float],
    competitor_relative_costs: list[float],
    alpha_default: float = 0.5,
) -> np.ndarray:
    """Derive competition matrix alpha from competitor data.

    alpha_ij measures how much species j affects species i.
    Diagonal = 1.0 (self-limitation).
    Off-diagonal derived from relative market share and cost position.
    """
    n = n_company + n_competitors
    alpha = np.eye(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j < n_company:
                alpha[i, j] = alpha_default
            else:
                comp_idx = j - n_company
                if comp_idx < len(competitor_market_shares):
                    share = competitor_market_shares[comp_idx]
                    cost = competitor_relative_costs[comp_idx] if comp_idx < len(competitor_relative_costs) else 1.0
                    alpha[i, j] = alpha_default * (1.0 + share) / max(cost, 0.1)
                else:
                    alpha[i, j] = alpha_default

    return alpha


def step_competition(
    populations: np.ndarray,
    growth_rates: np.ndarray,
    capacities: np.ndarray,
    alpha: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Solve Lotka-Volterra competition for one timestep.

    Returns updated population vector.
    """
    n = len(populations)
    if n == 0:
        return populations

    y0 = np.maximum(populations, 0.0)
    r = np.maximum(growth_rates, 0.0)
    K = np.maximum(capacities, 1.0)

    sol = solve_ivp(
        lotka_volterra_rhs,
        [0, dt],
        y0,
        args=(r, K, alpha),
        method="RK45",
        max_step=0.5,
    )

    if sol.success and sol.y.shape[1] > 0:
        result = np.maximum(sol.y[:, -1], 0.0)
    else:
        logger.warning("Lotka-Volterra solve failed, using Euler step")
        dy = lotka_volterra_rhs(0, y0, r, K, alpha)
        result = np.maximum(y0 + dy * dt, 0.0)

    return result
