"""N-species Lotka-Volterra competition model, vectorized."""

import logging

import numpy as np
from scipy.integrate import solve_ivp

from biosim.math.solver import euler_step

logger = logging.getLogger(__name__)


def lotka_volterra_rhs(
    t: float,
    populations: np.ndarray,
    growth_rates: np.ndarray,
    carrying_capacities: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    N-species Lotka-Volterra competition RHS.
    dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j) / K_i)

    populations: (n_species,)
    growth_rates: (n_species,)
    carrying_capacities: (n_species,)
    alpha: (n_species, n_species) — alpha[i,i] = 1.0 (self-competition)
    Returns: dpopulations/dt (n_species,)
    """
    n = np.maximum(populations, 0.0)
    k = np.maximum(carrying_capacities, 1e-10)

    competition_effect = alpha @ n
    dn = growth_rates * n * (1.0 - competition_effect / k)
    return dn


def step_competition(
    populations: np.ndarray,
    growth_rates: np.ndarray,
    carrying_capacities: np.ndarray,
    alpha: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Advance Lotka-Volterra competition by dt using solve_ivp. Returns new populations."""
    try:
        sol = solve_ivp(
            lotka_volterra_rhs,
            [0.0, dt],
            populations,
            method="RK45",
            args=(growth_rates, carrying_capacities, alpha),
            rtol=1e-6,
            atol=1e-9,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        result = sol.y[:, -1]
    except Exception:
        logger.warning("solve_ivp RK45 failed for competition, falling back to Euler step")
        result = euler_step(
            lotka_volterra_rhs, populations, dt, growth_rates, carrying_capacities, alpha
        )

    return np.maximum(result, 0.0)


def build_competition_matrix(
    n_species: int,
    base_competition: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create alpha matrix: 1.0 on diagonal, base_competition off-diagonal
    with small random perturbation (+-10%).
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbation = rng.uniform(-0.1, 0.1, size=(n_species, n_species))
    alpha = np.full((n_species, n_species), base_competition) + base_competition * perturbation

    np.fill_diagonal(alpha, 1.0)
    mask = ~np.eye(n_species, dtype=bool)
    alpha[mask] = np.clip(alpha[mask], 0.0, 0.99)

    return alpha


def coexistence_check(
    growth_rates: np.ndarray,
    carrying_capacities: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Check which species can coexist at equilibrium.

    For 2-species: mutual coexistence when K1/K2 > alpha12 and K2/K1 > alpha21.
    For N-species: a species can persist if the equilibrium system has a positive solution.
    We check if the linear system alpha @ N* = K has all-positive N*.
    """
    n_species = len(growth_rates)

    try:
        n_star = np.linalg.solve(alpha, carrying_capacities)
        return n_star > 0.0
    except np.linalg.LinAlgError:
        # Singular matrix — no clean equilibrium, mark all as uncertain (False)
        return np.zeros(n_species, dtype=bool)
