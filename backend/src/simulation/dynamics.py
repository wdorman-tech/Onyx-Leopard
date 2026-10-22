"""Pure math layer: production, relaxation, competition, market share.

No business logic, no I/O. Every function here is an established model:

  * Cobb-Douglas production        Y = A * K^α * L^β
  * First-order relaxation ODE     dy/dt = r * (target - y)   (exact solution)
  * N-species Lotka-Volterra       dN_i/dt = r_i N_i (1 - Σ_j a_ij N_j / K_i)
  * Multinomial-logit market share s_i ∝ q_i^β_q * m_i^β_m * p_i^(-ε)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.integrate import solve_ivp

log = logging.getLogger(__name__)

_EPS = 1e-10

#: Integrators tried in order for the Lotka-Volterra step. BDF covers stiff
#: regimes (one firm far above capacity); Euler is the last resort.
LV_INTEGRATORS: tuple[str, ...] = ("RK45", "BDF")


def cobb_douglas(tfp: float, capital: float, labor: float, alpha: float, beta: float) -> float:
    """Y = A * K^α * L^β. α + β != 1 allows non-constant returns to scale."""
    return max(tfp, _EPS) * max(capital, _EPS) ** alpha * max(labor, _EPS) ** beta


def relax(current: float, target: float, rate: float, dt: float = 1.0) -> float:
    """Exact solution of dy/dt = rate * (target - y) over dt."""
    return target + (current - target) * math.exp(-rate * dt)


def competition_matrix(
    n: int, base: float, perturbation: float, rng: np.random.Generator
) -> np.ndarray:
    """LV interaction matrix: 1 on the diagonal, `base` ± `perturbation` off it."""
    noise = rng.uniform(-perturbation, perturbation, size=(n, n))
    alpha = np.clip(base * (1.0 + noise), 0.0, 0.99)
    np.fill_diagonal(alpha, 1.0)
    return alpha


def _lv_rhs(
    _t: float, n: np.ndarray, r: np.ndarray, k: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    n = np.maximum(n, 0.0)
    return r * n * (1.0 - (alpha @ n) / np.maximum(k, _EPS))


def lotka_volterra_step(
    populations: np.ndarray,
    growth_rates: np.ndarray,
    carrying_capacities: np.ndarray,
    alpha: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Advance N-species competition by dt. RK45 → BDF → Euler."""
    args = (growth_rates, carrying_capacities, alpha)
    for method in LV_INTEGRATORS:
        sol = solve_ivp(_lv_rhs, (0.0, dt), populations, method=method, args=args)
        if sol.success:
            return np.maximum(sol.y[:, -1], 0.0)
        log.warning("LV %s integration failed (%s); trying next integrator", method, sol.message)
    return np.maximum(populations + dt * _lv_rhs(0.0, populations, *args), 0.0)


def logit_shares(
    quality: list[float],
    marketing: list[float],
    price: list[float],
    weight: list[float],
    *,
    quality_exponent: float,
    marketing_exponent: float,
    price_elasticity: float,
) -> list[float]:
    """Multinomial-logit share with a constant-elasticity price term.

    `weight` carries any extra multiplicative attraction (LV market strength,
    shock modifiers). Returns shares summing to 1.
    """
    attraction = [
        max(q, _EPS) ** quality_exponent
        * max(m, _EPS) ** marketing_exponent
        * max(p, _EPS) ** (-price_elasticity)
        * max(w, _EPS)
        for q, m, p, w in zip(quality, marketing, price, weight, strict=True)
    ]
    total = sum(attraction)
    return [a / total for a in attraction]
