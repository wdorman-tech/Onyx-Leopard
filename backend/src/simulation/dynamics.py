"""Pure math layer: production, relaxation, competition, market share.

No business logic, no I/O. Every function here is an established model:

  * Cobb-Douglas production        Y = A * K^α * L^β
  * First-order relaxation ODE     dy/dt = r * (target - y)   (exact solution)
  * N-species Lotka-Volterra       dN_i/dt = r_i N_i (1 - Σ_j a_ij N_j / K_i)
  * Multinomial-logit market share s_i ∝ q_i^β_q * m_i^β_m * p_i^(-ε)
  * Constant-elasticity demand     Q = Q_0 (P / P_0)^(-ε), so spend ∝ P^(1-ε)
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


def lv_capacities_for_equilibrium(alpha: np.ndarray, equilibrium: np.ndarray) -> np.ndarray:
    """Carrying capacities K = A · N* that make N* the interior LV equilibrium.

    At equilibrium every bracket in the LV equation is zero: Σ_j a_ij N*_j = K_i.
    Choosing K this way puts the rest point at any strictly positive N*, so no
    population there is competitively excluded.
    """
    return alpha @ equilibrium


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


def own_price_elasticity(
    share_elasticity: float, share: float, market_elasticity: float, unit_share: float
) -> float:
    """Elasticity of one firm's unit demand to its own price (positive magnitude).

    Units q_i = S(P̄) s_i / p_i, with category spend S ∝ P̄^(1-ε_m), logit share
    s_i ∝ p_i^(-ε_s), and ∂ln P̄/∂ln p_i = u_i (the firm's share of units):
    ε_i = 1 + ε_s (1 - s_i) - (1 - ε_m) u_i. For a monopolist (s = u = 1) this
    is the category elasticity ε_m.
    """
    return 1.0 + share_elasticity * (1.0 - share) - (1.0 - market_elasticity) * unit_share


def nash_share_elasticity(margin: float, share: float, market_elasticity: float) -> float:
    """Share-price elasticity ε_s at which `margin` is the Bertrand-Nash markup.

    Inverts the Lerner condition (p - c) / p = 1 / ε_i at symmetric prices
    (u_i = s_i), the way empirical IO recovers demand elasticities from
    observed markups: ε_s = (1/m - 1 + (1 - ε_m) s) / (1 - s). Needs s < 1.
    """
    return (1.0 / max(margin, _EPS) - 1.0 + (1.0 - market_elasticity) * share) / (1.0 - share)


def constant_elasticity_spend(
    base_spend: float, price: float, reference_price: float, elasticity: float
) -> float:
    """Total spend at `price` under Q = Q_0 (P/P_0)^(-ε): spend = S_0 (P/P_0)^(1-ε).

    ε < 1 (inelastic): a lower price shrinks spend. ε > 1: it grows spend.
    ε = 1: spend does not depend on price.
    """
    ratio = max(price, _EPS) / max(reference_price, _EPS)
    return base_spend * ratio ** (1.0 - elasticity)
