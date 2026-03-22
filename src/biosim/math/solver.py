"""Batched ODE solver facade — composes growth into a single solve_ivp call."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from biosim.math.growth import VARS_PER_AGENT, growth_rhs

logger = logging.getLogger(__name__)


def euler_step(
    rhs_func: Callable[..., np.ndarray],
    y0: np.ndarray,
    dt: float,
    *args: Any,
) -> np.ndarray:
    """Simple forward Euler fallback."""
    dy = rhs_func(0.0, y0, *args)
    return y0 + dt * dy


def solve_tick(
    firm_size: np.ndarray,
    cash: np.ndarray,
    growth_rate: np.ndarray,
    carrying_capacity: np.ndarray,
    revenue: np.ndarray,
    fixed_costs: np.ndarray,
    variable_cost_rate: np.ndarray,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Solves all ODE systems for one tick (dt = 1 week).
    Returns dict with keys: 'firm_size', 'cash', 'growth_rate' — each (n_agents,).

    Fallback chain: RK45 -> BDF (if stiff) -> Euler (if solver fails).
    """
    n_agents = len(firm_size)
    y0 = np.column_stack([firm_size, cash, growth_rate]).ravel()
    rhs_args = (n_agents, carrying_capacity, revenue, fixed_costs, variable_cost_rate)

    y_final = None

    try:
        sol = solve_ivp(
            growth_rhs,
            [0.0, dt],
            y0,
            method="RK45",
            args=rhs_args,
            rtol=1e-6,
            atol=1e-9,
        )
        if sol.success:
            y_final = sol.y[:, -1]
        else:
            raise RuntimeError(f"RK45 failed: {sol.message}")
    except Exception:
        logger.warning("RK45 failed, trying BDF for stiff system")

    if y_final is None:
        try:
            sol = solve_ivp(
                growth_rhs,
                [0.0, dt],
                y0,
                method="BDF",
                args=rhs_args,
                rtol=1e-6,
                atol=1e-9,
            )
            if sol.success:
                y_final = sol.y[:, -1]
            else:
                raise RuntimeError(f"BDF failed: {sol.message}")
        except Exception:
            logger.warning("BDF also failed, falling back to Euler step")

    if y_final is None:
        y_final = euler_step(growth_rhs, y0, dt, *rhs_args)

    result = y_final.reshape(n_agents, VARS_PER_AGENT)
    return {
        "firm_size": result[:, 0].copy(),
        "cash": result[:, 1].copy(),
        "growth_rate": result[:, 2].copy(),
    }
