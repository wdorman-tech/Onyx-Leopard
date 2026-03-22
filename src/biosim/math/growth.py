"""Stub -- real implementation in Unit 2. Provides Euler fallback for testing."""

import numpy as np


def step_growth(
    firm_size: np.ndarray,
    cash: np.ndarray,
    growth_rate: np.ndarray,
    carrying_capacity: np.ndarray,
    revenue: np.ndarray,
    fixed_costs: np.ndarray,
    variable_cost_rate: np.ndarray,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple Euler step for growth ODE. Used as fallback/stub."""
    d_size = growth_rate * firm_size * (1 - firm_size / np.maximum(carrying_capacity, 1e-10))
    d_cash = revenue - (fixed_costs + variable_cost_rate * firm_size)
    return firm_size + d_size * dt, cash + d_cash * dt, growth_rate.copy()
