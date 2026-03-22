"""Core growth ODE system — logistic growth + cash flow, vectorized over all agents."""

import numpy as np

VARS_PER_AGENT = 3  # [firm_size, cash, growth_rate_momentum]


def growth_rhs(
    t: float,
    y_flat: np.ndarray,
    n_agents: int,
    carrying_capacity: np.ndarray,
    revenue: np.ndarray,
    fixed_costs: np.ndarray,
    variable_cost_rate: np.ndarray,
) -> np.ndarray:
    """
    Vectorized RHS for the 3-variable growth system across all agents.

    y_flat shape: (n_agents * 3,) — packed [firm_size, cash, growth_rate] for all agents
    Returns: dy_flat shape: (n_agents * 3,)
    """
    y = y_flat.reshape(n_agents, VARS_PER_AGENT)
    firm_size = np.maximum(y[:, 0], 1e-10)
    growth_rate = y[:, 2]

    k = np.maximum(carrying_capacity, 1e-10)

    d_firm_size = growth_rate * firm_size * (1.0 - firm_size / k)
    costs = fixed_costs + variable_cost_rate * firm_size
    d_cash = revenue - costs
    d_growth_rate = np.zeros(n_agents)

    dy = np.column_stack([d_firm_size, d_cash, d_growth_rate])
    return dy.ravel()


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
    """Steps growth ODE by dt via solve_tick. Returns (new_firm_size, new_cash, new_growth_rate)."""
    from biosim.math.solver import solve_tick

    result = solve_tick(
        firm_size, cash, growth_rate,
        carrying_capacity, revenue, fixed_costs, variable_cost_rate,
        dt=dt,
    )
    return result["firm_size"], result["cash"], result["growth_rate"]
