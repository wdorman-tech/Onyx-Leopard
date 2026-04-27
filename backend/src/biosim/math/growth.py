"""Core growth ODE system — logistic growth + cash flow, vectorized over all agents."""

import numpy as np

VARS_PER_AGENT = 3  # [firm_size, cash, growth_rate_momentum]

#: Capacity-utilization feedback gain on the intrinsic growth rate r.
#: dr/dt = GROWTH_RATE_FEEDBACK_K * (utilization - target). Units: per-tick.
#: A firm operating above the target utilization has product-market momentum
#: and r drifts up; a sluggish firm sees r decay. Calibrated so that a firm
#: held at full saturation (util=1.0) gains 0.003/tick of r — meaningful
#: drift over a year of simulation without runaway. P-MATH-6 fix.
GROWTH_RATE_FEEDBACK_K: float = 0.01

#: Target capacity utilization at which dr/dt = 0 (no momentum gain or decay).
#: 0.7 matches the operations-research convention for "healthy" utilization
#: that leaves slack for surge demand without burning out the workforce.
GROWTH_RATE_TARGET_UTILIZATION: float = 0.7

#: Hard caps on the evolved growth rate — prevents runaway positive feedback
#: if a firm sits at saturation indefinitely, and stops r from going arbitrarily
#: negative when a firm is permanently underutilized. ±0.5 / -0.1 per tick is
#: already extreme; the caps are a safety net, not a tuning parameter.
GROWTH_RATE_MAX: float = 0.5
GROWTH_RATE_MIN: float = -0.1


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

    # Capacity-utilization feedback on r: high utilization → r drifts up
    # (positive momentum / experience curve), low utilization → r decays.
    # One-sided clamps zero out the gradient when r has already hit a cap
    # and the feedback would push it further past — keeps r in [MIN, MAX].
    utilization = firm_size / k
    d_growth_rate = GROWTH_RATE_FEEDBACK_K * (utilization - GROWTH_RATE_TARGET_UTILIZATION)
    d_growth_rate = np.where(
        (growth_rate >= GROWTH_RATE_MAX) & (d_growth_rate > 0), 0.0, d_growth_rate
    )
    d_growth_rate = np.where(
        (growth_rate <= GROWTH_RATE_MIN) & (d_growth_rate < 0), 0.0, d_growth_rate
    )

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
    from src.biosim.math.solver import solve_tick

    result = solve_tick(
        firm_size, cash, growth_rate,
        carrying_capacity, revenue, fixed_costs, variable_cost_rate,
        dt=dt,
    )
    return result["firm_size"], result["cash"], result["growth_rate"]
