"""Location tick function — supports physical, subscription, and service economics."""

from __future__ import annotations

import numpy as np
from biosim.math.growth import step_growth

from src.simulation.models import BatchTickResult, LocationArrays, LocationConfig


def tick_locations_batch(
    arrays: LocationArrays,
    modifiers: dict[str, float],
    company_cash: float,
    allocated_demands: np.ndarray,
    rng: np.random.Generator,
    company_name: str,
    labels: list[str],
    supply_unit_name: str = "units",
    config: LocationConfig | None = None,
    growth_model: str = "linear_convergence",
    growth_rate: float = 0.1,
) -> BatchTickResult:
    """Vectorized batch tick for all locations in a company.

    Operates on LocationArrays (struct-of-arrays) using NumPy.
    Mutates arrays.customers, arrays.inventory, arrays.satisfaction in-place.
    """
    n = arrays.size
    if n == 0:
        return BatchTickResult()

    if config is None:
        config = LocationConfig()

    events: list[str] = []
    model = arrays.economics_model

    # Scalar modifiers (same for all locations in the company).
    # Each list element is a YAML-declared modifier key. Cost modifiers compound
    # multiplicatively (each value is already 1+delta from aggregate_modifiers).
    # Revenue and satisfaction boosts are additive deltas. When a list is empty
    # (un-migrated YAML), we fall back to the single canonical key for cost.
    if config.cost_modifier_keys:
        cost_mult = 1.0
        for key in config.cost_modifier_keys:
            cost_mult *= modifiers.get(key, 1.0)
    else:
        cost_mult = modifiers.get(config.variable_cost_modifier_key, 1.0)

    sat_boost = sum(modifiers.get(k, 0.0) for k in config.satisfaction_modifier_keys)
    rev_boost = sum(modifiers.get(k, 0.0) for k in config.revenue_modifier_keys)

    # 1. Customer convergence (unified mode)
    sat = np.minimum(1.0, arrays.satisfaction + sat_boost)
    target = np.minimum(allocated_demands, arrays.max_capacity * config.demand_cap_ratio)

    if growth_model == "logistic_ode":
        # Logistic growth ODE via biosim: customers are "firm_size", target is carrying capacity
        effective_rate = np.full(n, growth_rate, dtype=np.float64) * sat
        new_customers, _, _ = step_growth(
            firm_size=arrays.customers.copy(),
            cash=np.zeros(n, dtype=np.float64),  # cash tracked externally
            growth_rate=effective_rate,
            carrying_capacity=target,
            revenue=np.zeros(n, dtype=np.float64),
            fixed_costs=np.zeros(n, dtype=np.float64),
            variable_cost_rate=np.zeros(n, dtype=np.float64),
            dt=1.0,
        )
        arrays.customers = np.maximum(0.0, new_customers)
    else:
        gap = target - arrays.customers
        convergence_rate = config.customer_convergence_rate * sat
        arrays.customers = arrays.customers + convergence_rate * gap
        arrays.customers = np.maximum(0.0, arrays.customers)

    # 1b. Subscription churn
    if model == "subscription":
        daily_churn_rate = arrays.churn_rate / config.days_per_month
        churned = np.floor(arrays.customers * daily_churn_rate)
        arrays.customers = np.maximum(0.0, arrays.customers - churned)

    noise = rng.uniform(config.demand_noise_low, config.demand_noise_high, size=n)
    actual_demand = np.maximum(0, np.floor(arrays.customers * noise)).astype(np.int64)

    # 2. Supply constraint
    inventory_pre = arrays.inventory.copy()
    if model == "subscription":
        servable = np.minimum(actual_demand, arrays.max_capacity.astype(np.int64))
    else:
        inventory_int = np.floor(arrays.inventory).astype(np.int64)
        servable = np.minimum(actual_demand, np.minimum(arrays.max_capacity.astype(np.int64), inventory_int))
    unserved = np.maximum(0, actual_demand - servable)

    # 3. Revenue
    daily_revenue = servable * arrays.price * (1.0 + rev_boost)

    # 4. Inventory consumption (not for subscription)
    if model != "subscription":
        arrays.inventory = arrays.inventory - servable

    # 5. Capacity decay (physical/service only)
    if model != "subscription":
        decayed = arrays.inventory * arrays.capacity_decay_rate
        arrays.inventory = arrays.inventory - decayed
        arrays.inventory = np.maximum(0.0, arrays.inventory)
    else:
        decayed = np.zeros(n)

    # 6. Replenish (physical/service) or auto-scale (subscription)
    if model == "subscription":
        total_reorder_cost = 0.0
        replenish_mask = np.zeros(n, dtype=np.bool_)
        needs_replenish = np.zeros(n, dtype=np.bool_)
        order_costs = np.zeros(n)
    else:
        needs_replenish = arrays.inventory < arrays.replenish_threshold
        order_costs = arrays.replenish_amount * arrays.supply_cost_per_unit * cost_mult
        replenish_mask = needs_replenish & (order_costs <= company_cash)
        arrays.inventory = arrays.inventory + arrays.replenish_amount * replenish_mask
        total_reorder_cost = float((order_costs * replenish_mask).sum())

    # 7. Costs
    effective_variable_cost = arrays.variable_cost_per_unit * cost_mult
    daily_costs = arrays.daily_fixed_costs + servable * effective_variable_cost

    # 8. Profit
    daily_profit = daily_revenue - daily_costs

    # 9. Satisfaction update
    stockout_mask = unserved > 0
    safe_demand = np.maximum(actual_demand, 1)
    stockout_penalty = config.satisfaction_penalty_rate * (unserved / safe_demand)
    arrays.satisfaction = np.where(
        stockout_mask,
        np.maximum(0.0, arrays.satisfaction - stockout_penalty),
        np.minimum(1.0, arrays.satisfaction + config.satisfaction_recovery_rate),
    )

    # 10. Events (post-process from arrays)
    cap_stockouts = np.where((unserved > 0) & (actual_demand > arrays.max_capacity))[0]
    for i in cap_stockouts:
        events.append(f"{company_name} {labels[i]}: Turned away {int(unserved[i])} customers (at capacity)")

    if model != "subscription":
        inv_stockouts = np.where(
            (unserved > 0) & (inventory_pre < actual_demand) & (actual_demand <= arrays.max_capacity)
        )[0]
        for i in inv_stockouts:
            events.append(
                f"{company_name} {labels[i]}: Turned away {int(unserved[i])} customers (low {supply_unit_name})"
            )

        decay_events = np.where(decayed >= 1.0)[0]
        for i in decay_events:
            if model == "service":
                events.append(f"{company_name} {labels[i]}: {decayed[i]:.0f} {supply_unit_name} lost (idle)")
            else:
                events.append(f"{company_name} {labels[i]}: {decayed[i]:.0f} {supply_unit_name} spoiled")

        replenished = np.where(replenish_mask)[0]
        for i in replenished:
            events.append(
                f"{company_name} {labels[i]}: Replenished {arrays.replenish_amount[i]:.0f}"
                f" {supply_unit_name} (${order_costs[i]:.0f})"
            )
        failed_replenish = np.where(needs_replenish & ~replenish_mask)[0]
        for i in failed_replenish:
            events.append(f"{company_name} {labels[i]}: Cannot replenish — insufficient cash")

    return BatchTickResult(
        total_revenue=float(daily_revenue.sum()),
        total_costs=float(daily_costs.sum()),
        total_profit=float(daily_profit.sum()),
        total_reorder_cost=total_reorder_cost,
        events=events,
    )
