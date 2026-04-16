"""Location tick function — supports physical, subscription, and service economics."""

from __future__ import annotations

import math
import random

import numpy as np

from src.simulation.models import BatchTickResult, LocationArrays, LocationConfig, LocationState, LocationTickResult


def tick_location(
    state: LocationState,
    modifiers: dict[str, float],
    company_cash: float,
    allocated_demand: float | None = None,
    supply_unit_name: str = "units",
) -> tuple[LocationTickResult, float]:
    """Run one day of location economics.

    Args:
        state: Mutable location state (updated in-place).
        modifiers: Aggregated modifiers from company nodes.
        company_cash: Available cash for reorders.
        allocated_demand: If set (unified mode), the market-allocated customer
            target for this location.
        supply_unit_name: Label for inventory units (e.g. "lbs chicken", "licenses").

    Returns:
        (LocationTickResult, reorder_cost) — the result and any cash spent on reorders.
    """
    events: list[str] = []
    reorder_spent = 0.0

    # Apply modifiers
    growth_boost = modifiers.get("customer_growth", 0.0)
    reach_boost = modifiers.get("customer_reach", 0.0)
    cost_mult = modifiers.get("food_cost", 1.0)  # key overridden via config in batch path
    satisfaction_boost = modifiers.get("satisfaction_baseline", 0.0)
    revenue_boost = modifiers.get("catering_revenue", 0.0)

    sat = min(1.0, state.satisfaction + satisfaction_boost)
    model = state.economics_model

    # ── 1. Customer demand ──
    if allocated_demand is not None:
        target = min(allocated_demand, float(state.max_capacity) * state.demand_cap_ratio)
        gap = target - state.customers
        convergence_rate = state.customer_convergence_rate * sat
        state.customers += convergence_rate * gap
        state.customers = max(0.0, state.customers)
        actual_demand = max(0, math.floor(state.customers * random.uniform(state.demand_noise_low, state.demand_noise_high)))
    else:
        effective_max_customers = state.max_local_customers * (1 + reach_boost)
        effective_growth_rate = state.word_of_mouth_rate * (1 + growth_boost)
        growth = effective_growth_rate * sat * (1 - state.customers / effective_max_customers)
        state.customers *= 1 + growth
        actual_demand = max(0, math.floor(state.customers * random.uniform(state.demand_noise_low, state.demand_noise_high)))

    # ── 1b. Subscription churn (applied before demand calculation) ──
    if model == "subscription" and state.churn_rate > 0:
        daily_churn = state.churn_rate / 30.0  # monthly rate → daily
        churned = math.floor(state.customers * daily_churn)
        if churned > 0:
            state.customers = max(0.0, state.customers - churned)
            events.append(f"{churned} customers churned")

    # ── 2. Supply constraint ──
    if model == "subscription":
        # Subscription: capacity is max subscribers, no physical inventory
        servable = min(actual_demand, state.max_capacity)
    else:
        servable = min(actual_demand, state.max_capacity, int(state.inventory))
    unserved = max(0, actual_demand - servable)

    if unserved > 0 and actual_demand > state.max_capacity:
        events.append(f"Turned away {unserved} customers (at capacity)")
    elif unserved > 0 and model != "subscription":
        events.append(f"Turned away {unserved} customers (low {supply_unit_name})")

    # ── 3. Revenue ──
    daily_revenue = servable * state.price * (1 + revenue_boost)

    # ── 4. Inventory consumption (physical and service models) ──
    if model != "subscription":
        state.inventory -= servable

    # ── 5. Capacity decay ──
    if model == "physical" and state.capacity_decay_rate > 0:
        decayed = state.inventory * state.capacity_decay_rate
        if decayed >= 1:
            events.append(f"{decayed:.0f} {supply_unit_name} spoiled")
        state.inventory -= decayed
        state.inventory = max(0, state.inventory)
    elif model == "service" and state.capacity_decay_rate > 0:
        # Service: unused capacity decays (bench time)
        decayed = state.inventory * state.capacity_decay_rate
        if decayed >= 1:
            events.append(f"{decayed:.0f} {supply_unit_name} lost (idle)")
        state.inventory -= decayed
        state.inventory = max(0, state.inventory)

    # ── 6. Auto-replenish (physical/service: restock; subscription: scale infra) ──
    if model == "subscription":
        # Subscription: scaling costs when approaching capacity
        if actual_demand > state.max_capacity * state.subscription_scaling_threshold and state.scaling_cost_per_unit > 0:
            scale_units = max(1, int(state.max_capacity * state.subscription_scaling_increment))
            scale_cost = scale_units * state.scaling_cost_per_unit
            if company_cash >= scale_cost:
                state.max_capacity += scale_units
                reorder_spent = scale_cost
                events.append(f"Scaled capacity +{scale_units} (${scale_cost:.0f})")
    else:
        effective_supply_cost = state.supply_cost_per_unit * cost_mult
        if state.inventory < state.replenish_threshold:
            order_cost = state.replenish_amount * effective_supply_cost
            if company_cash >= order_cost:
                state.inventory += state.replenish_amount
                reorder_spent = order_cost
                events.append(
                    f"Replenished {state.replenish_amount:.0f} {supply_unit_name} (${order_cost:.0f})"
                )
            else:
                events.append("Cannot replenish — insufficient cash")

    # ── 7. Costs ──
    effective_variable_cost = state.variable_cost_per_unit * cost_mult
    daily_variable_costs = servable * effective_variable_cost
    daily_costs = state.daily_fixed_costs + daily_variable_costs

    # Subscription: customer acquisition costs
    if model == "subscription" and state.acquisition_cost > 0:
        new_customers = max(0, servable - int(state.customers * state.new_customer_ratio))
        daily_costs += new_customers * state.acquisition_cost

    # ── 8. Profit ──
    daily_profit = daily_revenue - daily_costs

    # ── 9. Satisfaction update ──
    if unserved > 0 and actual_demand > 0:
        state.satisfaction = max(0, state.satisfaction - state.satisfaction_penalty_rate * (unserved / actual_demand))
    else:
        state.satisfaction = min(1.0, state.satisfaction + state.satisfaction_recovery_rate)

    result = LocationTickResult(
        revenue=round(daily_revenue, 2),
        costs=round(daily_costs, 2),
        profit=round(daily_profit, 2),
        customers_served=servable,
        events=events,
    )
    return result, reorder_spent


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

    # Scalar modifiers (same for all locations in the company)
    cost_mult = modifiers.get(config.variable_cost_modifier_key, 1.0)
    sat_boost = modifiers.get("satisfaction_baseline", 0.0)
    rev_boost = modifiers.get("catering_revenue", 0.0)

    # 1. Customer convergence (unified mode)
    sat = np.minimum(1.0, arrays.satisfaction + sat_boost)
    target = np.minimum(allocated_demands, arrays.max_capacity * config.demand_cap_ratio)
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
