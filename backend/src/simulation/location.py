"""Location tick function — extracted from RestaurantEngine for multi-location use."""

from __future__ import annotations

import math
import random

from src.simulation.models import LocationState, LocationTickResult


def tick_location(
    state: LocationState,
    modifiers: dict[str, float],
    company_cash: float,
    allocated_demand: float | None = None,
) -> tuple[LocationTickResult, float]:
    """Run one day of location economics.

    Args:
        state: Mutable location state (updated in-place).
        modifiers: Aggregated modifiers from company nodes.
            Keys: "food_cost", "customer_growth", "customer_reach",
                  "satisfaction_baseline", "catering_revenue", "labor", etc.
            Values: multipliers applied to the relevant param.
        company_cash: Available cash for reorders.
        allocated_demand: If set (unified mode), the market-allocated customer
            target for this location. The location's customers converge toward
            this value instead of using independent logistic growth. When None,
            standalone logistic growth is used (backward compatible).

    Returns:
        (LocationTickResult, reorder_cost) — the result and any cash spent on reorders.
    """
    events: list[str] = []
    reorder_spent = 0.0

    # Apply modifiers
    growth_boost = modifiers.get("customer_growth", 0.0)
    reach_boost = modifiers.get("customer_reach", 0.0)
    food_cost_mult = modifiers.get("food_cost", 1.0)
    satisfaction_boost = modifiers.get("satisfaction_baseline", 0.0)
    revenue_boost = modifiers.get("catering_revenue", 0.0)

    sat = min(1.0, state.satisfaction + satisfaction_boost)

    if allocated_demand is not None:
        # Unified mode: smooth convergence toward market-allocated demand.
        # Higher satisfaction -> faster adoption; capped below max_capacity
        # to leave headroom for demand noise and prevent stockout spirals.
        target = min(allocated_demand, float(state.max_capacity) * 0.85)
        gap = target - state.customers
        convergence_rate = 0.05 * sat  # ~20-tick half-life at sat=0.7
        state.customers += convergence_rate * gap
        state.customers = max(0.0, state.customers)
        actual_demand = max(0, math.floor(state.customers * random.uniform(0.9, 1.1)))
    else:
        # Standalone mode: independent logistic growth
        effective_max_customers = state.max_local_customers * (1 + reach_boost)
        effective_growth_rate = state.word_of_mouth_rate * (1 + growth_boost)

        growth = effective_growth_rate * sat * (1 - state.customers / effective_max_customers)
        state.customers *= 1 + growth
        actual_demand = max(0, math.floor(state.customers * random.uniform(0.9, 1.1)))

    # 2. Supply constraint
    servable = min(actual_demand, state.max_capacity, int(state.inventory))
    unserved = max(0, actual_demand - servable)

    if unserved > 0 and actual_demand > state.max_capacity:
        events.append(f"Turned away {unserved} customers (at capacity)")
    elif unserved > 0 and state.inventory < actual_demand:
        events.append(f"Turned away {unserved} customers (out of chicken)")

    # 3. Revenue (with catering boost)
    daily_revenue = servable * state.price * (1 + revenue_boost)

    # 4. Inventory consumption
    state.inventory -= servable

    # 5. Spoilage
    spoiled = state.inventory * state.spoilage_rate
    if spoiled >= 1:
        events.append(f"{spoiled:.0f} lbs chicken spoiled")
    state.inventory -= spoiled
    state.inventory = max(0, state.inventory)

    # 6. Auto-reorder (uses company cash)
    effective_chicken_cost = state.chicken_cost_per_lb * food_cost_mult
    if state.inventory < state.reorder_point:
        order_cost = state.reorder_qty * effective_chicken_cost
        if company_cash >= order_cost:
            state.inventory += state.reorder_qty
            reorder_spent = order_cost
            events.append(f"Ordered {state.reorder_qty:.0f} lbs chicken (${order_cost:.0f})")
        else:
            events.append("Cannot reorder — insufficient cash")

    # 7. Costs
    effective_food_cost = state.food_cost_per_plate * food_cost_mult
    daily_variable_costs = servable * effective_food_cost
    daily_costs = state.daily_fixed_costs + daily_variable_costs

    # 8. Profit
    daily_profit = daily_revenue - daily_costs

    # 9. Satisfaction update
    if unserved > 0 and actual_demand > 0:
        state.satisfaction = max(0, state.satisfaction - 0.02 * (unserved / actual_demand))
    else:
        state.satisfaction = min(1.0, state.satisfaction + 0.005)

    result = LocationTickResult(
        revenue=round(daily_revenue, 2),
        costs=round(daily_costs, 2),
        profit=round(daily_profit, 2),
        customers_served=servable,
        events=events,
    )
    return result, reorder_spent
