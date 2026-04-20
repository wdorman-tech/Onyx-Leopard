"""Heuristic decision agent — rule-based fallback for AI CEO decisions.

Extracts and generalizes the 6 quarterly decision rules from MarketEngine
into a standalone agent that produces CEODecision objects. Used as:
- Fallback when AI budget is exhausted
- Default agent for Monte Carlo runs (no API calls needed)
- The "heuristic tier" in the tiered AI architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.simulation.ceo_agent import CEODecision

if TYPE_CHECKING:
    from src.simulation.unified import CompanyAgent


def heuristic_decide(
    company: CompanyAgent,
    all_companies: list[CompanyAgent],
    tam: float,
    tick: int,
) -> CEODecision:
    """Apply rule-based heuristic decisions for a company.

    Generalizes MarketEngine's 6 quarterly review rules to work with
    CompanyAgent state. Rules execute in priority order; later rules
    can override earlier ones.
    """
    spec = company.spec
    ceo = spec.ceo

    cash = company.state.cash
    locations = company.location_count()
    revenue = company.daily_revenue
    share = company.share
    prev_share = company.prev_share
    capacity = company.capacity
    utilization = revenue / capacity if capacity > 0 else 0.0

    # Read aggregate price/cost across all active locations (capacity-weighted).
    # Falls back to ceo defaults when there are no active locations.
    price = company.mean_location_price()
    cost_target = company.mean_variable_cost()

    marketing = company.marketing_boost
    expansion = "normal"
    quality_invest = 0.0
    max_locs = ceo.max_locations_per_year_cap
    reasoning_parts: list[str] = []

    # Daily corporate overhead as proxy for fixed costs
    daily_overhead = company.compute_corporate_overhead()
    monthly_overhead = daily_overhead * 30

    # Rule 1: Cash Emergency — short-circuit. When cash is critical, ALL other
    # rules are suspended; expanding or boosting marketing now would accelerate death.
    emergency_threshold = max(monthly_overhead * 2, spec.constants.starting_cash * 0.1)
    if cash < emergency_threshold:
        return CEODecision(
            reasoning="Cash emergency — cutting marketing and freezing expansion",
            price_adjustment=price,
            expansion_pace="conservative",
            marketing_intensity=max(0.1, marketing * 0.5),
            quality_investment=0.0,
            cost_target=cost_target,
            max_locations_per_year=0,
        )

    # Rule 2: Excess Capacity — boost marketing to fill capacity
    if utilization < 0.5:
        marketing = min(1.0, marketing * 1.2)
        reasoning_parts.append("Excess capacity — boosting marketing")

    # Rule 3: Capacity Constraint — expand
    if utilization > 0.75:
        expansion = "aggressive"
        max_locs = min(ceo.max_locations_per_year_cap, locations + 2)
        reasoning_parts.append("At capacity — expanding")

    # Rule 4: Market Share Decline — invest in quality and marketing
    if prev_share > 0 and share < prev_share * 0.9:
        quality_invest = 0.05
        marketing = min(1.0, marketing * 1.15)
        reasoning_parts.append("Losing share — investing in quality and marketing")

    # Rule 5: Profitable Growth — moderate investment
    if cash > monthly_overhead * 5 and utilization > 0.6:
        quality_invest = max(quality_invest, 0.02)
        max_locs = min(ceo.max_locations_per_year_cap, locations + 1)
        reasoning_parts.append("Profitable — investing in growth")

    # Rule 6: Market Opportunity — expand when demand exceeds capacity
    demand_potential = tam * share
    if demand_potential > capacity * 1.2 and cash > monthly_overhead * 3:
        expansion = "aggressive"
        max_locs = min(ceo.max_locations_per_year_cap, max_locs + 2)
        reasoning_parts.append("Market opportunity — expanding toward demand")

    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Maintaining current course (heuristic)"

    return CEODecision(
        reasoning=reasoning,
        price_adjustment=price,
        expansion_pace=expansion,
        marketing_intensity=marketing,
        quality_investment=quality_invest,
        cost_target=cost_target,
        max_locations_per_year=max_locs,
    )
