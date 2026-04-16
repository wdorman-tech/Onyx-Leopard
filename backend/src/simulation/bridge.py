"""Bridge between company node graphs and market competition variables.

Maps a company's node graph -> (quality, marketing, capacity) for the market
share attraction formula, and maps market-allocated demand -> per-location
customer allocation.
"""

from __future__ import annotations

from src.simulation.config_loader import IndustrySpec
from src.simulation.models import CompanyState, SimNode


def derive_competitive_attributes(
    state: CompanyState,
    spec: IndustrySpec,
    marketing_boost: float = 0.5,
) -> tuple[float, float, float]:
    """Extract (quality, marketing, capacity) from a company's node graph.

    Args:
        state: The company's current state with node graph.
        spec: Industry specification with bridge mappings.
        marketing_boost: CEO agent marketing intensity (0.0-1.0). 0.5 = neutral.

    Returns values in market-compatible units:
        quality  — [0, ~1.5] range, fed into q^beta in share attraction
        marketing — [5, ~50] range, fed into m^alpha in share attraction
        capacity  — $/day of maximum revenue, used as K_i ceiling
    """
    bridge = spec.bridge
    location_type = spec.roles.location_type
    quality_keys = frozenset(bridge.quality_modifier_keys)
    infra_mults = bridge.infrastructure_multipliers
    marketing_contribs = bridge.marketing_contributions

    # Single pass over all nodes
    locations: list[SimNode] = []
    infra_mult = 1.0
    quality_mult = 1.0
    marketing = bridge.marketing_baseline

    for node in state.nodes.values():
        if not node.active:
            continue
        # Collect locations inline
        if node.type == location_type and node.location_state is not None:
            locations.append(node)
        # Infrastructure multipliers
        if node.type in infra_mults:
            infra_mult *= infra_mults[node.type]
        # Quality modifiers
        for key, val in node.revenue_modifiers.items():
            if key in quality_keys:
                quality_mult *= 1.0 + val
        # Marketing contributions
        if node.type in marketing_contribs:
            marketing += marketing_contribs[node.type]

    if not locations:
        return (0.01, bridge.marketing_baseline, 0.0)

    loc_count = len(locations)
    marketing += loc_count * bridge.marketing_per_location

    # CEO marketing boost
    if marketing_boost != bridge.marketing_boost_neutral:
        marketing = max(1.0, marketing + (marketing_boost - bridge.marketing_boost_neutral) * bridge.marketing_boost_multiplier)

    # ── Capacity K_i ──
    avg_price = (
        sum(loc.location_state.price for loc in locations) / loc_count  # type: ignore[union-attr]
    )
    total_max_capacity = sum(
        loc.location_state.max_capacity for loc in locations  # type: ignore[union-attr]
    )
    capacity = total_max_capacity * avg_price * infra_mult * bridge.sustainable_utilization

    # ── Quality q_i ──
    avg_satisfaction = (
        sum(loc.location_state.satisfaction for loc in locations)  # type: ignore[union-attr]
        / loc_count
    )
    quality = max(0.01, avg_satisfaction * quality_mult)

    return (quality, marketing, capacity)


def allocate_demand_to_locations(
    total_customers: float,
    locations: list[SimNode],
) -> dict[str, float]:
    """Distribute a firm's market-allocated customer demand across its locations.

    Allocation is proportional to each location's attractiveness score:
        attractiveness_j = satisfaction_j * max_capacity_j

    Better-performing, larger locations attract more of the firm's share.
    """
    if not locations or total_customers <= 0:
        return {loc.id: 0.0 for loc in locations}

    scores: dict[str, float] = {}
    total_score = 0.0
    for loc in locations:
        if loc.location_state is None:
            continue
        score = loc.location_state.satisfaction * loc.location_state.max_capacity
        scores[loc.id] = score
        total_score += score

    if total_score <= 0:
        per_loc = total_customers / len(locations)
        return {loc.id: per_loc for loc in locations}

    return {
        loc_id: total_customers * (score / total_score)
        for loc_id, score in scores.items()
    }
