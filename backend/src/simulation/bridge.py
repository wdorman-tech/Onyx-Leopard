"""Bridge between company node graphs and market competition variables.

Maps a company's node graph -> (quality, marketing, capacity) for the market
share attraction formula, and maps market-allocated demand -> per-location
customer allocation.
"""

from __future__ import annotations

from src.simulation.models import CompanyState, NodeType, SimNode

# ── Node-to-market contribution weights ──

# Marketing: which node types contribute to the firm's marketing score,
# and how much each contributes in market units.
_MARKETING_CONTRIBUTIONS: dict[NodeType, float] = {
    NodeType.MARKETING: 15.0,
    NodeType.DELIVERY_PARTNERSHIP: 10.0,
    NodeType.CATERING: 5.0,
}

# Quality: which revenue_modifier keys feed into the quality score.
# These are applied multiplicatively: q = avg_sat * Π(1 + modifier).
_QUALITY_MODIFIER_KEYS: frozenset[str] = frozenset({
    "satisfaction_baseline",  # QA: +0.03
    "menu_innovation",        # R&D: +0.05
    "new_location_satisfaction",  # Training: +0.05
})

MARKETING_BASELINE = 5.0  # word-of-mouth, no marketing node
MARKETING_PER_LOCATION = 1.0  # each location adds a bit of visibility

# Locations can sustainably operate at this fraction of max_capacity.
# Headroom prevents persistent stockouts and satisfaction degradation.
SUSTAINABLE_UTILIZATION = 0.85


def derive_competitive_attributes(
    state: CompanyState,
) -> tuple[float, float, float]:
    """Extract (quality, marketing, capacity) from a company's node graph.

    Returns values in market-compatible units:
        quality  — [0, ~1.5] range, fed into q^beta in share attraction
        marketing — [5, ~50] range, fed into m^alpha in share attraction
        capacity  — $/day of maximum revenue, used as K_i ceiling
    """
    locations = _active_locations(state)
    if not locations:
        return (0.01, MARKETING_BASELINE, 0.0)

    # ── Capacity K_i ──
    # Base: Σ(max_capacity) across locations, converted to $/day via price.
    # Infrastructure nodes (commissary, distribution center) multiply throughput.
    infra_mult = 1.0
    for node in state.nodes.values():
        if not node.active:
            continue
        if node.type == NodeType.COMMISSARY:
            infra_mult *= 1.15
        elif node.type == NodeType.DISTRIBUTION_CENTER:
            infra_mult *= 1.10

    avg_price = (
        sum(loc.location_state.price for loc in locations) / len(locations)  # type: ignore[union-attr]
    )
    total_max_capacity = sum(
        loc.location_state.max_capacity for loc in locations  # type: ignore[union-attr]
    )
    capacity = total_max_capacity * avg_price * infra_mult * SUSTAINABLE_UTILIZATION

    # ── Quality q_i ──
    # Base: average satisfaction across locations.
    # Multiplied by quality-contributing node modifiers.
    avg_satisfaction = (
        sum(loc.location_state.satisfaction for loc in locations)  # type: ignore[union-attr]
        / len(locations)
    )
    quality_mult = 1.0
    for node in state.nodes.values():
        if not node.active:
            continue
        for key, val in node.revenue_modifiers.items():
            if key in _QUALITY_MODIFIER_KEYS:
                quality_mult *= 1.0 + val

    quality = max(0.01, avg_satisfaction * quality_mult)

    # ── Marketing m_i ──
    # Baseline + per-location visibility + node contributions.
    loc_count = len(locations)
    marketing = MARKETING_BASELINE + loc_count * MARKETING_PER_LOCATION
    for node in state.nodes.values():
        if not node.active:
            continue
        if node.type in _MARKETING_CONTRIBUTIONS:
            marketing += _MARKETING_CONTRIBUTIONS[node.type]

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


def _active_locations(state: CompanyState) -> list[SimNode]:
    """Return all active restaurant nodes with location state."""
    return [
        n for n in state.nodes.values()
        if n.type == NodeType.RESTAURANT and n.active and n.location_state is not None
    ]
