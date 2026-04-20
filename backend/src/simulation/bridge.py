"""Bridge between company node graphs and market competition variables.

Maps a company's node graph -> (quality, marketing, capacity) for the market
share attraction formula, and maps market-allocated demand -> per-location
customer allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from biosim.math.production import cobb_douglas

from src.simulation.config_loader import IndustrySpec
from src.simulation.models import CompanyState, SimNode


@dataclass
class _CompanyAggregate:
    """Per-company aggregates from a single node-graph pass.

    Holds everything the capacity calculation needs *before* the Cobb-Douglas
    step, so the batch path can compute Cobb-Douglas once across all firms
    instead of N times with 1-element arrays.
    """

    has_locations: bool
    marketing: float
    avg_price: float
    avg_satisfaction: float
    quality_mult: float
    infra_mult: float
    total_max_capacity: float
    capital_val: float
    labor_val: float


def _aggregate_company(
    state: CompanyState, spec: IndustrySpec, marketing_boost: float,
) -> _CompanyAggregate:
    """Single pass over a company's node graph to build the aggregate inputs."""
    bridge = spec.bridge
    location_type = spec.roles.location_type
    quality_keys = frozenset(bridge.quality_modifier_keys)
    infra_mults = bridge.infrastructure_multipliers
    marketing_contribs = bridge.marketing_contributions

    locations: list[SimNode] = []
    infra_mult = 1.0
    quality_mult = 1.0
    marketing = bridge.marketing_baseline

    for node in state.nodes.values():
        if not node.active:
            continue
        if node.type == location_type and node.location_state is not None:
            locations.append(node)
        if node.type in infra_mults:
            infra_mult *= infra_mults[node.type]
        for key, val in node.revenue_modifiers.items():
            if key in quality_keys:
                quality_mult *= 1.0 + val
        if node.type in marketing_contribs:
            marketing += marketing_contribs[node.type]

    if not locations:
        return _CompanyAggregate(
            has_locations=False,
            marketing=bridge.marketing_baseline,
            avg_price=0.0,
            avg_satisfaction=0.0,
            quality_mult=1.0,
            infra_mult=1.0,
            total_max_capacity=0.0,
            capital_val=0.0,
            labor_val=1.0,
        )

    loc_count = len(locations)
    marketing += loc_count * bridge.marketing_per_location

    if marketing_boost != bridge.marketing_boost_neutral:
        boost_delta = marketing_boost - bridge.marketing_boost_neutral
        marketing = max(
            1.0,
            marketing + boost_delta * bridge.marketing_boost_multiplier,
        )

    avg_price = sum(loc.location_state.price for loc in locations) / loc_count  # type: ignore[union-attr]
    total_max_capacity = sum(
        loc.location_state.max_capacity for loc in locations  # type: ignore[union-attr]
    )
    avg_satisfaction = (
        sum(loc.location_state.satisfaction for loc in locations) / loc_count  # type: ignore[union-attr]
    )
    capital_val = sum(
        loc.location_state.daily_fixed_costs for loc in locations  # type: ignore[union-attr]
    ) * 365.0  # annualize
    labor_val = float(state.total_employees) if state.total_employees > 0 else 1.0

    return _CompanyAggregate(
        has_locations=True,
        marketing=marketing,
        avg_price=avg_price,
        avg_satisfaction=avg_satisfaction,
        quality_mult=quality_mult,
        infra_mult=infra_mult,
        total_max_capacity=total_max_capacity,
        capital_val=capital_val,
        labor_val=labor_val,
    )


def _capacity_from_raw_output(
    agg: _CompanyAggregate, raw_output: float, spec: IndustrySpec,
) -> float:
    """Turn a Cobb-Douglas raw output into a $-revenue/day capacity ceiling."""
    util = spec.bridge.sustainable_utilization
    if spec.math.production_model == "cobb_douglas":
        return raw_output * agg.avg_price * util
    return agg.total_max_capacity * agg.avg_price * agg.infra_mult * util


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
    agg = _aggregate_company(state, spec, marketing_boost)
    if not agg.has_locations:
        return (0.01, spec.bridge.marketing_baseline, 0.0)

    if spec.math.production_model == "cobb_douglas":
        raw = float(cobb_douglas(
            np.array([agg.infra_mult]),
            np.array([agg.capital_val]),
            np.array([agg.labor_val]),
            np.array([spec.math.production_alpha]),
            np.array([spec.math.production_beta]),
        )[0])
    else:
        raw = 0.0  # unused in non-cobb_douglas branch

    capacity = _capacity_from_raw_output(agg, raw, spec)
    quality = max(0.01, agg.avg_satisfaction * agg.quality_mult)
    return (quality, agg.marketing, capacity)


def derive_competitive_attributes_batch(
    states: list[CompanyState],
    spec: IndustrySpec,
    marketing_boosts: list[float],
) -> list[tuple[float, float, float]]:
    """Vectorized variant: one Cobb-Douglas call across all companies.

    Equivalent to calling `derive_competitive_attributes` per company, but
    batches the capacity production function so we hit NumPy once per tick
    instead of N times with 1-element arrays.
    """
    if not states:
        return []
    if len(states) != len(marketing_boosts):
        raise ValueError(
            f"states ({len(states)}) / marketing_boosts ({len(marketing_boosts)}) length mismatch",
        )

    aggregates = [
        _aggregate_company(s, spec, mb)
        for s, mb in zip(states, marketing_boosts, strict=True)
    ]

    if spec.math.production_model == "cobb_douglas":
        n = len(aggregates)
        infra_arr = np.fromiter((a.infra_mult for a in aggregates), dtype=float, count=n)
        capital_arr = np.fromiter((a.capital_val for a in aggregates), dtype=float, count=n)
        labor_arr = np.fromiter((a.labor_val for a in aggregates), dtype=float, count=n)
        alpha_arr = np.full(n, spec.math.production_alpha)
        beta_arr = np.full(n, spec.math.production_beta)
        raw_outputs = cobb_douglas(infra_arr, capital_arr, labor_arr, alpha_arr, beta_arr)
    else:
        raw_outputs = np.zeros(len(aggregates))

    results: list[tuple[float, float, float]] = []
    for agg, raw in zip(aggregates, raw_outputs, strict=True):
        if not agg.has_locations:
            results.append((0.01, spec.bridge.marketing_baseline, 0.0))
            continue
        capacity = _capacity_from_raw_output(agg, float(raw), spec)
        quality = max(0.01, agg.avg_satisfaction * agg.quality_mult)
        results.append((quality, agg.marketing, capacity))
    return results


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
