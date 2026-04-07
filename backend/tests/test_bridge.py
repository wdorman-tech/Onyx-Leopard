"""Unit tests for bridge.py — node graph → market variable mapping."""

from __future__ import annotations

import pytest

from src.simulation.bridge import (
    allocate_demand_to_locations,
    derive_competitive_attributes,
)
from src.simulation.config_loader import load_industry
from src.simulation.models import (
    CompanyState,
    LocationState,
    NodeCategory,
    SimNode,
)

_spec = load_industry("restaurant")
MARKETING_BASELINE = _spec.bridge.marketing_baseline
MARKETING_PER_LOCATION = _spec.bridge.marketing_per_location
SUSTAINABLE_UTILIZATION = _spec.bridge.sustainable_utilization


def _make_company(
    num_locations: int = 1,
    satisfaction: float = 0.7,
    extra_nodes: list[str] | None = None,
) -> CompanyState:
    """Build a CompanyState with the given number of locations and optional nodes."""
    state = CompanyState(name="Test Co")
    counter = 0

    # Owner
    counter += 1
    state.nodes[f"owner-{counter}"] = SimNode(
        id=f"owner-{counter}",
        type="owner_operator",
        label="Owner",
        category=NodeCategory.CORPORATE,
    )

    # Locations
    for i in range(num_locations):
        counter += 1
        node_id = f"restaurant-{counter}"
        state.nodes[node_id] = SimNode(
            id=node_id,
            type="restaurant",
            label=f"Location #{i + 1}",
            category=NodeCategory.LOCATION,
            location_state=LocationState(satisfaction=satisfaction),
        )

    # Extra nodes
    for nt in extra_nodes or []:
        counter += 1
        node_id = f"{nt}-{counter}"
        node_def = _spec.nodes[nt]
        state.nodes[node_id] = SimNode(
            id=node_id,
            type=nt,
            label=node_def.label,
            category=NodeCategory(node_def.category),
            revenue_modifiers=dict(node_def.revenue_modifiers),
        )

    return state


# ── derive_competitive_attributes ──


class TestDeriveCompetitiveAttributes:
    def test_no_locations_returns_minimum(self):
        state = CompanyState(name="Empty Co")
        q, m, k = derive_competitive_attributes(state, _spec)
        assert q == pytest.approx(0.01)
        assert m == MARKETING_BASELINE
        assert k == 0.0

    def test_single_location_baseline(self):
        state = _make_company(num_locations=1, satisfaction=0.7)
        q, m, k = derive_competitive_attributes(state, _spec)

        # Quality = avg_satisfaction * quality_mult (no quality nodes = mult 1.0)
        assert q == pytest.approx(0.7, abs=0.01)

        # Marketing = baseline + 1 location * per_loc
        assert m == pytest.approx(MARKETING_BASELINE + MARKETING_PER_LOCATION)

        # Capacity = 80 * $14 * 1.0 * 0.85
        expected_k = 80 * 14.0 * 1.0 * SUSTAINABLE_UTILIZATION
        assert k == pytest.approx(expected_k, abs=0.01)

    def test_multiple_locations_scale_capacity(self):
        state = _make_company(num_locations=3, satisfaction=0.7)
        _q, m, k = derive_competitive_attributes(state, _spec)

        expected_k = (80 * 3) * 14.0 * 1.0 * SUSTAINABLE_UTILIZATION
        assert k == pytest.approx(expected_k, abs=0.01)
        assert m == pytest.approx(MARKETING_BASELINE + 3 * MARKETING_PER_LOCATION)

    def test_commissary_boosts_capacity(self):
        state = _make_company(num_locations=2, extra_nodes=["commissary"])
        _, _, k_with = derive_competitive_attributes(state, _spec)

        state_without = _make_company(num_locations=2)
        _, _, k_without = derive_competitive_attributes(state_without, _spec)

        assert k_with == pytest.approx(k_without * 1.15, abs=1.0)

    def test_distribution_center_boosts_capacity(self):
        state = _make_company(num_locations=2, extra_nodes=["distribution_center"])
        _, _, k_with = derive_competitive_attributes(state, _spec)

        state_without = _make_company(num_locations=2)
        _, _, k_without = derive_competitive_attributes(state_without, _spec)

        assert k_with == pytest.approx(k_without * 1.10, abs=1.0)

    def test_both_infra_nodes_stack(self):
        state = _make_company(
            num_locations=2,
            extra_nodes=["commissary", "distribution_center"],
        )
        _, _, k = derive_competitive_attributes(state, _spec)

        base_k = (80 * 2) * 14.0 * SUSTAINABLE_UTILIZATION
        expected_k = base_k * 1.15 * 1.10
        assert k == pytest.approx(expected_k, abs=1.0)

    def test_marketing_node_increases_marketing(self):
        state = _make_company(num_locations=1, extra_nodes=["marketing"])
        _, m, _ = derive_competitive_attributes(state, _spec)

        base_m = MARKETING_BASELINE + MARKETING_PER_LOCATION
        assert m == pytest.approx(base_m + 15.0)

    def test_delivery_partnership_increases_marketing(self):
        state = _make_company(num_locations=1, extra_nodes=["delivery_partnership"])
        _, m, _ = derive_competitive_attributes(state, _spec)

        base_m = MARKETING_BASELINE + MARKETING_PER_LOCATION
        assert m == pytest.approx(base_m + 10.0)

    def test_catering_increases_marketing(self):
        state = _make_company(num_locations=1, extra_nodes=["catering"])
        _, m, _ = derive_competitive_attributes(state, _spec)

        base_m = MARKETING_BASELINE + MARKETING_PER_LOCATION
        assert m == pytest.approx(base_m + 5.0)

    def test_quality_nodes_boost_quality(self):
        state = _make_company(
            num_locations=1,
            satisfaction=0.7,
            extra_nodes=["quality_assurance", "rnd_menu", "training"],
        )
        q, _, _ = derive_competitive_attributes(state, _spec)

        # q = 0.7 * (1.03) * (1.05) * (1.05) = 0.7 * 1.1355... ≈ 0.7949
        expected_q = 0.7 * 1.03 * 1.05 * 1.05
        assert q == pytest.approx(expected_q, abs=0.01)

    def test_quality_bounded_below(self):
        state = _make_company(num_locations=1, satisfaction=0.0)
        q, _, _ = derive_competitive_attributes(state, _spec)
        assert q >= 0.01

    def test_inactive_nodes_ignored(self):
        state = _make_company(num_locations=1)
        # Add an inactive marketing node
        state.nodes["dead-marketing"] = SimNode(
            id="dead-marketing",
            type="marketing",
            label="Dead Marketing",
            category=NodeCategory.CORPORATE,
            active=False,
        )
        _, m, _ = derive_competitive_attributes(state, _spec)
        base_m = MARKETING_BASELINE + MARKETING_PER_LOCATION
        assert m == pytest.approx(base_m)  # no boost from inactive


# ── allocate_demand_to_locations ──


class TestAllocateDemand:
    def test_zero_customers_returns_zeros(self):
        locations = [
            SimNode(
                id="loc-1",
                type="restaurant",
                label="L1",
                category=NodeCategory.LOCATION,
                location_state=LocationState(satisfaction=0.7),
            ),
        ]
        alloc = allocate_demand_to_locations(0.0, locations)
        assert alloc["loc-1"] == 0.0

    def test_single_location_gets_all_demand(self):
        locations = [
            SimNode(
                id="loc-1",
                type="restaurant",
                label="L1",
                category=NodeCategory.LOCATION,
                location_state=LocationState(satisfaction=0.7),
            ),
        ]
        alloc = allocate_demand_to_locations(100.0, locations)
        assert alloc["loc-1"] == pytest.approx(100.0)

    def test_equal_locations_split_evenly(self):
        locations = [
            SimNode(
                id=f"loc-{i}",
                type="restaurant",
                label=f"L{i}",
                category=NodeCategory.LOCATION,
                location_state=LocationState(satisfaction=0.7, max_capacity=80),
            )
            for i in range(3)
        ]
        alloc = allocate_demand_to_locations(300.0, locations)
        for loc in locations:
            assert alloc[loc.id] == pytest.approx(100.0, abs=0.01)

    def test_higher_satisfaction_gets_more(self):
        loc_good = SimNode(
            id="good",
            type="restaurant",
            label="Good",
            category=NodeCategory.LOCATION,
            location_state=LocationState(satisfaction=0.9, max_capacity=80),
        )
        loc_bad = SimNode(
            id="bad",
            type="restaurant",
            label="Bad",
            category=NodeCategory.LOCATION,
            location_state=LocationState(satisfaction=0.3, max_capacity=80),
        )
        alloc = allocate_demand_to_locations(100.0, [loc_good, loc_bad])
        assert alloc["good"] > alloc["bad"]
        assert alloc["good"] + alloc["bad"] == pytest.approx(100.0)

    def test_larger_capacity_gets_more(self):
        loc_big = SimNode(
            id="big",
            type="restaurant",
            label="Big",
            category=NodeCategory.LOCATION,
            location_state=LocationState(satisfaction=0.7, max_capacity=120),
        )
        loc_small = SimNode(
            id="small",
            type="restaurant",
            label="Small",
            category=NodeCategory.LOCATION,
            location_state=LocationState(satisfaction=0.7, max_capacity=40),
        )
        alloc = allocate_demand_to_locations(100.0, [loc_big, loc_small])
        assert alloc["big"] > alloc["small"]
        # big has 3x the capacity, so ~3x the attractiveness
        assert alloc["big"] == pytest.approx(75.0, abs=0.01)
        assert alloc["small"] == pytest.approx(25.0, abs=0.01)

    def test_empty_locations_list(self):
        alloc = allocate_demand_to_locations(100.0, [])
        assert alloc == {}

    def test_total_allocation_equals_input(self):
        locations = [
            SimNode(
                id=f"loc-{i}",
                type="restaurant",
                label=f"L{i}",
                category=NodeCategory.LOCATION,
                location_state=LocationState(
                    satisfaction=0.3 + 0.2 * i,
                    max_capacity=60 + 10 * i,
                ),
            )
            for i in range(4)
        ]
        alloc = allocate_demand_to_locations(500.0, locations)
        total = sum(alloc.values())
        assert total == pytest.approx(500.0, abs=0.01)
