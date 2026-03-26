"""Regression test: tick_location() standalone mode must be unchanged.

Verifies that adding the `allocated_demand` parameter to tick_location()
didn't break the original standalone logistic growth behavior.
"""

from __future__ import annotations

import random

import pytest

from src.simulation.location import tick_location
from src.simulation.models import LocationState


def _make_state(**overrides) -> LocationState:
    defaults = dict(
        inventory=80.0,
        customers=30.0,
        satisfaction=0.7,
        price=14.0,
        max_capacity=80,
    )
    defaults.update(overrides)
    return LocationState(**defaults)


class TestStandaloneMode:
    """When allocated_demand is None (default), behavior must match original."""

    def test_default_parameter_is_none(self):
        """allocated_demand defaults to None — standalone mode."""
        state = _make_state()
        result, _ = tick_location(state, {}, 50_000.0)
        # Should not raise, customers should grow
        assert result.customers_served >= 0

    def test_customers_grow_with_logistic_curve(self):
        """In standalone mode, customers should follow logistic growth."""
        random.seed(42)
        state = _make_state(customers=30.0, satisfaction=0.7)

        # Run 50 ticks
        for _ in range(50):
            tick_location(state, {}, 50_000.0)

        # Customers should have grown from 30
        assert state.customers > 30.0

    def test_satisfaction_recovers_when_no_stockouts(self):
        """If no customers are turned away, satisfaction ticks up."""
        random.seed(42)
        state = _make_state(customers=20.0, satisfaction=0.5, inventory=200.0)

        for _ in range(20):
            tick_location(state, {}, 50_000.0)

        assert state.satisfaction > 0.5

    def test_inventory_reorder_triggers(self):
        """When inventory drops below reorder_point, a reorder happens."""
        random.seed(42)
        state = _make_state(inventory=25.0, reorder_point=30.0, reorder_qty=100.0)
        result, reorder_cost = tick_location(state, {}, 50_000.0)

        assert reorder_cost > 0
        # Inventory should have been replenished
        assert state.inventory > 25.0

    def test_no_reorder_when_no_cash(self):
        """If company is broke, reorder shouldn't happen."""
        random.seed(42)
        state = _make_state(inventory=5.0, reorder_point=30.0)
        _, reorder_cost = tick_location(state, {}, 0.0)  # zero cash
        assert reorder_cost == 0.0

    def test_modifiers_applied(self):
        """Food cost modifier should change effective costs."""
        random.seed(42)
        state = _make_state(customers=50.0, inventory=200.0)
        result_normal, _ = tick_location(state, {}, 50_000.0)

        random.seed(42)
        state2 = _make_state(customers=50.0, inventory=200.0)
        result_cheap, _ = tick_location(state2, {"food_cost": 0.5}, 50_000.0)

        # Cheaper food cost -> lower costs -> higher profit (same revenue)
        assert result_cheap.profit > result_normal.profit

    def test_spoilage_reduces_inventory(self):
        """Spoilage should reduce inventory each tick."""
        random.seed(42)
        state = _make_state(
            inventory=200.0,
            customers=10.0,
            spoilage_rate=0.1,
            reorder_point=5.0,
        )
        tick_location(state, {}, 50_000.0)
        # With 10 served and ~10% spoilage on remaining ~190: ~19 spoiled
        # Inventory should be well below 200
        assert state.inventory < 200.0


class TestUnifiedMode:
    """When allocated_demand is set, location converges toward that target."""

    def test_convergence_toward_allocated(self):
        """Customers should converge toward allocated_demand."""
        random.seed(42)
        state = _make_state(customers=30.0, satisfaction=0.7)

        # Allocated demand of 60 — customers should move toward 60
        for _ in range(100):
            tick_location(state, {}, 50_000.0, allocated_demand=60.0)

        # Should be closer to 60 than to 30
        assert state.customers > 45.0

    def test_convergence_capped_at_sustainable_capacity(self):
        """Even with high allocated demand, customers cap at 0.85 * max_capacity."""
        random.seed(42)
        state = _make_state(customers=30.0, satisfaction=0.7, max_capacity=80)

        for _ in range(200):
            tick_location(state, {}, 50_000.0, allocated_demand=200.0)

        # Should converge to ~68 (0.85 * 80), not 200
        assert state.customers < 80
        assert state.customers > 50

    def test_zero_allocated_shrinks_demand(self):
        """If allocated_demand is 0, customers should decrease."""
        random.seed(42)
        state = _make_state(customers=50.0, satisfaction=0.7)

        for _ in range(50):
            tick_location(state, {}, 50_000.0, allocated_demand=0.0)

        assert state.customers < 50.0

    def test_allocated_demand_does_not_use_logistic_growth(self):
        """In unified mode, growth rate/reach modifiers should NOT matter."""
        random.seed(42)
        state_unified = _make_state(customers=30.0, satisfaction=0.7)
        for _ in range(50):
            tick_location(
                state_unified, {"customer_growth": 10.0, "customer_reach": 10.0},
                50_000.0, allocated_demand=60.0,
            )

        random.seed(42)
        state_unified2 = _make_state(customers=30.0, satisfaction=0.7)
        for _ in range(50):
            tick_location(
                state_unified2, {},
                50_000.0, allocated_demand=60.0,
            )

        # Both should converge to same value (modifiers don't affect convergence)
        assert state_unified.customers == pytest.approx(state_unified2.customers, abs=0.1)
