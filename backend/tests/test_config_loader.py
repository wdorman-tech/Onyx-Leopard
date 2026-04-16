"""Tests for industry config loading and validation."""

from __future__ import annotations

import pytest

from src.simulation.config_loader import (
    clear_cache,
    list_industry_specs,
    load_industry,
)
from src.simulation.nodes import NODE_REGISTRY


@pytest.fixture(autouse=True)
def _clear_config_cache():
    clear_cache()
    yield
    clear_cache()


class TestLoadRestaurant:
    def test_loads_without_error(self):
        spec = load_industry("restaurant")
        assert spec.meta.slug == "restaurant"
        assert spec.meta.playable is True

    def test_meta_fields(self):
        spec = load_industry("restaurant")
        assert spec.meta.name == "Restaurant / Food Service"
        assert spec.meta.icon == "utensils-crossed"
        assert spec.meta.total_nodes == 50
        assert spec.meta.growth_stages == 4

    def test_roles_defined(self):
        spec = load_industry("restaurant")
        assert spec.roles.location_type == "restaurant"
        assert spec.roles.founder_type == "owner_operator"
        assert "chicken_supplier" in spec.roles.supplier_types
        assert "produce_supplier" in spec.roles.supplier_types

    def test_numbered_labels(self):
        spec = load_industry("restaurant")
        assert spec.roles.numbered_labels["restaurant"] == "Location"
        assert spec.roles.numbered_labels["area_manager"] == "Area Manager"

    def test_node_count_matches_registry(self):
        spec = load_industry("restaurant")
        assert len(spec.nodes) == len(NODE_REGISTRY)

    def test_node_values_match_registry(self):
        """Every node in the YAML must match the hardcoded NODE_REGISTRY values."""
        spec = load_industry("restaurant")
        for node_type, config in NODE_REGISTRY.items():
            key = node_type.value
            assert key in spec.nodes, f"Missing node: {key}"
            node_def = spec.nodes[key]
            assert node_def.label == config.label, f"Label mismatch for {key}"
            assert node_def.category == config.category.value, f"Category mismatch for {key}"
            assert node_def.stage == config.stage, f"Stage mismatch for {key}"
            assert node_def.annual_cost == config.annual_cost, f"Cost mismatch for {key}"
            assert node_def.cost_modifiers == config.cost_modifiers, f"Cost mods mismatch for {key}"
            assert node_def.revenue_modifiers == config.revenue_modifiers, (
                f"Rev mods mismatch for {key}"
            )
            assert node_def.enabled == config.enabled, f"Enabled mismatch for {key}"

    def test_trigger_count(self):
        spec = load_industry("restaurant")
        assert len(spec.triggers) == 19

    def test_trigger_node_types_exist_in_nodes(self):
        spec = load_industry("restaurant")
        for trigger in spec.triggers:
            assert trigger.node_type in spec.nodes, (
                f"Trigger references unknown node type: {trigger.node_type}"
            )

    def test_location_expansion_trigger_exists(self):
        spec = load_industry("restaurant")
        expansion_triggers = [t for t in spec.triggers if t.is_location_expansion]
        assert len(expansion_triggers) == 1
        assert expansion_triggers[0].node_type == "restaurant"

    def test_bridge_fields(self):
        spec = load_industry("restaurant")
        assert spec.bridge.marketing_baseline == 5.0
        assert spec.bridge.marketing_per_location == 1.0
        assert spec.bridge.sustainable_utilization == 0.85
        assert spec.bridge.marketing_contributions["marketing"] == 15.0
        assert "satisfaction_baseline" in spec.bridge.quality_modifier_keys
        assert spec.bridge.infrastructure_multipliers["commissary"] == 1.15

    def test_constants(self):
        spec = load_industry("restaurant")
        assert spec.constants.location_open_cost == 50000
        assert spec.constants.employees_per_location == 15
        assert spec.constants.starting_cash == 50000
        assert len(spec.constants.volume_discounts) == 5

    def test_stages(self):
        spec = load_industry("restaurant")
        assert len(spec.stages) == 4
        assert spec.stages[0].stage == 1
        assert spec.stages[-1].min_locations == 51

    def test_location_defaults(self):
        spec = load_industry("restaurant")
        assert spec.location_defaults.price == 14.0
        assert spec.location_defaults.max_capacity == 80
        assert spec.location_defaults.unified_replenish_amount == 200.0


class TestCaching:
    def test_returns_same_object(self):
        spec1 = load_industry("restaurant")
        spec2 = load_industry("restaurant")
        assert spec1 is spec2

    def test_clear_cache_forces_reload(self):
        spec1 = load_industry("restaurant")
        clear_cache()
        spec2 = load_industry("restaurant")
        assert spec1 is not spec2


class TestValidation:
    def test_unknown_industry_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_industry("nonexistent-industry")


class TestListIndustries:
    def test_lists_at_least_restaurant(self):
        specs = list_industry_specs()
        slugs = [s.meta.slug for s in specs]
        assert "restaurant" in slugs
