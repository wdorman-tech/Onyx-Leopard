"""Unit tests for the data-driven trigger condition evaluator."""

from __future__ import annotations

from src.simulation.conditions import evaluate_condition

# ── Simple metric comparisons ──


class TestMetricComparisons:
    def test_greater_than_true(self):
        assert evaluate_condition(
            {"monthly_revenue": {">": 15000}},
            {"monthly_revenue": 20000},
            {},
        )

    def test_greater_than_false(self):
        assert not evaluate_condition(
            {"monthly_revenue": {">": 15000}},
            {"monthly_revenue": 10000},
            {},
        )

    def test_greater_than_equal_boundary(self):
        assert evaluate_condition(
            {"location_count": {">=": 3}},
            {"location_count": 3},
            {},
        )

    def test_greater_than_equal_below(self):
        assert not evaluate_condition(
            {"location_count": {">=": 3}},
            {"location_count": 2},
            {},
        )

    def test_less_than(self):
        assert evaluate_condition(
            {"cash": {"<": 5000}},
            {"cash": 3000},
            {},
        )

    def test_less_than_equal(self):
        assert evaluate_condition(
            {"cash": {"<=": 5000}},
            {"cash": 5000},
            {},
        )

    def test_equals(self):
        assert evaluate_condition(
            {"stage": {"==": 3}},
            {"stage": 3},
            {},
        )

    def test_missing_metric_defaults_to_zero(self):
        assert not evaluate_condition(
            {"monthly_revenue": {">": 15000}},
            {},
            {},
        )


# ── Boolean combinators ──


class TestCombinators:
    def test_all_true(self):
        assert evaluate_condition(
            {"all": [
                {"cash": {">": 80000}},
                {"avg_location_margin": {">=": 0.10}},
            ]},
            {"cash": 100000, "avg_location_margin": 0.15},
            {},
        )

    def test_all_one_false(self):
        assert not evaluate_condition(
            {"all": [
                {"cash": {">": 80000}},
                {"avg_location_margin": {">=": 0.10}},
            ]},
            {"cash": 100000, "avg_location_margin": 0.05},
            {},
        )

    def test_any_one_true(self):
        assert evaluate_condition(
            {"any": [
                {"total_employees": {">": 50}},
                {"location_count": {">=": 4}},
            ]},
            {"total_employees": 30, "location_count": 5},
            {},
        )

    def test_any_none_true(self):
        assert not evaluate_condition(
            {"any": [
                {"total_employees": {">": 50}},
                {"location_count": {">=": 4}},
            ]},
            {"total_employees": 30, "location_count": 2},
            {},
        )

    def test_nested_all_with_has_node(self):
        assert evaluate_condition(
            {"all": [
                {"cash": {">": 80000}},
                {"avg_location_margin": {">=": 0.10}},
                {"has_node": "general_manager"},
            ]},
            {"cash": 100000, "avg_location_margin": 0.15},
            {"general_manager": 1},
        )


# ── Special checks ──


class TestSpecialChecks:
    def test_has_node_present(self):
        assert evaluate_condition(
            {"has_node": "general_manager"},
            {},
            {"general_manager": 1},
        )

    def test_has_node_absent(self):
        assert not evaluate_condition(
            {"has_node": "general_manager"},
            {},
            {},
        )

    def test_has_node_zero_count(self):
        assert not evaluate_condition(
            {"has_node": "general_manager"},
            {},
            {"general_manager": 0},
        )

    def test_node_count_ratio_needs_more(self):
        # 6 locations, 0 area managers -> should fire (6 > 0 * 6)
        assert evaluate_condition(
            {"node_count_ratio": {"node_type": "area_manager", "per": 6}},
            {"location_count": 6},
            {"area_manager": 0},
        )

    def test_node_count_ratio_already_staffed(self):
        # 6 locations, 1 area manager -> should NOT fire (6 > 1 * 6 is false)
        assert not evaluate_condition(
            {"node_count_ratio": {"node_type": "area_manager", "per": 6}},
            {"location_count": 6},
            {"area_manager": 1},
        )

    def test_node_count_ratio_needs_second(self):
        # 13 locations, 1 area manager -> should fire (13 > 1 * 6)
        assert evaluate_condition(
            {"node_count_ratio": {"node_type": "area_manager", "per": 6}},
            {"location_count": 13},
            {"area_manager": 1},
        )

    def test_node_count_ratio_below_minimum(self):
        # 4 locations -> should NOT fire (4 < 6 minimum)
        assert not evaluate_condition(
            {"node_count_ratio": {"node_type": "area_manager", "per": 6}},
            {"location_count": 4},
            {"area_manager": 0},
        )


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_condition(self):
        assert evaluate_condition({}, {}, {})

    def test_all_empty_list(self):
        assert evaluate_condition({"all": []}, {}, {})

    def test_any_empty_list(self):
        assert not evaluate_condition({"any": []}, {}, {})
