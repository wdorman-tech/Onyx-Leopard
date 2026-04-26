"""Unit tests for `bridge.py` v2 — auto-derived modifier aggregation.

Covers the v2 contract added in the §6 refactor:
    * `aggregate_modifiers(library, spawned_nodes) -> dict[str, float]`
    * `bucket_modifiers(modifiers) -> BridgeAggregate`
    * Suffix taxonomy partition (quality / marketing / infrastructure / other)
    * Hard-fail on unknown node_keys

The v1 tests in `test_bridge.py` are intentionally untouched — they pin the
legacy `derive_competitive_attributes*` API that wave 4 will delete.
"""

from __future__ import annotations

import pytest

from src.simulation.bridge import (
    BridgeAggregate,
    aggregate_modifiers,
    bucket_modifiers,
    derive_bridge_aggregate,
)
from src.simulation.library_loader import CategoryCaps, NodeDef, NodeLibrary

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _node(
    key: str,
    *,
    modifier_keys: dict[str, float] | None = None,
    category: str = "ops",
) -> NodeDef:
    """Build a minimal valid `NodeDef`. Tests only care about modifier_keys."""
    return NodeDef(
        key=key,
        category=category,
        label=key.replace("_", " ").title(),
        hire_cost=0.0,
        daily_fixed_costs=0.0,
        employees_count=0,
        capacity_contribution=0,
        modifier_keys=modifier_keys or {},
        prerequisites=[],
        category_caps=CategoryCaps(soft_cap=1, hard_cap=10),
        applicable_economics=["subscription"],
    )


def _library(*nodes: NodeDef) -> NodeLibrary:
    """Build a `NodeLibrary` directly from `NodeDef`s, skipping load validation."""
    return NodeLibrary({n.key: n for n in nodes})


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_modifiers
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateModifiers:
    def test_empty_spawned_nodes_returns_empty_dict(self):
        lib = _library(_node("foo", modifier_keys={"churn_reduction": 0.10}))
        assert aggregate_modifiers(lib, {}) == {}

    def test_zero_count_node_is_skipped(self):
        lib = _library(_node("bd_rep", modifier_keys={"pipeline_strength": 0.10}))
        assert aggregate_modifiers(lib, {"bd_rep": 0}) == {}

    def test_single_node_matches_library_entry(self):
        lib = _library(
            _node(
                "bd_rep",
                modifier_keys={"pipeline_strength": 0.10, "sales_velocity": 0.05},
            )
        )
        out = aggregate_modifiers(lib, {"bd_rep": 1})
        assert out == {"pipeline_strength": 0.10, "sales_velocity": 0.05}

    def test_three_copies_scales_linearly(self):
        lib = _library(_node("bd_rep", modifier_keys={"pipeline_strength": 0.10}))
        out = aggregate_modifiers(lib, {"bd_rep": 3})
        # 3 * 0.10 = 0.30. Floating-point — use approx.
        assert out["pipeline_strength"] == pytest.approx(0.30)

    def test_multiple_nodes_share_modifier_key_sum_correctly(self):
        lib = _library(
            _node("bd_rep", modifier_keys={"pipeline_strength": 0.10}),
            _node("ae_rep", modifier_keys={"pipeline_strength": 0.18}),
        )
        out = aggregate_modifiers(lib, {"bd_rep": 2, "ae_rep": 1})
        # 2 * 0.10 + 1 * 0.18 = 0.38
        assert out["pipeline_strength"] == pytest.approx(0.38)

    def test_unknown_node_key_raises_keyerror(self):
        lib = _library(_node("bd_rep", modifier_keys={"pipeline_strength": 0.10}))
        with pytest.raises(KeyError, match="unknown node_key"):
            aggregate_modifiers(lib, {"phantom_role": 1})

    def test_negative_count_raises_valueerror(self):
        lib = _library(_node("bd_rep", modifier_keys={"pipeline_strength": 0.10}))
        with pytest.raises(ValueError, match="cannot be negative"):
            aggregate_modifiers(lib, {"bd_rep": -1})

    def test_node_with_empty_modifier_keys_contributes_nothing(self):
        # The founder / a location / a passive supplier carry no modifiers —
        # their presence in spawned_nodes must not invent keys.
        lib = _library(
            _node("founder", modifier_keys={}),
            _node("bd_rep", modifier_keys={"pipeline_strength": 0.10}),
        )
        out = aggregate_modifiers(lib, {"founder": 1, "bd_rep": 1})
        assert out == {"pipeline_strength": 0.10}


# ─────────────────────────────────────────────────────────────────────────────
# bucket_modifiers — suffix taxonomy
# ─────────────────────────────────────────────────────────────────────────────


class TestBucketModifiers:
    def test_empty_input_yields_empty_buckets(self):
        agg = bucket_modifiers({})
        assert agg.quality == {}
        assert agg.marketing == {}
        assert agg.infrastructure == {}
        assert agg.other == {}

    def test_returns_bridge_aggregate_pydantic_model(self):
        agg = bucket_modifiers({"churn_reduction": 0.1})
        assert isinstance(agg, BridgeAggregate)

    # ── Quality bucket ──

    def test_churn_reduction_lands_in_quality(self):
        agg = bucket_modifiers({"churn_reduction": 0.15})
        assert agg.quality == {"churn_reduction": 0.15}
        assert agg.marketing == {}
        assert agg.infrastructure == {}
        assert agg.other == {}

    def test_retention_suffix_lands_in_quality(self):
        agg = bucket_modifiers({"client_retention": 0.10, "retainer_retention": 0.20})
        assert agg.quality == {"client_retention": 0.10, "retainer_retention": 0.20}
        assert agg.other == {}

    def test_satisfaction_suffix_lands_in_quality(self):
        agg = bucket_modifiers({"customer_satisfaction": 0.05})
        assert agg.quality == {"customer_satisfaction": 0.05}

    def test_quality_suffix_lands_in_quality(self):
        agg = bucket_modifiers({"build_quality": 0.07})
        assert agg.quality == {"build_quality": 0.07}

    # ── Marketing bucket ──

    def test_pipeline_strength_lands_in_marketing(self):
        agg = bucket_modifiers({"pipeline_strength": 0.10})
        assert agg.marketing == {"pipeline_strength": 0.10}
        assert agg.quality == {}

    def test_marketing_suffix_lands_in_marketing(self):
        agg = bucket_modifiers({"content_marketing": 0.08})
        assert agg.marketing == {"content_marketing": 0.08}

    def test_lead_gen_suffix_lands_in_marketing(self):
        agg = bucket_modifiers({"outbound_lead_gen": 0.12})
        assert agg.marketing == {"outbound_lead_gen": 0.12}

    def test_brand_suffix_lands_in_marketing(self):
        agg = bucket_modifiers({"category_brand": 0.04})
        assert agg.marketing == {"category_brand": 0.04}

    def test_awareness_suffix_lands_in_marketing(self):
        agg = bucket_modifiers({"market_awareness": 0.06})
        assert agg.marketing == {"market_awareness": 0.06}

    # ── Infrastructure bucket ──

    def test_capacity_uplift_lands_in_infrastructure(self):
        agg = bucket_modifiers({"capacity_uplift": 0.20})
        assert agg.infrastructure == {"capacity_uplift": 0.20}

    def test_infrastructure_suffix_lands_in_infrastructure(self):
        agg = bucket_modifiers({"data_infrastructure": 0.15})
        assert agg.infrastructure == {"data_infrastructure": 0.15}

    def test_throughput_suffix_lands_in_infrastructure(self):
        agg = bucket_modifiers({"pipeline_throughput": 0.11})
        assert agg.infrastructure == {"pipeline_throughput": 0.11}

    def test_efficiency_suffix_lands_in_infrastructure(self):
        agg = bucket_modifiers({"infra_efficiency": 0.10})
        assert agg.infrastructure == {"infra_efficiency": 0.10}

    # ── Other bucket ──

    def test_unmatched_keys_land_in_other(self):
        # `customer_growth` and `delivery_cost` don't end in any bucket tag.
        # `food_cost` likewise. They should all land in `other`.
        agg = bucket_modifiers(
            {"customer_growth": 0.20, "delivery_cost": -0.10, "food_cost": -0.08},
        )
        assert agg.other == {
            "customer_growth": 0.20,
            "delivery_cost": -0.10,
            "food_cost": -0.08,
        }
        assert agg.quality == {}
        assert agg.marketing == {}
        assert agg.infrastructure == {}

    def test_partial_string_match_does_not_classify(self):
        # `marketing_baseline` does NOT end in `_marketing`, it ends in
        # `_baseline`. Per strict suffix semantics this lands in `other` —
        # the taxonomy is intentional, not coincidental.
        agg = bucket_modifiers({"marketing_baseline": 5.0})
        assert agg.other == {"marketing_baseline": 5.0}
        assert agg.marketing == {}

    # ── Mixed input — full taxonomy exercised in one shot ──

    def test_three_node_mix_produces_all_four_buckets(self):
        # One key per bucket plus an unmatched key, mimicking a real CEO mid-game:
        #   sales hire   → pipeline_strength      → marketing
        #   support team → churn_reduction        → quality
        #   data layer   → infrastructure_efficiency-style → infrastructure
        #   founder boost→ customer_growth        → other
        agg = bucket_modifiers(
            {
                "pipeline_strength": 0.10,
                "churn_reduction": 0.15,
                "infra_efficiency": 0.10,
                "customer_growth": 0.25,
            },
        )
        assert agg.marketing == {"pipeline_strength": 0.10}
        assert agg.quality == {"churn_reduction": 0.15}
        assert agg.infrastructure == {"infra_efficiency": 0.10}
        assert agg.other == {"customer_growth": 0.25}


# ─────────────────────────────────────────────────────────────────────────────
# derive_bridge_aggregate — end-to-end composition
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveBridgeAggregate:
    def test_end_to_end_three_node_mix_all_buckets_populated(self):
        lib = _library(
            _node(
                "ae_rep",
                modifier_keys={"pipeline_strength": 0.10},
                category="sales",
            ),
            _node(
                "support_lead",
                modifier_keys={"churn_reduction": 0.15},
                category="ops",
            ),
            _node(
                "platform_eng",
                modifier_keys={"infra_efficiency": 0.10, "customer_growth": 0.05},
                category="ops",
            ),
        )
        agg = derive_bridge_aggregate(
            lib, {"ae_rep": 2, "support_lead": 1, "platform_eng": 1}
        )

        # 2 * 0.10 = 0.20
        assert agg.marketing == {"pipeline_strength": pytest.approx(0.20)}
        assert agg.quality == {"churn_reduction": pytest.approx(0.15)}
        assert agg.infrastructure == {"infra_efficiency": pytest.approx(0.10)}
        assert agg.other == {"customer_growth": pytest.approx(0.05)}

    def test_empty_spawned_yields_all_empty_buckets(self):
        # Early-tick reality: only the founder is spawned and the founder
        # carries no modifier_keys. All four buckets must be empty dicts —
        # never raise, never invent.
        lib = _library(_node("founder", modifier_keys={}, category="exec"))
        agg = derive_bridge_aggregate(lib, {"founder": 1})
        assert agg.quality == {}
        assert agg.marketing == {}
        assert agg.infrastructure == {}
        assert agg.other == {}
