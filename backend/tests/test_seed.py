"""Tests for `CompanySeed` and archetype samplers (Onyx Leopard v2)."""

from __future__ import annotations

import json
import random
import statistics

import pytest
from pydantic import ValidationError

from src.simulation.seed import (
    ARCHETYPES,
    CompanySeed,
    sample_seed_for_archetype,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _valid_seed_kwargs() -> dict:
    """Minimal-valid `CompanySeed` payload for direct construction tests."""
    return {
        "name": "Test Co.",
        "niche": "Test niche for unit tests",
        "archetype": "small_team",
        "industry_keywords": ["test", "fixture"],
        "location_label": "Location",
        "economics_model": "physical",
        "starting_price": 14.0,
        "base_unit_cost": 4.0,
        "daily_fixed_costs": 300.0,
        "starting_cash": 50_000.0,
        "starting_employees": 5,
        "base_capacity_per_location": 80,
        "margin_target": 0.55,
        "revenue_per_employee_target": 200_000.0,
        "tam": 1e8,
        "competitor_density": 4,
        "market_growth_rate": 0.05,
        "customer_unit_label": "diners",
        "seasonality_amplitude": 0.15,
        "initial_supplier_types": ["primary_goods_supplier"],
        "initial_revenue_streams": ["storefront_sales"],
        "initial_cost_centers": ["cogs", "labor"],
        "initial_locations": 1,
        "initial_marketing_intensity": 0.3,
        "initial_quality_target": 0.7,
        "initial_price_position": "mid",
        "initial_capital_runway_months": 12.0,
        "initial_hiring_pace": "steady",
        "initial_geographic_scope": "local",
        "initial_revenue_concentration": 0.4,
        "initial_customer_acquisition_channel": "word_of_mouth",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Field count — check we have ~30 fields as spec says
# ─────────────────────────────────────────────────────────────────────────────


def test_field_count_matches_spec():
    """Spec §2 calls for 5 + 8 + 5 + 12 = 30 fields."""
    fields = CompanySeed.model_fields
    # Allow ±2 for the spec's slight ambiguity (8 economics fields enumerates 7)
    assert 28 <= len(fields) <= 32, (
        f"expected ~30 fields per v2 spec §2, got {len(fields)}: {sorted(fields.keys())}"
    )


def test_required_field_groups_present():
    """Every spec'd field name must exist on the model."""
    expected = {
        # Identity (5)
        "name",
        "niche",
        "archetype",
        "industry_keywords",
        "location_label",
        # Economics (8)
        "economics_model",
        "starting_price",
        "base_unit_cost",
        "daily_fixed_costs",
        "starting_cash",
        "starting_employees",
        "base_capacity_per_location",
        "margin_target",
        # Market (5)
        "tam",
        "competitor_density",
        "market_growth_rate",
        "customer_unit_label",
        "seasonality_amplitude",
        # Initial Setup (3 lists, spec'd)
        "initial_supplier_types",
        "initial_revenue_streams",
        "initial_cost_centers",
    }
    actual = set(CompanySeed.model_fields.keys())
    missing = expected - actual
    assert not missing, f"missing spec fields: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip serialization
# ─────────────────────────────────────────────────────────────────────────────


def test_json_round_trip_minimal():
    seed = CompanySeed(**_valid_seed_kwargs())
    payload = seed.model_dump_json()
    restored = CompanySeed.model_validate_json(payload)
    assert restored == seed


def test_json_round_trip_via_dict():
    seed = CompanySeed(**_valid_seed_kwargs())
    as_dict = seed.model_dump()
    # Round-trip through json.dumps/loads to guarantee no funky types leak
    encoded = json.dumps(as_dict)
    decoded = json.loads(encoded)
    restored = CompanySeed.model_validate(decoded)
    assert restored == seed


def test_json_round_trip_all_archetypes():
    rng = random.Random(7)
    for archetype in ARCHETYPES:
        seed = sample_seed_for_archetype(archetype, rng=rng)
        restored = CompanySeed.model_validate_json(seed.model_dump_json())
        assert restored == seed


# ─────────────────────────────────────────────────────────────────────────────
# Field validation — bad inputs should raise
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_negative_starting_cash():
    bad = _valid_seed_kwargs() | {"starting_cash": -100.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_negative_starting_price():
    bad = _valid_seed_kwargs() | {"starting_price": -1.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_zero_starting_price():
    """starting_price has gt=0 — zero is invalid (no business model)."""
    bad = _valid_seed_kwargs() | {"starting_price": 0.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_negative_base_unit_cost():
    bad = _valid_seed_kwargs() | {"base_unit_cost": -0.5}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_unit_cost_above_price():
    """Negative gross margin at t=0 is rejected (cross-field validator)."""
    bad = _valid_seed_kwargs() | {"starting_price": 10.0, "base_unit_cost": 15.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_invalid_economics_model():
    bad = _valid_seed_kwargs() | {"economics_model": "barter"}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_invalid_archetype():
    bad = _valid_seed_kwargs() | {"archetype": "unicorn_decacorn"}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_margin_target_above_one():
    bad = _valid_seed_kwargs() | {"margin_target": 1.5}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_margin_target_negative():
    bad = _valid_seed_kwargs() | {"margin_target": -0.1}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_competitor_density_above_ten():
    bad = _valid_seed_kwargs() | {"competitor_density": 11}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_competitor_density_negative():
    bad = _valid_seed_kwargs() | {"competitor_density": -1}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_zero_starting_employees():
    bad = _valid_seed_kwargs() | {"starting_employees": 0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_zero_base_capacity():
    bad = _valid_seed_kwargs() | {"base_capacity_per_location": 0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_zero_tam():
    bad = _valid_seed_kwargs() | {"tam": 0.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_seasonality_above_one():
    bad = _valid_seed_kwargs() | {"seasonality_amplitude": 1.5}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_market_growth_extreme_negative():
    bad = _valid_seed_kwargs() | {"market_growth_rate": -0.99}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_empty_supplier_list():
    bad = _valid_seed_kwargs() | {"initial_supplier_types": []}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_too_many_supplier_entries():
    bad = _valid_seed_kwargs() | {
        "initial_supplier_types": ["a", "b", "c", "d"],
    }
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_blank_keyword_in_list():
    bad = _valid_seed_kwargs() | {"industry_keywords": ["valid", "  "]}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_invalid_price_position():
    bad = _valid_seed_kwargs() | {"initial_price_position": "free"}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_invalid_hiring_pace():
    bad = _valid_seed_kwargs() | {"initial_hiring_pace": "yolo"}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_extra_fields():
    bad = _valid_seed_kwargs() | {"unknown_field": 42}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_runway_months_too_long():
    bad = _valid_seed_kwargs() | {"initial_capital_runway_months": 200.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


def test_rejects_runway_months_zero():
    bad = _valid_seed_kwargs() | {"initial_capital_runway_months": 0.0}
    with pytest.raises(ValidationError):
        CompanySeed(**bad)


# ─────────────────────────────────────────────────────────────────────────────
# Archetype samplers produce valid seeds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_sampler_returns_valid_seed(archetype):
    rng = random.Random(0)
    seed = sample_seed_for_archetype(archetype, rng=rng)
    assert isinstance(seed, CompanySeed)
    assert seed.archetype == archetype


def test_sampler_is_deterministic_with_same_rng_seed():
    a = sample_seed_for_archetype("small_team", rng=random.Random(42))
    b = sample_seed_for_archetype("small_team", rng=random.Random(42))
    assert a == b


def test_sampler_produces_variation_with_different_seeds():
    a = sample_seed_for_archetype("venture_funded", rng=random.Random(1))
    b = sample_seed_for_archetype("venture_funded", rng=random.Random(2))
    # They might happen to produce same economics_model but starting_price will diverge
    assert a != b


def test_sampler_unknown_archetype_raises():
    with pytest.raises(ValueError, match="unknown archetype"):
        sample_seed_for_archetype("solo_unicorn")  # type: ignore[arg-type]


def test_sampler_respects_overrides():
    seed = sample_seed_for_archetype(
        "small_team",
        rng=random.Random(0),
        name="Custom Name",
        niche="custom niche",
        industry_keywords=["a", "b"],
        economics_model="subscription",
    )
    assert seed.name == "Custom Name"
    assert seed.niche == "custom niche"
    assert seed.industry_keywords == ["a", "b"]
    assert seed.economics_model == "subscription"


# ─────────────────────────────────────────────────────────────────────────────
# Distribution sanity — 100 sampled seeds per archetype
# ─────────────────────────────────────────────────────────────────────────────


def _sample_n(archetype, n: int, rng_seed: int = 1234) -> list[CompanySeed]:
    rng = random.Random(rng_seed)
    return [sample_seed_for_archetype(archetype, rng=rng) for _ in range(n)]


def test_solo_founder_distribution_realistic():
    seeds = _sample_n("solo_founder", 100)
    cash = [s.starting_cash for s in seeds]
    employees = [s.starting_employees for s in seeds]
    assert max(cash) <= 100_000.0
    assert min(cash) >= 15_000.0
    assert max(employees) <= 3
    assert min(employees) >= 1
    # Service-economics should dominate (weight 0.65)
    economics = [s.economics_model for s in seeds]
    service_count = economics.count("service")
    assert service_count >= 40, (
        f"expected service-heavy solo_founder distribution, got {service_count}/100"
    )


def test_small_team_distribution_realistic():
    seeds = _sample_n("small_team", 100)
    cash = [s.starting_cash for s in seeds]
    employees = [s.starting_employees for s in seeds]
    assert all(40_000.0 <= c <= 200_000.0 for c in cash)
    assert all(3 <= e <= 15 for e in employees)
    # Mean cash should land near the band midpoint (~120k)
    assert 80_000.0 <= statistics.mean(cash) <= 160_000.0


def test_venture_funded_distribution_realistic():
    seeds = _sample_n("venture_funded", 100)
    cash = [s.starting_cash for s in seeds]
    assert all(500_000.0 <= c <= 15_000_000.0 for c in cash)
    # Subscription should dominate (weight 0.70)
    economics = [s.economics_model for s in seeds]
    assert economics.count("subscription") >= 50


def test_enterprise_distribution_realistic():
    seeds = _sample_n("enterprise", 100)
    cash = [s.starting_cash for s in seeds]
    employees = [s.starting_employees for s in seeds]
    tam = [s.tam for s in seeds]
    assert all(c >= 10_000_000.0 for c in cash)
    assert all(e >= 200 for e in employees)
    assert all(t >= 1e9 for t in tam)


def test_all_archetypes_produce_valid_margins():
    for archetype in ARCHETYPES:
        for s in _sample_n(archetype, 100):
            assert 0.05 <= s.margin_target <= 0.85
            # And by construction unit cost cannot exceed price
            assert s.base_unit_cost <= s.starting_price


def test_all_archetypes_produce_valid_competitor_density():
    for archetype in ARCHETYPES:
        for s in _sample_n(archetype, 100):
            assert 0 <= s.competitor_density <= 10


def test_all_archetypes_produce_valid_seasonality():
    for archetype in ARCHETYPES:
        for s in _sample_n(archetype, 100):
            assert 0.0 <= s.seasonality_amplitude <= 1.0


def test_all_archetypes_set_appropriate_lists():
    """Sampler must populate all three node-library reference lists."""
    for archetype in ARCHETYPES:
        for s in _sample_n(archetype, 50):
            assert 1 <= len(s.initial_supplier_types) <= 3
            assert 1 <= len(s.initial_revenue_streams) <= 3
            assert 1 <= len(s.initial_cost_centers) <= 3


def test_all_archetypes_round_trip():
    """Every sampled seed across all archetypes must JSON round-trip."""
    for archetype in ARCHETYPES:
        for s in _sample_n(archetype, 20):
            assert CompanySeed.model_validate_json(s.model_dump_json()) == s
