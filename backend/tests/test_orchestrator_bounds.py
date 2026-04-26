"""Bounds-checking tests for `_validate_adjust_params` (P-AI-2 — Phase 2.1).

Every CEO `adjust_params` key must be filtered or clamped before it leaves the
orchestrator and reaches the engine. These tests exercise `_validate_adjust_params`
directly on a real (unmocked) `TacticalOrchestrator` instance — no LLM is
invoked. The validator is pure: it does not touch transcript, cost-tracker, or
client, so we can call it directly without setting up the async tick path.

Test list (per checklist 2.1):
    1. test_orchestrator_rejects_negative_price
    2. test_orchestrator_clamps_marketing_intensity_above_2
    3. test_orchestrator_blocks_raise_for_bootstrap_archetype

Bonus coverage (gaps that would otherwise be silent):
    * price upper bound rejection
    * marketing_intensity below-zero clamp
    * replenish_supplier above-1 clamp
    * unknown-key drop
    * non-finite (NaN / Inf) drop
    * raise_amount allowed for venture_growth (positive control)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from src.simulation.library_loader import (
    CategoryCaps,
    NodeDef,
    NodeLibrary,
)
from src.simulation.orchestrator import (
    MARKETING_INTENSITY_LOWER,
    MARKETING_INTENSITY_UPPER,
    PRICE_LOWER_BOUND_MULT,
    PRICE_UPPER_BOUND_MULT,
    REPLENISH_SUPPLIER_LOWER,
    REPLENISH_SUPPLIER_UPPER,
    TacticalOrchestrator,
)
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import CompanySeed
from src.simulation.stance import CeoStance

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — minimal seed/stance/library, mirroring `test_orchestrator_llm.py`
# ─────────────────────────────────────────────────────────────────────────────


def _make_seed(starting_price: float = 100.0) -> CompanySeed:
    return CompanySeed(
        name="Bounds Test Co",
        niche="B2B SaaS for QA testing",
        archetype="small_team",
        industry_keywords=["saas", "qa"],
        location_label="Product",
        economics_model="subscription",
        starting_price=starting_price,
        base_unit_cost=20.0,
        daily_fixed_costs=500.0,
        starting_cash=1_000_000.0,
        starting_employees=5,
        base_capacity_per_location=1000,
        margin_target=0.50,
        revenue_per_employee_target=200_000.0,
        tam=1e8,
        competitor_density=3,
        market_growth_rate=0.10,
        customer_unit_label="subscribers",
        seasonality_amplitude=0.10,
        initial_supplier_types=["cloud_provider"],
        initial_revenue_streams=["subscription_revenue"],
        initial_cost_centers=["engineering_payroll"],
        initial_locations=1,
        initial_marketing_intensity=0.30,
        initial_quality_target=0.75,
        initial_price_position="mid",
        initial_capital_runway_months=12.0,
        initial_hiring_pace="steady",
        initial_geographic_scope="national",
        initial_revenue_concentration=0.40,
        initial_customer_acquisition_channel="content",
    )


def _make_stance(archetype: str = "founder_operator") -> CeoStance:
    return CeoStance(
        archetype=archetype,
        risk_tolerance=0.55,
        growth_obsession=0.50,
        quality_floor=0.70,
        hiring_bias="balanced",
        time_horizon="annual",
        cash_comfort=6.0,
        signature_moves=["stay close to the customer", "hire only when it hurts"],
        voice="I run a tight ship.",
    )


def _make_library() -> NodeLibrary:
    """Single-node library — enough to satisfy `TacticalOrchestrator.__init__`.
    The validator never touches the library, so the contents are irrelevant."""
    return NodeLibrary(
        {
            "founder_engineer": NodeDef(
                key="founder_engineer",
                category="ops",
                label="Founder Engineer",
                hire_cost=0.0,
                daily_fixed_costs=0.0,
                employees_count=1,
                capacity_contribution=0,
                modifier_keys={},
                prerequisites=[],
                category_caps=CategoryCaps(soft_cap=1, hard_cap=1),
                applicable_economics=["subscription", "service"],
            )
        }
    )


def _make_orch(
    tmp_path: Path,
    *,
    seed: CompanySeed | None = None,
    stance: CeoStance | None = None,
) -> TacticalOrchestrator:
    """Construct a real TacticalOrchestrator (no mock client needed — we never
    call `tick`, only the synchronous `_validate_adjust_params` helper)."""
    return TacticalOrchestrator(
        seed=seed or _make_seed(),
        stance=stance or _make_stance(),
        library=_make_library(),
        transcript=Transcript(tmp_path / "sim.jsonl", mode="off"),
        cost_tracker=CostTracker(ceiling_usd=50.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Required tests (per checklist 2.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_orchestrator_rejects_negative_price(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Negative price must be dropped — never propagated to the engine."""
    seed = _make_seed(starting_price=100.0)
    orch = _make_orch(tmp_path, seed=seed)

    with caplog.at_level(logging.WARNING, logger="src.simulation.orchestrator"):
        result = orch._validate_adjust_params({"price": -10.0}, seed, orch.stance)

    assert result == {}
    # Warning surfaced with the offending value so operators can audit drift.
    assert any("price" in rec.message and "-10" in rec.message for rec in caplog.records)


def test_orchestrator_clamps_marketing_intensity_above_2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """marketing_intensity above the upper bound must be clamped, not dropped."""
    orch = _make_orch(tmp_path)

    with caplog.at_level(logging.WARNING, logger="src.simulation.orchestrator"):
        result = orch._validate_adjust_params(
            {"marketing_intensity": 5.0}, orch.seed, orch.stance
        )

    assert result == {"marketing_intensity": MARKETING_INTENSITY_UPPER}
    assert result["marketing_intensity"] == 2.0
    assert any("marketing_intensity" in rec.message for rec in caplog.records)


def test_orchestrator_blocks_raise_for_bootstrap_archetype(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Stance gate: only `venture_growth` / `consolidator` may submit raise_amount.

    A `founder_operator` stance proposing a raise gets it dropped. The same
    decision under a `venture_growth` stance survives unchanged — proving the
    drop is gate-driven, not unconditional.
    """
    seed = _make_seed()

    bootstrap_orch = _make_orch(tmp_path, stance=_make_stance("founder_operator"))
    with caplog.at_level(logging.WARNING, logger="src.simulation.orchestrator"):
        bootstrap_result = bootstrap_orch._validate_adjust_params(
            {"raise_amount": 1000.0}, seed, bootstrap_orch.stance
        )
    assert bootstrap_result == {}
    assert any(
        "raise_amount" in rec.message and "founder_operator" in rec.message
        for rec in caplog.records
    )

    # Positive control: a venture_growth stance keeps the raise.
    vg_orch = _make_orch(tmp_path, stance=_make_stance("venture_growth"))
    vg_result = vg_orch._validate_adjust_params(
        {"raise_amount": 1000.0}, seed, vg_orch.stance
    )
    assert vg_result == {"raise_amount": 1000.0}


# ─────────────────────────────────────────────────────────────────────────────
# Bonus coverage — gaps the named tests would leave behind
# ─────────────────────────────────────────────────────────────────────────────


def test_orchestrator_rejects_price_above_upper_bound(tmp_path: Path) -> None:
    """price > seed.starting_price * PRICE_UPPER_BOUND_MULT must be dropped.

    Lower bound has its own named test; the upper bound deserves symmetric
    coverage to prove both halves of the gate fire.
    """
    seed = _make_seed(starting_price=100.0)
    orch = _make_orch(tmp_path, seed=seed)
    upper = seed.starting_price * PRICE_UPPER_BOUND_MULT  # 1000.0

    # Just inside the bound passes.
    inside = orch._validate_adjust_params({"price": upper}, seed, orch.stance)
    assert inside == {"price": upper}

    # Just outside is dropped.
    outside = orch._validate_adjust_params({"price": upper + 1.0}, seed, orch.stance)
    assert outside == {}


def test_orchestrator_rejects_price_below_lower_bound(tmp_path: Path) -> None:
    """price < seed.starting_price * PRICE_LOWER_BOUND_MULT must be dropped.

    Distinct from the negative-price test: a small POSITIVE price below the
    floor (e.g. predatory pricing) must still be rejected by the band check.
    """
    seed = _make_seed(starting_price=100.0)
    orch = _make_orch(tmp_path, seed=seed)
    lower = seed.starting_price * PRICE_LOWER_BOUND_MULT  # 10.0

    # At the floor passes.
    at_floor = orch._validate_adjust_params({"price": lower}, seed, orch.stance)
    assert at_floor == {"price": lower}

    # 1 cent below is dropped.
    below = orch._validate_adjust_params({"price": lower - 0.01}, seed, orch.stance)
    assert below == {}


def test_orchestrator_clamps_marketing_intensity_below_zero(tmp_path: Path) -> None:
    """Negative marketing_intensity clamps UP to the lower bound (not dropped)."""
    orch = _make_orch(tmp_path)
    result = orch._validate_adjust_params(
        {"marketing_intensity": -3.0}, orch.seed, orch.stance
    )
    assert result == {"marketing_intensity": MARKETING_INTENSITY_LOWER}


def test_orchestrator_clamps_replenish_supplier_above_one(tmp_path: Path) -> None:
    """replenish_supplier > 1.0 clamps to the upper bound (not dropped)."""
    orch = _make_orch(tmp_path)
    result = orch._validate_adjust_params(
        {"replenish_supplier": 7.5}, orch.seed, orch.stance
    )
    assert result == {"replenish_supplier": REPLENISH_SUPPLIER_UPPER}
    assert result["replenish_supplier"] == 1.0


def test_orchestrator_clamps_replenish_supplier_below_zero(tmp_path: Path) -> None:
    """Negative replenish_supplier clamps to 0 (no restock)."""
    orch = _make_orch(tmp_path)
    result = orch._validate_adjust_params(
        {"replenish_supplier": -0.5}, orch.seed, orch.stance
    )
    assert result == {"replenish_supplier": REPLENISH_SUPPLIER_LOWER}


def test_orchestrator_drops_unknown_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Anything not in KNOWN_ADJUST_PARAMS_KEYS is dropped with a warning.

    The orchestrator must not silently forward novel keys to the engine —
    that would defeat the bounds gate the moment the model invents one.
    """
    orch = _make_orch(tmp_path)
    with caplog.at_level(logging.WARNING, logger="src.simulation.orchestrator"):
        result = orch._validate_adjust_params(
            {"hire_cmo_at_salary": 250000.0}, orch.seed, orch.stance
        )
    assert result == {}
    assert any("hire_cmo_at_salary" in rec.message for rec in caplog.records)


def test_orchestrator_drops_non_finite_values(tmp_path: Path) -> None:
    """NaN and ±Inf can't be clamped meaningfully — drop them on every key."""
    orch = _make_orch(tmp_path)
    bogus = {
        "price": float("nan"),
        "marketing_intensity": float("inf"),
        "replenish_supplier": float("-inf"),
    }
    result = orch._validate_adjust_params(bogus, orch.seed, orch.stance)
    assert result == {}


def test_orchestrator_validate_returns_new_dict(tmp_path: Path) -> None:
    """Validator must return a NEW dict — caller's input is never mutated.

    The orchestrator's `_filter_invalid_decision` relies on this: it compares
    the returned dict to `decision.adjust_params` to decide whether to
    rebuild the decision. Mutation would break that comparison.
    """
    orch = _make_orch(tmp_path)
    incoming = {"marketing_intensity": 5.0}
    result = orch._validate_adjust_params(incoming, orch.seed, orch.stance)

    assert result is not incoming
    # Input untouched even though the value got clamped on output.
    assert incoming == {"marketing_intensity": 5.0}
    assert result == {"marketing_intensity": MARKETING_INTENSITY_UPPER}


def test_orchestrator_validate_preserves_in_band_values(tmp_path: Path) -> None:
    """All-valid input round-trips unchanged — no spurious warnings or drops.

    Catches regressions where a tighter-than-spec bound silently mutates
    legitimate decisions.
    """
    seed = _make_seed(starting_price=100.0)
    vg_orch = _make_orch(tmp_path, seed=seed, stance=_make_stance("venture_growth"))
    incoming = {
        "price": 150.0,
        "marketing_intensity": 1.2,
        "raise_amount": 50_000.0,
        "replenish_supplier": 0.5,
    }
    result = vg_orch._validate_adjust_params(incoming, seed, vg_orch.stance)
    assert result == incoming
    # Sanity-check the bounds we're asserting against — protects against a
    # constants-renumbering accidentally turning this test into a no-op.
    assert not math.isclose(MARKETING_INTENSITY_UPPER, 1.2)
