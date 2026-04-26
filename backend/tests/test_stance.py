"""Tests for `stance.py` — model invariants, archetype consistency, prompt quality."""

from __future__ import annotations

import random
import statistics

import pytest
from pydantic import ValidationError

from src.simulation.stance import (
    ARCHETYPES,
    CeoStance,
    sample_stance,
    to_system_prompt,
)

# ---------------------------------------------------------------------------
# Frozen-model invariants
# ---------------------------------------------------------------------------


def _make_stance() -> CeoStance:
    """Construct a fully-specified stance for direct invariant tests."""
    return CeoStance(
        archetype="bootstrap",
        risk_tolerance=0.2,
        growth_obsession=0.2,
        quality_floor=0.7,
        hiring_bias="lean",
        time_horizon="annual",
        cash_comfort=18.0,
        signature_moves=["profitable from day one", "no debt, no dilution"],
        voice="I run lean and I sleep well at night.",
    )


def test_frozen_model_rejects_mutation() -> None:
    stance = _make_stance()
    with pytest.raises(ValidationError):
        stance.risk_tolerance = 0.99  # type: ignore[misc]


def test_frozen_model_rejects_field_assignment_on_categorical() -> None:
    stance = _make_stance()
    with pytest.raises(ValidationError):
        stance.hiring_bias = "build_bench"  # type: ignore[misc]


def test_signature_moves_min_length_enforced() -> None:
    with pytest.raises(ValidationError):
        CeoStance(
            archetype="bootstrap",
            risk_tolerance=0.2,
            growth_obsession=0.2,
            quality_floor=0.7,
            hiring_bias="lean",
            time_horizon="annual",
            cash_comfort=18.0,
            signature_moves=["only one move"],
            voice="I run lean and I sleep well at night.",
        )


def test_unit_interval_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        CeoStance(
            archetype="bootstrap",
            risk_tolerance=1.5,  # out of [0,1]
            growth_obsession=0.2,
            quality_floor=0.7,
            hiring_bias="lean",
            time_horizon="annual",
            cash_comfort=18.0,
            signature_moves=["a", "b"],
            voice="I run lean and I sleep well at night.",
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        CeoStance(
            archetype="bootstrap",
            risk_tolerance=0.2,
            growth_obsession=0.2,
            quality_floor=0.7,
            hiring_bias="lean",
            time_horizon="annual",
            cash_comfort=18.0,
            signature_moves=["a", "b"],
            voice="I run lean and I sleep well at night.",
            extra_field="nope",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Sampler covers all archetypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_sample_stance_produces_valid_stance(archetype: str) -> None:
    rng = random.Random(7)
    stance = sample_stance(archetype, rng)
    assert stance.archetype == archetype
    assert 0.0 <= stance.risk_tolerance <= 1.0
    assert 0.0 <= stance.growth_obsession <= 1.0
    assert 0.0 <= stance.quality_floor <= 1.0
    assert stance.cash_comfort > 0.0
    assert 2 <= len(stance.signature_moves) <= 3
    assert len(stance.signature_moves) == len(set(stance.signature_moves)), (
        "signature_moves must be drawn without replacement"
    )
    assert len(stance.voice) >= 10


def test_sample_stance_rejects_unknown_archetype() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="Unknown archetype"):
        sample_stance("hyperscaler", rng)


def test_sample_stance_is_deterministic_given_rng() -> None:
    a = sample_stance("venture_growth", random.Random(42))
    b = sample_stance("venture_growth", random.Random(42))
    assert a == b


# ---------------------------------------------------------------------------
# Internal-consistency assertions across archetypes
# ---------------------------------------------------------------------------


_N_SAMPLES = 100


def _sample_many(archetype: str, n: int = _N_SAMPLES) -> list[CeoStance]:
    rng = random.Random(2026)
    return [sample_stance(archetype, rng) for _ in range(n)]


def test_bootstrap_more_cash_comfort_than_venture_growth() -> None:
    boot = _sample_many("bootstrap")
    vc = _sample_many("venture_growth")
    boot_mean = statistics.mean(s.cash_comfort for s in boot)
    vc_mean = statistics.mean(s.cash_comfort for s in vc)
    assert boot_mean > vc_mean, (
        f"bootstrap cash_comfort mean ({boot_mean:.2f}) should exceed "
        f"venture_growth ({vc_mean:.2f})"
    )


def test_bootstrap_lower_risk_than_venture_growth() -> None:
    boot = _sample_many("bootstrap")
    vc = _sample_many("venture_growth")
    boot_mean = statistics.mean(s.risk_tolerance for s in boot)
    vc_mean = statistics.mean(s.risk_tolerance for s in vc)
    assert boot_mean < vc_mean
    # Envelopes do not overlap → every individual sample obeys the ordering too.
    assert max(s.risk_tolerance for s in boot) < min(s.risk_tolerance for s in vc)


def test_bootstrap_lower_growth_obsession_than_venture_growth() -> None:
    boot = _sample_many("bootstrap")
    vc = _sample_many("venture_growth")
    assert statistics.mean(s.growth_obsession for s in boot) < statistics.mean(
        s.growth_obsession for s in vc
    )


def test_venture_growth_always_builds_bench() -> None:
    for s in _sample_many("venture_growth"):
        assert s.hiring_bias == "build_bench"


def test_bootstrap_always_lean() -> None:
    for s in _sample_many("bootstrap"):
        assert s.hiring_bias == "lean"


def test_turnaround_low_cash_comfort() -> None:
    # Turnaround CEOs operate close to the bone — they're already in a crisis,
    # they don't have the luxury of a big runway buffer.
    samples = _sample_many("turnaround")
    assert max(s.cash_comfort for s in samples) <= 5.0


def test_quality_floors_distinct_between_venture_and_founder() -> None:
    # founder_operator should hold a higher quality bar than venture_growth on average
    fo = _sample_many("founder_operator")
    vc = _sample_many("venture_growth")
    assert statistics.mean(s.quality_floor for s in fo) > statistics.mean(
        s.quality_floor for s in vc
    )


# ---------------------------------------------------------------------------
# Inception prompt quality
# ---------------------------------------------------------------------------


def test_to_system_prompt_non_empty_and_under_limit() -> None:
    rng = random.Random(11)
    for archetype in ARCHETYPES:
        prompt = to_system_prompt(sample_stance(archetype, rng))
        assert prompt.strip(), f"prompt for {archetype} was empty"
        assert len(prompt) < 1500, (
            f"prompt for {archetype} was {len(prompt)} chars; budget is 1500"
        )


def test_to_system_prompt_references_every_attribute() -> None:
    """Every stance field must surface somewhere in the inception prompt.

    The prompt is the LLM's only view of the persona — leaving any field out
    silently breaks the role-lock guarantee.
    """
    rng = random.Random(99)
    stance = sample_stance("consolidator", rng)
    prompt = to_system_prompt(stance)

    # Categorical / textual fields appear verbatim.
    assert stance.archetype in prompt
    assert stance.hiring_bias in prompt
    assert stance.time_horizon in prompt
    assert stance.voice in prompt
    for move in stance.signature_moves:
        assert move in prompt, f"signature move {move!r} missing from prompt"

    # Numeric fields appear as formatted numerics.
    assert f"{stance.risk_tolerance:.2f}" in prompt
    assert f"{stance.growth_obsession:.2f}" in prompt
    assert f"{stance.quality_floor:.2f}" in prompt
    assert f"{stance.cash_comfort:.1f}" in prompt


def test_to_system_prompt_contains_role_lock_language() -> None:
    """The prompt must explicitly tell the model to stay in character."""
    stance = _make_stance()
    prompt = to_system_prompt(stance)
    lower = prompt.lower()
    assert "ceo" in lower
    # Anti-drift language — the prompt must signal "do not deviate".
    assert any(token in lower for token in ("do not drift", "consistent with"))
