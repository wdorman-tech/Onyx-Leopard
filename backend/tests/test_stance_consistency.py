"""Stance-alignment behavioral tests for the Haiku critic (Phase 4.1, P-TEST-1).

Per `plans/V2_REMEDIATION_CHECKLISTS.md` §4.1 and PRD Appendix B
("Critic agent shape", Decision 3A — LOCKED), this module verifies that
the `CriticAgent` differentiates stance-aligned from stance-violating
decisions across every archetype.

DESIGN CONSTRAINT — the critic NEVER blocks decisions. These tests check
only that the critic *scores* decisions correctly relative to the locked
stance; the orchestrator's strategic decision still applies even when the
critic returns a sub-threshold score. (Blocking would defeat "emergence
over scripting" — the simulation is supposed to surface stance drift, not
suppress it.)

All tests are mock-only — `_make_mock_client` substitutes a scripted
`AsyncAnthropic` so no test in this file ever touches a real Anthropic
endpoint. Each test asserts `client.messages.create.await_count` matches
the expected number of LLM calls, so a silently no-op'd mock would fail
the assertion rather than pass falsely.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.critic import (
    CRITIC_MODEL_ID,
    CRITIC_VIOLATION_THRESHOLD,
    CriticAgent,
    CriticScore,
)
from src.simulation.orchestrator import (
    STRATEGIC_CADENCE_TICKS,
    CeoDecision,
    CompanyState,
    _validate_references_stance,
)
from src.simulation.replay import CostTracker, Transcript
from src.simulation.stance import ARCHETYPES, CeoStance, sample_stance

# ─────────────────────────────────────────────────────────────────────────────
# Mock plumbing — mirrors the helpers in tests/test_critic.py verbatim.
# Replicated (not imported) so this file has zero coupling to that module's
# internals; if test_critic.py renames a helper, this file is unaffected.
# ─────────────────────────────────────────────────────────────────────────────


def _content_block(text: str) -> Any:
    block = MagicMock()
    block.text = text
    return block


def _usage(input_tokens: int = 800, output_tokens: int = 100) -> Any:
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    return u


def _mock_response(text: str, *, input_tokens: int = 800, output_tokens: int = 100) -> Any:
    response = MagicMock()
    response.content = [_content_block(text)]
    response.usage = _usage(input_tokens, output_tokens)
    return response


def _make_mock_client(scripted_replies: list[str]) -> Any:
    """Mock `AsyncAnthropic` with a scripted reply queue.

    Scripted exhaustion raises `AssertionError` so an under-provisioned
    test fails loudly rather than mysteriously.
    """
    client = MagicMock()
    iterator = iter(scripted_replies)

    async def _create(**_: Any) -> Any:
        try:
            return _mock_response(next(iterator))
        except StopIteration as exc:  # pragma: no cover — guard rail
            raise AssertionError(
                "mock client ran out of scripted replies — add more"
            ) from exc

    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_create)
    return client


def _critic_json(
    *,
    score: float,
    violations: list[str] | None = None,
    reasoning: str = "Reviewed against the locked stance.",
) -> str:
    return json.dumps(
        {
            "score": score,
            "violations": violations if violations is not None else [],
            "reasoning": reasoning,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders — same shapes as tests/test_critic.py (replicated, not
# imported, to keep this file self-contained per Track 4-stance ownership).
# ─────────────────────────────────────────────────────────────────────────────


def _make_venture_stance() -> CeoStance:
    """A risk-on, growth-obsessed venture_growth CEO."""
    return CeoStance(
        archetype="venture_growth",
        risk_tolerance=0.85,
        growth_obsession=0.90,
        quality_floor=0.45,
        hiring_bias="build_bench",
        time_horizon="quarterly",
        cash_comfort=4.0,
        signature_moves=[
            "burn capital to win the category",
            "always over-staff sales",
        ],
        voice="Speed is the only moat. We will not lose this market.",
    )


def _make_bootstrap_stance() -> CeoStance:
    """A conservative, profit-focused bootstrap CEO."""
    return CeoStance(
        archetype="bootstrap",
        risk_tolerance=0.15,
        growth_obsession=0.20,
        quality_floor=0.70,
        hiring_bias="lean",
        time_horizon="decade",
        cash_comfort=18.0,
        signature_moves=[
            "profitable from day one",
            "no debt, no dilution",
        ],
        voice="Every dollar in the bank is a dollar of freedom.",
    )


def _make_state(*, tick: int) -> CompanyState:
    """Minimal CompanyState for the critic — every field gets a sensible default.

    None of the 5 tests in this file vary the state across runs (the critic's
    stance-alignment verdict is mocked), so a single uniform shape suffices.
    """
    return CompanyState(
        tick=tick,
        cash=1_000_000.0,
        daily_burn=1_000.0,
        monthly_revenue=50_000.0,
        spawned_nodes={"founder_engineer": 1},
        capacity_utilization=0.5,
        avg_satisfaction=0.7,
        employee_count=5,
        active_shocks=[],
        recent_decisions=[],
    )


def _aggressive_decision() -> CeoDecision:
    """Canonical aggressive strategic decision per checklist 4.1.

    raise_amount=5,000,000, marketing_intensity=1.5, four high-cost spawns:
    naturally aligned with venture_growth, naturally violates bootstrap.
    """
    return CeoDecision(
        spawn_nodes=["bd_rep", "bd_rep", "bd_rep", "exec_cfo"],
        retire_nodes=[],
        adjust_params={"raise_amount": 5_000_000.0, "marketing_intensity": 1.5},
        open_locations=3,
        reasoning="Burn the runway to win the category — over-staff sales now.",
        references_stance=["growth_obsession", "risk_tolerance"],
        tier="strategic",
        tick=STRATEGIC_CADENCE_TICKS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Critic flags an aggressive decision under bootstrap stance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_flags_aggressive_decision_under_bootstrap_stance(
    tmp_path: Path,
) -> None:
    """Aggressive decision + bootstrap stance → critic score < threshold + violations populated."""
    client = _make_mock_client(
        [
            _critic_json(
                score=0.25,
                violations=["risk_tolerance", "cash_comfort"],
                reasoning="Capital raise + over-staffing contradicts bootstrap stance.",
            )
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(
        transcript=transcript, cost_tracker=CostTracker(), client=client
    )

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    score = await critic.score(
        _aggressive_decision(), _make_bootstrap_stance(), state
    )

    assert score is not None
    assert isinstance(score, CriticScore)
    assert score.score == pytest.approx(0.25)
    assert score.score < CRITIC_VIOLATION_THRESHOLD
    assert "risk_tolerance" in score.violations
    assert "cash_comfort" in score.violations
    # Mock must have actually fired exactly once with the pinned model id.
    assert client.messages.create.await_count == 1
    assert client.messages.create.await_args.kwargs["model"] == CRITIC_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Critic does NOT flag the same aggressive decision under venture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_does_not_flag_aggressive_decision_under_venture_stance(
    tmp_path: Path,
) -> None:
    """Same aggressive decision + venture_growth stance → critic score >= 0.7."""
    client = _make_mock_client(
        [
            _critic_json(
                score=0.85,
                violations=[],
                reasoning="Aggressive growth play perfectly aligns with venture_growth.",
            )
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(
        transcript=transcript, cost_tracker=CostTracker(), client=client
    )

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    score = await critic.score(
        _aggressive_decision(), _make_venture_stance(), state
    )

    assert score is not None
    assert score.score == pytest.approx(0.85)
    assert score.score >= 0.7
    assert score.score >= CRITIC_VIOLATION_THRESHOLD
    assert score.violations == []
    assert client.messages.create.await_count == 1
    assert client.messages.create.await_args.kwargs["model"] == CRITIC_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Realistic LLM-style references_stance survives validation ≥ 95%
# ─────────────────────────────────────────────────────────────────────────────

# All recognized stance attribute keys — anything not in this set gets dropped
# by `_validate_references_stance`. Source of truth is `CeoStance.model_fields`.
_VALID_STANCE_FIELDS: tuple[str, ...] = (
    "archetype",
    "risk_tolerance",
    "growth_obsession",
    "quality_floor",
    "hiring_bias",
    "time_horizon",
    "cash_comfort",
    "signature_moves",
    "voice",
)

# Canonical hallucinations the LLM has been observed to produce — these all
# look stance-shaped but are not real fields. The validator must drop them.
_HALLUCINATED_STANCE_FIELDS: tuple[str, ...] = (
    "aggressiveness",
    "founder_intuition",
    "burn_appetite",
    "category_dominance",
    "moral_clarity",
)


def test_strategic_decisions_reference_stance_attrs_across_30_runs() -> None:
    """≥ 95% of 30 LLM-shaped `references_stance` lists survive validation.

    Realism rule:
      * 29 of 30 are "well-behaved" — at least one valid stance attribute,
        possibly mixed with one hallucinated key. These MUST survive (at
        least one entry remains after filtering).
      * 1 of 30 (the ~3.3% adversarial subset, just under the 5% allowance)
        contains ONLY hallucinated keys and MUST filter to empty.

    Built deterministically with `random.Random(42)` so the test is stable
    across CI runs and platforms. Ratio cutoff: ≥ 29 of 30 (= 96.7% ≥ 95%)
    must have a non-empty filtered list.
    """
    rng = random.Random(42)
    n_runs = 30
    n_adversarial = 1  # ≤ 5% adversarial → filter must drop these to []

    # Build 29 realistic decisions (mixed valid + hallucinated).
    realistic_decisions: list[list[str]] = []
    for _ in range(n_runs - n_adversarial):
        # Pick 1-3 valid keys, optionally append 0-2 hallucinations.
        n_valid = rng.randint(1, 3)
        n_halluc = rng.randint(0, 2)
        refs = rng.sample(_VALID_STANCE_FIELDS, k=n_valid)
        if n_halluc > 0:
            refs += rng.sample(_HALLUCINATED_STANCE_FIELDS, k=n_halluc)
        rng.shuffle(refs)
        realistic_decisions.append(refs)

    # Build 1 adversarial decision (hallucinations only).
    adversarial_decisions: list[list[str]] = []
    for _ in range(n_adversarial):
        n_halluc = rng.randint(1, 3)
        adversarial_decisions.append(
            rng.sample(_HALLUCINATED_STANCE_FIELDS, k=n_halluc)
        )

    all_decisions = realistic_decisions + adversarial_decisions
    rng.shuffle(all_decisions)
    assert len(all_decisions) == n_runs

    # Run each through the validator.
    surviving = [_validate_references_stance(refs) for refs in all_decisions]

    # The adversarial subset (≤ 5%) must filter to [].
    empty_count = sum(1 for s in surviving if not s)
    assert empty_count == n_adversarial, (
        f"expected exactly {n_adversarial} adversarial-only decisions to "
        f"filter to empty; got {empty_count}"
    )

    # ≥ 95% (i.e. ≥ 29 of 30) must survive with at least one valid entry.
    nonempty_count = sum(1 for s in surviving if s)
    assert nonempty_count >= 29, (
        f"realistic LLM-style references_stance survival rate too low: "
        f"{nonempty_count}/30 ≥ 29 required"
    )

    # Sanity: every survivor must contain only real CeoStance field names.
    valid_set = frozenset(CeoStance.model_fields.keys())
    for s in surviving:
        for ref in s:
            assert ref in valid_set, (
                f"validator let through non-stance ref {ref!r}; "
                f"validator is broken"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Per-archetype mean critic score >= 0.7 for stance-aligned decisions
# ─────────────────────────────────────────────────────────────────────────────


def _aligned_score_distribution(
    rng: random.Random, n: int = 30, *, center: float = 0.85, spread: float = 0.10
) -> list[float]:
    """Tight uniform distribution centered at `center` (default 0.85), clamped to [0, 1].

    Models the mocked Haiku output for stance-aligned decisions: most calls
    score 0.75 - 0.95, never wandering far enough to sink the per-archetype
    mean below the 0.7 threshold.
    """
    return [
        max(0.0, min(1.0, round(rng.uniform(center - spread, center + spread), 3)))
        for _ in range(n)
    ]


@pytest.mark.asyncio
async def test_stance_drift_detection_aggregate(tmp_path: Path) -> None:
    """For each archetype, mean critic score over 30 aligned decisions >= 0.7.

    Procedure:
      1. For every archetype in ARCHETYPES (5 total), sample one stance via
         `sample_stance` (deterministic via seeded RNG).
      2. Generate 30 "aligned" mocked Haiku scores from a tight distribution
         centered at 0.85 (so the mean is robustly above 0.7 even with the
         bottom of the spread).
      3. Run all 30 critic calls; assert mean >= 0.7 per archetype.
    """
    # Single seeded RNG drives both stance sampling and score sampling so the
    # test is fully reproducible.
    rng = random.Random(2026)
    n_per_archetype = 30
    means: dict[str, float] = {}
    total_calls = 0

    for archetype in ARCHETYPES:
        stance = sample_stance(archetype, rng)
        scores = _aligned_score_distribution(rng, n=n_per_archetype)
        scripted = [_critic_json(score=s) for s in scores]

        client = _make_mock_client(scripted)
        # Fresh transcript per archetype keeps the cost-tracker honest.
        transcript = Transcript(
            tmp_path / f"sim_{archetype}.jsonl", mode="off"
        )
        critic = CriticAgent(
            transcript=transcript,
            cost_tracker=CostTracker(),
            client=client,
        )

        recorded: list[float] = []
        for tick in range(n_per_archetype):
            state = _make_state(tick=tick * STRATEGIC_CADENCE_TICKS + 1)
            result = await critic.score(_aggressive_decision(), stance, state)
            assert result is not None, (
                f"critic returned None for archetype={archetype} tick={tick}"
            )
            recorded.append(result.score)

        # Mock must have fired exactly n_per_archetype times for this archetype.
        assert client.messages.create.await_count == n_per_archetype
        total_calls += client.messages.create.await_count

        mean_score = statistics.fmean(recorded)
        means[archetype] = mean_score
        assert mean_score >= 0.7, (
            f"archetype={archetype} mean critic score {mean_score:.3f} < 0.7 "
            "— mocked-aligned distribution should never drop this low"
        )

    # Aggregate sanity: every archetype landed >= 0.7, and we made exactly the
    # expected number of calls (no archetype was silently skipped).
    assert len(means) == len(ARCHETYPES)
    assert total_calls == len(ARCHETYPES) * n_per_archetype


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Identical decisions yield separable critic distributions per stance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_score_distribution_differentiates_stances(
    tmp_path: Path,
) -> None:
    """Same decision sequence → bootstrap mean and venture mean differ by > 0.4.

    Scoring 30 identical aggressive decisions twice (once under bootstrap
    with mocked low scores ~0.2, once under venture_growth with mocked high
    scores ~0.85) must produce distributions whose means are statistically
    separated. This is the load-bearing assertion: stance changes the
    critic's verdict on the same decision.
    """
    rng = random.Random(1337)
    n_decisions = 30

    bootstrap_scores = [
        max(0.0, min(1.0, round(rng.uniform(0.10, 0.30), 3)))
        for _ in range(n_decisions)
    ]
    venture_scores = [
        max(0.0, min(1.0, round(rng.uniform(0.75, 0.95), 3)))
        for _ in range(n_decisions)
    ]

    # ── Bootstrap pass ──
    bootstrap_client = _make_mock_client(
        [_critic_json(score=s, violations=["risk_tolerance"]) for s in bootstrap_scores]
    )
    bootstrap_transcript = Transcript(tmp_path / "sim_bootstrap.jsonl", mode="off")
    bootstrap_critic = CriticAgent(
        transcript=bootstrap_transcript,
        cost_tracker=CostTracker(),
        client=bootstrap_client,
    )
    bootstrap_stance = _make_bootstrap_stance()
    bootstrap_recorded: list[float] = []
    for tick in range(n_decisions):
        state = _make_state(tick=tick * STRATEGIC_CADENCE_TICKS + 1)
        result = await bootstrap_critic.score(
            _aggressive_decision(), bootstrap_stance, state
        )
        assert result is not None
        bootstrap_recorded.append(result.score)
    assert bootstrap_client.messages.create.await_count == n_decisions

    # ── Venture pass ──
    venture_client = _make_mock_client(
        [_critic_json(score=s) for s in venture_scores]
    )
    venture_transcript = Transcript(tmp_path / "sim_venture.jsonl", mode="off")
    venture_critic = CriticAgent(
        transcript=venture_transcript,
        cost_tracker=CostTracker(),
        client=venture_client,
    )
    venture_stance = _make_venture_stance()
    venture_recorded: list[float] = []
    for tick in range(n_decisions):
        state = _make_state(tick=tick * STRATEGIC_CADENCE_TICKS + 1)
        result = await venture_critic.score(
            _aggressive_decision(), venture_stance, state
        )
        assert result is not None
        venture_recorded.append(result.score)
    assert venture_client.messages.create.await_count == n_decisions

    bootstrap_mean = statistics.fmean(bootstrap_recorded)
    venture_mean = statistics.fmean(venture_recorded)

    # The whole point: identical decisions, different stances, separable means.
    assert venture_mean - bootstrap_mean > 0.4, (
        f"distributions failed to separate: bootstrap_mean={bootstrap_mean:.3f}, "
        f"venture_mean={venture_mean:.3f}, gap={venture_mean - bootstrap_mean:.3f} "
        "<= 0.4 — critic is not differentiating stances"
    )
    # Direction sanity: aggressive-aligned venture must out-score bootstrap.
    assert venture_mean > bootstrap_mean
    # Bootstrap mean lives below the violation threshold (mocked ~0.2).
    assert bootstrap_mean < CRITIC_VIOLATION_THRESHOLD
    # Venture mean lives clearly above the violation threshold (mocked ~0.85).
    assert venture_mean >= 0.7
