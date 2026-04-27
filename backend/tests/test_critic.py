"""Tests for the stance-alignment critic agent (Phase 2.3, Decision 3A).

Per `plans/V2_REMEDIATION_CHECKLISTS.md` §2.3 and PRD Appendix B, the
critic agent scores STRATEGIC-tier decisions for stance alignment via a
Haiku 4.5 LLM call. It MUST:
  * Never block — below-threshold scores log a warning, decision still applies.
  * Skip silently when the cost ceiling would be exceeded.
  * Fire ONLY for strategic-tier decisions, never for heuristic or tactical.
  * Round-trip through transcript record/replay byte-identically.

Mocks the Anthropic SDK end-to-end (no real API calls). Reuses the
mocking pattern from `test_orchestrator_llm.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.critic import (
    CRITIC_MODEL_ID,
    CRITIC_VIOLATION_THRESHOLD,
    CriticAgent,
    CriticReplayDivergenceError,
    CriticScore,
)
from src.simulation.library_loader import (
    CategoryCaps,
    NodeDef,
    NodeLibrary,
)
from src.simulation.orchestrator import (
    STRATEGIC_CADENCE_TICKS,
    TACTICAL_CADENCE_TICKS,
    CeoDecision,
    CompanyState,
    HeuristicOrchestrator,
    OrchestratorBundle,
    StrategicOrchestrator,
    TacticalOrchestrator,
)
from src.simulation.replay import (
    CostTracker,
    Transcript,
)
from src.simulation.seed import CompanySeed
from src.simulation.shocks import Shock
from src.simulation.stance import CeoStance


# ─────────────────────────────────────────────────────────────────────────────
# Mock plumbing (mirrors test_orchestrator_llm.py)
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
    """Mock `AsyncAnthropic` with a scripted reply queue."""
    client = MagicMock()
    iterator = iter(scripted_replies)

    async def _create(**_: Any) -> Any:
        try:
            return _mock_response(next(iterator))
        except StopIteration as exc:
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


def _decision_json(
    *,
    spawn_nodes: list[str] | None = None,
    retire_nodes: list[str] | None = None,
    adjust_params: dict[str, float] | None = None,
    open_locations: int = 0,
    reasoning: str = "Default reasoning.",
    references_stance: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "spawn_nodes": spawn_nodes if spawn_nodes is not None else [],
            "retire_nodes": retire_nodes if retire_nodes is not None else [],
            "adjust_params": adjust_params if adjust_params is not None else {},
            "open_locations": open_locations,
            "reasoning": reasoning,
            "references_stance": (
                references_stance if references_stance is not None else ["risk_tolerance"]
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (minimal seed/stance/library)
# ─────────────────────────────────────────────────────────────────────────────


def _make_seed(economics_model: str = "subscription") -> CompanySeed:
    return CompanySeed(
        name="Test Co",
        niche="B2B SaaS for QA testing",
        archetype="small_team",
        industry_keywords=["saas", "qa"],
        location_label="Product",
        economics_model=economics_model,  # type: ignore[arg-type]
        starting_price=99.0,
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


def _make_library() -> NodeLibrary:
    nodes_raw: list[NodeDef] = [
        NodeDef(
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
        ),
        NodeDef(
            key="bd_rep",
            category="sales",
            label="BD Rep",
            hire_cost=8000.0,
            daily_fixed_costs=220.0,
            employees_count=1,
            capacity_contribution=0,
            modifier_keys={"pipeline_strength": 0.10},
            prerequisites=[],
            category_caps=CategoryCaps(soft_cap=6, hard_cap=12),
            applicable_economics=["subscription", "service"],
        ),
        NodeDef(
            key="exec_cfo",
            category="exec",
            label="CFO",
            hire_cost=200_000.0,
            daily_fixed_costs=900.0,
            employees_count=1,
            capacity_contribution=0,
            modifier_keys={},
            prerequisites=["founder_engineer"],
            category_caps=CategoryCaps(soft_cap=1, hard_cap=1),
            applicable_economics=["subscription", "service"],
        ),
    ]
    return NodeLibrary({n.key: n for n in nodes_raw})


def _make_state(
    *,
    tick: int,
    cash: float = 1_000_000.0,
    daily_burn: float = 1_000.0,
    monthly_revenue: float = 50_000.0,
    spawned: dict[str, int] | None = None,
    capacity_utilization: float = 0.5,
    avg_satisfaction: float = 0.7,
    employee_count: int = 5,
    active_shocks: list[Shock] | None = None,
    recent_decisions: list[CeoDecision] | None = None,
) -> CompanyState:
    return CompanyState(
        tick=tick,
        cash=cash,
        daily_burn=daily_burn,
        monthly_revenue=monthly_revenue,
        spawned_nodes=dict(spawned) if spawned else {"founder_engineer": 1},
        capacity_utilization=capacity_utilization,
        avg_satisfaction=avg_satisfaction,
        employee_count=employee_count,
        active_shocks=list(active_shocks) if active_shocks else [],
        recent_decisions=list(recent_decisions) if recent_decisions else [],
    )


def _aggressive_decision() -> CeoDecision:
    """A high-burn, high-growth strategic decision — naturally aligned with
    venture_growth, naturally misaligned with bootstrap."""
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
# Test 1: high alignment for aligned decision under venture stance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_high_alignment_for_aligned_decision(tmp_path: Path) -> None:
    """Mocked Haiku returns 0.9 for an aggressive decision under venture_growth."""
    client = _make_mock_client(
        [
            _critic_json(
                score=0.9,
                violations=[],
                reasoning="Aggressive growth bet aligns with venture_growth.",
            )
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(transcript=transcript, cost_tracker=CostTracker(), client=client)

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    score = await critic.score(_aggressive_decision(), _make_venture_stance(), state)

    assert score is not None
    assert isinstance(score, CriticScore)
    assert score.score == pytest.approx(0.9)
    assert score.violations == []
    assert client.messages.create.await_count == 1
    # Pinned model id.
    assert client.messages.create.await_args.kwargs["model"] == CRITIC_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: low alignment for misaligned decision under bootstrap stance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_low_alignment_for_misaligned_decision(tmp_path: Path) -> None:
    """Mocked Haiku returns 0.2 for an aggressive decision under bootstrap."""
    client = _make_mock_client(
        [
            _critic_json(
                score=0.2,
                violations=["risk_tolerance", "growth_obsession", "cash_comfort"],
                reasoning="Capital raise + over-staffing contradicts bootstrap stance.",
            )
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(transcript=transcript, cost_tracker=CostTracker(), client=client)

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    score = await critic.score(_aggressive_decision(), _make_bootstrap_stance(), state)

    assert score is not None
    assert score.score == pytest.approx(0.2)
    assert score.score < CRITIC_VIOLATION_THRESHOLD
    assert "risk_tolerance" in score.violations
    assert client.messages.create.await_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: skipped when budget exhausted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_skipped_when_budget_exhausted(tmp_path: Path) -> None:
    """When `cost_tracker.would_exceed(...)` is true → returns None, no API call."""
    # Tiny ceiling, then drain it past the line.
    tracker = CostTracker(ceiling_usd=0.001)
    # Burn the budget by recording a fake call worth more than the ceiling.
    # Haiku is $1/1M input — 5000 input tokens = $0.005 > $0.001.
    tracker.record(input_tokens=0, output_tokens=0, model=CRITIC_MODEL_ID)
    # Now any predictive check will fail (we have $0.001 ceiling; predicted
    # call cost is well over $0.001 with the default 2000-input/300-output).
    assert tracker.would_exceed(2000, 300, CRITIC_MODEL_ID) is True

    client = _make_mock_client([])  # Must NEVER be called.
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(transcript=transcript, cost_tracker=tracker, client=client)

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    score = await critic.score(_aggressive_decision(), _make_venture_stance(), state)

    assert score is None
    assert client.messages.create.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: critic does not block strategic decision on violation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_does_not_block_strategic_decision_on_violation(
    tmp_path: Path,
) -> None:
    """Bundle.tick() still returns the strategic decision when critic score < threshold."""
    # Strategic LLM emits a normal aggressive decision.
    strategic_reply = _decision_json(
        spawn_nodes=["exec_cfo"],
        adjust_params={"raise_amount": 3_000_000.0},
        reasoning="Hire CFO + raise capital — aggressive growth play.",
        references_stance=["growth_obsession", "risk_tolerance"],
    )
    # Critic emits a low score → violation, but decision must still pass through.
    critic_reply = _critic_json(
        score=0.2,
        violations=["risk_tolerance", "cash_comfort"],
        reasoning="Decision contradicts the bootstrap stance.",
    )

    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    tracker = CostTracker(ceiling_usd=50.0)

    seed = _make_seed()
    stance = _make_bootstrap_stance()
    library = _make_library()

    library_dict: dict[str, dict] = {
        k: {
            "category": n.category,
            "label": n.label,
            "hire_cost": n.hire_cost,
            "daily_fixed_costs": n.daily_fixed_costs,
            "employees_count": n.employees_count,
            "capacity_contribution": n.capacity_contribution,
            "modifier_keys": dict(n.modifier_keys),
            "prerequisites": list(n.prerequisites),
            "category_caps": {
                "soft_cap": n.category_caps.soft_cap,
                "hard_cap": n.category_caps.hard_cap,
            },
            "applicable_economics": list(n.applicable_economics),
        }
        for k, n in library.nodes.items()
    }

    heuristic = HeuristicOrchestrator(seed=seed, stance=stance, library=library_dict)
    tactical = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client([]),  # tactical must NOT fire at tick=90
    )
    strategic = StrategicOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client([strategic_reply]),
    )
    critic = CriticAgent(
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client([critic_reply]),
    )

    bundle = OrchestratorBundle(
        heuristic=heuristic,
        tactical=tactical,
        strategic=strategic,
        critic=critic,
        stance=stance,
        company_id="test-co",
    )

    # tick=90 hits strategic cadence but not heuristic (90 % 7 != 0) and not
    # tactical (90 % 30 == 0 — actually it IS tactical cadence). Use 270 instead:
    # 270 % 7 != 0, 270 % 30 == 0, 270 % 90 == 0. Still tactical though.
    # Use a strategic-only tick: 990. 990 % 7 != 0, 990 % 30 == 0, 990 % 90 == 0.
    # Strategic always co-fires with tactical at LCM(30, 90) = 90.
    # Easiest: just provide a tactical mock that's never called by skipping
    # tactical_replies and using force_wake on a strategic call instead.
    severe = Shock(
        name="market_crash",
        severity="severe",
        duration_ticks=120,
        impact={"market_demand_mult": 0.5},
        description="Crash",
        tick_started=15,
    )
    state = _make_state(tick=15, active_shocks=[severe])  # off-cadence; force_wake
    decisions, critic_scores = await bundle.tick(state)

    # Strategic decision MUST still be in the decisions list, despite low score.
    strategic_decisions = [d for d in decisions if d.tier == "strategic"]
    assert len(strategic_decisions) == 1, "strategic decision must not be blocked"
    assert "exec_cfo" in strategic_decisions[0].spawn_nodes

    # Critic score recorded.
    assert len(critic_scores) == 1
    assert critic_scores[0].score == pytest.approx(0.2)
    assert critic_scores[0].score < CRITIC_VIOLATION_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: replay round-trip — recorded score reads back byte-identically
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_replay_round_trip(tmp_path: Path) -> None:
    """Record a critic score then replay — score reads back byte-identical."""
    transcript_path = tmp_path / "replay-roundtrip.jsonl"

    # ── Record phase ──
    record_transcript = Transcript(transcript_path, mode="record")
    record_client = _make_mock_client(
        [
            _critic_json(
                score=0.73,
                violations=["quality_floor"],
                reasoning="Mostly aligned but pushes against quality_floor.",
            )
        ]
    )
    rec_critic = CriticAgent(
        transcript=record_transcript,
        cost_tracker=CostTracker(),
        client=record_client,
    )
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    decision = _aggressive_decision()
    stance = _make_venture_stance()

    recorded_score = await rec_critic.score(
        decision, stance, state, company_id="test-co"
    )
    assert recorded_score is not None
    assert recorded_score.score == pytest.approx(0.73)

    # ── Replay phase ──
    replay_transcript = Transcript(transcript_path, mode="replay")
    replay_client = _make_mock_client([])  # MUST NOT BE CALLED
    rep_critic = CriticAgent(
        transcript=replay_transcript,
        cost_tracker=CostTracker(),
        client=replay_client,
    )

    replayed_score = await rep_critic.score(
        decision, stance, state, company_id="test-co"
    )
    assert replayed_score is not None
    # Byte-identical fields.
    assert replayed_score.score == recorded_score.score
    assert replayed_score.violations == recorded_score.violations
    assert replayed_score.reasoning == recorded_score.reasoning
    # Replay must NOT have hit the API.
    assert replay_client.messages.create.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: critic only runs for strategic tier
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_only_runs_for_strategic_tier(tmp_path: Path) -> None:
    """No Haiku critic call when only heuristic / tactical fire (no strategic)."""
    # tick=210: 210 % 7 == 0 (heuristic cadence), 210 % 30 == 0 (tactical
    # cadence), 210 % 90 != 0 (strategic OFF). Strategic must NOT fire,
    # therefore critic must NOT fire.
    tactical_reply = _decision_json(
        adjust_params={"price": 105.0},
        reasoning="Price tweak.",
        references_stance=["growth_obsession"],
    )

    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    tracker = CostTracker(ceiling_usd=50.0)

    seed = _make_seed()
    stance = _make_venture_stance()
    library = _make_library()

    library_dict: dict[str, dict] = {
        k: {
            "category": n.category,
            "label": n.label,
            "hire_cost": n.hire_cost,
            "daily_fixed_costs": n.daily_fixed_costs,
            "employees_count": n.employees_count,
            "capacity_contribution": n.capacity_contribution,
            "modifier_keys": dict(n.modifier_keys),
            "prerequisites": list(n.prerequisites),
            "category_caps": {
                "soft_cap": n.category_caps.soft_cap,
                "hard_cap": n.category_caps.hard_cap,
            },
            "applicable_economics": list(n.applicable_economics),
        }
        for k, n in library.nodes.items()
    }

    heuristic = HeuristicOrchestrator(seed=seed, stance=stance, library=library_dict)
    tactical = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client([tactical_reply]),
    )
    strategic = StrategicOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client([]),  # MUST NOT FIRE at tick=210 (off-cadence)
    )
    critic_client = _make_mock_client([])  # MUST NOT FIRE at all
    critic = CriticAgent(
        transcript=transcript, cost_tracker=tracker, client=critic_client
    )

    bundle = OrchestratorBundle(
        heuristic=heuristic,
        tactical=tactical,
        strategic=strategic,
        critic=critic,
        stance=stance,
        company_id="test-co",
    )

    # Pre-warm heuristic with sustained high util so it fires its capacity rule.
    spawned = {"founder_engineer": 1}
    for offset in range(208, 210):
        bundle.heuristic.tick(
            _make_state(
                tick=offset,
                spawned=spawned,
                capacity_utilization=0.99,
            )
        )

    state = _make_state(
        tick=TACTICAL_CADENCE_TICKS * 7,  # 210 — heuristic + tactical, no strategic
        spawned=spawned,
        capacity_utilization=0.99,
    )
    decisions, critic_scores = await bundle.tick(state)

    tiers = sorted(d.tier for d in decisions)
    # Strategic MUST NOT have fired.
    assert "strategic" not in tiers
    # Tactical fired (cadence hit).
    assert "tactical" in tiers
    # Critic was NOT called.
    assert critic_client.messages.create.await_count == 0
    # No critic scores recorded.
    assert critic_scores == []


# ─────────────────────────────────────────────────────────────────────────────
# Crash-safety tests (Issue #2 — best-effort telemetry contract)
#
# The critic must NEVER abort the sim. SDK transport errors, malformed JSON,
# and schema-validation failures all degrade to "log + return None" — the
# strategic decision still applies, the next tick still runs.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_returns_none_on_sdk_transport_error(tmp_path: Path) -> None:
    """Anthropic SDK raising mid-call → critic logs + returns None."""
    transport_error_client = MagicMock()
    transport_error_client.messages = MagicMock()
    transport_error_client.messages.create = AsyncMock(
        side_effect=RuntimeError("simulated transport failure")
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(
        transcript=transcript,
        cost_tracker=CostTracker(),
        client=transport_error_client,
    )
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)

    score = await critic.score(
        _aggressive_decision(), _make_venture_stance(), state
    )

    assert score is None
    # Cost tracker MUST NOT have been charged — no successful tokens.
    assert critic.cost_tracker is not None
    assert critic.cost_tracker.total_cost() == 0.0


@pytest.mark.asyncio
async def test_critic_returns_none_on_malformed_json(tmp_path: Path) -> None:
    """Haiku returning prose-only response → critic logs + returns None."""
    bad_json_client = _make_mock_client(
        ["Sorry, I cannot score this decision because of vibes."]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(
        transcript=transcript,
        cost_tracker=CostTracker(),
        client=bad_json_client,
    )
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)

    score = await critic.score(
        _aggressive_decision(), _make_venture_stance(), state
    )

    assert score is None


@pytest.mark.asyncio
async def test_critic_returns_none_on_schema_violation(tmp_path: Path) -> None:
    """JSON parses but score outside [0, 1] → schema rejects → return None."""
    out_of_range_client = _make_mock_client(
        [json.dumps({"score": 2.5, "violations": [], "reasoning": "Bad."})]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    critic = CriticAgent(
        transcript=transcript,
        cost_tracker=CostTracker(),
        client=out_of_range_client,
    )
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)

    score = await critic.score(
        _aggressive_decision(), _make_venture_stance(), state
    )

    assert score is None


@pytest.mark.asyncio
async def test_bundle_tick_survives_critic_failure(tmp_path: Path) -> None:
    """Bundle MUST return the strategic decision even if the critic raises.

    This is the load-bearing assertion behind the "never aborts a sim"
    contract. We force the critic's underlying SDK to raise and verify
    the strategic decision still appears in `decisions` — without the
    crash-safety wrapping in `score()`, the exception would propagate
    out of `bundle.tick()` and the strategic decision would be lost.
    """
    strategic_reply = _decision_json(
        spawn_nodes=["exec_cfo"],
        reasoning="Hire CFO.",
        references_stance=["growth_obsession"],
    )

    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    tracker = CostTracker(ceiling_usd=50.0)
    seed = _make_seed()
    stance = _make_venture_stance()
    library = _make_library()

    library_dict: dict[str, dict] = {
        k: {
            "category": n.category,
            "label": n.label,
            "hire_cost": n.hire_cost,
            "daily_fixed_costs": n.daily_fixed_costs,
            "employees_count": n.employees_count,
            "capacity_contribution": n.capacity_contribution,
            "modifier_keys": dict(n.modifier_keys),
            "prerequisites": list(n.prerequisites),
            "category_caps": {
                "soft_cap": n.category_caps.soft_cap,
                "hard_cap": n.category_caps.hard_cap,
            },
            "applicable_economics": list(n.applicable_economics),
        }
        for k, n in library.nodes.items()
    }

    heuristic = HeuristicOrchestrator(seed=seed, stance=stance, library=library_dict)
    tactical = TacticalOrchestrator(
        seed=seed, stance=stance, library=library,
        transcript=transcript, cost_tracker=tracker,
        client=_make_mock_client([]),
    )
    strategic = StrategicOrchestrator(
        seed=seed, stance=stance, library=library,
        transcript=transcript, cost_tracker=tracker,
        client=_make_mock_client([strategic_reply]),
    )

    # Critic's SDK raises on every call.
    failing_client = MagicMock()
    failing_client.messages = MagicMock()
    failing_client.messages.create = AsyncMock(
        side_effect=RuntimeError("simulated transport failure")
    )
    critic = CriticAgent(
        transcript=transcript, cost_tracker=tracker, client=failing_client
    )

    bundle = OrchestratorBundle(
        heuristic=heuristic, tactical=tactical, strategic=strategic,
        critic=critic, stance=stance, company_id="test-co",
    )

    severe = Shock(
        name="market_crash", severity="severe", duration_ticks=120,
        impact={"market_demand_mult": 0.5},
        description="Crash", tick_started=15,
    )
    state = _make_state(tick=15, active_shocks=[severe])

    decisions, critic_scores = await bundle.tick(state)

    # Strategic decision MUST be present despite the critic blowing up.
    strategic_decisions = [d for d in decisions if d.tier == "strategic"]
    assert len(strategic_decisions) == 1
    # No critic score recorded — the call failed.
    assert critic_scores == []


# ─────────────────────────────────────────────────────────────────────────────
# Engine-wiring test (Issue #1 — critic must fire in default sims)
# ─────────────────────────────────────────────────────────────────────────────


def test_default_engine_bundle_constructs_critic_agent(tmp_path: Path) -> None:
    """A default-constructed `CompanyAgentV2` must wire a `CriticAgent` into
    its orchestrator bundle — not the `critic=None` no-op of pre-fix code.

    This is the load-bearing assertion behind Issue #1: in production, every
    strategic-tier decision must be scored. Without explicit wiring in
    `CompanyAgentV2.__init__`, the bundle defaulted to `critic=None` and the
    critic never fired in real sims.

    Uses the production library + sample_seed_for_archetype so the seed's
    initial refs round-trip through library validation, mirroring
    `test_unified_v2.py`'s setup.
    """
    import random

    from src.simulation.library_loader import _reset_library_cache, get_library
    from src.simulation.seed import sample_seed_for_archetype
    from src.simulation.stance import sample_stance
    from src.simulation.unified_v2 import CompanyAgentV2

    _reset_library_cache()
    library = get_library()
    rng = random.Random(42)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    # Patch seed refs to live nodes in the production library — same trick
    # `test_unified_v2.small_team_seed_and_stance` uses.
    suppliers: list[str] = []
    revenues: list[str] = []
    costs: list[str] = []
    for key, node in sorted(library.nodes.items()):
        if seed.economics_model not in node.applicable_economics:
            continue
        if node.category == "supplier" and not suppliers:
            suppliers.append(key)
        elif node.category == "revenue" and not revenues:
            revenues.append(key)
        elif node.category == "ops" and not costs:
            costs.append(key)
    seed = seed.model_copy(update={
        "initial_supplier_types": suppliers or ["primary_goods_supplier"],
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs or ["bookkeeper"],
    })
    stance = sample_stance("venture_growth", rng=rng)
    transcript = Transcript(tmp_path / "wire-test.jsonl", mode="off")

    company = CompanyAgentV2(
        seed=seed,
        stance=stance,
        library=library,
        sim_id="wire-test",
        company_id="wire-co",
        rng=rng,
        transcript=transcript,
    )

    bundle = company.orchestrator
    assert bundle.critic is not None, (
        "default-constructed engine bundle must wire a CriticAgent — "
        "without it the strategic-tier never gets stance-aligned"
    )
    assert isinstance(bundle.critic, CriticAgent)
    assert bundle.stance is stance, "bundle stance must be the locked stance"
    assert bundle.company_id == "wire-co"
    # Critic shares the company's transcript + cost tracker so persistence
    # and budgeting flow through the same per-sim state.
    assert bundle.critic.transcript is transcript
    assert bundle.critic.cost_tracker is company.cost_tracker


# ─────────────────────────────────────────────────────────────────────────────
# Replay divergence detection (Issue #5)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_replay_raises_on_decision_hash_mismatch(tmp_path: Path) -> None:
    """Replay with a DIFFERENT decision than recorded → divergence error.

    Without divergence detection, the critic would silently return the OLD
    score for the NEW decision — wrong-by-construction. The agent re-computes
    `decision_sha256` on every replay call and compares it to the one
    persisted at record time.
    """
    transcript_path = tmp_path / "diverge.jsonl"
    record_transcript = Transcript(transcript_path, mode="record")
    record_client = _make_mock_client(
        [_critic_json(score=0.9, violations=[], reasoning="Aligned.")]
    )
    rec_critic = CriticAgent(
        transcript=record_transcript,
        cost_tracker=CostTracker(),
        client=record_client,
    )
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    original = _aggressive_decision()
    stance = _make_venture_stance()

    await rec_critic.score(original, stance, state, company_id="test-co")

    # Replay with a DIFFERENT decision — same tick, same company, but the
    # spawn list is changed. The recorded score does not apply to this input.
    replay_transcript = Transcript(transcript_path, mode="replay")
    rep_critic = CriticAgent(
        transcript=replay_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client([]),
    )
    divergent = CeoDecision(
        spawn_nodes=["bd_rep"],  # was ["bd_rep", "bd_rep", "bd_rep", "exec_cfo"]
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning="Different decision.",
        references_stance=["growth_obsession"],
        tier="strategic",
        tick=STRATEGIC_CADENCE_TICKS,
    )

    with pytest.raises(CriticReplayDivergenceError, match="decision hash"):
        await rep_critic.score(divergent, stance, state, company_id="test-co")


@pytest.mark.asyncio
async def test_critic_replay_raises_on_state_hash_mismatch(tmp_path: Path) -> None:
    """Replay with the SAME decision but DIFFERENT state → divergence error."""
    transcript_path = tmp_path / "diverge-state.jsonl"
    record_transcript = Transcript(transcript_path, mode="record")
    rec_critic = CriticAgent(
        transcript=record_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client(
            [_critic_json(score=0.9, violations=[], reasoning="Aligned.")]
        ),
    )
    decision = _aggressive_decision()
    stance = _make_venture_stance()
    original_state = _make_state(tick=STRATEGIC_CADENCE_TICKS, cash=1_000_000.0)

    await rec_critic.score(decision, stance, original_state, company_id="test-co")

    replay_transcript = Transcript(transcript_path, mode="replay")
    rep_critic = CriticAgent(
        transcript=replay_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client([]),
    )
    # Same tick, same decision, but cash differs → state diverged.
    divergent_state = _make_state(tick=STRATEGIC_CADENCE_TICKS, cash=42.0)

    with pytest.raises(CriticReplayDivergenceError, match="state hash"):
        await rep_critic.score(decision, stance, divergent_state, company_id="test-co")


@pytest.mark.asyncio
async def test_critic_replay_raises_on_stance_hash_mismatch(tmp_path: Path) -> None:
    """Replay with the SAME decision and state but DIFFERENT stance → divergence.

    Stance is folded into the state hash (the critic's score is a function of
    decision + stance + state). A copy-paste seed/stance swap that produced
    the same decision text would otherwise silently return the wrong score.
    """
    transcript_path = tmp_path / "diverge-stance.jsonl"
    record_transcript = Transcript(transcript_path, mode="record")
    rec_critic = CriticAgent(
        transcript=record_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client(
            [_critic_json(score=0.9, violations=[], reasoning="Aligned.")]
        ),
    )
    decision = _aggressive_decision()
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    venture = _make_venture_stance()

    await rec_critic.score(decision, venture, state, company_id="test-co")

    replay_transcript = Transcript(transcript_path, mode="replay")
    rep_critic = CriticAgent(
        transcript=replay_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client([]),
    )
    bootstrap = _make_bootstrap_stance()  # Different archetype + sliders.

    with pytest.raises(CriticReplayDivergenceError, match="state hash"):
        await rep_critic.score(decision, bootstrap, state, company_id="test-co")


@pytest.mark.asyncio
async def test_critic_replay_passes_when_inputs_match(tmp_path: Path) -> None:
    """Identical inputs → replay returns the recorded score, no divergence."""
    transcript_path = tmp_path / "match.jsonl"
    record_transcript = Transcript(transcript_path, mode="record")
    rec_critic = CriticAgent(
        transcript=record_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client(
            [_critic_json(score=0.81, violations=["quality_floor"], reasoning="OK.")]
        ),
    )
    decision = _aggressive_decision()
    stance = _make_venture_stance()
    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)

    recorded = await rec_critic.score(decision, stance, state, company_id="test-co")
    assert recorded is not None

    replay_transcript = Transcript(transcript_path, mode="replay")
    rep_critic = CriticAgent(
        transcript=replay_transcript,
        cost_tracker=CostTracker(),
        client=_make_mock_client([]),
    )
    replayed = await rep_critic.score(decision, stance, state, company_id="test-co")
    assert replayed is not None
    assert replayed.score == recorded.score
    assert replayed.violations == recorded.violations
    assert replayed.reasoning == recorded.reasoning


def test_critic_decision_hash_is_deterministic() -> None:
    """Same decision payload → same hash, regardless of dict-insertion order."""
    a = CeoDecision(
        spawn_nodes=["bd_rep"], retire_nodes=[],
        adjust_params={"price": 100.0, "marketing_intensity": 1.2},
        open_locations=0, reasoning="A.",
        references_stance=["risk_tolerance", "growth_obsession"],
        tier="strategic", tick=90,
    )
    # Different insertion order on adjust_params + references_stance.
    b = CeoDecision(
        spawn_nodes=["bd_rep"], retire_nodes=[],
        adjust_params={"marketing_intensity": 1.2, "price": 100.0},
        open_locations=0, reasoning="A.",
        references_stance=["risk_tolerance", "growth_obsession"],
        tier="strategic", tick=90,
    )
    assert CriticAgent._decision_sha256(a) == CriticAgent._decision_sha256(b)


def test_critic_decision_hash_differs_for_different_payloads() -> None:
    """Distinct payloads → distinct hashes (no collision masking divergence)."""
    base = _aggressive_decision()
    other = CeoDecision(
        spawn_nodes=["bd_rep"],  # different spawn list
        retire_nodes=[],
        adjust_params=dict(base.adjust_params),
        open_locations=base.open_locations,
        reasoning=base.reasoning,
        references_stance=list(base.references_stance),
        tier=base.tier,
        tick=base.tick,
    )
    assert CriticAgent._decision_sha256(base) != CriticAgent._decision_sha256(other)
