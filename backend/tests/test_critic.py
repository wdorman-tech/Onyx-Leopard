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
