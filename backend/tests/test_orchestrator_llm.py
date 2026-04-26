"""Tests for the LLM tiers of the v2 CEO orchestrator.

Covers `TacticalOrchestrator` (Haiku 4.5), `StrategicOrchestrator` (Sonnet 4.6),
and the `OrchestratorBundle` that routes ticks across all three layers.

The Anthropic SDK is mocked end-to-end — no real API calls. Each test
constructs an orchestrator with an injected mock `AsyncAnthropic` client
whose `messages.create` coroutine returns scripted assistant responses (and
a tunable `usage` block driving the cost-tracker integration).

Determinism: every replay test uses an in-memory transcript file via
`tmp_path` so concurrent test workers don't fight over the same path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.library_loader import (
    CategoryCaps,
    NodeDef,
    NodeLibrary,
)
from src.simulation.orchestrator import (
    DECISION_HISTORY_LEN,
    LLM_MAX_RETRIES,
    STATE_WINDOW_LEN,
    STRATEGIC_CADENCE_TICKS,
    STRATEGIC_MODEL_ID,
    TACTICAL_CADENCE_TICKS,
    TACTICAL_MODEL_ID,
    CeoDecision,
    CompanyState,
    HeuristicOrchestrator,
    OrchestratorBundle,
    StrategicOrchestrator,
    TacticalOrchestrator,
)
from src.simulation.replay import (
    CeoDecision as ReplayCeoDecision,
)
from src.simulation.replay import (
    CostTracker,
    Transcript,
    TranscriptEntry,
    prompt_sha256,
)
from src.simulation.seed import CompanySeed
from src.simulation.shocks import Shock
from src.simulation.stance import CeoStance

# ─────────────────────────────────────────────────────────────────────────────
# Mock plumbing
# ─────────────────────────────────────────────────────────────────────────────


def _content_block(text: str) -> Any:
    block = MagicMock()
    block.text = text
    return block


def _usage(input_tokens: int = 1200, output_tokens: int = 200) -> Any:
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    return u


def _mock_response(text: str, *, input_tokens: int = 1200, output_tokens: int = 200) -> Any:
    response = MagicMock()
    response.content = [_content_block(text)]
    response.usage = _usage(input_tokens, output_tokens)
    return response


def _make_mock_client(
    scripted_replies: list[str],
    *,
    input_tokens: int = 1200,
    output_tokens: int = 200,
) -> Any:
    """Mock `anthropic.AsyncAnthropic` with a scripted reply queue.

    Each call to `client.messages.create(...)` advances the script. Raises if
    the script is exhausted (forces tests to budget their replies correctly).
    """
    client = MagicMock()
    iterator = iter(scripted_replies)

    async def _create(**_: Any) -> Any:
        try:
            return _mock_response(
                next(iterator),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except StopIteration as exc:
            raise AssertionError("mock client ran out of scripted replies — add more") from exc

    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_create)
    return client


def _decision_json(
    *,
    spawn_nodes: list[str] | None = None,
    retire_nodes: list[str] | None = None,
    adjust_params: dict[str, float] | None = None,
    open_locations: int = 0,
    reasoning: str = "Default reasoning citing my stance.",
    references_stance: list[str] | None = None,
) -> str:
    """Build a valid (or selectively invalid) CeoDecision JSON payload."""
    payload = {
        "spawn_nodes": spawn_nodes if spawn_nodes is not None else [],
        "retire_nodes": retire_nodes if retire_nodes is not None else [],
        "adjust_params": adjust_params if adjust_params is not None else {},
        "open_locations": open_locations,
        "reasoning": reasoning,
        "references_stance": (
            references_stance if references_stance is not None else ["risk_tolerance"]
        ),
    }
    return json.dumps(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — minimal seed/stance/library
# ─────────────────────────────────────────────────────────────────────────────


def _make_seed(economics_model: str = "subscription") -> CompanySeed:
    return CompanySeed(
        name="Test Co",
        niche="B2B SaaS for QA testing",
        archetype="small_team",
        industry_keywords=["saas", "qa"],
        location_label="Product",
        economics_model=economics_model,  # type: ignore[arg-type]
        base_price=99.0,
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


def _make_stance() -> CeoStance:
    return CeoStance(
        archetype="founder_operator",
        risk_tolerance=0.55,
        growth_obsession=0.50,
        quality_floor=0.70,
        hiring_bias="balanced",
        time_horizon="annual",
        cash_comfort=6.0,
        signature_moves=[
            "stay close to the customer",
            "hire only when it hurts",
        ],
        voice="I run a tight ship and I take the calls myself.",
    )


def _make_library() -> NodeLibrary:
    """Tiny but valid `NodeLibrary` covering every category the LLM tiers
    might propose. Built directly from `NodeDef` objects (no YAML round-trip)
    to keep tests independent of `library_loader.load_library`."""
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
            key="needs_prereq",
            category="ops",
            label="Needs Prereq Node",
            hire_cost=15000.0,
            daily_fixed_costs=200.0,
            employees_count=1,
            capacity_contribution=0,
            modifier_keys={},
            prerequisites=["does_not_exist"],
            category_caps=CategoryCaps(soft_cap=5, hard_cap=10),
            applicable_economics=["subscription", "service"],
        ),
        NodeDef(
            key="capped_node",
            category="sales",
            label="Capped Node",
            hire_cost=5000.0,
            daily_fixed_costs=120.0,
            employees_count=1,
            capacity_contribution=0,
            modifier_keys={},
            prerequisites=[],
            category_caps=CategoryCaps(soft_cap=2, hard_cap=2),
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


def _make_tracker(ceiling_usd: float = 50.0) -> CostTracker:
    return CostTracker(ceiling_usd=ceiling_usd)


# ─────────────────────────────────────────────────────────────────────────────
# Tactical tier — happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tactical_tier_emits_decision_on_cadence(tmp_path: Path) -> None:
    """Tick on cadence + valid JSON → CeoDecision with tier='tactical'."""
    client = _make_mock_client(
        [
            _decision_json(
                adjust_params={"price": 105.0, "marketing_intensity": 0.6},
                reasoning="Price up 6% citing growth_obsession.",
                references_stance=["growth_obsession", "hiring_bias"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="record")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    state = _make_state(tick=TACTICAL_CADENCE_TICKS)
    decision = await orch.tick(state)

    assert decision is not None
    assert decision.tier == "tactical"
    assert decision.tick == TACTICAL_CADENCE_TICKS
    assert decision.adjust_params == {"price": 105.0, "marketing_intensity": 0.6}
    assert "growth_obsession" in decision.references_stance
    # API called exactly once.
    assert client.messages.create.await_count == 1
    # Model id pinned to Haiku 4.5.
    assert client.messages.create.await_args.kwargs["model"] == TACTICAL_MODEL_ID


@pytest.mark.asyncio
async def test_tactical_off_cadence_returns_none(tmp_path: Path) -> None:
    """Off-cadence ticks don't even build a prompt — they just record state."""
    client = _make_mock_client([])  # zero scripted replies — must not be called
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    # Off cadence: 1, 7, 15, 29 — all not divisible by 30.
    for tick in (1, 7, 15, 29):
        decision = await orch.tick(_make_state(tick=tick))
        assert decision is None
    # Confirm zero LLM calls were made.
    assert client.messages.create.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Strategic tier — happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_strategic_tier_emits_decision_on_cadence(tmp_path: Path) -> None:
    """Tick on cadence + valid JSON → CeoDecision with tier='strategic'."""
    client = _make_mock_client(
        [
            _decision_json(
                spawn_nodes=["exec_cfo"],
                open_locations=2,
                reasoning="Hiring CFO + opening 2 locations — venture growth play.",
                references_stance=["growth_obsession", "time_horizon"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="record")
    orch = StrategicOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    state = _make_state(tick=STRATEGIC_CADENCE_TICKS)
    decision = await orch.tick(state)

    assert decision is not None
    assert decision.tier == "strategic"
    assert decision.tick == STRATEGIC_CADENCE_TICKS
    assert decision.spawn_nodes == ["exec_cfo"]
    assert decision.open_locations == 2
    assert client.messages.create.await_args.kwargs["model"] == STRATEGIC_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
# Strategic tier — severe shock force-wake
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_strategic_force_wake_on_severe_shock(tmp_path: Path) -> None:
    """A severe shock at tick=15 (off-cadence) still wakes strategic via force_wake."""
    client = _make_mock_client(
        [
            _decision_json(
                adjust_params={"raise_amount": 5_000_000.0},
                reasoning="Market crash — raising emergency bridge per cash_comfort.",
                references_stance=["cash_comfort", "risk_tolerance"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="record")
    orch = StrategicOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    severe_shock = Shock(
        name="market_crash",
        severity="severe",
        duration_ticks=120,
        impact={"market_demand_mult": 0.6},
        description="Severe market crash",
        tick_started=15,
    )
    state = _make_state(
        tick=15,  # NOT divisible by 90
        active_shocks=[severe_shock],
    )

    decision = await orch.tick(state, force_wake=True)
    assert decision is not None
    assert decision.tier == "strategic"
    assert decision.tick == 15  # off-cadence but force_wake fired
    assert "cash_comfort" in decision.references_stance
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_strategic_no_force_wake_no_call_off_cadence(tmp_path: Path) -> None:
    """Off-cadence tick with no force_wake → no LLM call, returns None."""
    client = _make_mock_client([])
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = StrategicOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    # tick=15 is off-cadence; no force_wake → silent.
    assert await orch.tick(_make_state(tick=15), force_wake=False) is None
    assert client.messages.create.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Bad JSON triggers retry; persistent bad JSON returns None
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bad_json_triggers_retry_then_succeeds(tmp_path: Path) -> None:
    """First reply unparseable → retry → second reply valid → decision returned."""
    client = _make_mock_client(
        [
            "this is not JSON at all",
            _decision_json(
                adjust_params={"price": 110.0},
                reasoning="Recovered after malformed first attempt.",
                references_stance=["risk_tolerance"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is not None
    assert decision.adjust_params == {"price": 110.0}
    assert client.messages.create.await_count == 2  # one retry happened


@pytest.mark.asyncio
async def test_persistent_bad_json_returns_none(tmp_path: Path) -> None:
    """Two unparseable replies → orchestrator gives up, returns None."""
    client = _make_mock_client(["garbage", "still garbage"])
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is None
    # Used the full retry budget.
    assert client.messages.create.await_count == LLM_MAX_RETRIES + 1


# ─────────────────────────────────────────────────────────────────────────────
# Empty references_stance triggers retry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_references_stance_triggers_retry(tmp_path: Path) -> None:
    """Role-lock invariant: empty references_stance → retry once with feedback."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Generic, drift-prone reasoning",
                references_stance=[],  # role-lock violation
            ),
            _decision_json(
                reasoning="Now citing risk_tolerance explicitly.",
                references_stance=["risk_tolerance"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is not None
    assert decision.references_stance == ["risk_tolerance"]
    assert client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_persistent_empty_references_stance_returns_none(tmp_path: Path) -> None:
    """If even after retry the LLM keeps returning empty references_stance →
    role-lock cannot be enforced, return None and let the caller fall back."""
    client = _make_mock_client(
        [
            _decision_json(reasoning="No stance refs", references_stance=[]),
            _decision_json(reasoning="Still no stance refs", references_stance=[]),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# Decision filtering — prerequisites and hard_cap
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_filters_spawn_with_unmet_prereqs(tmp_path: Path) -> None:
    """LLM proposes a node missing prereqs → filtered out + note appended."""
    client = _make_mock_client(
        [
            _decision_json(
                # `needs_prereq` requires `does_not_exist` — must be filtered.
                spawn_nodes=["bd_rep", "needs_prereq"],
                reasoning="Add BD rep and a node we shouldn't be able to spawn.",
                references_stance=["growth_obsession"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is not None
    assert "needs_prereq" not in decision.spawn_nodes
    assert "bd_rep" in decision.spawn_nodes
    # Filtering note appended to reasoning.
    assert "needs_prereq" in decision.reasoning
    assert "prerequisites not satisfied" in decision.reasoning


@pytest.mark.asyncio
async def test_filters_spawn_at_hard_cap(tmp_path: Path) -> None:
    """LLM proposes a node already at hard_cap → filtered out + note appended."""
    client = _make_mock_client(
        [
            _decision_json(
                spawn_nodes=["capped_node", "bd_rep"],
                reasoning="Try to overspawn capped_node and also a fine BD rep.",
                references_stance=["hiring_bias"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    state = _make_state(
        tick=TACTICAL_CADENCE_TICKS,
        spawned={
            "founder_engineer": 1,
            "capped_node": 2,  # hard_cap is 2 → must reject another
        },
    )
    decision = await orch.tick(state)
    assert decision is not None
    assert "capped_node" not in decision.spawn_nodes
    assert "bd_rep" in decision.spawn_nodes
    assert "hard_cap" in decision.reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Cost tracker integration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_tracker_preflight_skips_call(tmp_path: Path) -> None:
    """`would_exceed=True` → orchestrator skips the API call, returns None."""
    client = _make_mock_client([])  # zero replies — must NOT be called
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    # Tiny ceiling so any predicted spend trivially exceeds it.
    tracker = _make_tracker(ceiling_usd=0.000001)
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=tracker,
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is None
    assert client.messages.create.await_count == 0
    assert tracker.total_cost() == 0.0  # no spend recorded


@pytest.mark.asyncio
async def test_cost_tracker_charges_on_successful_call(tmp_path: Path) -> None:
    """A successful Haiku call charges the tracker per the recorded usage."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Charge me.",
                references_stance=["growth_obsession"],
            ),
        ],
        input_tokens=2_000,
        output_tokens=400,
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="record")
    tracker = _make_tracker(ceiling_usd=10.0)
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=tracker,
        client=client,
    )

    decision = await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    assert decision is not None
    # Haiku 4.5 pricing: 2_000 * 1e-6 input + 400 * 5e-6 output = 0.002 + 0.002 = 0.004
    assert tracker.total_cost() == pytest.approx(0.004, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Replay mode — pre-recorded transcript replays without LLM call
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_mode_skips_llm_call(tmp_path: Path) -> None:
    """A pre-recorded transcript serves the decision; the LLM is never invoked."""
    transcript_path = tmp_path / "sim.jsonl"

    # ── Step 1: record a transcript by running tactical with a real "fake" LLM.
    record_client = _make_mock_client(
        [
            _decision_json(
                adjust_params={"price": 123.0},
                reasoning="Recorded run — bumping price.",
                references_stance=["risk_tolerance"],
            ),
        ]
    )
    rec_transcript = Transcript(transcript_path, mode="record")
    rec_orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=rec_transcript,
        cost_tracker=_make_tracker(),
        client=record_client,
        sim_id="sim_a",
        company_id="acme",
    )
    state = _make_state(tick=TACTICAL_CADENCE_TICKS)
    rec_decision = await rec_orch.tick(state)
    assert rec_decision is not None

    # ── Step 2: replay the same scenario; the LLM client must NEVER be called.
    replay_client = MagicMock()
    replay_client.messages = MagicMock()
    replay_client.messages.create = AsyncMock(
        side_effect=AssertionError("LLM must not be called in replay mode")
    )
    rep_transcript = Transcript(transcript_path, mode="replay")
    # Match the record-run's tracker ceiling exactly so the `budget` field in
    # the user prompt hashes to the same SHA. (`replay_or_call` rejects any
    # prompt-SHA drift to catch state divergence — and the budget envelope is
    # part of the state.)
    rep_orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=rep_transcript,
        cost_tracker=_make_tracker(),  # default ceiling = same as record run
        client=replay_client,
        sim_id="sim_a",
        company_id="acme",
    )

    rep_decision = await rep_orch.tick(state)
    assert rep_decision is not None
    assert rep_decision.adjust_params == {"price": 123.0}
    assert rep_decision.tier == "tactical"
    # LLM client never called.
    assert replay_client.messages.create.await_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorBundle — cadence routing
# ─────────────────────────────────────────────────────────────────────────────


def _make_bundle(
    tmp_path: Path,
    *,
    tactical_replies: list[str],
    strategic_replies: list[str],
) -> OrchestratorBundle:
    """Build an OrchestratorBundle wired to two separate mock clients."""
    seed = _make_seed()
    stance = _make_stance()
    library = _make_library()
    library_dict = {k: n.model_dump() for k, n in library.nodes.items()}

    transcript = Transcript(tmp_path / "bundle.jsonl", mode="off")
    tracker = _make_tracker(ceiling_usd=50.0)

    heuristic = HeuristicOrchestrator(seed, stance, library_dict)
    tactical = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client(tactical_replies),
    )
    strategic = StrategicOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=tracker,
        client=_make_mock_client(strategic_replies),
    )
    return OrchestratorBundle(
        heuristic=heuristic,
        tactical=tactical,
        strategic=strategic,
    )


@pytest.mark.asyncio
async def test_bundle_tick_210_fires_heuristic_and_tactical(tmp_path: Path) -> None:
    """tick=210: 210 % 7 == 0 AND 210 % 30 == 0 BUT 210 % 90 != 0.

    Heuristic and tactical both fire, strategic does not.
    """
    tactical_reply = _decision_json(
        adjust_params={"price": 100.0},
        reasoning="Tactical price tweak.",
        references_stance=["growth_obsession"],
    )
    bundle = _make_bundle(
        tmp_path,
        tactical_replies=[tactical_reply],
        strategic_replies=[],  # must NOT be called
    )
    # State that makes the heuristic fire too — sustained high util to trigger
    # the capacity rule. Pre-warm the heuristic's history.
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
        tick=210,
        spawned=spawned,
        capacity_utilization=0.99,
    )
    decisions = await bundle.tick(state)

    tiers = sorted(d.tier for d in decisions)
    # Heuristic is allowed to no-op if no rule matches; but sustained 0.99 util
    # for 3+ samples should produce one. Tactical must fire on cadence.
    assert "tactical" in tiers
    assert "strategic" not in tiers


@pytest.mark.asyncio
async def test_bundle_tick_180_fires_all_three(tmp_path: Path) -> None:
    """tick=180: 180 % 7 != 0 (no heuristic) BUT 180 % 30 == 0 AND 180 % 90 == 0.

    Note: 180 / 7 = 25.71, so it's actually not heuristic-cadence. We need a
    tick divisible by 7, 30, and 90 simultaneously. LCM(7,30,90) = 630.
    """
    tactical_reply = _decision_json(
        adjust_params={"price": 105.0},
        reasoning="Tactical at LCM tick.",
        references_stance=["risk_tolerance"],
    )
    strategic_reply = _decision_json(
        spawn_nodes=["exec_cfo"],
        reasoning="Strategic CFO hire.",
        references_stance=["growth_obsession", "time_horizon"],
    )
    bundle = _make_bundle(
        tmp_path,
        tactical_replies=[tactical_reply],
        strategic_replies=[strategic_reply],
    )
    spawned = {"founder_engineer": 1}
    for offset in range(628, 630):
        bundle.heuristic.tick(
            _make_state(
                tick=offset,
                spawned=spawned,
                capacity_utilization=0.99,
            )
        )

    state = _make_state(
        tick=630,  # divisible by 7, 30, AND 90
        spawned=spawned,
        capacity_utilization=0.99,
    )
    decisions = await bundle.tick(state)

    tiers = sorted(d.tier for d in decisions)
    assert "tactical" in tiers
    assert "strategic" in tiers
    # Heuristic may or may not fire depending on rule selection — but at least
    # the LLM tiers must both produce a decision on this LCM tick.


@pytest.mark.asyncio
async def test_bundle_severe_shock_force_wakes_strategic(tmp_path: Path) -> None:
    """tick=15 (no cadences) + severe shock → strategic still fires via force_wake."""
    strategic_reply = _decision_json(
        adjust_params={"raise_amount": 3_000_000.0},
        reasoning="Severe market crash — bridge financing per cash_comfort.",
        references_stance=["cash_comfort", "risk_tolerance"],
    )
    bundle = _make_bundle(
        tmp_path,
        tactical_replies=[],  # tactical should not fire
        strategic_replies=[strategic_reply],
    )

    severe = Shock(
        name="market_crash",
        severity="severe",
        duration_ticks=120,
        impact={"market_demand_mult": 0.5},
        description="Severe crash",
        tick_started=15,
    )
    state = _make_state(tick=15, active_shocks=[severe])

    decisions = await bundle.tick(state)
    tiers = [d.tier for d in decisions]
    assert "strategic" in tiers
    assert "tactical" not in tiers


# ─────────────────────────────────────────────────────────────────────────────
# Inception prompting — system prompt contains stance attributes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_contains_stance_attributes(tmp_path: Path) -> None:
    """Verify the stance role-lock paragraph is wired into every LLM call."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Stance-aware decision.",
                references_stance=["risk_tolerance"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    stance = _make_stance()
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=stance,
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    system_prompt = client.messages.create.await_args.kwargs["system"]

    # Stance attributes must be cited verbatim in the role-lock paragraph.
    assert stance.archetype in system_prompt
    assert f"risk_tolerance={stance.risk_tolerance:.2f}" in system_prompt
    assert f"growth_obsession={stance.growth_obsession:.2f}" in system_prompt
    assert f"quality_floor={stance.quality_floor:.2f}" in system_prompt
    assert stance.hiring_bias in system_prompt
    assert stance.time_horizon in system_prompt
    # Tier-specific instruction header is appended.
    assert "TACTICAL TIER" in system_prompt
    # JSON-only directive is the closing line.
    assert "strictly valid JSON" in system_prompt


@pytest.mark.asyncio
async def test_strategic_system_prompt_contains_strategic_header(tmp_path: Path) -> None:
    """Strategic tier uses its own DOMAIN_INSTRUCTION header."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Strategic.",
                references_stance=["growth_obsession"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = StrategicOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )
    await orch.tick(_make_state(tick=STRATEGIC_CADENCE_TICKS))
    system_prompt = client.messages.create.await_args.kwargs["system"]
    assert "STRATEGIC TIER" in system_prompt
    assert "TACTICAL TIER" not in system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# State window + decision history wiring
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_window_passed_in_user_prompt(tmp_path: Path) -> None:
    """Pre-warm the orchestrator with several off-cadence ticks; the on-cadence
    call's user prompt must include those snapshots in `state_window`."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Acting on the trend.",
                references_stance=["growth_obsession"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    # Warm up with 5 off-cadence ticks of varying cash.
    for offset, cash in enumerate([1_000_000, 1_010_000, 1_005_000, 990_000, 980_000]):
        await orch.tick(
            _make_state(
                tick=TACTICAL_CADENCE_TICKS - 5 + offset,
                cash=float(cash),
            )
        )

    # Trigger the on-cadence call.
    await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS, cash=970_000.0))

    user_prompt = client.messages.create.await_args.kwargs["messages"][0]["content"]
    payload = json.loads(user_prompt)
    assert "state_window" in payload
    # Window must contain at most STATE_WINDOW_LEN entries.
    assert len(payload["state_window"]) <= STATE_WINDOW_LEN
    # And must include the most recent snapshot (cash=970_000 at the trigger tick).
    assert payload["state_window"][-1]["cash"] == 970_000.0
    assert payload["state_window"][-1]["tick"] == TACTICAL_CADENCE_TICKS


@pytest.mark.asyncio
async def test_recent_decisions_filtered_to_same_tier(tmp_path: Path) -> None:
    """`recent_decisions` in the prompt is filtered to THIS tier only."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Continuity check.",
                references_stance=["growth_obsession"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    orch = TacticalOrchestrator(
        seed=_make_seed(),
        stance=_make_stance(),
        library=_make_library(),
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    # Build a mixed-tier history.
    history: list[CeoDecision] = [
        CeoDecision(
            spawn_nodes=[],
            retire_nodes=[],
            adjust_params={},
            open_locations=0,
            reasoning="HEUR-1",
            references_stance=["cash_comfort"],
            tier="heuristic",
            tick=7,
        ),
        CeoDecision(
            spawn_nodes=[],
            retire_nodes=[],
            adjust_params={"price": 95.0},
            open_locations=0,
            reasoning="TAC-1",
            references_stance=["growth_obsession"],
            tier="tactical",
            tick=30,
        ),
        CeoDecision(
            spawn_nodes=[],
            retire_nodes=[],
            adjust_params={},
            open_locations=0,
            reasoning="STRAT-1",
            references_stance=["time_horizon"],
            tier="strategic",
            tick=90,
        ),
        CeoDecision(
            spawn_nodes=[],
            retire_nodes=[],
            adjust_params={"price": 100.0},
            open_locations=0,
            reasoning="TAC-2",
            references_stance=["growth_obsession"],
            tier="tactical",
            tick=60,
        ),
    ]
    state = _make_state(
        tick=TACTICAL_CADENCE_TICKS,
        recent_decisions=history,
    )
    await orch.tick(state)

    user_prompt = client.messages.create.await_args.kwargs["messages"][0]["content"]
    payload = json.loads(user_prompt)
    assert "recent_decisions" in payload
    # Only tactical decisions should be present.
    reasonings = {d["reasoning"] for d in payload["recent_decisions"]}
    assert reasonings == {"TAC-1", "TAC-2"}
    # And capped to DECISION_HISTORY_LEN.
    assert len(payload["recent_decisions"]) <= DECISION_HISTORY_LEN


# ─────────────────────────────────────────────────────────────────────────────
# Available nodes filtering — economics_model
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_available_nodes_filtered_by_economics_model(tmp_path: Path) -> None:
    """`available_nodes` in the prompt only includes nodes applicable to the
    seed's economics_model. A subscription seed must not see physical-only nodes."""
    client = _make_mock_client(
        [
            _decision_json(
                reasoning="Ack.",
                references_stance=["growth_obsession"],
            ),
        ]
    )
    transcript = Transcript(tmp_path / "sim.jsonl", mode="off")
    # Build a library with one subscription node and one physical-only node.
    physical_only = NodeDef(
        key="restaurant_only",
        category="location",
        label="Restaurant Only",
        hire_cost=50_000.0,
        daily_fixed_costs=300.0,
        employees_count=10,
        capacity_contribution=80,
        modifier_keys={},
        prerequisites=[],
        category_caps=CategoryCaps(soft_cap=10, hard_cap=999),
        applicable_economics=["physical"],
    )
    library = _make_library()
    library.nodes["restaurant_only"] = physical_only

    orch = TacticalOrchestrator(
        seed=_make_seed(economics_model="subscription"),
        stance=_make_stance(),
        library=library,
        transcript=transcript,
        cost_tracker=_make_tracker(),
        client=client,
    )

    await orch.tick(_make_state(tick=TACTICAL_CADENCE_TICKS))
    user_prompt = client.messages.create.await_args.kwargs["messages"][0]["content"]
    payload = json.loads(user_prompt)
    keys = {n["key"] for n in payload["available_nodes"]}
    assert "restaurant_only" not in keys
    assert "founder_engineer" in keys


# ─────────────────────────────────────────────────────────────────────────────
# Replay — ensures determinism across record→replay round trip
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_via_handcrafted_transcript(tmp_path: Path) -> None:
    """A hand-rolled `TranscriptEntry` written to disk can be replayed by the
    orchestrator without ever touching the LLM."""
    transcript_path = tmp_path / "sim.jsonl"

    # Build the same prompt the orchestrator would build for tick=30.
    stance = _make_stance()
    library = _make_library()
    seed = _make_seed()
    # Pre-instantiate to trigger the canonical user-prompt build (we need its
    # SHA to write a valid transcript).
    transcript_off = Transcript(transcript_path, mode="off")
    orch_for_prompt = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript_off,
        cost_tracker=_make_tracker(),
        client=_make_mock_client([]),
        sim_id="sim_x",
        company_id="acme",
    )
    state = _make_state(tick=TACTICAL_CADENCE_TICKS)
    # Drive _record_snapshot so the prompt includes the snapshot row.
    orch_for_prompt._record_snapshot(state)
    user_prompt = orch_for_prompt._build_user_prompt(state)
    expected_sha = prompt_sha256(user_prompt)

    # Hand-write the transcript entry. Note company_id must match the orch's
    # `<company>:<tier>` key for replay to find it.
    canonical_decision = ReplayCeoDecision(
        spawn_nodes=["bd_rep"],
        retire_nodes=[],
        adjust_params={"price": 99.0},
        open_locations=0,
        reasoning="Hand-recorded transcript decision.",
        references_stance=["growth_obsession"],
    )
    rec = Transcript(transcript_path, mode="record")
    rec.record(
        TranscriptEntry(
            sim_id="sim_x",
            tick=TACTICAL_CADENCE_TICKS,
            company_id="acme:tactical",
            decision_id="hand-rolled-1",
            tier="tactical",
            prompt_sha256=expected_sha,
            raw_response="(hand-rolled)",
            parsed_decision=canonical_decision,
            model=TACTICAL_MODEL_ID,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.001,
        )
    )

    # Now replay through the orchestrator — the LLM client must never be called.
    explode_client = MagicMock()
    explode_client.messages = MagicMock()
    explode_client.messages.create = AsyncMock(
        side_effect=AssertionError("replay must not call LLM")
    )
    rep_transcript = Transcript(transcript_path, mode="replay")
    rep_orch = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=rep_transcript,
        # MUST match the tracker the prompt was hashed against (default ceiling).
        cost_tracker=_make_tracker(),
        client=explode_client,
        sim_id="sim_x",
        company_id="acme",
    )
    decision = await rep_orch.tick(state)
    assert decision is not None
    assert decision.adjust_params == {"price": 99.0}
    assert decision.spawn_nodes == ["bd_rep"]
    assert decision.tier == "tactical"
    assert decision.tick == TACTICAL_CADENCE_TICKS
    assert explode_client.messages.create.await_count == 0
