"""Determinism invariants for v2 — same seed × stance × RNG × shocks =>
identical outcomes (V2 plan invariant 9).

Two flavors:
  * Heuristic-only traces (no LLM): same-seed determinism, RNG / stance
    divergence — `_silent_bundle` mocks tactical and strategic to None.
  * LLM record/replay round-trips: a `Transcript` records mocked-LLM
    decisions in phase A, then replays them in phase B with the LLM client
    asserted-never-called. End-to-end byte-equality of every TickResult
    field is the contract (V2 PRD "Done criteria" #2).
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.library_loader import _reset_library_cache, get_library
from src.simulation.orchestrator import (
    LLM_MAX_RETRIES,
    STRATEGIC_CADENCE_TICKS,
    TACTICAL_CADENCE_TICKS,
    HeuristicOrchestrator,
    OrchestratorBundle,
    StrategicOrchestrator,
    TacticalOrchestrator,
)
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import sample_seed_for_archetype
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import sample_stance
from src.simulation.unified_v2 import CompanyAgentV2, TickResult


@pytest.fixture(scope="module")
def library():
    _reset_library_cache()
    return get_library()


def _refs_for(library, economics_model: str):
    s, r, c = [], [], []
    for key, node in sorted(library.nodes.items()):
        if economics_model not in node.applicable_economics:
            continue
        if node.category == "supplier" and not s:
            s.append(key)
        elif node.category == "revenue" and not r:
            r.append(key)
        elif node.category == "ops" and not c:
            c.append(key)
    return s, r, c


def _silent_bundle(seed, stance, library):
    library_dict = {key: node.model_dump() for key, node in library.nodes.items()}
    heuristic = HeuristicOrchestrator(seed=seed, stance=stance, library=library_dict)
    tactical = MagicMock()
    tactical.tick = AsyncMock(return_value=None)
    strategic = MagicMock()
    strategic.tick = AsyncMock(return_value=None)
    return OrchestratorBundle(heuristic=heuristic, tactical=tactical, strategic=strategic)


def _build(library, *, rng_seed: int, shock_lambdas: dict[str, float] | None = None):
    rng = random.Random(rng_seed)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    s, r, c = _refs_for(library, "physical")
    seed = seed.model_copy(update={
        "economics_model": "physical",
        "initial_supplier_types": s,
        "initial_revenue_streams": r,
        "initial_cost_centers": c,
    })
    stance = sample_stance("bootstrap", rng=rng)
    bundle = _silent_bundle(seed, stance, library)

    # Important: derive sub-RNG seeds from `rng` so two calls with the same
    # `rng_seed` produce byte-identical state. We capture the seeds rather
    # than the Random objects so the construction is fully deterministic.
    shock_seed = rng.randint(0, 2**31 - 1)
    company_rng_seed = rng.randint(0, 2**31 - 1)

    return CompanyAgentV2(
        seed=seed,
        stance=stance,
        library=library,
        sim_id="determinism",
        company_id="co",
        rng=random.Random(company_rng_seed),
        transcript=None,
        cost_tracker=CostTracker(ceiling_usd=10.0),
        shock_scheduler=ShockScheduler(rng_seed=shock_seed, lambdas=shock_lambdas or {}),
        orchestrator=bundle,
    )


def _trace(agent: CompanyAgentV2, ticks: int) -> list[tuple[float, float, dict[str, int]]]:
    out: list[tuple[float, float, dict[str, int]]] = []
    for _ in range(ticks):
        if not agent.alive:
            break
        asyncio.run(agent.step())
        out.append((agent.cash, agent.daily_revenue, dict(agent.spawned_nodes)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mocked-LLM plumbing for record/replay round-trip tests
# ─────────────────────────────────────────────────────────────────────────────


def _content_block(text: str) -> Any:
    block = MagicMock()
    block.text = text
    return block


def _usage(input_tokens: int = 0, output_tokens: int = 0) -> Any:
    """Default to ZERO tokens so the cost-tracker stays at $0 across calls.

    The user prompt embeds `spent_usd` / `remaining_usd` (see
    `_build_user_prompt`). If a recording call charges the tracker, the next
    call's prompt SHA differs from what replay sees — replay always reports
    `spent_usd=0` because the recorded cost was paid in the original run.
    Zero-token usage keeps both runs aligned without adding bookkeeping.
    """
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    return u


def _mock_response(text: str) -> Any:
    response = MagicMock()
    response.content = [_content_block(text)]
    response.usage = _usage()
    return response


def _make_mock_client(scripted_replies: list[str]) -> Any:
    """Mock `AsyncAnthropic` with a scripted reply queue.

    Each call to `client.messages.create(...)` advances the script. Raises
    `AssertionError` if the script is exhausted — forces tests to budget
    enough replies for cadence × retry-headroom.
    """
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


def _decision_json(
    *,
    spawn_nodes: list[str] | None = None,
    retire_nodes: list[str] | None = None,
    adjust_params: dict[str, float] | None = None,
    open_locations: int = 0,
    reasoning: str = "Mocked decision in deterministic replay test.",
    references_stance: list[str] | None = None,
) -> str:
    """Build a CeoDecision JSON payload matching the orchestrator schema."""
    payload = {
        "spawn_nodes": spawn_nodes if spawn_nodes is not None else [],
        "retire_nodes": retire_nodes if retire_nodes is not None else [],
        "adjust_params": adjust_params if adjust_params is not None else {},
        "open_locations": open_locations,
        "reasoning": reasoning,
        "references_stance": (
            references_stance if references_stance is not None
            else ["risk_tolerance"]
        ),
    }
    return json.dumps(payload)


def _build_replay_seed_stance(library, *, rng_seed: int):
    """Sample the SAME seed + stance from a fresh RNG — used by record AND
    replay so both phases see byte-identical config inputs.
    """
    rng = random.Random(rng_seed)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    s, r, c = _refs_for(library, "physical")
    seed = seed.model_copy(update={
        "economics_model": "physical",
        "initial_supplier_types": s,
        "initial_revenue_streams": r,
        "initial_cost_centers": c,
    })
    stance = sample_stance("bootstrap", rng=rng)
    return seed, stance


def _build_llm_agent(
    library,
    *,
    seed,
    stance,
    transcript: Transcript,
    tactical_client: Any,
    strategic_client: Any,
    company_rng_seed: int,
    shock_seed: int,
    sim_id: str,
    company_id: str,
):
    """Construct a CompanyAgentV2 with mocked tactical+strategic clients
    (heuristic real, no critic). Used by the record/replay tests so both
    phases share identical wiring except for the transcript mode.
    """
    library_dict = {key: node.model_dump() for key, node in library.nodes.items()}
    cost_tracker = CostTracker(ceiling_usd=10.0)

    heuristic = HeuristicOrchestrator(
        seed=seed, stance=stance, library=library_dict
    )
    tactical = TacticalOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=cost_tracker,
        client=tactical_client,
        sim_id=sim_id,
        company_id=company_id,
    )
    strategic = StrategicOrchestrator(
        seed=seed,
        stance=stance,
        library=library,
        transcript=transcript,
        cost_tracker=cost_tracker,
        client=strategic_client,
        sim_id=sim_id,
        company_id=company_id,
    )
    bundle = OrchestratorBundle(
        heuristic=heuristic,
        tactical=tactical,
        strategic=strategic,
        # critic intentionally None — these tests verify replay determinism
        # of the LLM tier round-trip, not the critic agent's hashing.
    )

    return CompanyAgentV2(
        seed=seed,
        stance=stance,
        library=library,
        sim_id=sim_id,
        company_id=company_id,
        rng=random.Random(company_rng_seed),
        transcript=transcript,
        cost_tracker=cost_tracker,
        shock_scheduler=ShockScheduler(rng_seed=shock_seed, lambdas={}),
        orchestrator=bundle,
    )


# Fields on `TickResult` that must be byte-identical between record and
# replay. `decisions` and shock lists are compared separately as objects.
_TICK_NUMERIC_FIELDS: tuple[str, ...] = (
    "tick",
    "cash",
    "daily_revenue",
    "daily_costs",
    "monthly_revenue",
    "capacity_utilization",
    "avg_satisfaction",
    "employee_count",
    "bankrupt",
)


def _assert_tick_results_equal(
    rec: TickResult, rep: TickResult, *, tick_idx: int
) -> None:
    """Assert all numeric + structural TickResult fields match value-equal."""
    for field in _TICK_NUMERIC_FIELDS:
        rec_v = getattr(rec, field)
        rep_v = getattr(rep, field)
        assert rec_v == rep_v, (
            f"tick {tick_idx}: TickResult.{field} drift "
            f"(record={rec_v!r}, replay={rep_v!r})"
        )
    assert rec.spawned_nodes == rep.spawned_nodes, (
        f"tick {tick_idx}: spawned_nodes drift "
        f"(record={rec.spawned_nodes}, replay={rep.spawned_nodes})"
    )
    assert rec.decisions == rep.decisions, (
        f"tick {tick_idx}: decisions drift"
    )


def _assert_finite_tick(result: TickResult, *, tick_idx: int) -> None:
    """Defensive: every numeric TickResult field must be finite (no NaN/Inf)."""
    for field in (
        "cash",
        "daily_revenue",
        "daily_costs",
        "monthly_revenue",
        "capacity_utilization",
        "avg_satisfaction",
    ):
        v = getattr(result, field)
        assert math.isfinite(v), (
            f"tick {tick_idx}: TickResult.{field}={v!r} not finite"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic-only determinism tests
# ─────────────────────────────────────────────────────────────────────────────


def test_same_seed_same_trace(library):
    """Two agents constructed with identical (seed_rng, shocks) produce
    identical per-tick (cash, revenue, spawned_nodes) traces."""
    a1 = _build(library, rng_seed=12345)
    a2 = _build(library, rng_seed=12345)
    t1 = _trace(a1, 50)
    t2 = _trace(a2, 50)
    assert t1 == t2


def test_different_seeds_diverge(library):
    """Different rng_seed produces materially different traces."""
    a1 = _build(library, rng_seed=1)
    a2 = _build(library, rng_seed=99999)
    t1 = _trace(a1, 50)
    t2 = _trace(a2, 50)
    # Final cash should differ (or at least one tick should differ in
    # spawned_nodes). The probability of a 50-tick byte-match across two
    # totally independent seeds is vanishingly small.
    assert t1 != t2


def test_shock_schedule_determinism(library):
    """Same shock_lambdas + same rng_seed produces same arrival timeline."""
    a1 = _build(library, rng_seed=7, shock_lambdas={"market_crash": 0.05})
    a2 = _build(library, rng_seed=7, shock_lambdas={"market_crash": 0.05})

    arrivals1: list[int] = []
    arrivals2: list[int] = []
    for _ in range(80):
        if a1.alive:
            r1 = asyncio.run(a1.step())
            for _ in r1.arriving_shocks:
                arrivals1.append(r1.tick)
        if a2.alive:
            r2 = asyncio.run(a2.step())
            for _ in r2.arriving_shocks:
                arrivals2.append(r2.tick)
    assert arrivals1 == arrivals2


# ─────────────────────────────────────────────────────────────────────────────
# LLM record/replay round-trip
# ─────────────────────────────────────────────────────────────────────────────


def _run_record_replay_roundtrip(
    library,
    tmp_path: Path,
    *,
    n_ticks: int,
    rng_seed: int = 4321,
    company_rng_seed: int = 17,
    shock_seed: int = 91,
) -> tuple[list[TickResult], list[TickResult], int]:
    """Drive a full record → replay round-trip. Returns
    `(record_trace, replay_trace, record_await_count)`.

    Caller chooses the tick count; cadence math (`TACTICAL_CADENCE_TICKS`
    = 30, `STRATEGIC_CADENCE_TICKS` = 90) determines how many mock replies
    to script. We over-script generously to cover `LLM_MAX_RETRIES`
    headroom and any cadence-edge cases.
    """
    transcript_path = tmp_path / "roundtrip.jsonl"
    seed, stance = _build_replay_seed_stance(library, rng_seed=rng_seed)

    # Cadence math + retry headroom. Each tier may retry up to
    # LLM_MAX_RETRIES extra times per cadence call on a parse failure;
    # our scripted replies are always valid so retries shouldn't fire,
    # but we budget for them anyway so a future schema tweak doesn't
    # silently exhaust the queue.
    tactical_fires = max(1, n_ticks // TACTICAL_CADENCE_TICKS)
    strategic_fires = max(1, n_ticks // STRATEGIC_CADENCE_TICKS)
    tactical_budget = tactical_fires * (LLM_MAX_RETRIES + 1) * 2
    strategic_budget = strategic_fires * (LLM_MAX_RETRIES + 1) * 2

    tactical_replies = [
        _decision_json(
            adjust_params={"price": float(seed.starting_price)},
            reasoning="Tactical no-op price reaffirmation.",
            references_stance=["risk_tolerance"],
        )
        for _ in range(tactical_budget)
    ]
    strategic_replies = [
        _decision_json(
            spawn_nodes=[],
            reasoning="Strategic stay-the-course; no exec hire this period.",
            references_stance=["time_horizon"],
        )
        for _ in range(strategic_budget)
    ]

    # ── Phase A: RECORD ──────────────────────────────────────────────────
    record_transcript = Transcript(transcript_path, mode="record")
    record_tactical_client = _make_mock_client(tactical_replies)
    record_strategic_client = _make_mock_client(strategic_replies)
    record_agent = _build_llm_agent(
        library,
        seed=seed,
        stance=stance,
        transcript=record_transcript,
        tactical_client=record_tactical_client,
        strategic_client=record_strategic_client,
        company_rng_seed=company_rng_seed,
        shock_seed=shock_seed,
        sim_id="roundtrip-sim",
        company_id="co",
    )
    record_trace: list[TickResult] = []
    for _ in range(n_ticks):
        if not record_agent.alive:
            break
        record_trace.append(asyncio.run(record_agent.step()))

    record_await_count = (
        record_tactical_client.messages.create.await_count
        + record_strategic_client.messages.create.await_count
    )

    # ── Phase B: REPLAY ──────────────────────────────────────────────────
    # Build a FRESH seed/stance from the same rng_seed — sampling is
    # deterministic, so both phases see byte-identical configs.
    seed2, stance2 = _build_replay_seed_stance(library, rng_seed=rng_seed)
    assert seed == seed2, "seed sampling must be deterministic for replay"

    replay_transcript = Transcript(transcript_path, mode="replay")
    # Replay clients must NEVER be awaited. Use AsyncMock with explicit
    # side_effect so an accidental call surfaces as a test failure.
    explode_client_t = MagicMock()
    explode_client_t.messages = MagicMock()
    explode_client_t.messages.create = AsyncMock(
        side_effect=AssertionError(
            "tactical LLM was invoked in replay mode (should be short-circuited)"
        )
    )
    explode_client_s = MagicMock()
    explode_client_s.messages = MagicMock()
    explode_client_s.messages.create = AsyncMock(
        side_effect=AssertionError(
            "strategic LLM was invoked in replay mode (should be short-circuited)"
        )
    )
    replay_agent = _build_llm_agent(
        library,
        seed=seed2,
        stance=stance2,
        transcript=replay_transcript,
        tactical_client=explode_client_t,
        strategic_client=explode_client_s,
        company_rng_seed=company_rng_seed,
        shock_seed=shock_seed,
        sim_id="roundtrip-sim",
        company_id="co",
    )
    replay_trace: list[TickResult] = []
    for _ in range(n_ticks):
        if not replay_agent.alive:
            break
        replay_trace.append(asyncio.run(replay_agent.step()))

    # Confirm replay never touched either LLM client.
    assert explode_client_t.messages.create.await_count == 0, (
        "tactical client was awaited during replay phase"
    )
    assert explode_client_s.messages.create.await_count == 0, (
        "strategic client was awaited during replay phase"
    )

    return record_trace, replay_trace, record_await_count


def test_replay_mode_consumes_existing_transcript(tmp_path, library):
    """Record a 100-tick run with mocked LLM tiers, replay against the same
    transcript, assert per-tick TickResult fields + decisions match value/byte
    equal. Verifies the LLM-tier replay short-circuit works end-to-end and
    the recording's mocks were actually exercised (await_count > 0)."""
    record_trace, replay_trace, record_await_count = _run_record_replay_roundtrip(
        library, tmp_path, n_ticks=100
    )

    # Mocks MUST have fired in the record phase (cadence guarantees ~3
    # tactical + ~1 strategic over 100 ticks, so >= 4 awaits).
    assert record_await_count > 0, (
        "record phase never invoked the mocked LLM clients — cadence "
        "math may be wrong, or the transcript path is broken"
    )

    assert len(record_trace) == len(replay_trace), (
        f"trace length drift: record={len(record_trace)} "
        f"replay={len(replay_trace)}"
    )
    for i, (rec, rep) in enumerate(zip(record_trace, replay_trace, strict=True)):
        _assert_tick_results_equal(rec, rep, tick_idx=i)


@pytest.mark.slow
def test_full_sim_5y_byte_identical_with_replay_transcript(tmp_path, library):
    """5-year (1825-tick) record/replay round-trip; every TickResult numeric
    field must be finite AND value-equal between phases. Slow — runs the full
    engine twice with mocked tactical+strategic LLM tiers active."""
    record_trace, replay_trace, _ = _run_record_replay_roundtrip(
        library,
        tmp_path,
        n_ticks=1825,
        rng_seed=2026,
        company_rng_seed=29,
        shock_seed=131,
    )

    assert len(record_trace) == len(replay_trace), (
        f"5y trace length drift: record={len(record_trace)} "
        f"replay={len(replay_trace)}"
    )
    for i, (rec, rep) in enumerate(zip(record_trace, replay_trace, strict=True)):
        _assert_finite_tick(rec, tick_idx=i)
        _assert_finite_tick(rep, tick_idx=i)
        _assert_tick_results_equal(rec, rep, tick_idx=i)


# ─────────────────────────────────────────────────────────────────────────────
# RNG / stance divergence
# ─────────────────────────────────────────────────────────────────────────────


def _build_with_rng(
    library,
    *,
    seed_rng_seed: int,
    stance_archetype: str,
    agent_rng_seed: int,
    shock_lambdas: dict[str, float] | None = None,
):
    """Same seed + stance + shock_lambdas, but caller-controlled `agent.rng`
    seed. Heuristic-only bundle (mocked tactical/strategic to None)."""
    rng = random.Random(seed_rng_seed)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    s, r, c = _refs_for(library, "physical")
    seed = seed.model_copy(update={
        "economics_model": "physical",
        "initial_supplier_types": s,
        "initial_revenue_streams": r,
        "initial_cost_centers": c,
    })
    stance = sample_stance(stance_archetype, rng=rng)
    bundle = _silent_bundle(seed, stance, library)
    shock_seed = rng.randint(0, 2**31 - 1)

    return CompanyAgentV2(
        seed=seed,
        stance=stance,
        library=library,
        sim_id="rng-divergence",
        company_id="co",
        rng=random.Random(agent_rng_seed),
        transcript=None,
        cost_tracker=CostTracker(ceiling_usd=10.0),
        shock_scheduler=ShockScheduler(
            rng_seed=shock_seed, lambdas=shock_lambdas or {}
        ),
        orchestrator=bundle,
    )


def test_different_rng_diverge(library):
    """Same seed/stance/shocks, DIFFERENT agent RNG seeds — traces must
    materially diverge. `agent.rng` is consumed by probabilistic attrition
    under `talent_war`, so we feed both agents a non-zero talent_war lambda
    to actually exercise the RNG path. Probability of an independent-stream
    byte-match across 100 ticks under live shocks is essentially zero."""
    # talent_war elevates attrition_mult > 1.0 each active tick, which feeds
    # `self.rng.random()` via the engine's attrition draw (unified_v2.py
    # line ~747). Without an rng-consuming code path, two engine runs are
    # mathematically identical regardless of agent.rng — the test would
    # accidentally assert nothing.
    shocks = {"talent_war": 0.5}
    a1 = _build_with_rng(
        library,
        seed_rng_seed=4242,
        stance_archetype="bootstrap",
        agent_rng_seed=7,
        shock_lambdas=shocks,
    )
    a2 = _build_with_rng(
        library,
        seed_rng_seed=4242,
        stance_archetype="bootstrap",
        agent_rng_seed=99,
        shock_lambdas=shocks,
    )
    t1 = _trace(a1, 100)
    t2 = _trace(a2, 100)
    # Defensive: divergence shouldn't be a degenerate "one died, the other
    # didn't" case — assert both reached a non-trivial slice of the run.
    assert len(t1) >= 30 and len(t2) >= 30, (
        f"degenerate: trace lengths {len(t1)}, {len(t2)} — at least one "
        "agent died too early to make divergence meaningful"
    )
    assert t1 != t2, (
        "Two agents with different agent.rng seeds produced byte-identical "
        "traces — RNG isn't actually feeding the engine."
    )


@pytest.mark.slow
def test_different_stance_diverge(library):
    """Same seed/RNG, 5 different stance archetypes, 365 ticks each. Final
    `(cash, total_spawned_nodes, employee_count)` tuples should NOT all
    collapse to the same value — assert ≥3 of the 5 are pairwise distinct."""
    archetypes = (
        "founder_operator",
        "venture_growth",
        "bootstrap",
        "consolidator",
        "turnaround",
    )
    finals: list[tuple[float, int, int]] = []
    for archetype in archetypes:
        agent = _build_with_rng(
            library,
            seed_rng_seed=8080,
            stance_archetype=archetype,
            agent_rng_seed=11,
        )
        for _ in range(365):
            if not agent.alive:
                break
            asyncio.run(agent.step())
        total_spawned = sum(agent.spawned_nodes.values())
        finals.append(
            (agent.cash, total_spawned, agent._compute_employee_count())
        )

    distinct = len(set(finals))
    assert distinct >= 3, (
        f"only {distinct}/5 stance archetypes produced distinct final state "
        f"tuples ({finals!r}) — stance is not materially shaping outcomes"
    )
