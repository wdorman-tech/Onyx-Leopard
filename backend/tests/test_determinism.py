"""Determinism invariants for v2 — same seed × stance × RNG × shocks =>
identical outcomes (V2 plan invariant 9).

Heuristic-only (no LLM). LLM determinism is covered by `replay.py` round-trips.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from src.simulation.library_loader import _reset_library_cache, get_library
from src.simulation.orchestrator import (
    HeuristicOrchestrator,
    OrchestratorBundle,
)
from src.simulation.replay import CostTracker
from src.simulation.seed import sample_seed_for_archetype
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import sample_stance
from src.simulation.unified_v2 import CompanyAgentV2


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
    from unittest.mock import AsyncMock, MagicMock

    library_dict = {key: node.model_dump() for key, node in library.nodes.items()}
    heuristic = HeuristicOrchestrator(seed=seed, stance=stance, library=library_dict)
    tactical = MagicMock(); tactical.tick = AsyncMock(return_value=None)
    strategic = MagicMock(); strategic.tick = AsyncMock(return_value=None)
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


def test_replay_mode_consumes_existing_transcript(tmp_path, library):
    """Sanity: a Transcript in replay mode (no entries) doesn't crash sim
    construction. (Full LLM replay is exercised in test_replay.py.)"""
    from src.simulation.replay import Transcript

    transcript_path = tmp_path / "empty.jsonl"
    transcript_path.write_text("")  # empty
    transcript = Transcript(path=transcript_path, mode="replay")

    rng = random.Random(42)
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

    agent = CompanyAgentV2(
        seed=seed, stance=stance, library=library,
        sim_id="replay-test", company_id="co",
        rng=random.Random(7),
        transcript=transcript,
        cost_tracker=CostTracker(ceiling_usd=1.0),
        shock_scheduler=ShockScheduler(rng_seed=7, lambdas={}),
        orchestrator=bundle,
    )
    # LLM tiers are stubbed to None; transcript is in replay mode but never
    # consulted. Just verify no crash.
    asyncio.run(agent.step())
    assert agent.tick == 1
