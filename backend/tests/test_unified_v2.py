"""Tests for `unified_v2.py` — the v2 simulation engine.

Heuristic-only runs (no real LLM calls). Tactical and strategic tiers are
mocked to always return `None` so tests stay deterministic and offline.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.library_loader import get_library
from src.simulation.orchestrator import (
    CeoDecision,
    CompanyState,
    HeuristicOrchestrator,
    OrchestratorBundle,
)
from src.simulation.replay import CostTracker
from src.simulation.seed import (
    CompanySeed,
    sample_seed_for_archetype,
)
from src.simulation.shocks import (
    ShockScheduler,
    make_market_crash,
)
from src.simulation.stance import (
    CeoStance,
    sample_stance,
)
from src.simulation.unified_v2 import (
    CompanyAgentV2,
    MultiCompanySimV2,
    TickResult,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def library():
    """Real production library — 122 nodes, validated."""
    from src.simulation.library_loader import _reset_library_cache
    _reset_library_cache()
    return get_library()


@pytest.fixture
def rng():
    return random.Random(42)


def _valid_refs_for(library, economics_model):
    """Pick first library node in each role-bucket that fits the economics."""
    suppliers, revenues, costs = [], [], []
    for key, node in sorted(library.nodes.items()):
        if economics_model not in node.applicable_economics:
            continue
        if node.category == "supplier" and len(suppliers) < 1:
            suppliers.append(key)
        elif node.category == "revenue" and len(revenues) < 1:
            revenues.append(key)
        elif node.category == "ops" and len(costs) < 1:
            costs.append(key)
    # Fallbacks
    if not suppliers:
        suppliers = ["primary_goods_supplier"]
    if not revenues:
        # Pick any revenue node
        for key, node in library.nodes.items():
            if node.category == "revenue":
                revenues.append(key); break
    if not costs:
        costs = ["bookkeeper"]
    return suppliers, revenues, costs


@pytest.fixture
def small_team_seed_and_stance(rng, library):
    seed = sample_seed_for_archetype("small_team", rng=rng)
    suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
    seed = seed.model_copy(update={
        "initial_supplier_types": suppliers,
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs,
    })
    stance = sample_stance("bootstrap", rng=rng)
    return seed, stance


def _make_silent_bundle(seed, stance, library):
    """Build an OrchestratorBundle whose LLM tiers always return None.

    Heuristic still runs normally — so tests exercise the heuristic decision
    application path without ever touching Anthropic SDK.
    """
    library_dict = {key: node.model_dump() for key, node in library.nodes.items()}
    heuristic = HeuristicOrchestrator(
        seed=seed, stance=stance, library=library_dict
    )
    tactical = MagicMock()
    tactical.tick = AsyncMock(return_value=None)
    strategic = MagicMock()
    strategic.tick = AsyncMock(return_value=None)
    return OrchestratorBundle(
        heuristic=heuristic,
        tactical=tactical,
        strategic=strategic,
    )


def _make_agent(seed, stance, library, *, rng=None, shock_scheduler=None):
    """Construct a CompanyAgentV2 with silent LLM tiers for testing."""
    rng = rng or random.Random(7)
    bundle = _make_silent_bundle(seed, stance, library)
    return CompanyAgentV2(
        seed=seed,
        stance=stance,
        library=library,
        sim_id="test-sim",
        company_id="test-co",
        rng=rng,
        transcript=None,
        cost_tracker=CostTracker(ceiling_usd=10.0),
        shock_scheduler=shock_scheduler,
        orchestrator=bundle,
    )


# ─── Initialization ────────────────────────────────────────────────────────


def test_agent_initializes_with_starting_cash(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    assert agent.cash == seed.starting_cash
    assert agent.tick == 0
    assert agent.alive is True


def test_agent_spawns_founder_at_t0(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    # An exec-category node must be spawned at t=0 (founder/founder_ceo/ceo)
    has_exec = any(
        library.nodes[k].category == "exec"
        for k, c in agent.spawned_nodes.items()
        if c > 0 and k in library.nodes
    )
    assert has_exec, f"No exec spawned: {list(agent.spawned_nodes.keys())}"


def test_agent_seeded_with_initial_refs(library):
    """Seeds with valid library refs should populate spawned_nodes."""
    rng = random.Random(7)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
    seed = seed.model_copy(update={
        "initial_supplier_types": suppliers,
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs,
    })
    stance = sample_stance("bootstrap", rng=rng)

    agent = _make_agent(seed, stance, library)
    for key in suppliers + revenues + costs:
        assert agent.spawned_nodes.get(key, 0) >= 1, f"missing {key}"


def test_agent_spawns_default_location_if_not_in_initial_refs(
    library, small_team_seed_and_stance
):
    """If no location-category node was seeded, agent picks one matching
    economics_model."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    has_location = any(
        library.nodes[k].category == "location"
        for k, count in agent.spawned_nodes.items()
        if count > 0 and k in library.nodes
    )
    assert has_location, "Agent must spawn at least one location node at t=0"


# ─── Tick loop ─────────────────────────────────────────────────────────────


def test_step_advances_tick_counter(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    asyncio.run(agent.step())
    assert agent.tick == 1


def test_step_returns_tick_result(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    result = asyncio.run(agent.step())
    assert isinstance(result, TickResult)
    assert result.tick == 1
    assert result.bankrupt is False


def test_step_records_history(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    for _ in range(5):
        asyncio.run(agent.step())
    assert len(agent._hist.cash) == 5
    assert len(agent._hist.revenue) == 5
    assert len(agent._hist.satisfaction) == 5


def test_30_tick_run_no_crash(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    for _ in range(30):
        asyncio.run(agent.step())
    assert agent.tick == 30


# ─── Decision application ─────────────────────────────────────────────────


def test_spawn_decision_increments_count_and_decreases_cash(
    library, small_team_seed_and_stance
):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    # Pick a node that's NOT already spawned and has no prerequisites
    candidate = None
    for key, node in sorted(library.nodes.items()):
        if (
            agent.spawned_nodes.get(key, 0) == 0
            and not node.prerequisites
            and node.category in ("ops", "supplier", "marketing")
        ):
            candidate = key
            break
    assert candidate is not None, "Library has no spawnable test candidate"

    cash_before = agent.cash
    hire_cost = library.nodes[candidate].hire_cost

    decision = CeoDecision(
        spawn_nodes=[candidate],
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning="test spawn",
        references_stance=["risk_tolerance"],
        tier="heuristic",
        tick=1,
    )
    agent._apply_decision(decision)

    assert agent.spawned_nodes[candidate] == 1
    assert agent.cash == pytest.approx(cash_before - hire_cost)


def test_retire_decision_decrements_count(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    # Spawn then retire a known node
    agent.spawned_nodes["bookkeeper"] = agent.spawned_nodes.get("bookkeeper", 0) + 1
    starting = agent.spawned_nodes["bookkeeper"]
    decision = CeoDecision(
        spawn_nodes=[],
        retire_nodes=["bookkeeper"],
        adjust_params={},
        open_locations=0,
        reasoning="test retire",
        references_stance=["cash_comfort"],
        tier="heuristic",
        tick=1,
    )
    agent._apply_decision(decision)
    assert agent.spawned_nodes.get("bookkeeper", 0) == starting - 1


def test_spawn_blocked_by_unmet_prerequisite(library, small_team_seed_and_stance):
    """Defensive: orchestrator should filter, but engine also re-checks."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    # Find a node with a prerequisite that's NOT in spawned_nodes
    candidate = None
    for key, node in library.nodes.items():
        if not node.prerequisites:
            continue
        if any(p not in agent.spawned_nodes for p in node.prerequisites):
            candidate = key
            break
    if candidate is None:
        pytest.skip("Library has no node with unmet prereqs in this state")

    cash_before = agent.cash
    decision = CeoDecision(
        spawn_nodes=[candidate],
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning="test prereq block",
        references_stance=["hiring_bias"],
        tier="heuristic",
        tick=1,
    )
    agent._apply_decision(decision)

    # Spawn was blocked, cash unchanged, count unchanged
    assert agent.spawned_nodes.get(candidate, 0) == 0
    assert agent.cash == cash_before


def test_spawn_blocked_by_hard_cap(library, small_team_seed_and_stance):
    """Spawning past hard_cap should be blocked silently."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    # Pick a no-prereq node and saturate it to hard_cap
    candidate = None
    for key, node in sorted(library.nodes.items()):
        if not node.prerequisites and node.category_caps.hard_cap <= 5:
            candidate = key
            break
    assert candidate is not None, "Library lacks small-cap testable node"

    hard_cap = library.nodes[candidate].hard_cap if hasattr(
        library.nodes[candidate], "hard_cap"
    ) else library.nodes[candidate].category_caps.hard_cap
    agent.spawned_nodes[candidate] = hard_cap

    cash_before = agent.cash
    decision = CeoDecision(
        spawn_nodes=[candidate],
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning="test hard_cap block",
        references_stance=["risk_tolerance"],
        tier="heuristic",
        tick=1,
    )
    agent._apply_decision(decision)
    # Count unchanged, cash unchanged
    assert agent.spawned_nodes[candidate] == hard_cap
    assert agent.cash == cash_before


# ─── Shocks ────────────────────────────────────────────────────────────────


def test_shock_scheduler_arrivals_recorded(library, small_team_seed_and_stance):
    """Inject a shock with a guaranteed arrival; verify it shows up active."""
    seed, stance = small_team_seed_and_stance

    # Build a scheduler that's guaranteed to arrive market_crash on every tick
    rng = random.Random(13)
    scheduler = ShockScheduler(
        rng_seed=13,
        lambdas={"market_crash": 1.0},  # very high arrival rate
    )

    agent = _make_agent(seed, stance, library, rng=rng, shock_scheduler=scheduler)

    # After several ticks, there should be at least one active shock
    arrivals_seen = 0
    for _ in range(10):
        result = asyncio.run(agent.step())
        arrivals_seen += len(result.arriving_shocks)

    assert arrivals_seen > 0, "High-lambda scheduler should produce arrivals"


def test_shock_decreases_revenue(library, small_team_seed_and_stance):
    """A market_crash shock should depress demand multiplier, reducing revenue."""
    seed, stance = small_team_seed_and_stance

    # Run baseline (no shocks) for 5 ticks
    agent_baseline = _make_agent(seed, stance, library)
    for _ in range(5):
        asyncio.run(agent_baseline.step())
    baseline_revenue = agent_baseline.daily_revenue

    # Run with a forced market crash on tick 1
    agent_shocked = _make_agent(seed, stance, library)
    crash = make_market_crash(rng=random.Random(99), severity="severe")
    agent_shocked.active_shocks.append(crash)
    crash.tick_started = 0  # active immediately

    for _ in range(5):
        asyncio.run(agent_shocked.step())

    # Shocked revenue should be lower (or both zero if no demand at all)
    assert agent_shocked.daily_revenue <= baseline_revenue, (
        f"shocked={agent_shocked.daily_revenue}, baseline={baseline_revenue}"
    )


# ─── Bridge integration ──────────────────────────────────────────────────


def test_bridge_aggregate_changes_with_spawns(library, small_team_seed_and_stance):
    """Spawning a node with marketing modifier_keys should populate marketing
    bucket in the next tick."""
    from src.simulation.bridge import derive_bridge_aggregate

    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    # Find a node with marketing-bucket modifier_keys
    target = None
    for key, node in sorted(library.nodes.items()):
        if not node.prerequisites and node.modifier_keys:
            mks = list(node.modifier_keys.keys())
            # Check if any modifier_key is in the marketing bucket
            from src.simulation.bridge import bucket_modifiers
            buckets = bucket_modifiers({k: 1.0 for k in mks})
            if buckets.marketing:
                target = key
                break

    if target is None:
        pytest.skip("No no-prereq node with marketing modifier_keys")

    # Aggregate before
    agg_before = derive_bridge_aggregate(library, agent.spawned_nodes)
    # Spawn
    agent.spawned_nodes[target] = agent.spawned_nodes.get(target, 0) + 1
    agg_after = derive_bridge_aggregate(library, agent.spawned_nodes)

    assert sum(agg_after.marketing.values()) > sum(agg_before.marketing.values()), (
        "Marketing bucket should grow after spawning marketing-modifier node"
    )


# ─── Graph snapshot ──────────────────────────────────────────────────────


def test_graph_snapshot_returns_valid_dict(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    snap = agent.to_graph_snapshot()
    assert "nodes" in snap
    assert "edges" in snap
    assert isinstance(snap["nodes"], list)
    assert isinstance(snap["edges"], list)
    # At least one node (founder + initial refs)
    assert len(snap["nodes"]) >= 1
    # Each node has the expected keys
    for n in snap["nodes"]:
        assert "id" in n and "label" in n and "category" in n


def test_graph_snapshot_edges_only_between_spawned_nodes(
    library, small_team_seed_and_stance
):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    snap = agent.to_graph_snapshot()
    spawned_keys = {k for k, c in agent.spawned_nodes.items() if c > 0}
    for e in snap["edges"]:
        assert e["source"] in spawned_keys
        assert e["target"] in spawned_keys


# ─── Multi-company sim ──────────────────────────────────────────────────


def test_multi_company_sim_runs(library):
    """Three companies, 50 ticks, no crash."""
    rng = random.Random(11)
    companies = []
    for i in range(3):
        seed = sample_seed_for_archetype("small_team", rng=rng)
        suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
        seed = seed.model_copy(update={
            "initial_supplier_types": suppliers,
            "initial_revenue_streams": revenues,
            "initial_cost_centers": costs,
        })
        stance = sample_stance("bootstrap", rng=rng)
        bundle = _make_silent_bundle(seed, stance, library)
        c = CompanyAgentV2(
            seed=seed,
            stance=stance,
            library=library,
            sim_id="multi-test",
            company_id=f"co-{i}",
            rng=random.Random(100 + i),
            transcript=None,
            cost_tracker=CostTracker(ceiling_usd=10.0),
            shock_scheduler=None,
            orchestrator=bundle,
        )
        companies.append(c)

    sim = MultiCompanySimV2(
        sim_id="multi-test",
        companies=companies,
        max_ticks=50,
        tam_initial=1_000_000.0,
    )

    results = asyncio.run(sim.run())
    assert len(results) == 50
    assert all("tick" in r and "shares" in r for r in results)


# ─── Smoke: archetype seeds ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "archetype",
    ["solo_founder", "small_team", "venture_funded", "enterprise"],
)
def test_archetype_seed_runs_60_ticks(library, archetype):
    """Each archetype-sampled seed runs for 60 ticks without crashing."""
    rng = random.Random(hash(archetype) & 0xFFFF_FFFF)
    seed = sample_seed_for_archetype(archetype, rng=rng)
    suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
    seed = seed.model_copy(update={
        "initial_supplier_types": suppliers,
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs,
    })
    stance = sample_stance(
        {
            "solo_founder": "founder_operator",
            "small_team": "bootstrap",
            "venture_funded": "venture_growth",
            "enterprise": "consolidator",
        }[archetype],
        rng=rng,
    )

    agent = _make_agent(seed, stance, library, rng=rng)
    for _ in range(60):
        asyncio.run(agent.step())
        if not agent.alive:
            break
    # Should at least have advanced ticks
    assert agent.tick > 0
