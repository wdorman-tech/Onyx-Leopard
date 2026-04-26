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
    Shock,
    ShockScheduler,
    make_market_crash,
    make_key_employee_departure,
    make_new_competitor_entry,
    make_regulatory_change,
    make_supply_chain_disruption,
    make_talent_war,
    make_viral_growth_event,
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


# ─── adjust_params wiring (Phase 1.1b — P-ENG-1, Decision 2A) ──────────────


def _force_decision(agent, **kwargs) -> CeoDecision:
    """Build a CeoDecision with sane defaults, override via kwargs."""
    base = dict(
        spawn_nodes=[],
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning="test",
        references_stance=["risk_tolerance"],
        tier="strategic",
        tick=agent.tick,
    )
    base.update(kwargs)
    return CeoDecision(**base)


def test_current_price_initialized_from_seed_starting_price(
    library, small_team_seed_and_stance
):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    assert agent.current_price == seed.starting_price


def test_marketing_multiplier_initialized_to_one(
    library, small_team_seed_and_stance
):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    assert agent.marketing_multiplier == 1.0


def test_ceo_price_change_affects_revenue(library, small_team_seed_and_stance):
    """Price set via adjust_params must alter the next tick's daily_revenue."""
    seed, stance = small_team_seed_and_stance
    agent_a = _make_agent(seed, stance, library, rng=random.Random(7))
    agent_b = _make_agent(seed, stance, library, rng=random.Random(7))

    # Agent A keeps default price; agent B raises price 2x.
    decision_b = _force_decision(agent_b, adjust_params={"price": seed.starting_price * 2.0})
    agent_b._apply_decision(decision_b)
    assert agent_b.current_price == pytest.approx(seed.starting_price * 2.0)

    # Run one tick on each. Higher price → higher daily_revenue (same demand).
    asyncio.run(agent_a.step())
    asyncio.run(agent_b.step())
    if agent_a.daily_revenue > 0:
        assert agent_b.daily_revenue > agent_a.daily_revenue
    else:
        # Solo-mode demand can be zero on tick 1 — accept equal-zero, but if
        # one is non-zero the other should be too at the same scale.
        assert agent_b.daily_revenue >= agent_a.daily_revenue


def test_ceo_marketing_change_affects_marketing_multiplier(
    library, small_team_seed_and_stance
):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    decision = _force_decision(agent, adjust_params={"marketing_intensity": 1.5})
    agent._apply_decision(decision)
    assert agent.marketing_multiplier == pytest.approx(1.5)


def test_ceo_marketing_intensity_clamped_to_max(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    decision = _force_decision(agent, adjust_params={"marketing_intensity": 99.0})
    agent._apply_decision(decision)
    # MARKETING_INTENSITY_MAX = 2.0
    assert agent.marketing_multiplier == 2.0


def test_ceo_price_out_of_bounds_is_rejected(library, small_team_seed_and_stance):
    """Price below 0.1x or above 10x starting_price is logged + ignored."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    original = agent.current_price

    # Negative price — rejected
    agent._apply_decision(_force_decision(agent, adjust_params={"price": -1.0}))
    assert agent.current_price == original

    # Way too high — rejected
    agent._apply_decision(
        _force_decision(agent, adjust_params={"price": seed.starting_price * 100.0})
    )
    assert agent.current_price == original


def test_ceo_raise_amount_increases_cash_only_for_venture_archetype(library):
    """raise_amount only applies when stance.archetype permits external capital."""
    rng = random.Random(7)
    seed = sample_seed_for_archetype("venture_funded", rng=rng)
    suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
    seed = seed.model_copy(update={
        "initial_supplier_types": suppliers,
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs,
    })

    # Bootstrap stance — raise blocked
    stance_boot = sample_stance("bootstrap", rng=rng)
    agent_boot = _make_agent(seed, stance_boot, library)
    cash_before = agent_boot.cash
    agent_boot._apply_decision(
        _force_decision(agent_boot, adjust_params={"raise_amount": 1_000_000.0})
    )
    assert agent_boot.cash == cash_before, (
        "Bootstrap stance must NOT receive raise_amount cash"
    )

    # Venture growth stance — raise allowed
    stance_vc = sample_stance("venture_growth", rng=rng)
    agent_vc = _make_agent(seed, stance_vc, library)
    cash_before_vc = agent_vc.cash
    agent_vc._apply_decision(
        _force_decision(agent_vc, adjust_params={"raise_amount": 1_000_000.0})
    )
    assert agent_vc.cash == pytest.approx(cash_before_vc + 1_000_000.0)


def test_ceo_negative_raise_amount_rejected(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    # Force a venture-permitted stance
    rng = random.Random(7)
    stance_vc = sample_stance("venture_growth", rng=rng)
    agent = _make_agent(seed, stance_vc, library)
    cash_before = agent.cash
    agent._apply_decision(_force_decision(agent, adjust_params={"raise_amount": -100.0}))
    assert agent.cash == cash_before


def test_financing_shock_haircuts_raise_amount(library, small_team_seed_and_stance):
    """When financing_availability_mult < 1.0, the actual cash inflow from a
    raise is scaled down — investors pull back during a credit crunch."""
    seed, stance = small_team_seed_and_stance
    rng = random.Random(7)
    stance_vc = sample_stance("venture_growth", rng=rng)
    agent = _make_agent(seed, stance_vc, library)
    # Simulate an active market_crash by populating _last_env directly.
    agent._last_env = {"financing_availability_mult": 0.4}
    cash_before = agent.cash
    agent._apply_decision(
        _force_decision(agent, adjust_params={"raise_amount": 1_000_000.0})
    )
    # Should receive 40% of the requested raise.
    assert agent.cash == pytest.approx(cash_before + 400_000.0)


def test_seed_starting_price_remains_immutable_after_decision(
    library, small_team_seed_and_stance
):
    """Even after CEO mutates current_price, the seed's starting_price is untouched."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    original_seed_price = seed.starting_price
    agent._apply_decision(
        _force_decision(agent, adjust_params={"price": original_seed_price * 0.5})
    )
    assert agent.seed.starting_price == original_seed_price
    assert agent.current_price == pytest.approx(original_seed_price * 0.5)
    assert agent.current_price != agent.seed.starting_price


def test_replenish_supplier_charges_supplier_burn(library, small_team_seed_and_stance):
    """The replenish_supplier signal subtracts a week of supplier fixed costs."""
    from src.simulation.unified_v2 import REPLENISH_INVENTORY_DAYS

    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    # Compute expected supplier burn
    supplier_burn = sum(
        library.nodes[k].daily_fixed_costs * c
        for k, c in agent.spawned_nodes.items()
        if k in library.nodes and library.nodes[k].category == "supplier"
    )
    if supplier_burn == 0:
        pytest.skip("Test seed has no supplier nodes")
    cash_before = agent.cash
    agent._apply_decision(
        _force_decision(agent, adjust_params={"replenish_supplier": 1.0})
    )
    expected_debit = supplier_burn * REPLENISH_INVENTORY_DAYS
    assert agent.cash == pytest.approx(cash_before - expected_debit)


def test_unknown_adjust_params_key_logged_not_crashed(
    library, small_team_seed_and_stance
):
    """Unknown keys must not crash; engine treats them as ignored."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    cash_before = agent.cash
    price_before = agent.current_price
    agent._apply_decision(
        _force_decision(agent, adjust_params={"made_up_key": 42.0})
    )
    assert agent.cash == cash_before
    assert agent.current_price == price_before


# ─── Multi-tick insolvency (Phase 1.2 — P-ENG-2) ───────────────────────────


def test_company_survives_one_tick_negative_cash_then_recovers(
    library, small_team_seed_and_stance
):
    """Single-tick negative cash must not kill the company anymore."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)
    # Force a single-tick cash crater.
    agent.cash = -1_000_000.0
    asyncio.run(agent.step())
    assert agent.alive is True, "Single negative-cash tick must not kill"
    assert agent.consecutive_insolvent >= 1


def test_company_dies_after_threshold_consecutive_insolvent_ticks(
    library, small_team_seed_and_stance
):
    """After INSOLVENT_TICKS_TO_DEATH consecutive negative-cash ticks, alive=False."""
    from src.simulation.unified_v2 import INSOLVENT_TICKS_TO_DEATH

    seed, stance = small_team_seed_and_stance
    # Use a tiny starting_cash so insolvency is reached fast.
    seed = seed.model_copy(update={"starting_cash": 1.0})
    agent = _make_agent(seed, stance, library)

    # Run enough ticks with deep cash crater to trigger death.
    for _ in range(INSOLVENT_TICKS_TO_DEATH + 5):
        agent.cash = -1_000_000.0  # force cash negative each tick
        result = asyncio.run(agent.step())
        if not agent.alive:
            break
    assert agent.alive is False
    assert agent.consecutive_insolvent >= INSOLVENT_TICKS_TO_DEATH


def test_consecutive_insolvent_resets_when_cash_recovers(
    library, small_team_seed_and_stance
):
    """If cash recovers above zero, the insolvency counter resets to zero."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library)

    # Tick 1: cash so deeply negative no single-tick revenue can rescue it.
    agent.cash = -1_000_000_000.0
    asyncio.run(agent.step())
    assert agent.cash < 0, "test setup: cash must stay negative after tick"
    assert agent.consecutive_insolvent >= 1

    # Tick 2: force cash way positive — must outweigh anything heuristic
    # spawn or daily burn could subtract during this single tick.
    agent.cash = 1_000_000_000.0
    asyncio.run(agent.step())
    assert agent.cash > 0, "test setup: cash must stay positive after tick"
    assert agent.consecutive_insolvent == 0


# ─── Shock impact wiring (Phase 1.3 — P-ENG-5) ─────────────────────────────


def _attach_shock(agent, shock: Shock, tick_started: int = 0) -> None:
    """Force a shock to be active on `agent` from `tick_started`."""
    shock.tick_started = tick_started
    agent.active_shocks.append(shock)


def test_market_crash_drops_revenue_same_tick(library, small_team_seed_and_stance):
    seed, stance = small_team_seed_and_stance
    agent_baseline = _make_agent(seed, stance, library, rng=random.Random(13))
    agent_shocked = _make_agent(seed, stance, library, rng=random.Random(13))

    asyncio.run(agent_baseline.step())
    crash = make_market_crash(rng=random.Random(99), severity="severe")
    _attach_shock(agent_shocked, crash, tick_started=0)
    asyncio.run(agent_shocked.step())

    assert agent_shocked.daily_revenue <= agent_baseline.daily_revenue


def test_talent_war_increases_hire_cost_for_next_spawn(
    library, small_team_seed_and_stance
):
    """When talent_war is active, _spawn_node multiplies hire_cost by hire_cost_mult."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library, rng=random.Random(7))
    # Tick once so _last_env is populated.
    asyncio.run(agent.step())

    talent = make_talent_war(rng=random.Random(101), severity="severe")
    _attach_shock(agent, talent, tick_started=agent.tick)
    asyncio.run(agent.step())  # populates _last_env with talent_war's hire_cost_mult

    # Find a no-prereq node we can spawn
    candidate = None
    for key, node in sorted(library.nodes.items()):
        if (
            agent.spawned_nodes.get(key, 0) == 0
            and not node.prerequisites
            and node.hire_cost > 0
            and node.category in ("ops", "supplier", "marketing")
        ):
            candidate = key
            break
    if candidate is None:
        pytest.skip("Library has no spawnable test candidate for hire_cost test")

    hire_cost_mult = float(agent._last_env.get("hire_cost_mult", 1.0))
    assert hire_cost_mult > 1.0, "talent_war should push hire_cost_mult > 1"

    cash_before = agent.cash
    expected_charge = library.nodes[candidate].hire_cost * hire_cost_mult
    agent._spawn_node(candidate)
    assert agent.spawned_nodes[candidate] == 1
    assert agent.cash == pytest.approx(cash_before - expected_charge, rel=1e-6)


def test_regulatory_change_adds_compliance_cost(library, small_team_seed_and_stance):
    """compliance_cost_add should inflate daily_costs each tick the shock is active."""
    seed, stance = small_team_seed_and_stance
    rng_a = random.Random(13)
    rng_b = random.Random(13)
    agent_baseline = _make_agent(seed, stance, library, rng=rng_a)
    agent_shocked = _make_agent(seed, stance, library, rng=rng_b)

    reg = make_regulatory_change(rng=random.Random(77), severity="severe")
    _attach_shock(agent_shocked, reg, tick_started=0)
    asyncio.run(agent_baseline.step())
    asyncio.run(agent_shocked.step())

    assert agent_shocked.daily_costs > agent_baseline.daily_costs


def test_key_employee_departure_charges_replacement_cost_once(
    library, small_team_seed_and_stance
):
    """replacement_cost_add fires exactly once per shock instance."""
    seed, stance = small_team_seed_and_stance
    agent = _make_agent(seed, stance, library, rng=random.Random(7))

    # Build a shock with a known replacement_cost_add.
    shock = make_key_employee_departure(rng=random.Random(33), severity="moderate")
    expected_charge = float(shock.impact.get("replacement_cost_add", 0.0))
    assert expected_charge > 0
    _attach_shock(agent, shock, tick_started=0)

    cash_before = agent.cash
    asyncio.run(agent.step())
    cash_after_tick1 = agent.cash
    # We can't isolate replacement charge from one tick of normal P&L, but
    # the one-time bookkeeping must mark this shock as charged.
    assert id(shock) in agent._charged_one_time_shocks

    # Tick again — the same shock instance must NOT charge replacement again.
    # Compare cash deltas: the second-tick delta should NOT include a fresh
    # replacement charge.
    asyncio.run(agent.step())
    delta_tick2 = cash_after_tick1 - agent.cash
    delta_tick1 = cash_before - cash_after_tick1
    # Delta tick1 ≥ delta tick2 + replacement_charge - tolerance.
    # Use a loose check: tick1 should be measurably larger than tick2.
    assert delta_tick1 > delta_tick2


def test_viral_growth_event_lowers_acquisition_cost(
    library, small_team_seed_and_stance
):
    """acquisition_cost_mult < 1 should boost effective marketing pressure → revenue."""
    seed, stance = small_team_seed_and_stance
    agent_baseline = _make_agent(seed, stance, library, rng=random.Random(13))
    agent_shocked = _make_agent(seed, stance, library, rng=random.Random(13))

    viral = make_viral_growth_event(rng=random.Random(55), severity="severe")
    _attach_shock(agent_shocked, viral, tick_started=0)
    asyncio.run(agent_baseline.step())
    asyncio.run(agent_shocked.step())

    assert agent_shocked.daily_revenue >= agent_baseline.daily_revenue


def test_supply_chain_disruption_lowers_throughput(
    library, small_team_seed_and_stance
):
    """lead_time_mult > 1 + inventory_throughput_mult < 1 reduce served customers."""
    seed, stance = small_team_seed_and_stance
    agent_baseline = _make_agent(seed, stance, library, rng=random.Random(13))
    agent_shocked = _make_agent(seed, stance, library, rng=random.Random(13))

    supply = make_supply_chain_disruption(rng=random.Random(88), severity="severe")
    _attach_shock(agent_shocked, supply, tick_started=0)
    asyncio.run(agent_baseline.step())
    asyncio.run(agent_shocked.step())
    # Either revenue or capacity_utilization should drop
    assert (
        agent_shocked.daily_revenue <= agent_baseline.daily_revenue
        or agent_shocked.capacity_utilization <= agent_baseline.capacity_utilization
    )


def test_competitor_entry_pulls_share_in_multi_company(library):
    """new_competitor_entry shock on company A should reduce A's share."""
    rng = random.Random(11)
    companies = []
    for i in range(2):
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

    # Attach competitor entry to company 0.
    entry = make_new_competitor_entry(rng=random.Random(44), severity="severe")
    _attach_shock(companies[0], entry, tick_started=0)

    sim = MultiCompanySimV2(
        sim_id="multi-test",
        companies=companies,
        max_ticks=2,
        tam_initial=1_000_000.0,
    )
    results = asyncio.run(sim.run())
    last = results[-1]
    shares = last["shares"]
    # Company 0 should have a lower share than company 1 due to the shock.
    assert shares[0] < shares[1]


# ─── is_complete + all_dead semantics (Phase 1.6 — P-ENG-4) ───────────────


def test_sim_emits_all_dead_event_when_companies_die(library):
    """When all companies die before max_ticks, the sim emits one all_dead event."""
    rng = random.Random(11)
    companies = []
    for i in range(2):
        seed = sample_seed_for_archetype("small_team", rng=rng)
        suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
        seed = seed.model_copy(update={
            "initial_supplier_types": suppliers,
            "initial_revenue_streams": revenues,
            "initial_cost_centers": costs,
            "starting_cash": 1.0,  # ultra-thin runway → fast death
        })
        stance = sample_stance("bootstrap", rng=rng)
        bundle = _make_silent_bundle(seed, stance, library)
        c = CompanyAgentV2(
            seed=seed,
            stance=stance,
            library=library,
            sim_id="death-test",
            company_id=f"co-{i}",
            rng=random.Random(100 + i),
            transcript=None,
            cost_tracker=CostTracker(ceiling_usd=10.0),
            shock_scheduler=None,
            orchestrator=bundle,
        )
        # Force imminent death — set companies AT the insolvency cap with
        # cash so deeply negative the tick math cannot rescue them.
        from src.simulation.unified_v2 import INSOLVENT_TICKS_TO_DEATH
        c.consecutive_insolvent = INSOLVENT_TICKS_TO_DEATH
        c.cash = -1_000_000_000.0
        companies.append(c)

    sim = MultiCompanySimV2(
        sim_id="death-test",
        companies=companies,
        max_ticks=10,
        tam_initial=1_000_000.0,
    )
    results = asyncio.run(sim.run())
    # Exactly one tick payload should carry all_dead=True.
    all_dead_payloads = [r for r in results if r.get("all_dead")]
    assert len(all_dead_payloads) == 1, (
        f"expected exactly one all_dead event, got {len(all_dead_payloads)}"
    )
    # is_complete is now True.
    assert sim.is_complete is True


def test_sim_continues_while_one_company_alive(library):
    """is_complete must NOT trigger while at least one company is alive."""
    rng = random.Random(11)
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
        sim_id="solo",
        company_id="co-0",
        rng=random.Random(101),
        transcript=None,
        cost_tracker=CostTracker(ceiling_usd=10.0),
        shock_scheduler=None,
        orchestrator=bundle,
    )
    sim = MultiCompanySimV2(
        sim_id="solo",
        companies=[c],
        max_ticks=20,
        tam_initial=1_000_000.0,
    )
    # After each step, is_complete should be False until tick=max or company dies.
    for _ in range(5):
        await_result = asyncio.run(sim.step())
        # First 5 ticks: company should still be alive (default starting_cash)
        assert sim.alive_companies, "Default-cash company should not die in 5 ticks"
        assert sim.is_complete is False
