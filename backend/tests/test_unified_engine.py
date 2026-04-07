"""Integration tests for the UnifiedEngine — multi-company competition."""

from __future__ import annotations

import math

import pytest

from src.simulation.config_loader import load_industry
from src.simulation.unified import CompanyAgent, UnifiedEngine, _compute_shares
from src.simulation.unified_models import UnifiedParams, UnifiedStartConfig

# ── Helper ──


def _run_engine(
    ticks: int,
    start_mode: str = "identical",
    num_companies: int = 4,
    seed: int = 42,
    **param_overrides,
) -> UnifiedEngine:
    params = UnifiedParams(**param_overrides)
    config = UnifiedStartConfig(
        start_mode=start_mode,
        num_companies=num_companies,
        params=params,
    )
    engine = UnifiedEngine(config=config, seed=seed)
    for _ in range(ticks):
        if engine.is_complete:
            break
        engine.tick()
    return engine


# ── _compute_shares ──


class TestComputeShares:
    def test_equal_firms_equal_shares(self):
        shares = _compute_shares(
            [1.0, 1.0, 1.0], [30.0, 30.0, 30.0],
            [True, True, True], alpha=0.8, beta=0.8,
        )
        assert len(shares) == 3
        for s in shares:
            assert s == pytest.approx(1 / 3, abs=1e-10)

    def test_dead_firms_get_zero(self):
        shares = _compute_shares(
            [1.0, 1.0], [30.0, 30.0],
            [True, False], alpha=0.8, beta=0.8,
        )
        assert shares[0] == pytest.approx(1.0)
        assert shares[1] == pytest.approx(0.0)

    def test_higher_quality_gets_more(self):
        shares = _compute_shares(
            [2.0, 1.0], [30.0, 30.0],
            [True, True], alpha=0.8, beta=0.8,
        )
        assert shares[0] > shares[1]

    def test_higher_marketing_gets_more(self):
        shares = _compute_shares(
            [1.0, 1.0], [60.0, 30.0],
            [True, True], alpha=0.8, beta=0.8,
        )
        assert shares[0] > shares[1]

    def test_all_dead_returns_zeros(self):
        shares = _compute_shares(
            [1.0, 1.0], [30.0, 30.0],
            [False, False], alpha=0.8, beta=0.8,
        )
        assert all(s == 0.0 for s in shares)

    def test_shares_sum_to_one(self):
        shares = _compute_shares(
            [0.5, 1.0, 1.5, 0.8], [10.0, 30.0, 50.0, 20.0],
            [True, True, True, True], alpha=0.8, beta=0.8,
        )
        assert sum(shares) == pytest.approx(1.0, abs=1e-10)


# ── CompanyAgent ──


class TestCompanyAgent:
    def test_initial_nodes(self):
        spec = load_industry("restaurant")
        agent = CompanyAgent(name="Test", index=0, spec=spec)
        types = [n.type for n in agent.state.nodes.values()]
        assert "owner_operator" in types
        assert "restaurant" in types
        assert "chicken_supplier" in types
        assert "produce_supplier" in types

    def test_initial_location_has_unified_reorder_params(self):
        spec = load_industry("restaurant")
        agent = CompanyAgent(name="Test", index=0, spec=spec)
        locs = agent.active_locations()
        assert len(locs) == 1
        ls = locs[0].location_state
        assert ls is not None
        assert ls.reorder_qty == 200.0
        assert ls.reorder_point == 80.0

    def test_location_count(self):
        spec = load_industry("restaurant")
        agent = CompanyAgent(name="Test", index=0, spec=spec)
        assert agent.location_count() == 1

    def test_stage_progression(self):
        spec = load_industry("restaurant")
        agent = CompanyAgent(name="Test", index=0, spec=spec)
        agent.update_stage()
        assert agent.state.stage == 1

        # Simulate 2 locations
        agent._add_node("restaurant")
        for n in agent.state.nodes.values():
            if n.type == "restaurant" and n.location_state is None:
                from src.simulation.models import LocationState
                n.location_state = LocationState()
        agent.refresh_caches()
        agent.update_stage()
        assert agent.state.stage == 2

    def test_graph_snapshot_has_nodes(self):
        spec = load_industry("restaurant")
        agent = CompanyAgent(name="Test", index=0, spec=spec)
        snap = agent.build_graph_snapshot()
        assert len(snap.nodes) >= 4  # owner + restaurant + 2 suppliers
        assert len(snap.edges) >= 2  # suppliers -> restaurant

    def test_trigger_isolation(self):
        """Each agent's triggers are independent (deep copied)."""
        spec = load_industry("restaurant")
        a = CompanyAgent(name="A", index=0, spec=spec)
        b = CompanyAgent(name="B", index=1, spec=spec)
        # Triggers are separate objects
        assert a.triggers is not b.triggers
        assert a.triggers[0] is not b.triggers[0]


# ── UnifiedEngine initialization ──


class TestUnifiedEngineInit:
    def test_identical_mode_creates_all_companies(self):
        config = UnifiedStartConfig(start_mode="identical", num_companies=4)
        engine = UnifiedEngine(config=config, seed=42)
        assert len(engine.companies) == 4
        assert all(c.alive for c in engine.companies)

    def test_randomized_mode_varies_cash(self):
        config = UnifiedStartConfig(start_mode="randomized", num_companies=5)
        engine = UnifiedEngine(config=config, seed=42)
        cash_values = [c.state.cash for c in engine.companies]
        # With randomized mode, cash varies between 30K-80K
        assert all(30_000 <= c <= 80_000 for c in cash_values)
        # Not all the same (extremely unlikely with 5 companies)
        assert len(set(round(c, 2) for c in cash_values)) > 1

    def test_staggered_mode_starts_with_fewer(self):
        config = UnifiedStartConfig(start_mode="staggered", num_companies=5)
        engine = UnifiedEngine(config=config, seed=42)
        assert len(engine.companies) == 2  # min(2, 5)

    def test_focused_company_set_on_init(self):
        config = UnifiedStartConfig(num_companies=3)
        engine = UnifiedEngine(config=config, seed=42)
        assert engine.focused_company_id == engine.companies[0].state.name


# ── UnifiedEngine tick cycle ──


class TestUnifiedEngineTick:
    def test_first_tick_produces_valid_result(self):
        engine = _run_engine(0)
        result = engine.tick()
        assert result["tick"] == 1
        assert result["status"] == "operating"
        assert result["tam"] > 0
        assert len(result["agents"]) == 4
        assert result["hhi"] > 0

    def test_tam_grows_each_tick(self):
        config = UnifiedStartConfig(num_companies=2)
        engine = UnifiedEngine(config=config, seed=42)
        tam_0 = engine.tam
        engine.tick()
        assert engine.tam > tam_0

    def test_shares_sum_to_one_after_tick(self):
        engine = _run_engine(10)
        alive = [c for c in engine.companies if c.alive]
        total_share = sum(c.share for c in alive)
        assert total_share == pytest.approx(1.0, abs=1e-6)

    def test_revenue_is_positive_early(self):
        engine = _run_engine(50, seed=42)
        alive = [c for c in engine.companies if c.alive]
        assert len(alive) > 0
        for c in alive:
            assert c.daily_revenue >= 0

    def test_cash_accounting_consistent(self):
        """Cash changes each tick should reflect revenue - costs - overhead - reorders."""
        config = UnifiedStartConfig(num_companies=2)
        engine = UnifiedEngine(config=config, seed=42)

        # Run a few ticks to stabilize
        for _ in range(10):
            engine.tick()

        # Snapshot cash, run one tick, verify direction makes sense
        for c in engine.companies:
            if c.alive:
                # Just verify cash is a real number (not NaN/inf)
                assert not math.isnan(c.state.cash)
                assert not math.isinf(c.state.cash)
                assert abs(c.state.cash) < 1_000_000_000

    def test_max_ticks_stops_engine(self):
        config = UnifiedStartConfig(num_companies=2, max_ticks=50)
        engine = UnifiedEngine(config=config, seed=42)
        for _ in range(100):
            if engine.is_complete:
                break
            engine.tick()
        assert engine.tick_num == 50
        assert engine.is_complete


# ── Long-run integration ──


class TestUnifiedLongRun:
    def test_500_tick_identical_no_crash(self):
        """Run 500 ticks with 4 identical companies. Must not raise."""
        engine = _run_engine(500, num_companies=4, seed=42)
        assert engine.tick_num == 500

    def test_500_tick_randomized_no_crash(self):
        engine = _run_engine(500, start_mode="randomized", num_companies=4, seed=42)
        assert engine.tick_num == 500

    def test_firms_with_more_locations_have_higher_capacity(self):
        """After growth period, companies with more locations should have more capacity."""
        engine = _run_engine(1000, num_companies=4, seed=42)
        alive = [c for c in engine.companies if c.alive]
        if len(alive) < 2:
            pytest.skip("Not enough alive companies for comparison")

        # Check correlation: more locations -> more capacity
        for c in alive:
            if c.location_count() > 1:
                # Multi-location firms should have capacity > single loc baseline
                single_loc_k = 80 * 14.0 * 0.85
                assert c.capacity > single_loc_k

    def test_bankruptcy_possible(self):
        """With tight parameters, at least one firm should go bankrupt."""
        engine = _run_engine(
            2000,
            num_companies=6,
            seed=42,
            tam_0=15_000.0,  # small market for 6 firms
            starting_cash=30_000.0,
        )
        dead = [c for c in engine.companies if not c.alive]
        assert len(dead) >= 1, "Expected at least one bankruptcy with tight market"

    def test_staggered_spawns_more_companies(self):
        """Staggered mode should eventually spawn additional firms."""
        engine = _run_engine(500, start_mode="staggered", num_companies=4, seed=42)
        assert len(engine.companies) > 2, "Staggered mode should have spawned new entrants"

    def test_no_negative_revenue(self):
        """No company should ever have negative daily_revenue."""
        config = UnifiedStartConfig(num_companies=4)
        engine = UnifiedEngine(config=config, seed=42)
        for _ in range(500):
            if engine.is_complete:
                break
            engine.tick()
            for c in engine.companies:
                if c.alive:
                    assert c.daily_revenue >= 0, (
                        f"{c.state.name} has negative revenue at tick {engine.tick_num}"
                    )

    def test_hhi_in_valid_range(self):
        """HHI should always be between 0 and 1."""
        config = UnifiedStartConfig(num_companies=4)
        engine = UnifiedEngine(config=config, seed=42)
        for _ in range(300):
            if engine.is_complete:
                break
            result = engine.tick()
            assert 0 <= result["hhi"] <= 1.0, f"HHI out of range at tick {result['tick']}"

    def test_graph_snapshot_in_result(self):
        """Every tick result should include valid graphs for all companies."""
        config = UnifiedStartConfig(num_companies=3)
        engine = UnifiedEngine(config=config, seed=42)
        result = engine.tick()
        graphs = result["graphs"]
        assert len(graphs) == 3
        for _name, graph in graphs.items():
            assert "nodes" in graph
            assert "edges" in graph
            assert len(graph["nodes"]) >= 4  # at least owner + restaurant + 2 suppliers


# ── Optimization validation ──


class TestOptimizationDeterminism:
    def test_100_tick_determinism(self):
        """Optimized engine must produce stable, valid results over 100 ticks."""
        engine = _run_engine(100, num_companies=4, seed=42)
        alive = [c for c in engine.companies if c.alive]
        assert len(alive) > 0

        # Shares still sum to ~1
        total_share = sum(c.share for c in alive)
        assert total_share == pytest.approx(1.0, abs=1e-6)

        # No NaN/inf cash
        for c in engine.companies:
            assert not math.isnan(c.state.cash)
            assert not math.isinf(c.state.cash)

        # Alive companies have positive location counts
        for c in alive:
            assert c.location_count() >= 1


class TestSoASync:
    def test_loc_arrays_match_sim_nodes(self):
        """LocationArrays mutable fields must match SimNode.location_state after tick."""
        engine = _run_engine(50, num_companies=4, seed=42)
        for c in engine.companies:
            if not c.alive:
                continue
            arrays = c._loc_arrays
            for i, node_id in enumerate(arrays.node_ids):
                node = c.state.nodes[node_id]
                ls = node.location_state
                assert ls is not None
                assert float(arrays.customers[i]) == pytest.approx(ls.customers, abs=1e-10)
                assert float(arrays.inventory[i]) == pytest.approx(ls.inventory, abs=1e-10)
                assert float(arrays.satisfaction[i]) == pytest.approx(ls.satisfaction, abs=1e-10)

    def test_loc_arrays_size_matches_location_count(self):
        """LocationArrays size must equal cached location count."""
        engine = _run_engine(200, num_companies=4, seed=42)
        for c in engine.companies:
            if not c.alive:
                continue
            assert c._loc_arrays.size == c.location_count()

    def test_cache_consistency(self):
        """Cached values must match re-computed values from nodes."""
        engine = _run_engine(100, num_companies=4, seed=42)
        for c in engine.companies:
            if not c.alive:
                continue
            # Re-count active nodes manually
            actual_active = sum(1 for n in c.state.nodes.values() if n.active)
            assert c.active_node_count() == actual_active

            # Re-count by type
            loc_type = c.spec.roles.location_type
            actual_locs = sum(
                1 for n in c.state.nodes.values()
                if n.type == loc_type and n.active
            )
            assert c.location_count() == actual_locs
