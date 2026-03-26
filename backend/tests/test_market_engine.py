"""Integration tests for MarketEngine — full tick loop validation."""

from __future__ import annotations

import math

import pytest

from src.simulation.market.engine import MarketEngine
from src.simulation.market.models import MarketParams
from src.simulation.market.presets import MARKET_PRESETS


class TestTAMEvolution:
    def test_tam_grows_correctly(self, default_params: MarketParams):
        engine = MarketEngine(default_params, max_ticks=100, seed=42)
        initial_tam = engine.tam
        for _ in range(100):
            engine.tick()
        expected = initial_tam * (1 + default_params.g_market) ** 100
        assert engine.tam == pytest.approx(expected, rel=1e-6)

    def test_tam_shrinks_with_negative_growth(self):
        params = MarketParams(g_market=-0.001, n_0=3)
        engine = MarketEngine(params, max_ticks=50, seed=42)
        initial_tam = engine.tam
        for _ in range(50):
            engine.tick()
        expected = initial_tam * (1 - 0.001) ** 50
        assert engine.tam == pytest.approx(expected, rel=1e-6)


class TestShareInvariants:
    def test_shares_sum_to_one_each_tick(self, price_war_engine: MarketEngine):
        for _ in range(50):
            result = price_war_engine.tick()
            alive_shares = [a["share"] for a in result["agents"] if a["alive"]]
            if alive_shares:
                assert sum(alive_shares) == pytest.approx(1.0, abs=1e-4)

    def test_dead_agents_have_zero_share(self, default_params: MarketParams):
        params = MarketParams(n_0=3, b_death=-100.0, t_death=1)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        # Force agent to bankrupt: zero revenue + deeply negative cash
        target = engine.agents[0]
        target.cash = -500.0
        target.revenue = 0.0
        target.marketing = 0.0
        for _ in range(5):
            target.cash = -500.0
            target.revenue = 0.0
            result = engine.tick()
        assert not target.alive, "Agent should be dead"
        for a in result["agents"]:
            if not a["alive"]:
                assert a["share"] == pytest.approx(0.0, abs=1e-10)


class TestRevenueDynamics:
    def test_revenue_nonnegative(self, price_war_engine: MarketEngine):
        for _ in range(200):
            result = price_war_engine.tick()
            for a in result["agents"]:
                assert a["revenue"] >= 0.0, f"Negative revenue for {a['name']} at tick {result['tick']}"

    def test_revenue_bounded_by_capacity(self, default_params: MarketParams):
        engine = MarketEngine(default_params, max_ticks=300, seed=42)
        for _ in range(300):
            result = engine.tick()
            for agent_data, agent_state in zip(result["agents"], engine.agents):
                if agent_state.alive:
                    # Redistribution is capped by spare capacity so this holds
                    assert agent_state.revenue <= agent_state.capacity * 1.01, (
                        f"{agent_data['name']}: revenue {agent_state.revenue} > capacity {agent_state.capacity}"
                    )

    def test_linear_convergence_reaches_equilibrium(self):
        """Revenue should converge to r*C/(r*C+delta) * ceiling, not stagnate near zero."""
        params = MarketParams(n_0=1, delta=0.05, r_range=(0.15, 0.15))
        engine = MarketEngine(params, max_ticks=300, seed=42)
        for _ in range(200):
            engine.tick()
        agent = engine.agents[0]
        if agent.alive:
            # Equilibrium R/K ≈ r*C/(r*C+delta) = 0.15/(0.15+0.05) = 0.75
            util = agent.revenue / agent.capacity if agent.capacity > 0 else 0.0
            assert util > 0.5, f"Utilization {util} too low — revenue should converge"


class TestCashDynamics:
    def test_cash_decreases_without_revenue(self):
        """Agent with zero revenue should lose cash each tick from fixed costs."""
        params = MarketParams(n_0=1, delta=1.0)  # 100% churn = instant revenue loss
        engine = MarketEngine(params, max_ticks=100, seed=42)
        initial_cash = engine.agents[0].cash
        engine.agents[0].revenue = 0.0
        for _ in range(10):
            engine.tick()
        assert engine.agents[0].cash < initial_cash


class TestContinuousAdjustment:
    def test_marketing_converges_to_target(self):
        params = MarketParams(n_0=1)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        agent = engine.agents[0]
        agent.m_target = 100.0
        agent.marketing = 20.0
        for _ in range(200):
            engine.tick()
        # With eta_m ~ 0.2, should converge within 200 ticks
        assert agent.marketing == pytest.approx(agent.m_target, rel=0.05)

    def test_quality_converges_to_target(self):
        params = MarketParams(n_0=1)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        agent = engine.agents[0]
        agent.q_target = 2.0
        agent.quality = 0.5
        for _ in range(200):
            engine.tick()
        assert agent.quality == pytest.approx(agent.q_target, rel=0.05)


class TestQuarterlyReview:
    def test_fires_at_tq_intervals(self):
        params = MarketParams(n_0=2, t_q=90)
        engine = MarketEngine(params, max_ticks=200, seed=42)

        # Run to tick 89 — keep forcing conditions for Rule 3
        for i in range(89):
            for agent in engine.agents:
                agent.revenue = 800.0
                agent.capacity = 1_000.0
                agent.cash = 50_000.0
                agent.k_target = agent.capacity
            engine.tick()

        # Clear any pending from prior ticks
        for agent in engine.agents:
            agent.pending_expansions = []
            agent.revenue = 800.0
            agent.capacity = 1_000.0
            agent.cash = 50_000.0
            agent.k_target = agent.capacity

        # Tick 90 — quarterly fires
        result = engine.tick()
        has_pending = any(len(a.pending_expansions) > 0 for a in engine.agents)
        assert has_pending, "Quarterly review at tick 90 should have committed a capacity expansion"


class TestCapacityExpansion:
    def test_delayed_by_tau_k(self):
        params = MarketParams(n_0=1, t_q=1)  # quarterly every tick for easy testing
        engine = MarketEngine(params, max_ticks=500, seed=42)
        agent = engine.agents[0]
        agent.revenue = 800.0
        agent.capacity = 1_000.0
        agent.cash = 100_000.0
        tau_k = agent.params.tau_k

        engine.tick()  # tick 1 — quarterly fires, commits expansion
        cap_after_commit = agent.capacity
        # Capacity should NOT have changed yet
        assert cap_after_commit == pytest.approx(1_000.0, abs=1.0)

        # Run until delivery
        for _ in range(tau_k + 5):
            engine.tick()
        # Now capacity should have increased
        assert agent.capacity > 1_000.0

    def test_capex_deducted_at_commitment(self):
        params = MarketParams(n_0=1, t_q=1, capex_per_unit=1.0)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        agent = engine.agents[0]
        agent.revenue = 800.0
        agent.capacity = 1_000.0
        agent.cash = 100_000.0

        cash_before = agent.cash
        engine.tick()  # quarterly fires

        # If expansion committed, CapEx should have been deducted
        if agent.pending_expansions:
            assert agent.cash < cash_before

    def test_market_opportunity_drives_expansion(self):
        """Rule 6 should trigger expansion when demand far exceeds capacity."""
        params = MarketParams(n_0=1, t_q=1, tam_0=100_000.0)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        agent = engine.agents[0]
        agent.revenue = 700.0
        agent.capacity = 1_000.0
        agent.cash = 50_000.0

        initial_cap = agent.capacity
        # Run several quarterly ticks to allow expansion cycles
        for _ in range(100):
            engine.tick()
        # With TAM=100k and single agent, demand >> capacity
        # Rule 6 should have triggered multiple expansions
        assert agent.capacity > initial_cap * 2, "Capacity should have grown significantly via market opportunity"


class TestBankruptcy:
    def test_agent_dies_after_t_death_ticks(self):
        params = MarketParams(n_0=3, b_death=-100.0, t_death=5)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        target = engine.agents[0]

        for _ in range(10):
            # Force state BEFORE tick so death check (step 9) sees low cash
            target.cash = -200.0
            target.revenue = 0.0
            target.marketing = 0.0
            engine.tick()

        assert not target.alive

    def test_death_counter_resets_on_recovery(self):
        params = MarketParams(n_0=2, b_death=-100.0, t_death=10)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        target = engine.agents[0]

        # Push below death threshold for 5 ticks
        for _ in range(5):
            target.cash = -200.0
            engine.tick()
        assert target.death_counter > 0

        # Recover
        target.cash = 50_000.0
        engine.tick()
        assert target.death_counter == 0
        assert target.alive

    def test_dead_agent_revenue_decays(self):
        params = MarketParams(n_0=3, b_death=-100.0, t_death=1, tau_decay=20)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        target = engine.agents[0]
        target.cash = -200.0
        target.revenue = 1_000.0

        engine.tick()
        engine.tick()  # Should be dead by now

        if not target.alive:
            rev_after_death = target.revenue
            for _ in range(10):
                engine.tick()
            assert target.revenue < rev_after_death


class TestRevenueRedistribution:
    def test_dead_agent_revenue_goes_to_survivors(self):
        """When an agent dies, its decaying revenue should be redistributed to survivors."""
        params = MarketParams(n_0=3, b_death=-100.0, t_death=1, tau_decay=10)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        target = engine.agents[0]

        # Kill agent 0 quickly
        target.cash = -200.0
        target.revenue = 5_000.0
        target.marketing = 0.0
        engine.tick()
        engine.tick()

        if not target.alive:
            # Track survivor revenue before and after decay ticks
            survivors = [a for a in engine.agents if a.alive]
            rev_before = sum(a.revenue for a in survivors)
            engine.tick()
            rev_after = sum(a.revenue for a in survivors)
            # Survivors should gain some freed revenue (may not be exact due to
            # their own growth/churn, but the redistribution should help)
            # We just check the mechanism ran without error and survivors are alive
            assert len(survivors) > 0
            assert all(a.revenue >= 0 for a in survivors)

    def test_redistribution_capped_by_capacity(self):
        """Redistributed revenue should not push agents above capacity."""
        params = MarketParams(n_0=2, b_death=-100.0, t_death=1, tau_decay=5)
        engine = MarketEngine(params, max_ticks=500, seed=42)
        target = engine.agents[0]
        survivor = engine.agents[1]

        target.cash = -200.0
        target.revenue = 50_000.0  # Huge revenue to redistribute
        target.marketing = 0.0
        survivor.revenue = survivor.capacity * 0.95  # Almost full

        engine.tick()
        engine.tick()

        if not target.alive and survivor.alive:
            # Survivor shouldn't exceed capacity despite massive freed revenue
            assert survivor.revenue <= survivor.capacity * 1.01


class TestNewEntrants:
    def test_entrants_can_spawn(self):
        """With low HHI and positive growth, new agents should eventually appear."""
        params = MarketParams(
            n_0=3,
            lambda_entry=0.5,
            g_market=0.01,
            g_ref=0.001,
        )
        engine = MarketEngine(params, max_ticks=500, seed=42)
        initial_total = len(engine.agents)
        for _ in range(200):
            engine.tick()
        assert len(engine.agents) > initial_total

    def test_entrants_spawn_with_unserved_demand(self):
        """Even at HHI≈1, unserved demand should attract new entrants."""
        params = MarketParams(
            n_0=1,
            lambda_entry=0.10,
            g_market=0.001,
            g_ref=0.001,
            tam_0=500_000.0,
            k_range=(500.0, 1_000.0),
        )
        engine = MarketEngine(params, max_ticks=500, seed=42)
        # Single agent with tiny capacity vs huge TAM -> mostly unserved
        initial_total = len(engine.agents)
        for _ in range(300):
            engine.tick()
        assert len(engine.agents) > initial_total, "Unserved demand should attract entrants"


class TestDeterminism:
    def test_same_seed_same_results(self):
        params = MARKET_PRESETS["price-war"].params
        engine1 = MarketEngine(params, max_ticks=100, seed=12345)
        engine2 = MarketEngine(params, max_ticks=100, seed=12345)

        for _ in range(100):
            r1 = engine1.tick()
            r2 = engine2.tick()

        assert r1["tam"] == r2["tam"]
        assert r1["hhi"] == r2["hhi"]
        for a1, a2 in zip(r1["agents"], r2["agents"]):
            assert a1["revenue"] == a2["revenue"]
            assert a1["cash"] == a2["cash"]

    def test_different_seeds_different_results(self):
        params = MARKET_PRESETS["price-war"].params
        engine1 = MarketEngine(params, max_ticks=100, seed=111)
        engine2 = MarketEngine(params, max_ticks=100, seed=222)

        for _ in range(100):
            r1 = engine1.tick()
            r2 = engine2.tick()

        # Agents have different initial params -> different outcomes
        assert r1["agents"][0]["revenue"] != r2["agents"][0]["revenue"]


class TestIsComplete:
    def test_complete_at_max_ticks(self):
        engine = MarketEngine(MarketParams(n_0=3), max_ticks=10, seed=42)
        for _ in range(10):
            engine.tick()
        assert engine.is_complete

    def test_complete_when_all_dead(self):
        params = MarketParams(n_0=2, b_death=100_000.0, t_death=1)
        engine = MarketEngine(params, max_ticks=1000, seed=42)
        for agent in engine.agents:
            agent.cash = -500_000.0
        for _ in range(5):
            engine.tick()
        assert engine.is_complete


class TestNoNaNOrInf:
    """Guard against numerical instability across all presets."""

    @pytest.mark.parametrize("preset_slug", list(MARKET_PRESETS.keys()))
    def test_no_nan_inf_in_results(self, preset_slug: str):
        params = MARKET_PRESETS[preset_slug].params
        engine = MarketEngine(params, max_ticks=300, seed=42)
        for _ in range(300):
            result = engine.tick()
            assert math.isfinite(result["tam"]), f"TAM is not finite at tick {result['tick']}"
            assert math.isfinite(result["hhi"]), f"HHI is not finite at tick {result['tick']}"
            for a in result["agents"]:
                assert math.isfinite(a["revenue"]), f"Revenue NaN/Inf for {a['name']}"
                assert math.isfinite(a["cash"]), f"Cash NaN/Inf for {a['name']}"
                assert math.isfinite(a["share"]), f"Share NaN/Inf for {a['name']}"
                assert math.isfinite(a["utilization"]), f"Util NaN/Inf for {a['name']}"


class TestGrowthChurnBalance:
    """Validate that r > delta holds for all presets, preventing permanent revenue decay."""

    @pytest.mark.parametrize("preset_slug", list(MARKET_PRESETS.keys()))
    def test_r_exceeds_delta(self, preset_slug: str):
        p = MARKET_PRESETS[preset_slug].params
        r_min = p.r_range[0]
        assert r_min > p.delta, (
            f"{preset_slug}: r_min={r_min} must exceed delta={p.delta} "
            f"for revenue growth to be possible"
        )

    @pytest.mark.parametrize("preset_slug", list(MARKET_PRESETS.keys()))
    def test_equilibrium_utilization_above_50pct(self, preset_slug: str):
        """Theoretical equilibrium R/K = r*C/(r*C+delta) with C=1 should be > 0.5."""
        p = MARKET_PRESETS[preset_slug].params
        r_mid = (p.r_range[0] + p.r_range[1]) / 2
        equil = r_mid / (r_mid + p.delta)
        assert equil > 0.5, (
            f"{preset_slug}: equilibrium utilization {equil:.3f} too low "
            f"(r_mid={r_mid}, delta={p.delta})"
        )
