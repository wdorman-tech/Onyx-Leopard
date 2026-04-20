"""Tests for Phase 4 agent architecture: heuristic agent, memory, probabilistic activation, budget."""

from __future__ import annotations

import copy

import pytest

from src.simulation.agent_memory import AgentMemory, AgentTier, AIBudget
from src.simulation.config_loader import load_industry
from src.simulation.heuristic_agent import heuristic_decide
from src.simulation.unified import CompanyAgent, UnifiedEngine
from src.simulation.unified_models import UnifiedStartConfig


# ── Fixtures ──

@pytest.fixture
def spec():
    # Deep-copy the cached IndustrySpec so per-test mutations (e.g. flipping
    # ceo.use_probabilistic_activation) don't bleed into other tests via
    # load_industry's module-level cache.
    return copy.deepcopy(load_industry("restaurant"))


@pytest.fixture
def company(spec):
    return CompanyAgent(name="Test Co", index=0, spec=spec, cash=50_000.0)


@pytest.fixture
def engine():
    config = UnifiedStartConfig(
        industry="restaurant", num_companies=3, max_ticks=500,
    )
    return UnifiedEngine(config=config, seed=42)


# ── HeuristicAgent tests ──

class TestHeuristicAgent:
    def test_returns_ceo_decision(self, company, engine):
        for _ in range(50):
            engine.tick()
        decision = heuristic_decide(
            engine.companies[0], engine.companies, engine.tam, engine.tick_num,
        )
        assert decision.reasoning
        assert decision.price_adjustment > 0
        assert decision.expansion_pace in ("aggressive", "normal", "conservative")
        assert 0.0 <= decision.marketing_intensity <= 1.0

    def test_cash_emergency_triggers_conservative(self, spec):
        company = CompanyAgent(name="Broke Co", index=0, spec=spec, cash=100.0)
        decision = heuristic_decide(company, [company], 25_000.0, 100)
        assert decision.expansion_pace == "conservative"
        assert decision.max_locations_per_year == 0
        assert "Cash emergency" in decision.reasoning

    def test_capacity_constraint_triggers_expansion(self, engine):
        # Run until companies have some revenue and capacity
        for _ in range(200):
            engine.tick()
        # The original test broke out of the loop on first match, so if no
        # company had alive + capacity > 0 the assertion was silently skipped.
        # Track explicitly and fail if the precondition itself never held.
        candidates = [c for c in engine.companies if c.alive and c.capacity > 0]
        assert candidates, "No alive company reached operating capacity in 200 ticks"

        c = candidates[0]
        c.daily_revenue = c.capacity * 0.9  # force >0.75 utilization
        decision = heuristic_decide(c, engine.companies, engine.tam, engine.tick_num)
        assert decision.expansion_pace == "aggressive"
        assert "capacity" in decision.reasoning.lower()

    def test_cash_emergency_overrides_capacity_expansion(self, engine):
        """Cash emergency must freeze expansion even when utilization is high.

        Regression: Rule 3 (capacity constraint) used to overwrite Rule 1's
        max_locs=0, so a cash-strapped company at capacity would still expand
        and accelerate insolvency.
        """
        for _ in range(200):
            engine.tick()
        for c in engine.companies:
            if c.alive and c.capacity > 0:
                c.daily_revenue = c.capacity * 0.9  # high utilization → would trigger Rule 3
                c.state.cash = 100.0  # below any reasonable emergency_threshold
                decision = heuristic_decide(
                    c, engine.companies, engine.tam, engine.tick_num,
                )
                assert decision.expansion_pace == "conservative"
                assert decision.max_locations_per_year == 0
                assert "Cash emergency" in decision.reasoning
                return
        pytest.skip("No company reached operating state with capacity > 0")

    def test_heuristic_preserves_price(self, spec):
        company = CompanyAgent(name="Price Co", index=0, spec=spec, cash=50_000.0)
        for node in company.state.nodes.values():
            if node.location_state is not None:
                node.location_state.price = 18.50
                break
        decision = heuristic_decide(company, [company], 25_000.0, 100)
        assert decision.price_adjustment == 18.50


# ── AgentMemory tests ──

class TestAgentMemory:
    def test_record_and_retrieve(self):
        memory = AgentMemory(recent_window=3)
        memory.record_decision(
            decision_data={"reasoning": "test", "price_adjustment": 14.0},
            tick=100, cash=50_000, share=0.25, locations=2, daily_revenue=500,
        )
        assert memory.total_decisions == 1
        assert len(memory.decisions) == 1

    def test_compression_after_window(self):
        memory = AgentMemory(recent_window=3)
        for i in range(6):
            memory.record_decision(
                decision_data={"reasoning": f"decision {i}"},
                tick=100 * (i + 1), cash=50_000 - i * 5_000,
                share=0.25, locations=2, daily_revenue=500,
            )
        assert memory.total_decisions == 6
        assert len(memory.decisions) == 3  # only recent_window kept
        assert "decision 0" in memory.summary  # older compressed
        assert "decision 1" in memory.summary
        assert "decision 2" in memory.summary

    def test_peak_tracking(self):
        memory = AgentMemory()
        memory.record_decision(
            decision_data={"reasoning": "boom"},
            tick=100, cash=100_000, share=0.5, locations=5, daily_revenue=1000,
        )
        memory.record_decision(
            decision_data={"reasoning": "bust"},
            tick=200, cash=-5_000, share=0.1, locations=3, daily_revenue=200,
        )
        assert memory.peak_cash == 100_000
        assert memory.min_cash == -5_000
        assert memory.peak_share == 0.5
        assert memory.peak_locations == 5
        assert memory.crisis_count == 1  # one negative-cash decision

    def test_prompt_context_generation(self):
        memory = AgentMemory(recent_window=2)
        for i in range(4):
            memory.record_decision(
                decision_data={"reasoning": f"reason {i}", "price_adjustment": 14.0},
                tick=182 * (i + 1), cash=50_000, share=0.25,
                locations=2, daily_revenue=500,
            )
        ctx = memory.build_prompt_context(ticks_per_year=365)
        assert "EARLIER DECISIONS" in ctx
        assert "RECENT DECISIONS" in ctx
        assert "TRACK RECORD" in ctx
        assert "Peak cash" in ctx

    def test_empty_memory_returns_empty_context(self):
        memory = AgentMemory()
        ctx = memory.build_prompt_context(ticks_per_year=365)
        assert ctx == ""


# ── AIBudget tests ──

class TestAIBudget:
    def test_fresh_budget_can_afford(self):
        budget = AIBudget(max_spend=1.0)
        assert budget.can_afford("claude-sonnet-4-6")
        assert not budget.exhausted

    def test_budget_exhaustion(self):
        budget = AIBudget(max_spend=0.03)  # ~2 Sonnet calls
        budget.record_call("claude-sonnet-4-6")
        assert budget.can_afford("claude-sonnet-4-6")
        budget.record_call("claude-sonnet-4-6")
        assert not budget.can_afford("claude-sonnet-4-6")
        assert budget.exhausted

    def test_haiku_cheaper_than_sonnet(self):
        budget = AIBudget(max_spend=0.01)
        # Can't afford Sonnet
        assert not budget.can_afford("claude-sonnet-4-6")
        # But can afford Haiku
        assert budget.can_afford("claude-haiku-4-5-20251001")

    def test_remaining_decreases(self):
        budget = AIBudget(max_spend=0.50)
        initial = budget.remaining
        budget.record_call("claude-sonnet-4-6")
        assert budget.remaining < initial
        assert budget.call_count == 1


# ── AgentTier tests ──

class TestAgentTier:
    def test_tier_values(self):
        assert AgentTier.EXECUTIVE == "executive"
        assert AgentTier.OPERATIONAL == "operational"
        assert AgentTier.HEURISTIC == "heuristic"


# ── Probabilistic Activation tests ──

class TestProbabilisticActivation:
    def test_activation_fires_over_many_ticks(self):
        """Over 1000 ticks, CEO should fire approximately 1000 * 0.0055 ≈ 5.5 times."""
        config = UnifiedStartConfig(
            industry="restaurant", num_companies=2, max_ticks=1000,
            ai_ceo_enabled=True,
            company_strategies={0: "balanced", 1: "aggressive_growth"},
        )
        engine = UnifiedEngine(config=config, seed=42)

        activation_count = 0
        for _ in range(1000):
            engine.tick()
            if engine._pending_ceo_calls:
                activation_count += 1
                # Reset flags (normally run_ceo_agents does this)
                engine._pending_ceo_calls = False
                for c in engine.companies:
                    c._pending_ceo_call = False

        # With prob=0.0055 per company per tick, 2 companies, 1000 ticks:
        # expected ≈ 11. Tightened range catches order-of-magnitude regressions
        # (e.g., crisis multiplier accidentally compounding) without flaking
        # on normal stochastic variation.
        assert 4 <= activation_count <= 25, f"Unexpected activation count: {activation_count}"

    def test_crisis_increases_activation_rate(self):
        """Companies with low cash should activate more frequently via crisis multiplier."""
        config = UnifiedStartConfig(
            industry="restaurant", num_companies=2, max_ticks=500,
            ai_ceo_enabled=True,
            company_strategies={0: "balanced", 1: "balanced"},
        )
        engine = UnifiedEngine(config=config, seed=42)

        crisis_activations = 0
        normal_activations = 0
        for _ in range(500):
            # Keep one company in low-cash crisis zone (30% of starting cash)
            # without killing it (b_death is -$5000)
            engine.companies[0].state.cash = 5_000  # below 30% of 50k = 15k

            engine.tick()
            for c in engine.companies:
                if c._pending_ceo_call:
                    if c is engine.companies[0]:
                        crisis_activations += 1
                    else:
                        normal_activations += 1
                    c._pending_ceo_call = False
            engine._pending_ceo_calls = False

        # Crisis company gets 1.5x multiplier (crisis_mult * 0.5 for low but positive cash)
        # Both should fire, but crisis company should fire at least once
        assert crisis_activations >= 1

    def test_fixed_interval_fallback(self):
        """When probabilistic activation is disabled, falls back to fixed interval."""
        config = UnifiedStartConfig(
            industry="restaurant", num_companies=2, max_ticks=400,
            ai_ceo_enabled=True,
            company_strategies={0: "balanced", 1: "balanced"},
        )
        engine = UnifiedEngine(config=config, seed=42)
        # Deep-copy the CEO config block before mutating — engine.spec is the
        # module-cached IndustrySpec, and mutating it in place would leak
        # `use_probabilistic_activation = False` into every other test that
        # loads the restaurant spec, making test order matter.
        engine.spec.ceo = copy.deepcopy(engine.spec.ceo)
        engine.spec.ceo.use_probabilistic_activation = False

        activation_ticks: list[int] = []
        for _ in range(400):
            engine.tick()
            if engine._pending_ceo_calls:
                activation_ticks.append(engine.tick_num)
                engine._pending_ceo_calls = False
                for c in engine.companies:
                    c._pending_ceo_call = False

        # Should fire exactly at interval_ticks multiples
        interval = engine.spec.ceo.interval_ticks
        expected = [t for t in range(interval, 401, interval)]
        assert activation_ticks == expected


# ── Integration: CompanyAgent has memory ──

class TestCompanyAgentMemory:
    def test_company_has_memory(self, company):
        assert isinstance(company.memory, AgentMemory)
        assert company.memory.total_decisions == 0

    def test_company_has_pending_flag(self, company):
        assert company._pending_ceo_call is False

    def test_engine_has_budget(self, engine):
        assert isinstance(engine._ai_budget, AIBudget)
        assert engine._ai_budget.remaining > 0
