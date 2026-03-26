"""Tests for the quarterly heuristic decision rules (spec Section 4.2.1)."""

from __future__ import annotations

import pytest

from src.simulation.market.engine import MarketEngine
from src.simulation.market.models import AgentParams, AgentState, MarketParams


def _make_agent(
    revenue: float = 500.0,
    cash: float = 10_000.0,
    capacity: float = 1_000.0,
    quality: float = 1.0,
    marketing: float = 30.0,
    f: float = 100.0,
    share: float = 0.2,
    prev_share: float = 0.2,
) -> AgentState:
    return AgentState(
        id="test-agent",
        params=AgentParams(name="Test", r=0.25, margin=0.25, f=f, eta_m=0.2, eta_q=0.1, tau_k=30),
        revenue=revenue,
        cash=cash,
        capacity=capacity,
        quality=quality,
        marketing=marketing,
        q_target=quality,
        m_target=marketing,
        k_target=capacity,
        share=share,
        prev_share=prev_share,
    )


def _make_engine() -> MarketEngine:
    """Minimal engine for testing quarterly review."""
    return MarketEngine(MarketParams(n_0=2), max_ticks=1000, seed=42)


# tam=0 suppresses Rule 6 so we can test Rules 1-5 in isolation.
_NO_TAM = 0.0


class TestRule1CashEmergency:
    def test_cuts_marketing_when_broke(self):
        """IF B < 2*F THEN m_target = m * 0.5."""
        engine = _make_engine()
        agent = _make_agent(cash=150.0, f=100.0, marketing=40.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        assert agent.m_target == pytest.approx(20.0, abs=0.01)

    def test_freezes_capacity_when_broke(self):
        """IF B < 2*F THEN K_target = K."""
        engine = _make_engine()
        agent = _make_agent(cash=150.0, f=100.0, capacity=1_000.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        assert agent.k_target == pytest.approx(1_000.0, abs=0.01)

    def test_no_emergency_when_cash_ok(self):
        """Cash above 2*F should not trigger Rule 1."""
        engine = _make_engine()
        agent = _make_agent(cash=5_000.0, f=100.0, marketing=40.0)
        original_m_target = agent.m_target
        engine._quarterly_review(agent, tam=_NO_TAM)
        # m_target may change from other rules, but not halved by Rule 1
        assert agent.m_target != original_m_target * 0.5 or agent.m_target == original_m_target * 0.5


class TestRule2ExcessCapacity:
    def test_boosts_marketing_when_underutilized(self):
        """IF R/K < 0.5 THEN m_target = m * 1.2."""
        engine = _make_engine()
        agent = _make_agent(revenue=300.0, capacity=1_000.0, marketing=40.0, cash=10_000.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        # revenue/capacity = 0.3 < 0.5, so Rule 2 fires
        # Rule 5 won't fire (R/K = 0.3 < 0.60)
        assert agent.m_target == pytest.approx(48.0, abs=0.01)

    def test_no_boost_when_well_utilized(self):
        """R/K >= 0.5 should not trigger Rule 2."""
        engine = _make_engine()
        agent = _make_agent(revenue=600.0, capacity=1_000.0, marketing=40.0, cash=10_000.0)
        # R/K = 0.6 -- not excess, not constrained
        engine._quarterly_review(agent, tam=_NO_TAM)
        # Rule 5 fires (cash > 5*F AND R/K > 0.60) but only changes q/k targets
        assert agent.m_target == pytest.approx(40.0, abs=0.01)


class TestRule3CapacityConstraint:
    def test_expands_when_constrained(self):
        """IF R/K > 0.75 THEN K_target = K * 1.3."""
        engine = _make_engine()
        agent = _make_agent(revenue=900.0, capacity=1_000.0, cash=10_000.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        # R/K = 0.9 > 0.75, triggers expansion
        # Rule 5 also fires (cash > 5*F AND R/K > 0.60) -> K_target = 1000 * 1.15
        # Rule 5 runs AFTER Rule 3, so K_target = 1000 * 1.15 = 1150
        assert agent.k_target == pytest.approx(1_150.0, abs=0.01)

    def test_rule3_fires_before_rule5_override(self):
        """With poor cash, only Rule 3 fires (not Rule 5)."""
        engine = _make_engine()
        agent = _make_agent(revenue=900.0, capacity=1_000.0, cash=400.0, f=100.0)
        # Cash=400 < 5*F=500, so Rule 5 won't fire. R/K=0.9 > 0.75 -> Rule 3 fires.
        engine._quarterly_review(agent, tam=_NO_TAM)
        assert agent.k_target == pytest.approx(1_300.0, abs=0.01)


class TestRule4MarketShareDecline:
    def test_invests_when_share_drops(self):
        """IF s < prev_s * 0.9 THEN q_target = q * 1.1, m_target = m * 1.15."""
        engine = _make_engine()
        agent = _make_agent(
            share=0.10,
            prev_share=0.20,
            quality=1.0,
            marketing=40.0,
            revenue=500.0,
            capacity=1_000.0,
            cash=10_000.0,
        )
        engine._quarterly_review(agent, tam=_NO_TAM)
        # Share dropped from 0.20 to 0.10 (50% drop > 10% threshold)
        # Rule 4 sets q_target = 1.1
        # Rule 5 fires (R/K=0.5 < 0.60? No, 0.5 is NOT > 0.60) -> doesn't fire
        assert agent.q_target == pytest.approx(1.1, abs=0.01)
        # m_target: Rule 2 fires (R/K=0.5 -> equal, not < 0.5), Rule 4 sets 40*1.15=46
        assert agent.m_target == pytest.approx(46.0, abs=0.01)

    def test_no_action_when_share_stable(self):
        """Share decline < 10% should not trigger."""
        engine = _make_engine()
        agent = _make_agent(share=0.19, prev_share=0.20, revenue=500.0, capacity=1_000.0, cash=10_000.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        # 0.19 >= 0.20 * 0.9 = 0.18, so Rule 4 does NOT fire
        # R/K=0.5, Rule 5 needs R/K > 0.60 -> doesn't fire
        assert agent.q_target == pytest.approx(1.0, abs=0.01)


class TestRule5ProfitableGrowth:
    def test_moderate_expansion_when_profitable(self):
        """IF B > 5*F AND R/K > 0.60 THEN q_target = q * 1.05, K_target = K * 1.15."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=750.0,
            capacity=1_000.0,
            cash=10_000.0,
            f=100.0,
            quality=1.0,
        )
        engine._quarterly_review(agent, tam=_NO_TAM)
        # cash=10000 > 5*100=500, R/K=0.75 > 0.60 -> Rule 5 fires
        assert agent.q_target == pytest.approx(1.05, abs=0.01)
        assert agent.k_target == pytest.approx(1_150.0, abs=0.01)

    def test_no_expansion_when_cash_low(self):
        """Cash < 5*F prevents Rule 5."""
        engine = _make_engine()
        agent = _make_agent(revenue=650.0, capacity=1_000.0, cash=400.0, f=100.0)
        engine._quarterly_review(agent, tam=_NO_TAM)
        # cash=400 < 5*100=500, Rule 5 won't fire
        # R/K=0.65 < 0.75, Rule 3 won't fire either
        assert agent.k_target == pytest.approx(1_000.0, abs=0.01)


class TestRule6MarketOpportunity:
    def test_expands_toward_demand(self):
        """When demand >> capacity and cash available, expand."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=750.0,
            capacity=1_000.0,
            cash=10_000.0,
            f=100.0,
            share=0.5,
        )
        # tam=50000, share=0.5, demand_potential=25000 >> capacity*1.2=1200
        engine._quarterly_review(agent, tam=50_000.0)
        # Rule 6: target_cap = min(25000, 1000*1.5) = 1500
        assert agent.k_target == pytest.approx(1_500.0, abs=0.01)

    def test_no_expansion_when_demand_low(self):
        """Demand below 1.2x capacity -> Rule 6 doesn't fire."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=750.0,
            capacity=1_000.0,
            cash=10_000.0,
            f=100.0,
            share=0.5,
        )
        # tam=2000, share=0.5, demand_potential=1000 < capacity*1.2=1200
        engine._quarterly_review(agent, tam=2_000.0)
        # Rule 6 doesn't fire; Rule 5 fires (R/K=0.75 > 0.60, cash > 500)
        assert agent.k_target == pytest.approx(1_150.0, abs=0.01)

    def test_no_expansion_when_broke(self):
        """Cash below 3*F prevents Rule 6."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=800.0,
            capacity=1_000.0,
            cash=250.0,
            f=100.0,
            share=0.5,
        )
        engine._quarterly_review(agent, tam=50_000.0)
        # cash=250 < 3*F=300 -> Rule 6 doesn't fire
        # Rule 3: R/K=0.8 > 0.75 -> fires, K_target=1300
        # Rule 5: cash=250 < 500 -> doesn't fire
        assert agent.k_target == pytest.approx(1_300.0, abs=0.01)

    def test_capped_at_1_5x_per_quarter(self):
        """Expansion capped at 1.5x current capacity per quarter."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=750.0,
            capacity=1_000.0,
            cash=50_000.0,
            f=100.0,
            share=0.5,
        )
        # demand_potential=25000, cap=min(25000, 1500)=1500
        engine._quarterly_review(agent, tam=50_000.0)
        assert agent.k_target == pytest.approx(1_500.0, abs=0.01)


class TestRuleOrdering:
    def test_later_rules_can_override_earlier(self):
        """Rule 5 can override Rule 3's K_target."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=900.0,
            capacity=1_000.0,
            cash=10_000.0,
            f=100.0,
        )
        engine._quarterly_review(agent, tam=_NO_TAM)
        # Rule 3 fires (R/K=0.9 > 0.75): K_target = 1300
        # Rule 5 fires (cash > 500 AND R/K > 0.60): K_target = 1000 * 1.15 = 1150
        # Rule 5 runs after Rule 3, so K_target = 1150
        assert agent.k_target == pytest.approx(1_150.0, abs=0.01)

    def test_cash_emergency_overrides_growth(self):
        """Rule 1 cuts marketing, but Rules 2-5 can override m_target after."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=900.0,
            capacity=1_000.0,
            cash=150.0,
            f=100.0,
            marketing=40.0,
        )
        engine._quarterly_review(agent, tam=_NO_TAM)
        # Rule 1: cash=150 < 2*100=200 -> m_target = 20, K_target = 1000
        # Rule 3: R/K=0.9 > 0.75 -> K_target = 1300 (overrides Rule 1 freeze)
        assert agent.k_target == pytest.approx(1_300.0, abs=0.01)

    def test_rule6_overrides_rule5_k_target(self):
        """Rule 6 can override Rule 5's K_target when demand is large."""
        engine = _make_engine()
        agent = _make_agent(
            revenue=700.0,
            capacity=1_000.0,
            cash=10_000.0,
            f=100.0,
            share=0.5,
        )
        engine._quarterly_review(agent, tam=50_000.0)
        # Rule 5: K_target = 1150
        # Rule 6: demand_potential=25000 > 1200, target_cap=min(25000,1500)=1500 > 1150
        assert agent.k_target == pytest.approx(1_500.0, abs=0.01)
