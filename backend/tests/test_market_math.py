"""Unit tests for pure math functions — validates spec equations."""

from __future__ import annotations

import math

import pytest

from src.simulation.market.engine import (
    compute_capital_constraint,
    compute_hhi,
    compute_share_attraction,
    compute_spawn_probability,
)
from src.simulation.market.models import AgentParams, AgentState


def _make_agent(id: str, quality: float = 1.0, marketing: float = 30.0, alive: bool = True) -> AgentState:
    return AgentState(
        id=id,
        params=AgentParams(name=id, r=0.05, margin=0.25, f=100.0, eta_m=0.2, eta_q=0.1, tau_k=30),
        quality=quality,
        marketing=marketing,
        alive=alive,
    )


# ── Share Attraction (spec Section 3.2) ──


class TestShareAttraction:
    def test_two_equal_agents_get_equal_share(self):
        agents = [_make_agent("a"), _make_agent("b")]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert len(shares) == 2
        assert shares[0] == pytest.approx(0.5, abs=1e-10)
        assert shares[1] == pytest.approx(0.5, abs=1e-10)

    def test_marketing_dominance(self):
        """With alpha > beta, higher marketing wins more share."""
        agents = [
            _make_agent("a", marketing=60.0, quality=1.0),
            _make_agent("b", marketing=30.0, quality=1.0),
        ]
        shares = compute_share_attraction(agents, alpha=1.5, beta=0.5)
        assert shares[0] > shares[1]
        assert shares[0] > 0.6  # 2x marketing with alpha=1.5 should dominate

    def test_quality_dominance(self):
        """With beta > alpha, higher quality wins more share."""
        agents = [
            _make_agent("a", quality=2.0, marketing=30.0),
            _make_agent("b", quality=1.0, marketing=30.0),
        ]
        shares = compute_share_attraction(agents, alpha=0.3, beta=1.5)
        assert shares[0] > shares[1]
        assert shares[0] > 0.7  # 2x quality with beta=1.5 should dominate

    def test_zero_marketing_gets_zero_share(self):
        """Agent with m=0 gets zero share regardless of quality (spec edge case)."""
        agents = [
            _make_agent("a", marketing=0.0, quality=2.0),
            _make_agent("b", marketing=30.0, quality=1.0),
        ]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert shares[0] == pytest.approx(0.0, abs=1e-10)
        assert shares[1] == pytest.approx(1.0, abs=1e-10)

    def test_single_agent_gets_full_share(self):
        agents = [_make_agent("a")]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert shares[0] == pytest.approx(1.0, abs=1e-10)

    def test_dead_agent_gets_zero_share(self):
        agents = [_make_agent("a", alive=True), _make_agent("b", alive=False)]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert shares[0] == pytest.approx(1.0, abs=1e-10)
        assert shares[1] == pytest.approx(0.0, abs=1e-10)

    def test_shares_sum_to_one(self):
        agents = [_make_agent(f"a{i}", quality=0.5 + i * 0.3, marketing=10 + i * 10) for i in range(5)]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert sum(shares) == pytest.approx(1.0, abs=1e-10)

    def test_all_zero_marketing_gives_equal_shares(self):
        """Edge case: when total attraction is 0, assign equal shares."""
        agents = [_make_agent("a", marketing=0.0), _make_agent("b", marketing=0.0)]
        shares = compute_share_attraction(agents, alpha=0.8, beta=0.8)
        assert shares[0] == pytest.approx(0.5, abs=1e-10)
        assert shares[1] == pytest.approx(0.5, abs=1e-10)

    def test_no_agents_returns_empty(self):
        shares = compute_share_attraction([], alpha=0.8, beta=0.8)
        assert shares == []


# ── Capital Constraint Sigmoid (spec Section 3.5) ──


class TestCapitalConstraint:
    def test_healthy_cash_near_one(self):
        """Cash >> threshold -> C approaches 1.0."""
        c = compute_capital_constraint(cash=50_000.0, b_threshold=5_000.0, k_sigmoid=0.001)
        assert c > 0.99

    def test_broke_near_zero(self):
        """Cash << threshold -> C approaches 0.0."""
        c = compute_capital_constraint(cash=-5_000.0, b_threshold=5_000.0, k_sigmoid=0.001)
        assert c < 0.01

    def test_midpoint_is_half(self):
        """Cash == threshold -> C = 0.5 exactly."""
        c = compute_capital_constraint(cash=5_000.0, b_threshold=5_000.0, k_sigmoid=0.001)
        assert c == pytest.approx(0.5, abs=1e-10)

    def test_monotonically_increasing(self):
        """Higher cash -> higher constraint value."""
        vals = [
            compute_capital_constraint(cash=c, b_threshold=5_000.0, k_sigmoid=0.001)
            for c in [-5000, 0, 5000, 10000, 50000]
        ]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_steeper_k_sharper_transition(self):
        """Higher k_sigmoid makes transition sharper around threshold."""
        gentle = compute_capital_constraint(cash=3_000.0, b_threshold=5_000.0, k_sigmoid=0.0005)
        steep = compute_capital_constraint(cash=3_000.0, b_threshold=5_000.0, k_sigmoid=0.005)
        # Both below threshold, but steep should be closer to 0
        assert steep < gentle

    def test_no_overflow_extreme_values(self):
        """Should not overflow with extreme cash values."""
        c_high = compute_capital_constraint(cash=1e12, b_threshold=5_000.0, k_sigmoid=0.001)
        c_low = compute_capital_constraint(cash=-1e12, b_threshold=5_000.0, k_sigmoid=0.001)
        assert 0.0 <= c_high <= 1.0
        assert 0.0 <= c_low <= 1.0


# ── HHI (spec Section 5.2) ──


class TestHHI:
    def test_perfect_competition(self):
        """N equal agents -> HHI = 1/N."""
        for n in [2, 5, 10, 20]:
            shares = [1.0 / n] * n
            assert compute_hhi(shares) == pytest.approx(1.0 / n, abs=1e-10)

    def test_monopoly(self):
        """Single agent with 100% share -> HHI = 1.0."""
        assert compute_hhi([1.0]) == pytest.approx(1.0, abs=1e-10)

    def test_duopoly_unequal(self):
        """70/30 split should give 0.49 + 0.09 = 0.58."""
        assert compute_hhi([0.7, 0.3]) == pytest.approx(0.58, abs=1e-10)

    def test_empty_returns_zero(self):
        assert compute_hhi([]) == pytest.approx(0.0, abs=1e-10)

    def test_increasing_concentration(self):
        """HHI increases as market concentrates."""
        equal = compute_hhi([0.25, 0.25, 0.25, 0.25])
        unequal = compute_hhi([0.50, 0.20, 0.20, 0.10])
        very_unequal = compute_hhi([0.80, 0.10, 0.05, 0.05])
        assert equal < unequal < very_unequal


# ── Spawn Probability (spec Section 5.2) ──


class TestSpawnProbability:
    def test_low_hhi_positive_growth(self):
        """Low concentration + growing market -> positive spawn rate."""
        p = compute_spawn_probability(hhi=0.15, lambda_entry=0.05, g_market=0.001, g_ref=0.001)
        assert p > 0

    def test_high_hhi_suppresses_competition_entry(self):
        """High concentration -> competition-driven entry approaches zero."""
        p = compute_spawn_probability(hhi=0.95, lambda_entry=0.05, g_market=0.001, g_ref=0.001)
        # With unserved_ratio=0 (default), only competition channel active
        assert p < 0.005

    def test_monopoly_hhi_zero_without_demand(self):
        """HHI=1 with no unserved demand -> zero entry."""
        p = compute_spawn_probability(hhi=1.0, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=0.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_monopoly_hhi_nonzero_with_unserved_demand(self):
        """HHI=1 but 90% unserved demand -> demand channel still attracts entry."""
        p = compute_spawn_probability(hhi=1.0, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=0.9)
        # Competition channel: 0 (HHI=1)
        # Demand channel: 0.05 * 0.9 * 0.5 = 0.0225
        assert p == pytest.approx(0.0225, abs=1e-6)
        assert p > 0

    def test_negative_growth_zero_competition_spawn(self):
        """Contracting market -> no competition-driven entry."""
        p = compute_spawn_probability(hhi=0.2, lambda_entry=0.05, g_market=-0.001, g_ref=0.001, unserved_ratio=0.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_zero_lambda_zero_spawn(self):
        p = compute_spawn_probability(hhi=0.2, lambda_entry=0.0, g_market=0.001, g_ref=0.001)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_proportional_to_lambda(self):
        """Doubling lambda doubles spawn rate (both channels scale with lambda)."""
        p1 = compute_spawn_probability(hhi=0.2, lambda_entry=0.03, g_market=0.001, g_ref=0.001)
        p2 = compute_spawn_probability(hhi=0.2, lambda_entry=0.06, g_market=0.001, g_ref=0.001)
        assert p2 == pytest.approx(p1 * 2, abs=1e-10)

    def test_unserved_demand_boosts_entry(self):
        """High unserved demand increases total spawn probability."""
        p_no_demand = compute_spawn_probability(hhi=0.3, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=0.0)
        p_high_demand = compute_spawn_probability(hhi=0.3, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=0.8)
        assert p_high_demand > p_no_demand

    def test_unserved_ratio_clamped(self):
        """Unserved ratio above 1.0 is clamped to 1.0."""
        p_one = compute_spawn_probability(hhi=0.5, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=1.0)
        p_over = compute_spawn_probability(hhi=0.5, lambda_entry=0.05, g_market=0.001, g_ref=0.001, unserved_ratio=1.5)
        assert p_one == pytest.approx(p_over, abs=1e-10)
