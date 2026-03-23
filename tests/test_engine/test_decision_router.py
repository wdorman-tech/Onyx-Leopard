from __future__ import annotations

import numpy as np
import pytest

from biosim.engine.decision_router import DecisionRouter, _Tier


def _make_snapshot(
    indices: list[int] | None = None,
    n: int = 2,
) -> dict:
    """Build a minimal snapshot dict for n companies."""
    if indices is None:
        indices = list(range(n))
    return {
        "indices": indices,
        "cash": [1e6] * n,
        "firm_size": [15.0] * n,
        "revenue": [1e5] * n,
        "costs": [8e4] * n,
        "market_share": [0.15] * n,
        "health_score": [0.6] * n,
        "dept_headcount": [[5.0] * 12 for _ in range(n)],
        "dept_budget": [[25000.0] * 12 for _ in range(n)],
    }


class TestTierConstants:
    def test_tier_values(self) -> None:
        assert _Tier.ODE == 0
        assert _Tier.HEURISTIC == 1
        assert _Tier.HAIKU == 2
        assert _Tier.SONNET == 3


class TestClassifyTier:
    def test_off_frequency_returns_ode(self) -> None:
        router = DecisionRouter()
        sv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Sales (dept 4) has freq=2. Tick 1 is off-cycle (1 % 2 != 0).
        tier = router.classify_tier(0, 4, 1, sv)
        assert tier == _Tier.ODE

    def test_on_frequency_first_call_returns_ode(self) -> None:
        """First call has no prev state, novelty=0 -> below half-threshold -> ODE."""
        router = DecisionRouter(novelty_threshold=0.15)
        sv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        tier = router.classify_tier(0, 4, 0, sv)
        assert tier == _Tier.ODE

    def test_low_novelty_returns_ode(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.001  # 0.1% change -> novelty ~0.001, well below 0.075
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.ODE

    def test_moderate_novelty_returns_heuristic(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        # ~10% change -> novelty ~0.1, between 0.075 and 0.15
        sv2 = sv1 * 1.1
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.HEURISTIC

    def test_high_novelty_returns_haiku(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        # ~20% change -> novelty ~0.2, between 0.15 and 0.30
        sv2 = sv1 * 1.2
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.HAIKU

    def test_very_high_novelty_returns_sonnet(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        # ~50% change -> novelty ~0.5, well above 0.30
        sv2 = sv1 * 1.5
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.SONNET

    def test_injected_event_high_novelty_returns_sonnet(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.2  # novelty ~0.2 >= threshold
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2, has_injected_event=True)
        assert tier == _Tier.SONNET

    def test_injected_event_low_novelty_returns_haiku(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.05  # novelty ~0.05 < threshold
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2, has_injected_event=True)
        assert tier == _Tier.HAIKU


class TestBudgetDowngrade:
    def test_sonnet_downgraded_to_haiku_at_80pct(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15, cost_budget=10.0)
        router.record_cost(8.5)  # 85% of budget
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.5
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.HAIKU

    def test_haiku_downgraded_to_heuristic_over_budget(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15, cost_budget=10.0)
        router.record_cost(11.0)  # 110% of budget
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.2
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        assert tier == _Tier.HEURISTIC

    def test_sonnet_fully_downgraded_over_budget(self) -> None:
        """When over budget, Sonnet -> Haiku -> Heuristic."""
        router = DecisionRouter(novelty_threshold=0.15, cost_budget=10.0)
        router.record_cost(11.0)
        sv1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        sv2 = sv1 * 1.5  # would be Sonnet
        router.classify_tier(0, 4, 0, sv1)
        tier = router.classify_tier(0, 4, 2, sv2)
        # Sonnet -> Haiku (>80%) -> Heuristic (>100%)
        assert tier == _Tier.HEURISTIC


class TestCostTracking:
    def test_initial_cost_zero(self) -> None:
        router = DecisionRouter()
        assert router.total_cost == 0.0

    def test_record_cost_accumulates(self) -> None:
        router = DecisionRouter()
        router.record_cost(1.5)
        router.record_cost(0.5)
        assert router.total_cost == pytest.approx(2.0)


class TestRouteDecisions:
    def test_returns_all_four_tiers(self) -> None:
        router = DecisionRouter()
        snap = _make_snapshot(n=1, indices=[0])
        result = router.route_decisions([0], 0, snap)
        assert _Tier.ODE in result
        assert _Tier.HEURISTIC in result
        assert _Tier.HAIKU in result
        assert _Tier.SONNET in result

    def test_all_agents_classified(self) -> None:
        router = DecisionRouter()
        snap = _make_snapshot(n=2, indices=[0, 1])
        result = router.route_decisions([0, 1], 0, snap)
        total = sum(len(v) for v in result.values())
        # 2 companies * 12 departments = 24
        assert total == 24

    def test_missing_company_skipped(self) -> None:
        router = DecisionRouter()
        snap = _make_snapshot(n=1, indices=[0])
        result = router.route_decisions([0, 99], 0, snap)
        total = sum(len(v) for v in result.values())
        # Only company 0 classified (12 depts), company 99 skipped
        assert total == 12

    def test_first_tick_all_on_freq_go_ode(self) -> None:
        """First tick (tick=0) with no history -> all novelty=0 -> all ODE."""
        router = DecisionRouter()
        snap = _make_snapshot(n=1, indices=[0])
        result = router.route_decisions([0], 0, snap)
        # All 12 depts at tick 0 are on-frequency (N % 0 handled by mod)
        # First call -> novelty=0 -> ODE
        assert len(result[_Tier.ODE]) == 12

    def test_second_tick_off_freq_depts_ode(self) -> None:
        """At tick=1, only Sales (freq=2) is on-frequency. But this is a fresh router
        so the on-freq dept still gets ODE (first call, no prev state)."""
        router = DecisionRouter()
        snap = _make_snapshot(n=1, indices=[0])
        result = router.route_decisions([0], 1, snap)
        # tick=1: freq 2 (Sales dept 4) -> 1%2=1 off. All freqs > 1 are off.
        assert len(result[_Tier.ODE]) == 12


class TestLLMFrequencies:
    def test_sales_frequency_is_2(self) -> None:
        assert DecisionRouter._LLM_FREQUENCIES[4] == 2

    def test_executive_frequency_is_12(self) -> None:
        assert DecisionRouter._LLM_FREQUENCIES[7] == 12

    def test_all_departments_have_frequency(self) -> None:
        for i in range(12):
            assert i in DecisionRouter._LLM_FREQUENCIES

    @pytest.mark.parametrize(
        "dept_idx,expected_freq",
        [
            (0, 4),   # Finance
            (1, 8),   # R&D
            (2, 8),   # Distribution
            (3, 4),   # Production
            (4, 2),   # Sales
            (5, 4),   # Marketing
            (6, 4),   # HR
            (7, 12),  # Executive
            (8, 4),   # Customer Service
            (9, 12),  # Legal
            (10, 8),  # IT
            (11, 4),  # Procurement
        ],
    )
    def test_department_frequencies_match_prd(self, dept_idx: int, expected_freq: int) -> None:
        assert DecisionRouter._LLM_FREQUENCIES[dept_idx] == expected_freq


class TestNoveltyEdgeCases:
    def test_zero_previous_state(self) -> None:
        """When previous state is all zeros, novelty falls back to norm of delta."""
        router = DecisionRouter(novelty_threshold=0.15)
        sv_zero = np.zeros(8)
        sv_nonzero = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        router.classify_tier(0, 4, 0, sv_zero)
        tier = router.classify_tier(0, 4, 2, sv_nonzero)
        # norm([1..8]) is large, so this should route high
        assert tier in (_Tier.HAIKU, _Tier.SONNET)

    def test_identical_state_returns_ode(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        router.classify_tier(0, 4, 0, sv)
        tier = router.classify_tier(0, 4, 2, sv.copy())
        assert tier == _Tier.ODE

    def test_different_companies_tracked_separately(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        router.classify_tier(0, 4, 0, sv)
        router.classify_tier(1, 4, 0, sv)
        # Company 0 with same state -> ODE
        tier0 = router.classify_tier(0, 4, 2, sv.copy())
        assert tier0 == _Tier.ODE
        # Company 1 with big change -> not ODE
        tier1 = router.classify_tier(1, 4, 2, sv * 1.5)
        assert tier1 != _Tier.ODE

    def test_different_depts_tracked_separately(self) -> None:
        router = DecisionRouter(novelty_threshold=0.15)
        sv = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        router.classify_tier(0, 0, 0, sv)  # Finance, freq=4
        router.classify_tier(0, 4, 0, sv)  # Sales, freq=2
        # Both with same state -> ODE
        tier_fin = router.classify_tier(0, 0, 4, sv.copy())
        tier_sales = router.classify_tier(0, 4, 2, sv.copy())
        assert tier_fin == _Tier.ODE
        assert tier_sales == _Tier.ODE
