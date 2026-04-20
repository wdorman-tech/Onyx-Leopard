from __future__ import annotations

import pytest

from biosim.engine.heuristics import (
    HEURISTIC_REGISTRY,
    heuristic_customer_service,
    heuristic_distribution,
    heuristic_executive,
    heuristic_finance,
    heuristic_hr,
    heuristic_it,
    heuristic_legal,
    heuristic_marketing,
    heuristic_procurement,
    heuristic_production,
    heuristic_rd,
    heuristic_sales,
)


def _make_snapshot(
    company_idx: int = 0,
    cash: float = 1e6,
    firm_size: float = 15.0,
    revenue: float = 1e5,
    costs: float = 8e4,
    market_share: float = 0.15,
    health_score: float = 0.6,
    growth_rate: float = 0.05,
    labor: float = 75.0,
    capital: float = 1e6,
) -> dict:
    return {
        "indices": [company_idx],
        "cash": [cash],
        "firm_size": [firm_size],
        "revenue": [revenue],
        "costs": [costs],
        "market_share": [market_share],
        "health_score": [health_score],
        "growth_rate": [growth_rate],
        "labor": [labor],
        "capital": [capital],
        "dept_headcount": [[5.0] * 12],
        "dept_budget": [[25000.0] * 12],
    }


class TestHeuristicReturnShape:
    """All heuristics must return a dict with action, parameters, rationale, confidence."""

    @pytest.mark.parametrize("dept_idx", range(12))
    def test_registry_returns_expected_keys(self, dept_idx: int) -> None:
        snapshot = _make_snapshot()
        fn = HEURISTIC_REGISTRY[dept_idx]
        result = fn(snapshot, 0)

        assert isinstance(result, dict)
        assert "action" in result
        assert "parameters" in result
        assert "rationale" in result
        assert "confidence" in result
        assert isinstance(result["action"], str)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_registry_has_all_12_departments(self) -> None:
        assert len(HEURISTIC_REGISTRY) == 12
        for i in range(12):
            assert i in HEURISTIC_REGISTRY


class TestFinanceHeuristic:
    def test_high_revenue_increases_rd_budget(self) -> None:
        snap = _make_snapshot(revenue=2e5, costs=1e5)
        result = heuristic_finance(snap, 0)
        assert result["action"] == "adjust_budget"
        assert result["parameters"]["delta_pct"] > 0

    def test_low_revenue_cuts_budget(self) -> None:
        snap = _make_snapshot(revenue=5e4, costs=1e5)
        result = heuristic_finance(snap, 0)
        assert result["action"] == "adjust_budget"
        assert result["parameters"]["delta_pct"] < 0

    def test_balanced_holds(self) -> None:
        snap = _make_snapshot(revenue=1e5, costs=1e5)
        result = heuristic_finance(snap, 0)
        assert result["action"] == "hold"


class TestRdHeuristic:
    def test_low_market_share_pivots(self) -> None:
        snap = _make_snapshot(market_share=0.05)
        result = heuristic_rd(snap, 0)
        assert result["action"] == "pivot_innovation"

    def test_healthy_sustains(self) -> None:
        snap = _make_snapshot(health_score=0.8, market_share=0.2)
        result = heuristic_rd(snap, 0)
        assert result["action"] == "sustain_innovation"


class TestSalesHeuristic:
    def test_high_share_raises_price(self) -> None:
        snap = _make_snapshot(market_share=0.3)
        result = heuristic_sales(snap, 0)
        assert result["parameters"]["delta_pct"] > 0

    def test_low_share_discounts(self) -> None:
        snap = _make_snapshot(market_share=0.03)
        result = heuristic_sales(snap, 0)
        assert result["parameters"]["delta_pct"] < 0

    def test_price_within_bounds(self) -> None:
        snap = _make_snapshot(market_share=0.03)
        result = heuristic_sales(snap, 0)
        assert abs(result["parameters"]["delta_pct"]) <= 5.0


class TestHrHeuristic:
    def test_high_revenue_per_employee_hires(self) -> None:
        snap = _make_snapshot(revenue=5e5, labor=50.0)
        result = heuristic_hr(snap, 0)
        assert result["action"] == "hire"

    def test_low_revenue_per_employee_lays_off(self) -> None:
        snap = _make_snapshot(revenue=5e4, labor=100.0)
        result = heuristic_hr(snap, 0)
        assert result["action"] == "layoff"


class TestExecutiveHeuristic:
    def test_strong_position_aggressive(self) -> None:
        snap = _make_snapshot(health_score=0.8, market_share=0.3)
        result = heuristic_executive(snap, 0)
        assert result["action"] == "aggressive_growth"

    def test_poor_health_defensive(self) -> None:
        snap = _make_snapshot(health_score=0.2)
        result = heuristic_executive(snap, 0)
        assert result["action"] == "defensive"


class TestDistributionHeuristic:
    def test_large_firm_expands(self) -> None:
        snap = _make_snapshot(firm_size=40.0)
        result = heuristic_distribution(snap, 0)
        assert result["action"] == "expand_routes"

    def test_small_firm_holds(self) -> None:
        snap = _make_snapshot(firm_size=10.0)
        result = heuristic_distribution(snap, 0)
        assert result["action"] == "hold"


class TestProductionHeuristic:
    def test_high_utilization_expands(self) -> None:
        # labor * 1000 / capital = 100 * 1000 / 1e5 = 1.0 > 0.85
        snap = _make_snapshot(labor=100.0, capital=1e5)
        result = heuristic_production(snap, 0)
        assert result["action"] == "expand_capacity"

    def test_low_utilization_reduces(self) -> None:
        # labor * 1000 / capital = 10 * 1000 / 1e6 = 0.01 < 0.5
        snap = _make_snapshot(labor=10.0, capital=1e6)
        result = heuristic_production(snap, 0)
        assert result["action"] == "reduce_capacity"


class TestMarketingHeuristic:
    def test_growth_phase_increases_campaign(self) -> None:
        snap = _make_snapshot(growth_rate=0.08, firm_size=10.0)
        result = heuristic_marketing(snap, 0)
        assert result["action"] == "increase_campaign"

    def test_mature_firm_maintains(self) -> None:
        snap = _make_snapshot(firm_size=35.0)
        result = heuristic_marketing(snap, 0)
        assert result["action"] == "maintain_campaign"


class TestCustomerServiceHeuristic:
    def test_low_ratio_hires(self) -> None:
        snap = _make_snapshot(labor=10.0, firm_size=20.0)
        result = heuristic_customer_service(snap, 0)
        assert result["action"] == "increase_service_staff"

    def test_high_ratio_reduces(self) -> None:
        snap = _make_snapshot(labor=100.0, firm_size=5.0)
        result = heuristic_customer_service(snap, 0)
        assert result["action"] == "reduce_service_staff"


class TestLegalHeuristic:
    def test_large_firm_increases_compliance(self) -> None:
        snap = _make_snapshot(firm_size=40.0)
        result = heuristic_legal(snap, 0)
        assert result["action"] == "increase_compliance"

    def test_small_firm_holds(self) -> None:
        snap = _make_snapshot(firm_size=10.0)
        result = heuristic_legal(snap, 0)
        assert result["action"] == "hold"


class TestItHeuristic:
    def test_growing_firm_upgrades(self) -> None:
        snap = _make_snapshot(firm_size=30.0, labor=150.0)
        result = heuristic_it(snap, 0)
        assert result["action"] == "upgrade_infrastructure"

    def test_small_firm_holds(self) -> None:
        snap = _make_snapshot(firm_size=10.0, labor=30.0)
        result = heuristic_it(snap, 0)
        assert result["action"] == "hold"


class TestProcurementHeuristic:
    def test_high_cost_ratio_renegotiates(self) -> None:
        snap = _make_snapshot(costs=8e4, revenue=1e5)
        result = heuristic_procurement(snap, 0)
        assert result["action"] == "renegotiate_vendors"

    def test_low_cost_ratio_holds(self) -> None:
        snap = _make_snapshot(costs=3e4, revenue=1e5)
        result = heuristic_procurement(snap, 0)
        assert result["action"] == "hold"


class TestMissingData:
    """Heuristics should handle missing snapshot keys gracefully."""

    @pytest.mark.parametrize("dept_idx", range(12))
    def test_empty_snapshot_does_not_crash(self, dept_idx: int) -> None:
        fn = HEURISTIC_REGISTRY[dept_idx]
        result = fn({}, 0)
        assert "action" in result
        assert "confidence" in result

    @pytest.mark.parametrize("dept_idx", range(12))
    def test_company_not_in_indices(self, dept_idx: int) -> None:
        snap = _make_snapshot(company_idx=0)
        fn = HEURISTIC_REGISTRY[dept_idx]
        result = fn(snap, 99)  # company 99 not in indices
        assert "action" in result
