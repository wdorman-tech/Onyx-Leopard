import numpy as np
import pytest

from biosim.engine.state_manager import StateManager
from biosim.types.config import SimConfig


class TestAddCompany:
    def test_adds_company_with_correct_state(self, state_manager: StateManager) -> None:
        idx = state_manager.add_company("TestCo", "#FF0000", size="medium")
        state = state_manager.state

        assert idx == 0
        assert state.n_active == 1
        assert state.alive[idx] is np.True_
        assert state.company_names[idx] == "TestCo"
        assert state.company_colors[idx] == "#FF0000"
        assert state.cash[idx] == 1e6
        assert state.firm_size[idx] == 15.0

    def test_multiple_companies_get_sequential_indices(self, state_manager: StateManager) -> None:
        idx0 = state_manager.add_company("A", "#AAA")
        idx1 = state_manager.add_company("B", "#BBB")
        idx2 = state_manager.add_company("C", "#CCC")

        assert (idx0, idx1, idx2) == (0, 1, 2)
        assert state_manager.state.n_active == 3

    def test_max_capacity_raises(self) -> None:
        config = SimConfig(max_companies=2)
        sm = StateManager(config)
        sm.add_company("A", "#AAA")
        sm.add_company("B", "#BBB")
        with pytest.raises(ValueError, match="Max capacity"):
            sm.add_company("C", "#CCC")


class TestDefaultParams:
    @pytest.mark.parametrize(
        "size,expected_cash,expected_firm_size",
        [
            ("small", 5e5, 5.0),
            ("medium", 1e6, 15.0),
            ("large", 5e6, 40.0),
        ],
    )
    def test_size_presets(
        self,
        state_manager: StateManager,
        size: str,
        expected_cash: float,
        expected_firm_size: float,
    ) -> None:
        idx = state_manager.add_company("Co", "#000", size=size)
        assert state_manager.state.cash[idx] == expected_cash
        assert state_manager.state.firm_size[idx] == expected_firm_size

    def test_unknown_size_falls_back_to_medium(self, state_manager: StateManager) -> None:
        idx = state_manager.add_company("Co", "#000", size="mega")
        assert state_manager.state.cash[idx] == 1e6

    def test_dept_headcount_sums_to_total_labor(self, state_manager: StateManager) -> None:
        idx = state_manager.add_company("Co", "#000", size="medium")
        total = state_manager.state.dept_headcount[idx].sum()
        assert abs(total - 75.0) < 1.0


class TestHistory:
    def test_record_snapshot_appends(self, state_manager: StateManager) -> None:
        state_manager.add_company("Co", "#000")
        snap = state_manager.record_snapshot()

        assert len(state_manager.history) == 1
        assert snap["tick"] == 0

    def test_rolling_window_evicts_old(self) -> None:
        config = SimConfig()
        sm = StateManager(config, max_history=5)
        sm.add_company("Co", "#000")
        for i in range(10):
            sm.tick_count = i
            sm.record_snapshot()

        assert len(sm.history) == 5
        assert sm.history[0]["tick"] == 5

    def test_get_history_returns_list(self, state_manager: StateManager) -> None:
        assert state_manager.get_history() == []


class TestSnapshotDict:
    def test_snapshot_has_expected_keys(self, populated_state_manager: StateManager) -> None:
        snap = populated_state_manager.state.to_snapshot_dict()
        expected_keys = {
            "n_active",
            "indices",
            "company_names",
            "company_colors",
            "cash",
            "firm_size",
            "growth_rate",
            "revenue",
            "costs",
            "market_share",
            "health_score",
            "carrying_capacity",
            "dept_headcount",
            "dept_budget",
            "capital",
            "labor",
        }
        assert set(snap.keys()) == expected_keys

    def test_snapshot_n_active_matches(self, populated_state_manager: StateManager) -> None:
        snap = populated_state_manager.state.to_snapshot_dict()
        assert snap["n_active"] == 3
        assert len(snap["company_names"]) == 3

    def test_snapshot_excludes_dead_companies(
        self, populated_state_manager: StateManager
    ) -> None:
        populated_state_manager.state.remove_company(1)
        snap = populated_state_manager.state.to_snapshot_dict()
        assert snap["n_active"] == 2
        assert "BetaInc" not in snap["company_names"]
