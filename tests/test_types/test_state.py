import numpy as np
import pytest

from biosim.types.state import ODE_VARS_PER_AGENT, StateArrays


class TestAddRemoveCompany:
    def test_add_company_returns_sequential_indices(self, state_arrays: StateArrays) -> None:
        assert state_arrays.n_active == 3
        idx = state_arrays.add_company("Delta Co", "#FF0000", {"cash": 100})
        assert idx == 3
        assert state_arrays.n_active == 4

    def test_add_company_sets_fields(self, state_arrays: StateArrays) -> None:
        assert state_arrays.company_names[0] == "Alpha Corp"
        assert state_arrays.company_colors[0] == "#E74C3C"
        assert state_arrays.cash[0] == 1_000_000
        assert state_arrays.firm_size[0] == 100
        assert state_arrays.growth_rate[0] == 0.05

    def test_add_company_reuses_dead_slot(self, state_arrays: StateArrays) -> None:
        state_arrays.remove_company(1)
        idx = state_arrays.add_company("Reuse Co", "#000000", {"cash": 42})
        assert idx == 1
        assert state_arrays.cash[1] == 42

    def test_add_company_raises_on_full(self) -> None:
        sa = StateArrays(max_capacity=2)
        sa.add_company("A", "#000", {"cash": 1})
        sa.add_company("B", "#000", {"cash": 2})
        with pytest.raises(RuntimeError, match="No free slots"):
            sa.add_company("C", "#000", {"cash": 3})

    def test_add_company_rejects_unknown_param(self) -> None:
        sa = StateArrays(max_capacity=5)
        with pytest.raises(ValueError, match="Unknown parameter"):
            sa.add_company("Bad", "#000", {"nonexistent_field": 1})

    def test_remove_company_zeros_data(self, state_arrays: StateArrays) -> None:
        state_arrays.remove_company(0)
        assert not state_arrays.alive[0]
        assert state_arrays.cash[0] == 0
        assert state_arrays.firm_size[0] == 0
        assert state_arrays.company_names[0] == ""
        assert state_arrays.n_active == 2

    def test_remove_already_dead_raises(self, state_arrays: StateArrays) -> None:
        state_arrays.remove_company(0)
        with pytest.raises(ValueError, match="already dead"):
            state_arrays.remove_company(0)

    def test_remove_out_of_range_raises(self, state_arrays: StateArrays) -> None:
        with pytest.raises(IndexError):
            state_arrays.remove_company(999)


class TestActiveIndices:
    def test_returns_alive_slots(self, state_arrays: StateArrays) -> None:
        idx = state_arrays.active_indices()
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_excludes_removed(self, state_arrays: StateArrays) -> None:
        state_arrays.remove_company(1)
        idx = state_arrays.active_indices()
        np.testing.assert_array_equal(idx, [0, 2])

    def test_empty_state(self) -> None:
        sa = StateArrays(max_capacity=5)
        assert len(sa.active_indices()) == 0


class TestOdePackUnpack:
    def test_roundtrip(self, state_arrays: StateArrays) -> None:
        original_cash = state_arrays.cash[:3].copy()
        original_size = state_arrays.firm_size[:3].copy()
        original_growth = state_arrays.growth_rate[:3].copy()

        packed = state_arrays.pack_ode_state()
        assert packed.shape == (3 * ODE_VARS_PER_AGENT,)

        # Modify packed data
        packed *= 2.0
        state_arrays.unpack_ode_state(packed)

        np.testing.assert_array_almost_equal(state_arrays.cash[:3], original_cash * 2)
        np.testing.assert_array_almost_equal(state_arrays.firm_size[:3], original_size * 2)
        np.testing.assert_array_almost_equal(state_arrays.growth_rate[:3], original_growth * 2)

    def test_unpack_wrong_size_raises(self, state_arrays: StateArrays) -> None:
        with pytest.raises(ValueError, match="Expected ODE vector"):
            state_arrays.unpack_ode_state(np.zeros(999))

    def test_pack_respects_alive_mask(self, state_arrays: StateArrays) -> None:
        state_arrays.remove_company(1)
        packed = state_arrays.pack_ode_state()
        assert packed.shape == (2 * ODE_VARS_PER_AGENT,)


class TestSnapshotDict:
    def test_contains_expected_keys(self, state_arrays: StateArrays) -> None:
        snap = state_arrays.to_snapshot_dict()
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
            "signal_activation",
            "capital",
            "labor",
            "dept_headcount",
            "dept_budget",
            "alive",
            "consecutive_insolvent",
            "directives",
            "charter",
            "last_decision_tick",
            "decision_tier_counts",
            "agent_state_deltas",
        }
        assert set(snap.keys()) == expected_keys

    def test_snapshot_is_a_copy(self, state_arrays: StateArrays) -> None:
        snap = state_arrays.to_snapshot_dict()
        snap["cash"][0] = -999
        assert state_arrays.cash[0] != -999

    def test_n_active_matches(self, state_arrays: StateArrays) -> None:
        snap = state_arrays.to_snapshot_dict()
        assert snap["n_active"] == 3
        assert len(snap["company_names"]) == 3


class TestConstructor:
    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="max_capacity must be >= 1"):
            StateArrays(max_capacity=0)

    def test_default_capacity(self) -> None:
        sa = StateArrays()
        assert sa.max_capacity == 50
        assert sa.cash.shape == (50,)
        assert sa.dept_headcount.shape == (50, 12)
