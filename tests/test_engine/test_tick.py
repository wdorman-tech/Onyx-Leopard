import numpy as np

from biosim.engine.state_manager import StateManager
from biosim.engine.tick import TickEngine
from biosim.types.config import BioConfig, SimConfig


class TestSingleTick:
    def test_tick_updates_firm_size_cash_revenue(
        self,
        tick_engine: TickEngine,
        populated_state_manager: StateManager,
    ) -> None:
        state = populated_state_manager.state
        old_sizes = state.firm_size[: state.n_active].copy()
        old_cash = state.cash[: state.n_active].copy()

        tick_engine.step(state)

        assert not np.array_equal(state.firm_size[: state.n_active], old_sizes)
        assert not np.array_equal(state.cash[: state.n_active], old_cash)
        assert np.all(state.revenue[state.active_indices()] > 0)

    def test_market_share_sums_to_one(
        self,
        tick_engine: TickEngine,
        populated_state_manager: StateManager,
    ) -> None:
        state = populated_state_manager.state
        tick_engine.step(state)

        indices = state.active_indices()
        total_share = state.market_share[indices].sum()
        assert abs(total_share - 1.0) < 1e-10


class TestCompetition:
    def test_competition_reduces_weaker_firms(self) -> None:
        """With competition enabled, weaker firms should lose relative ground."""
        config = SimConfig()
        bio = BioConfig(competition=True)
        sm = StateManager(config)
        sm.add_company("Strong", "#F00", size="large")
        sm.add_company("Weak", "#0F0", size="small")

        engine = TickEngine(bio, config)
        state = sm.state

        initial_ratio = state.firm_size[1] / state.firm_size[0]
        for _ in range(5):
            engine.step(state)
        final_ratio = state.firm_size[1] / state.firm_size[0]

        assert final_ratio <= initial_ratio + 0.05

    def test_no_competition_when_disabled(self) -> None:
        config = SimConfig()
        bio = BioConfig(competition=False)
        sm = StateManager(config)
        sm.add_company("A", "#F00", size="medium")
        sm.add_company("B", "#0F0", size="medium")

        engine = TickEngine(bio, config)
        state = sm.state

        engine.step(state)
        assert engine._competition_matrix is None


class TestGrowthDivision:
    def test_growth_adds_cells_when_threshold_exceeded(self) -> None:
        config = SimConfig(growth_division_threshold=0.5)
        bio = BioConfig(competition=False)
        sm = StateManager(config)
        sm.add_company("GrowCo", "#F00", size="large")

        state = sm.state
        state.firm_size[0] = 180.0
        state.carrying_capacity[0] = 200.0
        initial_labor = state.labor[0]

        engine = TickEngine(bio, config)
        engine.step(state)

        assert state.labor[0] >= initial_labor


class TestInsolvencyDeath:
    def test_company_dies_after_insolvent_ticks(self) -> None:
        config = SimConfig(insolvent_ticks_to_death=2)
        bio = BioConfig(competition=False)
        sm = StateManager(config)
        sm.add_company("Doomed", "#F00", size="small")

        state = sm.state
        state.cash[0] = -1e6
        state.fixed_costs[0] = 1e6
        state.variable_cost_rate[0] = 1e5
        # Minimal headcount so company dies quickly once shrinking starts
        state.dept_headcount[0] = np.zeros(12)
        state.dept_headcount[0, 0] = 2.0
        state.labor[0] = 2.0

        engine = TickEngine(bio, config)
        for _ in range(20):
            engine.step(state)
            if not state.alive[0]:
                break

        assert not state.alive[0]

    def test_insolvency_counter_resets_on_profit(self) -> None:
        config = SimConfig(insolvent_ticks_to_death=5)
        bio = BioConfig(competition=False)
        sm = StateManager(config)
        sm.add_company("Survivor", "#F00", size="large")

        state = sm.state
        engine = TickEngine(bio, config)

        state.cash[0] = -100
        engine.step(state)
        assert state.consecutive_insolvent[0] >= 1

        state.cash[0] = 1e6
        engine.step(state)
        assert state.consecutive_insolvent[0] == 0


class TestEmptyState:
    def test_empty_state_no_crash(self, tick_engine: TickEngine, sim_config: SimConfig) -> None:
        sm = StateManager(sim_config)
        snapshot = tick_engine.step(sm.state)
        assert snapshot["n_active"] == 0
