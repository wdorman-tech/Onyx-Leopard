from __future__ import annotations

import numpy as np

from biosim.engine.state_manager import StateManager
from biosim.engine.tick import TickEngine
from biosim.types.config import BioConfig, SimConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    competition: bool = False,
    growth_division_threshold: float = 0.8,
    insolvent_ticks_to_death: int = 3,
) -> tuple[TickEngine, StateManager]:
    sim = SimConfig(
        growth_division_threshold=growth_division_threshold,
        insolvent_ticks_to_death=insolvent_ticks_to_death,
    )
    bio = BioConfig(competition=competition)
    sm = StateManager(sim)
    return TickEngine(bio, sim), sm


# ---------------------------------------------------------------------------
# Phase structure
# ---------------------------------------------------------------------------


class TestPhaseStructure:
    """Verify the 8 named phase methods exist and are called."""

    def test_all_phase_methods_exist(self) -> None:
        engine, _ = _make_engine()
        for name in (
            "_phase_sense",
            "_phase_solve_odes",
            "_phase_agent_decisions",
            "_phase_interactions",
            "_phase_growth_division",
            "_phase_env_update",
            "_phase_selection",
            "_phase_emit",
        ):
            assert hasattr(engine, name), f"Missing phase method: {name}"

    def test_step_calls_all_phases_in_order(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state

        call_order: list[str] = []
        original_methods = {}
        for name in (
            "_phase_sense",
            "_phase_solve_odes",
            "_phase_agent_decisions",
            "_phase_interactions",
            "_phase_growth_division",
            "_phase_env_update",
            "_phase_selection",
            "_phase_emit",
        ):
            original = getattr(engine, name)
            original_methods[name] = original

            def make_wrapper(n, orig):
                def wrapper(*a, **kw):
                    call_order.append(n)
                    return orig(*a, **kw)
                return wrapper

            setattr(engine, name, make_wrapper(name, original))

        engine.step(state)

        assert call_order == [
            "_phase_sense",
            "_phase_solve_odes",
            "_phase_agent_decisions",
            "_phase_interactions",
            "_phase_growth_division",
            "_phase_env_update",
            "_phase_selection",
            "_phase_emit",
        ]

    def test_tick_count_increments(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        assert engine._tick_count == 0
        engine.step(sm.state)
        assert engine._tick_count == 1
        engine.step(sm.state)
        assert engine._tick_count == 2


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Agent phases are no-ops when modules are absent."""

    def test_router_is_none_without_agent_config(self) -> None:
        engine, _ = _make_engine()
        assert engine._router is None

    def test_agent_decisions_returns_empty_without_router(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        indices = state.active_indices()
        decisions = engine._phase_agent_decisions(state, indices)
        assert decisions == []

    def test_step_succeeds_without_agent_modules(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("A", "#F00", size="large")
        sm.add_company("B", "#0F0", size="small")

        for _ in range(10):
            snapshot = engine.step(sm.state)

        assert snapshot["n_active"] >= 1
        assert all(r > 0 for r in snapshot["revenue"])


# ---------------------------------------------------------------------------
# Phase 1: Sense
# ---------------------------------------------------------------------------


class TestPhaseSense:
    def test_market_share_sums_to_one(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("A", "#F00", size="large")
        sm.add_company("B", "#0F0", size="small")
        state = sm.state
        indices = state.active_indices()

        engine._phase_sense(state, indices)

        total = state.market_share[indices].sum()
        assert abs(total - 1.0) < 1e-10

    def test_market_share_proportional_to_firm_size(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Big", "#F00", size="large")
        sm.add_company("Small", "#0F0", size="small")
        state = sm.state
        indices = state.active_indices()

        engine._phase_sense(state, indices)

        assert state.market_share[0] > state.market_share[1]


# ---------------------------------------------------------------------------
# Phase 5: Growth/Division
# ---------------------------------------------------------------------------


class TestPhaseGrowthDivision:
    def test_division_when_above_threshold(self) -> None:
        engine, sm = _make_engine(growth_division_threshold=0.5)
        sm.add_company("GrowCo", "#F00", size="large")
        state = sm.state
        state.firm_size[0] = 180.0
        state.carrying_capacity[0] = 200.0
        initial_labor = state.labor[0]
        indices = state.active_indices()

        engine._phase_growth_division(state, indices, [])

        assert state.labor[0] > initial_labor

    def test_no_division_when_below_threshold(self) -> None:
        engine, sm = _make_engine(growth_division_threshold=0.8)
        sm.add_company("SmallCo", "#F00", size="small")
        state = sm.state
        state.firm_size[0] = 5.0
        state.carrying_capacity[0] = 200.0
        initial_labor = state.labor[0]
        indices = state.active_indices()

        engine._phase_growth_division(state, indices, [])

        assert state.labor[0] == initial_labor


# ---------------------------------------------------------------------------
# Phase 7: Selection
# ---------------------------------------------------------------------------


class TestPhaseSelection:
    def test_insolvent_counter_increments(self) -> None:
        engine, sm = _make_engine(insolvent_ticks_to_death=5)
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.cash[0] = -1000
        indices = state.active_indices()

        engine._phase_selection(state, indices)

        assert state.consecutive_insolvent[0] == 1

    def test_insolvent_counter_resets_on_positive_cash(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.consecutive_insolvent[0] = 3
        state.cash[0] = 1000
        indices = state.active_indices()

        engine._phase_selection(state, indices)

        assert state.consecutive_insolvent[0] == 0

    def test_company_dies_after_sustained_insolvency(self) -> None:
        engine, sm = _make_engine(insolvent_ticks_to_death=2)
        sm.add_company("Doomed", "#F00", size="small")
        state = sm.state
        state.cash[0] = -1e6
        state.dept_headcount[0] = np.zeros(12)
        state.dept_headcount[0, 0] = 1.0
        state.labor[0] = 1.0

        for _ in range(20):
            indices = state.active_indices()
            if len(indices) == 0:
                break
            engine._phase_selection(state, indices)

        assert not state.alive[0]


# ---------------------------------------------------------------------------
# Apply decisions
# ---------------------------------------------------------------------------


class TestApplyDecisions:
    def test_hire_employees(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        initial_dept = state.dept_headcount[0, 3].copy()
        initial_labor = state.labor[0]

        decisions = [(0, 3, {"action": "hire_employees", "parameters": {"count": 5}})]
        engine._apply_decisions(state, decisions)

        assert state.dept_headcount[0, 3] == initial_dept + 5
        assert state.labor[0] == initial_labor + 5

    def test_hire_capped_at_20(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        initial_dept = state.dept_headcount[0, 0].copy()

        decisions = [(0, 0, {"action": "hire_employees", "parameters": {"count": 100}})]
        engine._apply_decisions(state, decisions)

        assert state.dept_headcount[0, 0] == initial_dept + 20

    def test_fire_employees(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.dept_headcount[0, 2] = 10.0
        state.labor[0] = state.dept_headcount[0].sum()
        initial_labor = state.labor[0]

        decisions = [(0, 2, {"action": "fire_employees", "parameters": {"count": 3}})]
        engine._apply_decisions(state, decisions)

        assert state.dept_headcount[0, 2] == 7.0
        assert state.labor[0] == initial_labor - 3

    def test_fire_capped_at_current_headcount(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.dept_headcount[0, 5] = 3.0

        decisions = [(0, 5, {"action": "fire_employees", "parameters": {"count": 100}})]
        engine._apply_decisions(state, decisions)

        assert state.dept_headcount[0, 5] == 0.0

    def test_adjust_budget(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.dept_budget[0, 1] = 10000.0

        decisions = [(0, 1, {"action": "adjust_budget", "parameters": {"delta_pct": 25}})]
        engine._apply_decisions(state, decisions)

        assert abs(state.dept_budget[0, 1] - 12500.0) < 0.01

    def test_adjust_budget_clamped(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.dept_budget[0, 1] = 10000.0

        decisions = [
            (0, 1, {"action": "adjust_budget", "parameters": {"delta_pct": 500}})
        ]
        engine._apply_decisions(state, decisions)

        # delta_pct clamped to 100 => budget doubles
        assert abs(state.dept_budget[0, 1] - 20000.0) < 0.01

    def test_invest_capacity(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.cash[0] = 100_000.0
        state.capital[0] = 500_000.0

        decisions = [
            (0, 0, {"action": "invest_capacity", "parameters": {"amount": 5000}})
        ]
        engine._apply_decisions(state, decisions)

        assert state.capital[0] == 505_000.0
        assert state.cash[0] == 95_000.0

    def test_invest_capacity_capped_at_10pct_cash(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        state.cash[0] = 10_000.0
        state.capital[0] = 500_000.0

        decisions = [
            (0, 0, {"action": "invest_capacity", "parameters": {"amount": 50000}})
        ]
        engine._apply_decisions(state, decisions)

        # Capped to 10% of 10k = 1k
        assert state.capital[0] == 501_000.0
        assert state.cash[0] == 9_000.0

    def test_empty_decisions_is_noop(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        labor_before = state.labor[0]

        engine._apply_decisions(state, [])

        assert state.labor[0] == labor_before

    def test_unknown_action_is_noop(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("Co", "#F00", size="medium")
        state = sm.state
        labor_before = state.labor[0]

        decisions = [(0, 0, {"action": "do_nothing", "parameters": {}})]
        engine._apply_decisions(state, decisions)

        assert state.labor[0] == labor_before


# ---------------------------------------------------------------------------
# StateManager decision logging
# ---------------------------------------------------------------------------


class TestStateManagerDecisionLog:
    def test_record_decisions_stores_entries(self) -> None:
        sm = StateManager(SimConfig())
        sm.tick_count = 5
        decisions = [
            (0, 3, {"action": "hire_employees", "parameters": {"count": 2}}),
            (1, 7, {"action": "adjust_budget", "parameters": {"delta_pct": 10}}),
        ]

        sm.record_decisions(decisions)

        assert len(sm.decision_log) == 2
        assert sm.decision_log[0]["tick"] == 5
        assert sm.decision_log[0]["company"] == 0
        assert sm.decision_log[0]["dept"] == 3
        assert sm.decision_log[0]["action"] == "hire_employees"
        assert sm.decision_log[1]["company"] == 1

    def test_decision_log_trims_to_500(self) -> None:
        sm = StateManager(SimConfig())
        # Fill with 498 entries first
        for i in range(498):
            sm.decision_log.append({"tick": i, "company": 0, "dept": 0})

        # Adding 5 more pushes past 500 => trim
        decisions = [(0, 0, {"action": "noop"}) for _ in range(5)]
        sm.record_decisions(decisions)

        assert len(sm.decision_log) == 500

    def test_empty_decision_log_on_init(self) -> None:
        sm = StateManager(SimConfig())
        assert sm.decision_log == []


# ---------------------------------------------------------------------------
# Backward compatibility — full step produces same observable behavior
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The 8-phase step should produce the same results as the old 5-phase for
    the standard case (no agent modules installed)."""

    def test_step_returns_valid_snapshot(self) -> None:
        engine, sm = _make_engine(competition=True)
        sm.add_company("A", "#F00", size="large")
        sm.add_company("B", "#0F0", size="medium")
        sm.add_company("C", "#00F", size="small")

        snapshot = engine.step(sm.state)

        assert "n_active" in snapshot
        assert "cash" in snapshot
        assert "firm_size" in snapshot
        assert "revenue" in snapshot
        assert snapshot["n_active"] == 3

    def test_multi_tick_stability(self) -> None:
        engine, sm = _make_engine(competition=True)
        sm.add_company("A", "#F00", size="large")
        sm.add_company("B", "#0F0", size="medium")

        for _ in range(50):
            snapshot = engine.step(sm.state)

        assert snapshot["n_active"] >= 1
        for size in snapshot["firm_size"]:
            assert size > 0

    def test_empty_state_returns_immediately(self) -> None:
        engine, sm = _make_engine()
        snapshot = engine.step(sm.state)
        assert snapshot["n_active"] == 0
        assert engine._tick_count == 1

    def test_health_score_bounded_0_1(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("A", "#F00", size="medium")

        engine.step(sm.state)

        indices = sm.state.active_indices()
        for idx in indices:
            assert 0.0 <= sm.state.health_score[idx] <= 1.0

    def test_capital_never_below_minimum(self) -> None:
        engine, sm = _make_engine()
        sm.add_company("A", "#F00", size="medium")

        for _ in range(20):
            engine.step(sm.state)

        indices = sm.state.active_indices()
        for idx in indices:
            assert sm.state.capital[idx] >= 1e3
