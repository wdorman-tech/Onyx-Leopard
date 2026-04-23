from __future__ import annotations

import logging

import numpy as np

from biosim.math.competition import build_competition_matrix, step_competition
from biosim.math.growth import step_growth
from biosim.math.production import cobb_douglas
from biosim.types.config import BioConfig, SimConfig
from biosim.types.state import StateArrays

logger = logging.getLogger(__name__)

# Conditional imports — these modules may not exist yet
try:
    from biosim.engine.decision_router import DecisionRouter

    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False

try:
    from biosim.agents.llm_backend import CamelLLMBackend

    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from biosim.engine.heuristics import HEURISTIC_REGISTRY

    _HEURISTICS_AVAILABLE = True
except ImportError:
    _HEURISTICS_AVAILABLE = False


class TickEngine:
    """8-phase tick loop. Gracefully degrades to 5-phase when agent modules are absent."""

    def __init__(self, bio_config: BioConfig, sim_config: SimConfig) -> None:
        self.bio_config = bio_config
        self.sim_config = sim_config
        self._competition_matrix: np.ndarray | None = None
        self._tick_count = 0

        # Initialize decision router if available AND enabled
        self._router = None
        self._llm_backend = None

        if _ROUTER_AVAILABLE:
            agent_cfg = getattr(sim_config, "agent_config", None)
            if agent_cfg and getattr(agent_cfg, "enabled", False):
                self._router = DecisionRouter(
                    novelty_threshold=getattr(agent_cfg, "novelty_threshold", 0.15),
                    cost_budget=getattr(agent_cfg, "cost_budget_per_run", 5.0),
                )
                if _LLM_AVAILABLE:
                    self._llm_backend = CamelLLMBackend()

    def step(self, state: StateArrays) -> dict:
        """Execute one tick with up to 8 phases."""
        self._tick_count += 1
        indices = state.active_indices()
        n = len(indices)
        if n == 0:
            return state.to_snapshot_dict()

        # Phase 1: Sense
        self._phase_sense(state, indices)

        # Phase 2: Solve Internal ODEs
        self._phase_solve_odes(state, indices)

        # Phase 3: Agent Decisions (no-op if router unavailable)
        decisions = self._phase_agent_decisions(state, indices)

        # Phase 4: Interactions (Lotka-Volterra)
        self._phase_interactions(state, indices, n)

        # Phase 5: Growth/Division
        self._phase_growth_division(state, indices, decisions)

        # Phase 6: Environment Update (placeholder)
        self._phase_env_update(state, indices)

        # Phase 7: Selection (insolvency removal)
        self._phase_selection(state, indices)

        # Phase 8: Emit
        return self._phase_emit(state, indices)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_sense(self, state: StateArrays, indices: np.ndarray) -> None:
        """Phase 1: Update market share from firm sizes."""
        total_size = state.firm_size[indices].sum()
        if total_size > 0:
            state.market_share[indices] = state.firm_size[indices] / total_size

    def _phase_solve_odes(self, state: StateArrays, indices: np.ndarray) -> None:
        """Phase 2: Cobb-Douglas revenue + growth ODE step."""
        state.revenue[indices] = cobb_douglas(
            state.tfp[indices],
            state.capital[indices],
            state.labor[indices],
            state.alpha_prod[indices],
            state.beta_prod[indices],
        )
        variable_costs = state.variable_cost_rate[indices] * state.firm_size[indices]
        state.costs[indices] = state.fixed_costs[indices] + variable_costs

        new_size, new_cash, new_gr = step_growth(
            state.firm_size[indices],
            state.cash[indices],
            state.growth_rate[indices],
            state.carrying_capacity[indices],
            state.revenue[indices],
            state.fixed_costs[indices],
            state.variable_cost_rate[indices],
        )
        state.firm_size[indices] = np.maximum(new_size, 0.01)
        state.cash[indices] = new_cash
        state.growth_rate[indices] = new_gr

    def _phase_agent_decisions(
        self, state: StateArrays, indices: np.ndarray
    ) -> list[tuple[int, int, dict]]:
        """Phase 3: Route agent decisions through the 4-tier system.

        Returns empty list if router unavailable (graceful degradation).
        """
        if self._router is None:
            return []

        decisions: list[tuple[int, int, dict]] = []
        snapshot = state.to_snapshot_dict()

        routing = self._router.route_decisions(
            company_indices=indices.tolist(),
            current_tick=self._tick_count,
            state_snapshot=snapshot,
        )

        # Tier 0: Already handled by Phase 2 (ODE output) — skip

        # Tier 1: Execute heuristics synchronously
        if _HEURISTICS_AVAILABLE:
            for company_idx, dept_idx in routing.get(1, []):
                heuristic_fn = HEURISTIC_REGISTRY.get(dept_idx)
                if heuristic_fn:
                    result = heuristic_fn(snapshot, company_idx)
                    decisions.append((company_idx, dept_idx, result))

        # Tier 2/3: LLM calls (fallback to heuristic if unavailable)
        for tier in (2, 3):
            for company_idx, dept_idx in routing.get(tier, []):
                if _HEURISTICS_AVAILABLE:
                    heuristic_fn = HEURISTIC_REGISTRY.get(dept_idx)
                    if heuristic_fn:
                        result = heuristic_fn(snapshot, company_idx)
                        decisions.append((company_idx, dept_idx, result))

        return decisions

    def _phase_interactions(
        self, state: StateArrays, indices: np.ndarray, n: int
    ) -> None:
        """Phase 4: Lotka-Volterra competition between firms."""
        if self.bio_config.competition and n > 1:
            if self._competition_matrix is None or self._competition_matrix.shape[0] != n:
                self._competition_matrix = build_competition_matrix(n)
            new_pop = step_competition(
                state.firm_size[indices],
                state.growth_rate[indices],
                state.carrying_capacity[indices],
                self._competition_matrix,
            )
            state.firm_size[indices] = np.maximum(new_pop, 0.01)

    def _phase_growth_division(
        self,
        state: StateArrays,
        indices: np.ndarray,
        decisions: list[tuple[int, int, dict]],
    ) -> None:
        """Phase 5: Cell division based on firm performance + apply agent decisions."""
        for idx in indices:
            ratio = state.firm_size[idx] / max(state.carrying_capacity[idx], 1e-10)
            if ratio > self.sim_config.growth_division_threshold:
                largest_dept = int(np.argmax(state.dept_headcount[idx]))
                state.dept_headcount[idx, largest_dept] += 1
                state.labor[idx] = state.dept_headcount[idx].sum()

        self._apply_decisions(state, decisions)

    def _apply_decisions(
        self, state: StateArrays, decisions: list[tuple[int, int, dict]]
    ) -> None:
        """Apply structured decisions from agents to state arrays."""
        for company_idx, dept_idx, result in decisions:
            action = result.get("action", "")
            params = result.get("parameters", {})
            if action == "hire_employees":
                count = min(int(params.get("count", 0)), 20)
                if count > 0:
                    state.dept_headcount[company_idx, dept_idx] += count
                    state.labor[company_idx] = state.dept_headcount[company_idx].sum()
            elif action == "fire_employees":
                count = min(
                    int(params.get("count", 0)),
                    int(state.dept_headcount[company_idx, dept_idx]),
                )
                if count > 0:
                    state.dept_headcount[company_idx, dept_idx] -= count
                    state.labor[company_idx] = state.dept_headcount[company_idx].sum()
            elif action == "adjust_budget":
                delta_pct = max(-50, min(100, params.get("delta_pct", 0)))
                target_dept = params.get("dept", dept_idx)
                state.dept_budget[company_idx, target_dept] *= 1 + delta_pct / 100
            elif action == "invest_capacity":
                amount = min(params.get("amount", 0), state.cash[company_idx] * 0.1)
                if amount > 0:
                    state.capital[company_idx] += amount
                    state.cash[company_idx] -= amount

    def _phase_env_update(self, state: StateArrays, indices: np.ndarray) -> None:
        """Phase 6: Placeholder for nutrient diffusion PDE. No-op for now."""

    def _phase_selection(self, state: StateArrays, indices: np.ndarray) -> None:
        """Phase 7: Remove organisms insolvent for n consecutive ticks."""
        for idx in indices:
            if state.cash[idx] < 0:
                state.consecutive_insolvent[idx] += 1
                if (
                    state.consecutive_insolvent[idx]
                    >= self.sim_config.insolvent_ticks_to_death
                ):
                    nonzero = np.where(state.dept_headcount[idx] > 0)[0]
                    if len(nonzero) > 0:
                        smallest = nonzero[np.argmin(state.dept_headcount[idx, nonzero])]
                        state.dept_headcount[idx, smallest] = max(
                            0, state.dept_headcount[idx, smallest] - 1
                        )
                        state.labor[idx] = state.dept_headcount[idx].sum()
                    if state.dept_headcount[idx].sum() <= 0:
                        state.remove_company(idx)
                        logger.info("Company at index %d died (insolvent)", idx)
            else:
                state.consecutive_insolvent[idx] = 0

    def _phase_emit(self, state: StateArrays, indices: np.ndarray) -> dict:
        """Phase 8: Update health score and capital, return snapshot."""
        state.health_score[indices] = np.clip(
            state.cash[indices] / np.maximum(state.fixed_costs[indices] * 12, 1.0),
            0.0,
            1.0,
        )
        profit = state.revenue[indices] - state.costs[indices]
        state.capital[indices] += np.clip(
            profit * 0.1, -state.capital[indices] * 0.05, None
        )
        state.capital[indices] = np.maximum(state.capital[indices], 1e3)
        return state.to_snapshot_dict()
