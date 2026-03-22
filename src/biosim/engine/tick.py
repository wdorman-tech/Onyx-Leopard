import logging

import numpy as np

from biosim.math.competition import build_competition_matrix, step_competition
from biosim.math.growth import step_growth
from biosim.math.production import cobb_douglas
from biosim.types.config import BioConfig, SimConfig
from biosim.types.state import StateArrays

logger = logging.getLogger(__name__)


class TickEngine:
    """Phase 1 simplified tick loop with 5 phases."""

    def __init__(self, bio_config: BioConfig, sim_config: SimConfig) -> None:
        self.bio_config = bio_config
        self.sim_config = sim_config
        self._competition_matrix: np.ndarray | None = None

    def step(self, state: StateArrays) -> dict:
        """Execute one tick. Returns snapshot dict for GUI."""
        indices = state.active_indices()
        n = len(indices)
        if n == 0:
            return state.to_snapshot_dict()

        # Phase 1: Sense -- update market share
        total_size = state.firm_size[indices].sum()
        if total_size > 0:
            state.market_share[indices] = state.firm_size[indices] / total_size

        # Phase 2: Solve ODEs -- compute revenue via Cobb-Douglas, then step growth
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

        # Phase 3: Interact -- Lotka-Volterra competition
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

        # Phase 4: Grow/Die
        self._handle_growth_and_death(state, indices)

        # Phase 5: Emit -- update health score and capital
        state.health_score[indices] = np.clip(
            state.cash[indices] / np.maximum(state.fixed_costs[indices] * 12, 1.0), 0.0, 1.0
        )
        profit = state.revenue[indices] - state.costs[indices]
        state.capital[indices] += np.clip(profit * 0.1, -state.capital[indices] * 0.05, None)
        state.capital[indices] = np.maximum(state.capital[indices], 1e3)

        return state.to_snapshot_dict()

    def _handle_growth_and_death(self, state: StateArrays, indices: np.ndarray) -> None:
        """Cell division and death based on firm performance."""
        for idx in indices:
            ratio = state.firm_size[idx] / max(state.carrying_capacity[idx], 1e-10)
            if ratio > self.sim_config.growth_division_threshold:
                largest_dept = int(np.argmax(state.dept_headcount[idx]))
                state.dept_headcount[idx, largest_dept] += 1
                state.labor[idx] = state.dept_headcount[idx].sum()

            if state.cash[idx] < 0:
                state.consecutive_insolvent[idx] += 1
                if state.consecutive_insolvent[idx] >= self.sim_config.insolvent_ticks_to_death:
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
