from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

NUM_DEPARTMENTS = 12


class _Tier:
    """Self-contained tier constants (avoids dependency on enums.py changes)."""

    ODE = 0
    HEURISTIC = 1
    HAIKU = 2
    SONNET = 3


class LLMBackend(Protocol):
    """Interface that the CAMEL-AI agent framework implements."""

    async def call(self, dept_index: int, company_index: int, context: dict) -> dict: ...


class DecisionRouter:
    """Four-tier hybrid decision engine.

    Classifies each agent's situation per tick and routes to the appropriate tier.
    Tier 0 (ODE) and Tier 1 (heuristic) execute synchronously.
    Tier 2 (Haiku) and Tier 3 (Sonnet) are batched for async LLM calls.
    """

    __slots__ = (
        "_novelty_threshold",
        "_cost_budget",
        "_llm_backend",
        "_total_cost",
        "_last_state",
    )

    # Department LLM frequencies (ticks between calls) per PRD
    _LLM_FREQUENCIES: dict[int, int] = {
        0: 4,   # Finance
        1: 8,   # R&D
        2: 8,   # Distribution
        3: 4,   # Production
        4: 2,   # Sales
        5: 4,   # Marketing
        6: 4,   # HR
        7: 12,  # Executive
        8: 4,   # Customer Service
        9: 12,  # Legal
        10: 8,  # IT
        11: 4,  # Procurement
    }

    def __init__(
        self,
        novelty_threshold: float = 0.15,
        cost_budget: float = 5.0,
        llm_backend: LLMBackend | None = None,
    ) -> None:
        self._novelty_threshold = novelty_threshold
        self._cost_budget = cost_budget
        self._llm_backend = llm_backend
        self._total_cost: float = 0.0
        # Keyed by (company_index, dept_index) -> last state vector
        self._last_state: dict[tuple[int, int], np.ndarray] = {}

    def classify_tier(
        self,
        company_index: int,
        dept_index: int,
        current_tick: int,
        state_vector: np.ndarray,
        has_injected_event: bool = False,
    ) -> int:
        """Determine which tier a decision should route to.

        Algorithm:
        1. If not on frequency schedule for this dept, return Tier 0 (ODE).
        2. Compute novelty = L2 norm of state delta / magnitude of previous state.
        3. If event injected, escalate: Sonnet if very novel, else Haiku.
        4. Route by novelty thresholds.
        5. Budget-aware downgrades.
        """
        freq = self._LLM_FREQUENCIES.get(dept_index, 4)
        if current_tick % freq != 0:
            return _Tier.ODE

        key = (company_index, dept_index)
        prev = self._last_state.get(key)
        self._last_state[key] = state_vector.copy()

        if prev is None:
            # First tick for this agent -- no delta, default to heuristic
            novelty = 0.0
        else:
            delta = state_vector - prev
            prev_magnitude = np.linalg.norm(prev)
            if prev_magnitude < 1e-12:
                novelty = float(np.linalg.norm(delta))
            else:
                novelty = float(np.linalg.norm(delta) / prev_magnitude)

        threshold = self._novelty_threshold

        if has_injected_event:
            if novelty >= threshold:
                return self._budget_check(_Tier.SONNET)
            return self._budget_check(_Tier.HAIKU)

        if novelty < 0.5 * threshold:
            return _Tier.ODE
        if novelty < threshold:
            return _Tier.HEURISTIC
        if novelty < 2.0 * threshold:
            return self._budget_check(_Tier.HAIKU)
        return self._budget_check(_Tier.SONNET)

    def _budget_check(self, desired_tier: int) -> int:
        """Downgrade tier if cost budget is running low."""
        budget_used_pct = self._total_cost / max(self._cost_budget, 1e-12)

        if desired_tier == _Tier.SONNET and budget_used_pct > 0.8:
            logger.debug("Budget >80%%, downgrading Sonnet -> Haiku")
            desired_tier = _Tier.HAIKU

        if desired_tier == _Tier.HAIKU and budget_used_pct > 1.0:
            logger.debug("Budget exceeded, downgrading Haiku -> Heuristic")
            desired_tier = _Tier.HEURISTIC

        return desired_tier

    def route_decisions(
        self,
        company_indices: list[int],
        current_tick: int,
        state_snapshot: dict,
    ) -> dict[int, list[tuple[int, int]]]:
        """Classify all agents for one tick.

        Returns: {tier: [(company_idx, dept_idx), ...]}
        Builds a state vector per (company, dept) from snapshot for novelty computation.
        """
        result: dict[int, list[tuple[int, int]]] = {
            _Tier.ODE: [],
            _Tier.HEURISTIC: [],
            _Tier.HAIKU: [],
            _Tier.SONNET: [],
        }

        snapshot_indices = state_snapshot.get("indices", [])

        for ci in company_indices:
            if ci not in snapshot_indices:
                continue
            pos = snapshot_indices.index(ci)

            for dept_idx in range(NUM_DEPARTMENTS):
                sv = self._build_state_vector(state_snapshot, pos, dept_idx)
                tier = self.classify_tier(ci, dept_idx, current_tick, sv)
                result[tier].append((ci, dept_idx))

        return result

    def _build_state_vector(
        self, snapshot: dict, pos: int, dept_idx: int
    ) -> np.ndarray:
        """Build a state vector for novelty computation from snapshot data."""
        components: list[float] = []

        for key in ("cash", "firm_size", "revenue", "costs", "market_share", "health_score"):
            vals = snapshot.get(key, [])
            components.append(float(vals[pos]) if pos < len(vals) else 0.0)

        # Per-department data
        for key in ("dept_headcount", "dept_budget"):
            vals = snapshot.get(key, [])
            if pos < len(vals) and isinstance(vals[pos], list) and dept_idx < len(vals[pos]):
                components.append(float(vals[pos][dept_idx]))
            else:
                components.append(0.0)

        return np.array(components, dtype=np.float64)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def record_cost(self, cost_usd: float) -> None:
        self._total_cost += cost_usd
