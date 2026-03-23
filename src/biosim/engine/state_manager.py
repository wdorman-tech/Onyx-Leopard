from __future__ import annotations

from collections import deque

import numpy as np

from biosim.types.config import SimConfig
from biosim.types.state import StateArrays


class StateManager:
    """Manages the lifecycle of StateArrays."""

    def __init__(self, config: SimConfig, max_history: int = 200) -> None:
        self.config = config
        self.state = StateArrays(max_capacity=config.max_companies)
        self.tick_count: int = 0
        self.max_history: int = max_history
        self.history: deque[dict] = deque(maxlen=max_history)
        self.decision_log: list[dict] = []

    def add_company(
        self,
        name: str,
        color: str,
        industry: str = "generic",
        size: str = "medium",
    ) -> int:
        """Add a company with sensible defaults based on industry/size."""
        params = self._default_params(industry, size)
        return self.state.add_company(name, color, params)

    def _default_params(self, industry: str, size: str) -> dict:
        """Generate initial parameters based on industry and company size."""
        size_map = {
            "small": {
                "cash": 5e5,
                "firm_size": 5.0,
                "capital": 5e5,
                "labor": 25.0,
                "carrying_capacity": 50.0,
                "fixed_costs": 2.5e4,
            },
            "medium": {
                "cash": 1e6,
                "firm_size": 15.0,
                "capital": 1e6,
                "labor": 75.0,
                "carrying_capacity": 100.0,
                "fixed_costs": 7.5e4,
            },
            "large": {
                "cash": 5e6,
                "firm_size": 40.0,
                "capital": 5e6,
                "labor": 200.0,
                "carrying_capacity": 200.0,
                "fixed_costs": 2e5,
            },
        }
        params = size_map.get(size, size_map["medium"]).copy()
        params["growth_rate"] = 0.05
        params["variable_cost_rate"] = 1000.0
        params["tfp"] = 1.0
        params["alpha_prod"] = 0.3
        params["beta_prod"] = 0.7

        total_labor = params["labor"]
        weights = np.array([0.1, 0.1, 0.05, 0.15, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.05, 0.05])
        params["dept_headcount"] = (weights * total_labor).astype(float)
        params["dept_budget"] = params["dept_headcount"] * 5000
        return params

    def record_snapshot(self) -> dict:
        """Record current state to history and return snapshot dict."""
        snapshot = self.state.to_snapshot_dict()
        snapshot["tick"] = self.tick_count
        self.history.append(snapshot)
        return snapshot

    def record_decisions(self, decisions: list[tuple[int, int, dict]]) -> None:
        """Store decisions from current tick for dashboard display."""
        for company_idx, dept_idx, result in decisions:
            self.decision_log.append({
                "tick": self.tick_count,
                "company": company_idx,
                "dept": dept_idx,
                **result,
            })
        if len(self.decision_log) > 500:
            self.decision_log = self.decision_log[-500:]

    def get_history(self) -> list[dict]:
        return list(self.history)
