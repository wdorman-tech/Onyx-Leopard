"""Stub -- real implementation in Unit 3."""

import numpy as np


class BioSimModel:
    def __init__(self, bio_config=None, sim_config=None):
        self.tick_count = 0
        self._companies: list[dict] = []

    def add_company(self, name: str, color: str, **kwargs) -> None:
        self._companies.append({"name": name, "color": color})

    def step(self) -> dict:
        """Returns a state snapshot dict."""
        self.tick_count += 1
        n = len(self._companies)
        if n == 0:
            return {
                "n_active": 0,
                "indices": [],
                "company_names": [],
                "company_colors": [],
                "cash": [],
                "firm_size": [],
                "growth_rate": [],
                "revenue": [],
                "costs": [],
                "market_share": [],
                "health_score": [],
                "dept_headcount": [],
                "capital": [],
                "labor": [],
                "carrying_capacity": [],
            }

        rng = np.random.default_rng(self.tick_count)
        return {
            "n_active": n,
            "indices": list(range(n)),
            "company_names": [c["name"] for c in self._companies],
            "company_colors": [c["color"] for c in self._companies],
            "cash": (rng.uniform(5e5, 5e6, n)).tolist(),
            "firm_size": (rng.uniform(5, 50, n)).tolist(),
            "growth_rate": (rng.uniform(0.02, 0.08, n)).tolist(),
            "revenue": (rng.uniform(5e4, 5e5, n)).tolist(),
            "costs": (rng.uniform(3e4, 3e5, n)).tolist(),
            "market_share": (np.ones(n) / n).tolist(),
            "health_score": (rng.uniform(0.5, 1.0, n)).tolist(),
            "dept_headcount": (rng.uniform(2, 15, (n, 12))).tolist(),
            "capital": (rng.uniform(5e5, 5e6, n)).tolist(),
            "labor": (rng.uniform(25, 200, n)).tolist(),
            "carrying_capacity": (rng.uniform(50, 200, n)).tolist(),
        }
