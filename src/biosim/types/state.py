import numpy as np


class StateArrays:
    """Column-oriented numpy state container. All arrays pre-allocated at max_capacity."""

    __slots__ = (
        "max_capacity",
        "n_active",
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
        "alive",
        "consecutive_insolvent",
        "tfp",
        "capital",
        "labor",
        "alpha_prod",
        "beta_prod",
        "fixed_costs",
        "variable_cost_rate",
    )

    def __init__(self, max_capacity: int = 50) -> None:
        self.max_capacity = max_capacity
        self.n_active = 0
        self.company_names: list[str] = []
        self.company_colors: list[str] = []
        # Scalars per agent
        self.cash = np.zeros(max_capacity)
        self.firm_size = np.zeros(max_capacity)
        self.growth_rate = np.zeros(max_capacity)
        self.revenue = np.zeros(max_capacity)
        self.costs = np.zeros(max_capacity)
        self.market_share = np.zeros(max_capacity)
        self.health_score = np.ones(max_capacity)
        self.carrying_capacity = np.full(max_capacity, 100.0)
        # Production params
        self.tfp = np.ones(max_capacity)
        self.capital = np.zeros(max_capacity)
        self.labor = np.zeros(max_capacity)
        self.alpha_prod = np.full(max_capacity, 0.3)
        self.beta_prod = np.full(max_capacity, 0.7)
        self.fixed_costs = np.zeros(max_capacity)
        self.variable_cost_rate = np.zeros(max_capacity)
        # Per-department (12 departments)
        self.dept_headcount = np.zeros((max_capacity, 12))
        self.dept_budget = np.zeros((max_capacity, 12))
        # Masks
        self.alive = np.zeros(max_capacity, dtype=bool)
        self.consecutive_insolvent = np.zeros(max_capacity, dtype=int)

    def add_company(self, name: str, color: str, params: dict) -> int:
        """Add a company, returns its index."""
        if self.n_active >= self.max_capacity:
            raise ValueError("Max capacity reached")
        idx = self.n_active
        self.alive[idx] = True
        self.company_names.append(name)
        self.company_colors.append(color)
        for key, value in params.items():
            arr = getattr(self, key, None)
            if arr is None or not isinstance(arr, np.ndarray):
                raise ValueError(f"Unknown or non-array param: {key!r}")
            arr[idx] = value
        self.n_active += 1
        return idx

    def remove_company(self, idx: int) -> None:
        self.alive[idx] = False

    def active_indices(self) -> np.ndarray:
        return np.where(self.alive)[0]

    def active_mask(self) -> np.ndarray:
        return self.alive[: self.n_active]

    def to_snapshot_dict(self) -> dict:
        """Thread-safe serialization for GUI consumption."""
        mask = self.alive[: self.n_active]
        indices = np.where(mask)[0]
        return {
            "n_active": int(len(indices)),
            "indices": indices.tolist(),
            "company_names": [self.company_names[i] for i in indices],
            "company_colors": [self.company_colors[i] for i in indices],
            "cash": self.cash[indices].copy().tolist(),
            "firm_size": self.firm_size[indices].copy().tolist(),
            "growth_rate": self.growth_rate[indices].copy().tolist(),
            "revenue": self.revenue[indices].copy().tolist(),
            "costs": self.costs[indices].copy().tolist(),
            "market_share": self.market_share[indices].copy().tolist(),
            "health_score": self.health_score[indices].copy().tolist(),
            "carrying_capacity": self.carrying_capacity[indices].copy().tolist(),
            "dept_headcount": self.dept_headcount[indices].copy().tolist(),
            "dept_budget": self.dept_budget[indices].copy().tolist(),
            "capital": self.capital[indices].copy().tolist(),
            "labor": self.labor[indices].copy().tolist(),
        }
