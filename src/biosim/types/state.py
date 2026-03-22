from __future__ import annotations

import numpy as np

NUM_DEPARTMENTS = 12
# ODE state per company: cash + firm_size + growth_rate
ODE_VARS_PER_AGENT = 3


class StateArrays:
    """Pre-allocated numpy arrays for all simulation state.

    Uses __slots__ to avoid per-instance dict overhead on the hot path.
    All agent-level arrays are sized to max_capacity; only indices where
    alive[i] is True hold meaningful data.
    """

    __slots__ = (
        "max_capacity",
        "n_active",
        "company_names",
        "company_colors",
        # Scalar arrays (max_capacity,)
        "cash",
        "firm_size",
        "growth_rate",
        "revenue",
        "costs",
        "market_share",
        "health_score",
        "carrying_capacity",
        "signal_activation",
        # Production params (max_capacity,)
        "tfp",
        "capital",
        "labor",
        "alpha_prod",
        "beta_prod",
        "fixed_costs",
        "variable_cost_rate",
        # Per-department arrays (max_capacity, 12)
        "dept_headcount",
        "dept_budget",
        # ODE state arrays
        "cell_cycle",
        "apoptosis",
        "mapk",
        # Boolean masks
        "alive",
        "apoptosis_triggered",
        # Counters
        "consecutive_insolvent",
    )

    def __init__(self, max_capacity: int = 50) -> None:
        if max_capacity < 1:
            raise ValueError(f"max_capacity must be >= 1, got {max_capacity}")

        self.max_capacity = max_capacity
        self.n_active: int = 0
        self.company_names: list[str] = [""] * max_capacity
        self.company_colors: list[str] = [""] * max_capacity

        # Scalar arrays
        self.cash = np.zeros(max_capacity, dtype=np.float64)
        self.firm_size = np.zeros(max_capacity, dtype=np.float64)
        self.growth_rate = np.zeros(max_capacity, dtype=np.float64)
        self.revenue = np.zeros(max_capacity, dtype=np.float64)
        self.costs = np.zeros(max_capacity, dtype=np.float64)
        self.market_share = np.zeros(max_capacity, dtype=np.float64)
        self.health_score = np.zeros(max_capacity, dtype=np.float64)
        self.carrying_capacity = np.zeros(max_capacity, dtype=np.float64)
        self.signal_activation = np.zeros(max_capacity, dtype=np.float64)

        # Production parameters
        self.tfp = np.ones(max_capacity, dtype=np.float64)
        self.capital = np.zeros(max_capacity, dtype=np.float64)
        self.labor = np.zeros(max_capacity, dtype=np.float64)
        self.alpha_prod = np.full(max_capacity, 0.3, dtype=np.float64)
        self.beta_prod = np.full(max_capacity, 0.7, dtype=np.float64)
        self.fixed_costs = np.zeros(max_capacity, dtype=np.float64)
        self.variable_cost_rate = np.zeros(max_capacity, dtype=np.float64)

        # Per-department arrays
        self.dept_headcount = np.zeros((max_capacity, NUM_DEPARTMENTS), dtype=np.float64)
        self.dept_budget = np.zeros((max_capacity, NUM_DEPARTMENTS), dtype=np.float64)

        # ODE state: cell_cycle [cyclin, cdk_active], apoptosis [bax, bcl2],
        # mapk [3 tiers x active/inactive]
        self.cell_cycle = np.zeros((max_capacity, 2), dtype=np.float64)
        self.apoptosis = np.zeros((max_capacity, 2), dtype=np.float64)
        self.mapk = np.zeros((max_capacity, 6), dtype=np.float64)

        # Masks
        self.alive = np.zeros(max_capacity, dtype=np.bool_)
        self.apoptosis_triggered = np.zeros(max_capacity, dtype=np.bool_)

        # Counters
        self.consecutive_insolvent = np.zeros(max_capacity, dtype=np.int64)

    def add_company(self, name: str, color: str, initial_params: dict) -> int:
        """Allocate the next dead slot for a new company. Returns its index."""
        dead_slots = np.where(~self.alive)[0]
        if len(dead_slots) == 0:
            raise RuntimeError(
                f"No free slots — all {self.max_capacity} are occupied. "
                "Increase max_capacity or remove a company first."
            )

        idx = int(dead_slots[0])
        self.alive[idx] = True
        self.company_names[idx] = name
        self.company_colors[idx] = color

        # Apply initial params to arrays
        for key, value in initial_params.items():
            arr = getattr(self, key, None)
            if arr is None:
                raise ValueError(f"Unknown parameter '{key}' — not a StateArrays field")
            if isinstance(arr, np.ndarray) and arr.shape[0] == self.max_capacity:
                if arr.ndim == 1:
                    arr[idx] = value
                elif arr.ndim == 2:
                    arr[idx] = value
            else:
                raise ValueError(
                    f"Parameter '{key}' is not a valid agent array — "
                    "check shape matches max_capacity"
                )

        self.n_active = int(np.count_nonzero(self.alive))
        return idx

    def remove_company(self, index: int) -> None:
        """Mark a slot as dead and zero its data."""
        if not 0 <= index < self.max_capacity:
            raise IndexError(f"Index {index} out of range [0, {self.max_capacity})")
        if not self.alive[index]:
            raise ValueError(f"Slot {index} is already dead")

        self.alive[index] = False
        self.company_names[index] = ""
        self.company_colors[index] = ""

        # Zero all numeric arrays at this index
        for attr in self.__slots__:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray) and val.shape[0] == self.max_capacity:
                if val.ndim == 1:
                    val[index] = 0
                else:
                    val[index, :] = 0

        self.n_active = int(np.count_nonzero(self.alive))

    def active_indices(self) -> np.ndarray:
        """Return indices where alive is True."""
        return np.where(self.alive)[0]

    def pack_ode_state(self) -> np.ndarray:
        """Flatten [cash, firm_size, growth_rate] for active agents into a 1D vector."""
        idx = self.active_indices()
        return np.concatenate([self.cash[idx], self.firm_size[idx], self.growth_rate[idx]])

    def unpack_ode_state(self, y: np.ndarray) -> None:
        """Write solved ODE state back from a flat vector into the arrays."""
        idx = self.active_indices()
        n = len(idx)
        if y.shape != (n * ODE_VARS_PER_AGENT,):
            raise ValueError(
                f"Expected ODE vector of length {n * ODE_VARS_PER_AGENT}, got {y.shape[0]}"
            )
        self.cash[idx] = y[:n]
        self.firm_size[idx] = y[n : 2 * n]
        self.growth_rate[idx] = y[2 * n : 3 * n]

    def to_snapshot_dict(self) -> dict:
        """Thread-safe snapshot of current state for GUI consumption.

        Fancy-indexed arrays are already copies; .tolist() converts to pure Python.
        """
        idx = self.active_indices()
        return {
            "n_active": self.n_active,
            "indices": idx.tolist(),
            "company_names": [self.company_names[i] for i in idx],
            "company_colors": [self.company_colors[i] for i in idx],
            "cash": self.cash[idx].tolist(),
            "firm_size": self.firm_size[idx].tolist(),
            "growth_rate": self.growth_rate[idx].tolist(),
            "revenue": self.revenue[idx].tolist(),
            "costs": self.costs[idx].tolist(),
            "market_share": self.market_share[idx].tolist(),
            "health_score": self.health_score[idx].tolist(),
            "carrying_capacity": self.carrying_capacity[idx].tolist(),
            "signal_activation": self.signal_activation[idx].tolist(),
            "capital": self.capital[idx].tolist(),
            "labor": self.labor[idx].tolist(),
            "dept_headcount": self.dept_headcount[idx].tolist(),
            "dept_budget": self.dept_budget[idx].tolist(),
            "alive": self.alive[idx].tolist(),
            "consecutive_insolvent": self.consecutive_insolvent[idx].tolist(),
        }
