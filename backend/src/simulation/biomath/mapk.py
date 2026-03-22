from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# Kinetic parameters for the 3-tier MAPK cascade
# Each tier: inactive <-> active with Michaelis-Menten kinetics
K_M = 0.1  # Michaelis constant
V_MAX_FWD = 1.0  # max forward (activation) rate
V_MAX_REV = 0.3  # max reverse (deactivation) rate
TOTAL_PER_TIER = 1.0  # total protein per tier (active + inactive = 1)


def mapk_cascade_ode(
    t: float,
    y: np.ndarray,
    signal: float,
) -> list[float]:
    """6-variable MAPK cascade ODE.

    y = [MAPKKK*, MAPKKK, MAPKK*, MAPKK, MAPK*, MAPK]
    where * = active form.

    Tier 1 (executive): activated by external signal
    Tier 2 (management): activated by tier 1 active
    Tier 3 (operational): activated by tier 2 active

    Produces ultrasensitivity: effective Hill coefficient ~4-5.
    """
    mapkkk_a, mapkkk_i = float(y[0]), float(y[1])
    mapkk_a, mapkk_i = float(y[2]), float(y[3])
    mapk_a, mapk_i = float(y[4]), float(y[5])

    def mm_activate(substrate: float, enzyme: float) -> float:
        return V_MAX_FWD * enzyme * substrate / (K_M + substrate + 1e-9)

    def mm_deactivate(substrate: float) -> float:
        return V_MAX_REV * substrate / (K_M + substrate + 1e-9)

    # Tier 1: signal -> MAPKKK activation
    d_mapkkk_a = mm_activate(mapkkk_i, signal) - mm_deactivate(mapkkk_a)
    d_mapkkk_i = -d_mapkkk_a

    # Tier 2: MAPKKK* -> MAPKK activation
    d_mapkk_a = mm_activate(mapkk_i, mapkkk_a) - mm_deactivate(mapkk_a)
    d_mapkk_i = -d_mapkk_a

    # Tier 3: MAPKK* -> MAPK activation
    d_mapk_a = mm_activate(mapk_i, mapkk_a) - mm_deactivate(mapk_a)
    d_mapk_i = -d_mapk_a

    return [d_mapkkk_a, d_mapkkk_i, d_mapkk_a, d_mapkk_i, d_mapk_a, d_mapk_i]


def step_mapk_cascade(
    signal: float,
    current_state: np.ndarray | None = None,
    dt: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Run one timestep of the MAPK cascade.

    Args:
        signal: input signal strength (0-1)
        current_state: 6-element array [MAPKKK*, MAPKKK, MAPKK*, MAPKK, MAPK*, MAPK]
                        or None for initial state
        dt: timestep

    Returns:
        (operational_output, new_state)
        operational_output = MAPK* (the final tier active fraction)
    """
    if current_state is None:
        current_state = np.array([0.0, TOTAL_PER_TIER, 0.0, TOTAL_PER_TIER, 0.0, TOTAL_PER_TIER])

    sol = solve_ivp(
        mapk_cascade_ode,
        [0, dt],
        current_state,
        args=(max(0.0, min(1.0, signal)),),
        method="RK45",
        max_step=0.25,
    )

    if sol.success and sol.y.shape[1] > 0:
        new_state = np.clip(sol.y[:, -1], 0.0, TOTAL_PER_TIER)
    else:
        new_state = current_state.copy()

    operational_output = float(new_state[4])  # MAPK*
    return operational_output, new_state


def compute_cascade_response(
    signal: float,
    org_depth: int = 3,
    structure_type: str | None = None,
) -> float:
    """Simplified analytical approximation of the MAPK cascade.

    For quick evaluation without full ODE solve.
    Effective Hill coefficient = ~(1.5 * org_depth) for deep orgs.
    """
    depth_map = {"flat": 1, "matrix": 2, "functional": 3, "divisional": 3, "hierarchical": 4}
    effective_depth = depth_map.get(structure_type or "", org_depth)

    n_eff = 1.5 * effective_depth
    K_half = 0.5
    s = max(0.0, min(1.0, signal))
    return s**n_eff / (K_half**n_eff + s**n_eff + 1e-9)
