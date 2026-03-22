from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.simulation.biomath.models import CellCycleState

# Goldbeter model parameters (simplified)
K_SYN = 0.1  # cyclin synthesis rate
K_DEG_BASE = 0.01  # basal cyclin degradation
K_DEG_CDK = 0.5  # CDK-driven cyclin degradation
K_ACT = 0.5  # CDK activation rate by cyclin
K_INACT = 0.1  # CDK inactivation rate

# Hysteresis thresholds — trigger > abort ensures bistability
DIVISION_THRESHOLD = 0.7  # cyclin level to trigger S-phase
ABORT_THRESHOLD = 0.3  # cyclin level to abort back to G1


def cell_cycle_ode(
    t: float,
    y: np.ndarray,
    resource_rate: float,
) -> list[float]:
    """Simplified 2-variable Goldbeter cell cycle model with hysteresis.

    y = [cyclin, cdk_active]
    resource_rate = normalized income rate (drives cyclin synthesis)
    """
    cyclin, cdk = float(y[0]), float(y[1])

    d_cyclin = K_SYN * resource_rate - (K_DEG_BASE + K_DEG_CDK * cdk) * cyclin
    d_cdk = K_ACT * cyclin * (1.0 - cdk) - K_INACT * cdk

    return [d_cyclin, d_cdk]


def step_cell_cycle(state: CellCycleState, resource_rate: float, dt: float = 1.0) -> CellCycleState:
    """Advance cell cycle by one timestep.

    resource_rate: normalized rate of resource accumulation (revenue/costs ratio).
    Higher values drive cyclin accumulation toward division threshold.
    """
    y0 = [state.cyclin, state.cdk_active]

    sol = solve_ivp(
        cell_cycle_ode,
        [0, dt],
        y0,
        args=(max(0.0, resource_rate),),
        method="RK45",
        max_step=0.25,
    )

    if sol.success and sol.y.shape[1] > 0:
        state.cyclin = max(0.0, float(sol.y[0, -1]))
        state.cdk_active = max(0.0, min(1.0, float(sol.y[1, -1])))
    else:
        state.cyclin = max(0.0, state.cyclin + K_SYN * resource_rate * dt)
        state.cdk_active = max(0.0, min(1.0, state.cdk_active))

    # Phase transitions with hysteresis
    match state.phase:
        case "G1":
            if state.cyclin >= DIVISION_THRESHOLD:
                state.phase = "S"
        case "S":
            if state.cyclin < ABORT_THRESHOLD:
                state.phase = "G1"
            else:
                state.phase = "G2"
        case "G2":
            if state.cyclin < ABORT_THRESHOLD:
                state.phase = "G1"
            elif state.cdk_active > 0.8:
                state.phase = "M"
        case "M":
            # Division happens — reset cyclin
            state.cyclin = 0.0
            state.cdk_active = 0.0
            state.phase = "G1"

    return state


def check_division_ready(state: CellCycleState) -> bool:
    """Returns True if the node has passed through M phase (ready to expand)."""
    return state.phase == "M"


def calibrate_thresholds(company_stage: str | None) -> tuple[float, float]:
    """Calibrate division thresholds by company stage.

    Early stage = low threshold (easy to expand)
    Mature = high threshold (requires more resources to justify expansion)
    """
    stage_map = {
        "pre_revenue": (0.3, 0.1),
        "early": (0.4, 0.2),
        "growth": (0.6, 0.3),
        "mature": (0.8, 0.4),
        "turnaround": (0.9, 0.5),
    }
    return stage_map.get(company_stage or "", (DIVISION_THRESHOLD, ABORT_THRESHOLD))
