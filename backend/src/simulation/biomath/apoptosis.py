from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.simulation.biomath.models import ApoptosisState, BioState

# Kinetic constants for the Bax/Bcl2 bistable switch
K_ACT = 0.5  # Bax activation rate by stress
K_INHIB = 0.8  # Bcl2 inhibition of Bax
K_SYN = 0.3  # Bcl2 synthesis rate (driven by revenue)
K_SEQ = 0.4  # Bax sequestration of Bcl2
K_DECAY = 0.1  # Bcl2 natural decay
BAX_TOTAL = 1.0  # total Bax pool (active + inactive)

APOPTOSIS_THRESHOLD = 2.0  # bax/bcl2 ratio triggering irreversible death
WIND_DOWN_TICKS = 4  # ticks to fully wind down after trigger


def apoptosis_ode(
    t: float,
    y: np.ndarray,
    stress: float,
    revenue_rate: float,
) -> list[float]:
    """Bistable switch ODE for department death mechanics.

    y = [bax_active, bcl2]
    stress = max(0, cost_rate - revenue_rate) / cost_rate — normalised burn
    revenue_rate = Cobb-Douglas output, drives Bcl2 synthesis
    """
    bax_active, bcl2 = float(y[0]), float(y[1])
    bax_inactive = max(0.0, BAX_TOTAL - bax_active)

    d_bax = K_ACT * stress * bax_inactive - K_INHIB * max(bcl2, 0.0) * bax_active
    d_bcl2 = K_SYN * revenue_rate - K_SEQ * bax_active * max(bcl2, 0.0) - K_DECAY * max(bcl2, 0.0)

    return [d_bax, d_bcl2]


def step_apoptosis(state: BioState, dt: float = 1.0) -> ApoptosisState:
    """Advance apoptosis ODE by dt. Returns updated ApoptosisState."""
    apo = state.apoptosis or ApoptosisState()

    if apo.triggered:
        apo.caspase = min(1.0, apo.caspase + 0.25)
        return apo

    cost = max(state.cost_rate, 1e-6)
    stress = max(0.0, state.cost_rate - state.revenue_rate) / cost
    rev_norm = state.revenue_rate / max(cost, 1.0)

    y0 = [apo.bax, apo.bcl2]
    sol = solve_ivp(
        apoptosis_ode,
        [0, dt],
        y0,
        args=(stress, rev_norm),
        method="RK45",
        max_step=0.5,
    )

    if sol.success and sol.y.shape[1] > 0:
        apo.bax = max(0.0, float(sol.y[0, -1]))
        apo.bcl2 = max(1e-6, float(sol.y[1, -1]))
    else:
        apo.bax = max(0.0, apo.bax + stress * K_ACT * dt)
        apo.bcl2 = max(1e-6, apo.bcl2 - K_DECAY * apo.bcl2 * dt)

    ratio = apo.bax / apo.bcl2 if apo.bcl2 > 1e-6 else float("inf")
    if ratio > APOPTOSIS_THRESHOLD:
        apo.triggered = True
        apo.caspase = 0.1

    return apo


def check_apoptosis(bio_state: BioState, threshold: float = 0.3) -> bool:
    """Check if a node should enter apoptosis based on health score."""
    if bio_state.apoptosis and bio_state.apoptosis.triggered:
        return True
    return bio_state.health_score < threshold


def apply_wind_down(bio_state: BioState) -> BioState:
    """Apply wind-down effects: drain headcount and budget over WIND_DOWN_TICKS."""
    if not bio_state.apoptosis or not bio_state.apoptosis.triggered:
        return bio_state

    if bio_state.wind_down_ticks <= 0:
        bio_state.wind_down_ticks = WIND_DOWN_TICKS

    drain_fraction = 1.0 / max(bio_state.wind_down_ticks, 1)
    bio_state.population = max(0.0, bio_state.population * (1.0 - drain_fraction))
    released_cash = bio_state.cash * drain_fraction
    bio_state.cash = max(0.0, bio_state.cash - released_cash)
    bio_state.wind_down_ticks = max(0, bio_state.wind_down_ticks - 1)

    return bio_state
