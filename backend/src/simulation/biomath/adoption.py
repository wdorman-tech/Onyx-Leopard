from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.simulation.biomath.models import SIRState


def sir_rhs(
    t: float,
    y: np.ndarray,
    beta: float,
    gamma: float,
    N: float,
) -> list[float]:
    """Standard SIR adoption model ODE.

    S -> I -> R
    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - gamma * I
    dR/dt = gamma * I

    S = susceptible (potential adopters)
    I = infected (active adopters)
    R = recovered (churned/saturated adopters)
    """
    S, I, R = float(y[0]), float(y[1]), float(y[2])
    N = max(N, 1.0)

    new_infections = beta * S * I / N
    recoveries = gamma * I

    dS = -new_infections
    dI = new_infections - recoveries
    dR = recoveries

    return [dS, dI, dR]


def compute_r0(beta: float, gamma: float) -> float:
    """Basic reproduction number. R0 > 1 = viral growth, R0 < 1 = death."""
    if gamma <= 0:
        return float("inf") if beta > 0 else 0.0
    return beta / gamma


def step_sir(state: SIRState, dt: float = 1.0) -> SIRState:
    """Advance SIR model by one timestep."""
    if not state.active:
        return state

    y0 = [state.susceptible, state.infected, state.recovered]
    N = state.total_market

    sol = solve_ivp(
        sir_rhs,
        [0, dt],
        y0,
        args=(state.beta, state.gamma, N),
        method="RK45",
        max_step=0.25,
    )

    if sol.success and sol.y.shape[1] > 0:
        state.susceptible = max(0.0, float(sol.y[0, -1]))
        state.infected = max(0.0, float(sol.y[1, -1]))
        state.recovered = max(0.0, float(sol.y[2, -1]))
    else:
        # Euler fallback
        dS = -state.beta * state.susceptible * state.infected / max(N, 1.0)
        dI = -dS - state.gamma * state.infected
        dR = state.gamma * state.infected
        state.susceptible = max(0.0, state.susceptible + dS * dt)
        state.infected = max(0.0, state.infected + dI * dt)
        state.recovered = max(0.0, state.recovered + dR * dt)

    # Deactivate when infection dies out
    if state.infected < 0.001 * N and state.recovered > 0.1 * N:
        state.active = False

    return state


def create_sir_for_launch(
    market_growth_rate: float,
    churn_rate: float,
    total_market: float,
    initial_adopters_fraction: float = 0.01,
) -> SIRState:
    """Create an SIR state for a new product launch.

    beta derived from market_growth_rate, gamma from churn.
    Starts with a small seed of initial adopters.
    """
    beta = max(0.05, market_growth_rate * 2.0)
    gamma = max(0.01, churn_rate)

    initial_infected = total_market * initial_adopters_fraction
    return SIRState(
        susceptible=total_market - initial_infected,
        infected=initial_infected,
        recovered=0.0,
        beta=beta,
        gamma=gamma,
        total_market=total_market,
        active=True,
    )
