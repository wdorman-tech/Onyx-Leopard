from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ApoptosisState:
    """Bistable death switch state (Bax/Bcl2 model)."""

    bcl2: float = 1.0  # survival signal (revenue/contracts)
    bax: float = 0.0  # death signal (losses/decline)
    caspase: float = 0.0  # execution level 0-1
    triggered: bool = False  # once True, irreversible


@dataclass
class CellCycleState:
    """Goldbeter-style cell cycle checkpoint state."""

    cyclin: float = 0.0  # resource accumulation level
    cdk_active: float = 0.0  # activated CDK
    phase: str = "G1"  # G1 / S / G2 / M


@dataclass
class SIRState:
    """SIR adoption model state for product launches."""

    susceptible: float = 1.0
    infected: float = 0.0  # active adopters
    recovered: float = 0.0  # churned adopters
    beta: float = 0.3  # adoption rate
    gamma: float = 0.1  # churn rate
    total_market: float = 1.0
    active: bool = False


@dataclass
class BioState:
    """Per-node biological/mathematical state."""

    population: float = 0.0
    carrying_capacity: float = 10.0
    growth_rate: float = 0.05
    cash: float = 0.0
    capital: float = 0.0  # invested assets for Cobb-Douglas
    revenue_rate: float = 0.0
    cost_rate: float = 0.0
    health_score: float = 1.0
    # Phase 2
    signal_activation: float = 0.0
    apoptosis: ApoptosisState | None = None
    wind_down_ticks: int = 0
    # Phase 3
    sir_state: SIRState | None = None
    # Phase 4
    cell_cycle: CellCycleState | None = None


@dataclass
class BioParams:
    """Per-node mathematical parameters."""

    r: float = 0.05
    K: float = 10.0
    tfp: float = 1.0
    alpha: float = 0.3
    beta: float = 0.7
    fixed_costs: float = 0.0
    variable_cost_rate: float = 0.0


@dataclass
class ActionConstraints:
    """What each agent is mathematically allowed to do."""

    max_hire: int = 0
    max_fire: int = 0
    max_budget_increase: float = 0.0
    max_budget_decrease: float = 0.0
    health_status: str = "healthy"
    can_expand: bool = False
    capacity_utilization: float = 0.0


@dataclass
class FluxSolution:
    """Result of FBA resource allocation LP."""

    fluxes: dict[str, float] = field(default_factory=dict)
    shadow_prices: dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    objective_value: float = 0.0


@dataclass
class BioConfig:
    """Feature flags for bio-math models."""

    logistic_growth: bool = True
    cobb_douglas: bool = True
    conservation_laws: bool = True
    hill_signals: bool = True
    apoptosis: bool = True
    lotka_volterra: bool = True
    fba: bool = True
    mapk_cascade: bool = True
    sir_adoption: bool = True
    cell_cycle: bool = True
    cobb_douglas_revenue: bool = True
    replicator: bool = True
    replicator_interval: int = 5
