"""Data models for the competitive market simulation.

Implements the state variables from competitive_market_simulation_spec.md.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketParams(BaseModel):
    """Global market parameters — frozen after initialization."""

    # Market
    tam_0: float = 50_000.0
    g_market: float = 0.0003
    lambda_entry: float = 0.03
    g_ref: float = 0.001

    # Competition
    alpha: float = 0.8
    beta: float = 0.8
    delta: float = 0.08

    # Death / decay
    b_death: float = -500.0
    t_death: int = 30
    tau_decay: int = 60

    # Capital constraint sigmoid
    b_threshold: float = 1_500.0
    k_sigmoid: float = 0.001

    # Timing
    t_q: int = 90

    # Capacity costs
    c_k: float = 0.02
    capex_per_unit: float = 0.5

    # Initial agent count
    n_0: int = 5

    # Agent initialization ranges (for randomized spawning)
    r_range: tuple[float, float] = (0.20, 0.35)
    margin_range: tuple[float, float] = (0.20, 0.35)
    f_range: tuple[float, float] = (80.0, 200.0)
    q_range: tuple[float, float] = (0.6, 1.4)
    m_range: tuple[float, float] = (15.0, 50.0)
    k_range: tuple[float, float] = (800.0, 2000.0)
    b_range: tuple[float, float] = (8_000.0, 20_000.0)
    eta_m_range: tuple[float, float] = (0.15, 0.30)
    eta_q_range: tuple[float, float] = (0.05, 0.12)
    tau_k_range: tuple[int, int] = (20, 40)


class AgentParams(BaseModel):
    """Per-agent fixed parameters — set at creation."""

    name: str
    r: float
    margin: float
    f: float
    eta_m: float
    eta_q: float
    tau_k: int


class PendingExpansion(BaseModel):
    delivery_tick: int
    new_capacity: float


class AgentState(BaseModel):
    """Mutable per-agent state vector."""

    id: str
    params: AgentParams
    revenue: float = 0.0
    cash: float = 10_000.0
    capacity: float = 1_000.0
    quality: float = 1.0
    marketing: float = 30.0

    q_target: float = 1.0
    m_target: float = 30.0
    k_target: float = 1_000.0

    alive: bool = True
    death_counter: int = 0
    decay_ticks_remaining: int = 0

    share: float = 0.0
    prev_share: float = 0.0

    pending_expansions: list[PendingExpansion] = Field(default_factory=list)
    color: str = "#888888"


class AgentSnapshot(BaseModel):
    """Per-agent data sent to frontend each tick."""

    id: str
    name: str
    alive: bool
    revenue: float
    cash: float
    capacity: float
    quality: float
    marketing: float
    share: float
    utilization: float
    binding_constraint: str
    color: str


class MarketTickResult(BaseModel):
    """Full tick output sent to frontend."""

    tick: int
    tam: float
    captured: float
    hhi: float
    agent_count: int
    agents: list[AgentSnapshot]
    events: list[str]


class MarketPreset(BaseModel):
    """Named parameter bundle for quick-start scenarios."""

    name: str
    slug: str
    description: str
    params: MarketParams
