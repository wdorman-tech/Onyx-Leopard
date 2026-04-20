"""Pydantic models for the unified simulation tick output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.simulation.models import GraphSnapshot


class UnifiedAgentSnapshot(BaseModel):
    """Per-company data sent to the frontend each tick."""

    id: str
    name: str
    alive: bool
    color: str

    # Market-level attributes (derived from nodes via bridge)
    quality: float
    marketing: float
    capacity: float
    share: float
    utilization: float
    binding_constraint: str  # "demand" | "capacity"

    # Company-level metrics
    cash: float
    daily_revenue: float
    daily_costs: float
    stage: int
    location_count: int
    node_count: int
    avg_satisfaction: float
    total_employees: int


class UnifiedTickData(BaseModel):
    """Full tick output for the unified simulation — sent via SSE."""

    tick: int
    status: str  # "operating" | "collapsed"
    mode: str = "unified"

    # Market-level
    tam: float
    captured: float
    hhi: float
    agent_count: int

    # Per-company
    agents: list[UnifiedAgentSnapshot]

    # Graphs for all companies, keyed by company name
    focused_company_id: str
    graphs: dict[str, GraphSnapshot]

    events: list[str] = Field(default_factory=list)


class UnifiedParams(BaseModel):
    """Parameters for the unified simulation market layer.

    Unlike MarketParams, these are calibrated for dollar-denominated values
    since K/revenue come from actual location economics, not abstract units.
    """

    # Total addressable market ($/day of restaurant spending in the area)
    tam_0: float = 25_000.0
    g_market: float = 0.0002  # daily TAM growth rate (~7.5% annual)

    # Share attraction elasticities
    alpha: float = 0.8  # marketing weight
    beta: float = 0.8   # quality weight

    # Entry / exit
    # lambda_entry is the base hazard rate for *competitive* market entry — the
    # default fits multi-company unified/adaptive sims where rivals occasionally
    # appear. Single-company growth mode overrides this to 0.0 in
    # SessionManager.create_session, since spawning rivals into a 1-company sim
    # is meaningless.
    lambda_entry: float = 0.01
    g_ref: float = 0.001
    b_death: float = -5_000.0   # cash threshold for bankruptcy clock (dollars)
    t_death: int = 30            # consecutive ticks below b_death before death

    # Starting conditions
    starting_cash: float = 50_000.0


class UnifiedStartConfig(BaseModel):
    """Configuration for starting a unified simulation."""

    industry: str = "restaurant"
    start_mode: str = "identical"  # "identical" | "randomized" | "staggered"
    num_companies: int = Field(default=4, ge=1, le=20)
    max_ticks: int = Field(default=0, ge=0)
    params: UnifiedParams = UnifiedParams()

    # AI CEO agent settings
    ai_ceo_enabled: bool = False
    duration_years: int = Field(default=5)  # 5, 10, or 20
    company_strategies: dict[int, str] = Field(default_factory=dict)  # index → strategy
    ai_budget_max: float = Field(default=1.0, ge=0.0)  # max AI spend per sim run ($)

    # Custom company names (adaptive mode): index → name override
    custom_company_names: dict[int, str] = Field(default_factory=dict)
