from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing core models
# ---------------------------------------------------------------------------

class Outlook(str, Enum):
    pessimistic = "pessimistic"
    normal = "normal"
    optimistic = "optimistic"


class NodeData(BaseModel):
    id: str
    label: str
    type: Literal["department", "team", "role", "revenue_stream", "cost_center", "external"]
    metrics: dict[str, float]
    agent_prompt: str

class EdgeData(BaseModel):
    source: str
    target: str
    relationship: Literal["reports_to", "funds", "supplies", "collaborates", "serves"]
    label: str

class CompanyGraph(BaseModel):
    name: str
    description: str
    nodes: list[NodeData]
    edges: list[EdgeData]
    global_metrics: dict[str, float]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ParseRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class RefineRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    current_graph: CompanyGraph


# ---------------------------------------------------------------------------
# CompanyProfile — canonical company representation
# ---------------------------------------------------------------------------

class BusinessModel(str, Enum):
    b2b = "b2b"
    b2c = "b2c"
    b2b2c = "b2b2c"
    saas = "saas"
    marketplace = "marketplace"
    manufacturing = "manufacturing"
    services = "services"
    ecommerce = "ecommerce"
    other = "other"


class CompanyStage(str, Enum):
    pre_revenue = "pre_revenue"
    early = "early"
    growth = "growth"
    mature = "mature"
    turnaround = "turnaround"


class StructureType(str, Enum):
    functional = "functional"
    divisional = "divisional"
    matrix = "matrix"
    flat = "flat"
    hierarchical = "hierarchical"


class CompetitiveLandscape(str, Enum):
    monopoly = "monopoly"
    oligopoly = "oligopoly"
    monopolistic_competition = "monopolistic_competition"
    perfect_competition = "perfect_competition"


class ProductionModel(str, Enum):
    make_to_order = "make_to_order"
    make_to_stock = "make_to_stock"
    digital_delivery = "digital_delivery"
    service_delivery = "service_delivery"
    engineer_to_order = "engineer_to_order"


class PrimaryObjective(str, Enum):
    growth = "growth"
    profitability = "profitability"
    market_share = "market_share"
    survival = "survival"
    innovation = "innovation"
    sustainability = "sustainability"


class PricingModel(str, Enum):
    subscription = "subscription"
    usage_based = "usage_based"
    per_unit = "per_unit"
    freemium = "freemium"
    tiered = "tiered"
    custom = "custom"
    cost_plus = "cost_plus"
    value_based = "value_based"


class RevenueStreamType(str, Enum):
    recurring = "recurring"
    transactional = "transactional"
    licensing = "licensing"
    services = "services"
    advertising = "advertising"
    other = "other"


# --- Section models ---

class CompanyIdentity(BaseModel):
    name: str
    description: str = ""
    industry: str = ""
    naics_code: str = ""
    business_model: BusinessModel | None = None
    company_stage: CompanyStage | None = None
    year_founded: int | None = None
    headquarters: str = ""
    geographic_scope: str = ""
    operating_regions: list[str] = Field(default_factory=list)


class SubTeam(BaseModel):
    id: str
    name: str
    headcount: int = 0


class Department(BaseModel):
    id: str
    name: str
    headcount: int = 0
    budget: float = 0.0
    function: str = ""
    sub_teams: list[SubTeam] = Field(default_factory=list)
    kpis: dict[str, float] = Field(default_factory=dict)


class KeyRole(BaseModel):
    id: str
    title: str
    department_id: str = ""
    reports_to: str = ""


class OrganizationStructure(BaseModel):
    total_headcount: int = 0
    structure_type: StructureType | None = None
    departments: list[Department] = Field(default_factory=list)
    key_roles: list[KeyRole] = Field(default_factory=list)
    avg_salary: float = 0.0
    turnover_rate: float = 0.0
    hiring_cost: float = 0.0
    labor_productivity_index: float = 1.0


class RevenueStream(BaseModel):
    name: str
    annual_revenue: float = 0.0
    growth_rate: float = 0.0
    margin: float = 0.0
    type: RevenueStreamType = RevenueStreamType.recurring


class OperatingExpenses(BaseModel):
    sga: float = 0.0
    rd: float = 0.0
    depreciation: float = 0.0


class FinancialProfile(BaseModel):
    currency: str = "USD"
    fiscal_year_end: str = "December"
    annual_revenue: float = 0.0
    revenue_growth_rate: float = 0.0
    revenue_streams: list[RevenueStream] = Field(default_factory=list)
    cogs: float = 0.0
    gross_margin: float = 0.0
    operating_expenses: OperatingExpenses = Field(default_factory=OperatingExpenses)
    total_assets: float = 0.0
    total_debt: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    debt_to_equity: float = 0.0
    capex: float = 0.0
    rd_spend: float = 0.0
    dso: float = 0.0
    dio: float = 0.0
    dpo: float = 0.0
    net_income: float = 0.0
    ebitda: float = 0.0
    roe: float = 0.0
    roa: float = 0.0


class Competitor(BaseModel):
    name: str
    est_revenue: float = 0.0
    est_market_share: float = 0.0
    relative_cost: float = 1.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class MarketProfile(BaseModel):
    tam: float = 0.0
    sam: float = 0.0
    som: float = 0.0
    market_share: float = 0.0
    market_growth_rate: float = 0.0
    competitive_landscape: CompetitiveLandscape | None = None
    competitors: list[Competitor] = Field(default_factory=list)
    primary_competition_dimension: str = ""
    barriers_to_entry: str = ""
    pricing_model: PricingModel | None = None
    price_elasticity_estimate: float = 0.0


class KeyInput(BaseModel):
    name: str
    annual_cost: float = 0.0
    pct_of_cogs: float = 0.0
    substitutability: str = ""


class OperationsProfile(BaseModel):
    production_model: ProductionModel | None = None
    key_inputs: list[KeyInput] = Field(default_factory=list)
    supplier_concentration: float = 0.0
    capacity_utilization: float = 0.0
    inventory_model: str = ""
    avg_inventory_value: float = 0.0
    inventory_turnover: float = 0.0
    lead_time_days: float = 0.0
    defect_rate: float = 0.0
    customer_satisfaction_score: float = 0.0


class PlannedInitiative(BaseModel):
    name: str
    description: str = ""
    priority: int = 0


class MajorRisk(BaseModel):
    name: str
    description: str = ""
    likelihood: str = ""
    impact: str = ""


class StrategyProfile(BaseModel):
    primary_objective: PrimaryObjective | None = None
    strategic_priorities: list[str] = Field(default_factory=list)
    planned_initiatives: list[PlannedInitiative] = Field(default_factory=list)
    major_risks: list[MajorRisk] = Field(default_factory=list)
    moats: list[str] = Field(default_factory=list)
    simulation_objectives: list[str] = Field(default_factory=list)
    time_horizon_weeks: int = 52


class SimulationParameters(BaseModel):
    """Computed from profile data, not user-entered."""
    # Cobb-Douglas
    tfp: float = 1.0
    capital_elasticity: float = 0.3
    labor_elasticity: float = 0.7
    # Cost curves
    fixed_costs: float = 0.0
    variable_cost_per_unit: float = 0.0
    learning_curve_rate: float = 0.85
    # Cournot
    market_demand_intercept: float = 0.0
    market_demand_slope: float = 0.0
    marginal_cost: float = 0.0
    # Growth
    depreciation_rate: float = 0.05
    reinvestment_rate: float = 0.3
    # Risk
    revenue_volatility: float = 0.1
    demand_seasonality: list[float] = Field(
        default_factory=lambda: [1.0] * 12,
    )
    # Bio-math parameters
    logistic_growth_rate: float = 0.05
    carrying_capacity_multiplier: float = 1.5
    hill_coefficient: float = 2.0
    apoptosis_threshold: float = 0.3
    competition_alpha_default: float = 0.5


class ProfileMetadata(BaseModel):
    created_by: str = ""
    source_documents: list[str] = Field(default_factory=list)
    completeness_score: float = 0.0
    last_modified: datetime | None = None
    notes: str = ""


class CompanyProfile(BaseModel):
    schema_version: str = "1.0.0"
    identity: CompanyIdentity = Field(default_factory=lambda: CompanyIdentity(name=""))
    organization: OrganizationStructure = Field(default_factory=OrganizationStructure)
    financials: FinancialProfile = Field(default_factory=FinancialProfile)
    market: MarketProfile = Field(default_factory=MarketProfile)
    operations: OperationsProfile = Field(default_factory=OperationsProfile)
    strategy: StrategyProfile = Field(default_factory=StrategyProfile)
    sim_params: SimulationParameters = Field(default_factory=SimulationParameters)
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)


# ---------------------------------------------------------------------------
# Export / Import format
# ---------------------------------------------------------------------------

class SimulationSnapshot(BaseModel):
    tick: int = 0
    outlook: str = "normal"
    global_metrics: dict[str, float] = Field(default_factory=dict)
    history_summary: list[dict] = Field(default_factory=list)


class OleoExport(BaseModel):
    format: str = "onyx-leopard-export"
    format_version: str = "1.0.0"
    exported_at: str = ""
    profile: CompanyProfile
    graph: CompanyGraph
    simulation_snapshot: SimulationSnapshot | None = None
    checksum: str = ""
