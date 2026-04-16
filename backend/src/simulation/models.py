"""Data models for the node-based growth simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class NodeCategory(str, Enum):
    LOCATION = "location"
    CORPORATE = "corporate"
    EXTERNAL = "external"
    REVENUE = "revenue"


class NodeType(str, Enum):
    """DEPRECATED: Restaurant-specific node types. Kept for legacy GrowthEngine
    and NODE_REGISTRY compatibility. The unified engine uses plain strings
    loaded from industry YAML configs — new industries do NOT need entries here."""

    # Locations (v1)
    RESTAURANT = "restaurant"
    COMMISSARY = "commissary"
    DISTRIBUTION_CENTER = "distribution_center"

    # Locations (v2)
    DRIVE_THRU = "drive_thru"
    GHOST_KITCHEN = "ghost_kitchen"
    FOOD_COURT = "food_court"
    FRANCHISE_LOCATION = "franchise_location"
    TEST_LOCATION = "test_location"

    # Corporate (v1)
    OWNER_OPERATOR = "owner_operator"
    GENERAL_MANAGER = "general_manager"
    BOOKKEEPER = "bookkeeper"
    MARKETING = "marketing"
    HR = "hr"
    TRAINING = "training"
    AREA_MANAGER = "area_manager"
    QUALITY_ASSURANCE = "quality_assurance"
    PROCUREMENT = "procurement"
    IT_SUPPORT = "it_support"
    REAL_ESTATE = "real_estate"
    CONSTRUCTION = "construction"
    RND_MENU = "rnd_menu"
    LEGAL = "legal"
    FINANCE_FPA = "finance_fpa"

    # Corporate (v2)
    CORPORATE_HQ = "corporate_hq"
    CEO = "ceo"
    COO = "coo"
    CFO = "cfo"
    CMO = "cmo"
    CUSTOMER_SERVICE = "customer_service"
    FRANCHISE_DEV = "franchise_dev"
    FRANCHISE_ADVISORY = "franchise_advisory"
    REGIONAL_VP = "regional_vp"

    # External (v1)
    CHICKEN_SUPPLIER = "chicken_supplier"
    PRODUCE_SUPPLIER = "produce_supplier"

    # External (v2)
    BEVERAGE_SUPPLIER = "beverage_supplier"
    LANDLORD = "landlord"
    FRANCHISE_OWNER = "franchise_owner"
    AREA_DEVELOPER = "area_developer"

    # Revenue (v1)
    CATERING = "catering"
    DELIVERY_PARTNERSHIP = "delivery_partnership"

    # Revenue (v2)
    LOYALTY_PROGRAM = "loyalty_program"
    MERCHANDISE = "merchandise"
    LICENSING_IP = "licensing_ip"
    GIFT_CARD = "gift_card"

    # Specialized (v2)
    FOOD_SAFETY_LAB = "food_safety_lab"
    SECRET_SHOPPER = "secret_shopper"
    SUSTAINABILITY = "sustainability"
    INVESTOR_RELATIONS = "investor_relations"
    CAPTIVE_INSURANCE = "captive_insurance"


class LocationState(BaseModel):
    """Per-location economics — generic model for any industry type."""

    economics_model: str = "physical"  # "physical" | "subscription" | "service"
    inventory: float = 80.0
    customers: float = 30.0
    satisfaction: float = 0.7
    price: float = 14.00
    max_capacity: int = 80
    variable_cost_per_unit: float = 1.50
    daily_fixed_costs: float = 300.00
    replenish_threshold: float = 30.0
    replenish_amount: float = 100.0
    supply_cost_per_unit: float = 3.50
    capacity_decay_rate: float = 0.05
    word_of_mouth_rate: float = 0.02
    max_local_customers: float = 120.0
    churn_rate: float = 0.0
    acquisition_cost: float = 0.0
    scaling_cost_per_unit: float = 0.0
    # Simulation dynamics
    satisfaction_penalty_rate: float = 0.02
    satisfaction_recovery_rate: float = 0.005
    customer_convergence_rate: float = 0.05
    demand_cap_ratio: float = 0.85
    demand_noise_low: float = 0.9
    demand_noise_high: float = 1.1
    subscription_scaling_threshold: float = 0.8
    subscription_scaling_increment: float = 0.1
    new_customer_ratio: float = 0.95


@dataclass
class LocationConfig:
    """Per-industry location sim constants. Same for all locations in a company."""

    satisfaction_penalty_rate: float = 0.02
    satisfaction_recovery_rate: float = 0.005
    customer_convergence_rate: float = 0.05
    demand_cap_ratio: float = 0.85
    demand_noise_low: float = 0.9
    demand_noise_high: float = 1.1
    subscription_scaling_threshold: float = 0.8
    subscription_scaling_increment: float = 0.1
    new_customer_ratio: float = 0.95
    days_per_month: int = 30
    variable_cost_modifier_key: str = "food_cost"


@dataclass
class LocationArrays:
    """Struct-of-arrays layout for vectorized location economics."""

    node_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # Mutable state (updated every tick)
    customers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    inventory: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    satisfaction: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Read-only params (set at location creation)
    price: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    max_capacity: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    variable_cost_per_unit: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    daily_fixed_costs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    replenish_threshold: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    replenish_amount: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    supply_cost_per_unit: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    capacity_decay_rate: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    churn_rate: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Non-numeric metadata (per location, same for all in an industry)
    economics_model: str = "physical"

    @property
    def size(self) -> int:
        return len(self.node_ids)

    def append_location(self, node_id: str, label: str, ls: LocationState) -> None:
        """Append a single location's data to the arrays."""
        self.node_ids.append(node_id)
        self.labels.append(label)
        self.customers = np.append(self.customers, ls.customers)
        self.inventory = np.append(self.inventory, ls.inventory)
        self.satisfaction = np.append(self.satisfaction, ls.satisfaction)
        self.price = np.append(self.price, ls.price)
        self.max_capacity = np.append(self.max_capacity, float(ls.max_capacity))
        self.variable_cost_per_unit = np.append(self.variable_cost_per_unit, ls.variable_cost_per_unit)
        self.daily_fixed_costs = np.append(self.daily_fixed_costs, ls.daily_fixed_costs)
        self.replenish_threshold = np.append(self.replenish_threshold, ls.replenish_threshold)
        self.replenish_amount = np.append(self.replenish_amount, ls.replenish_amount)
        self.supply_cost_per_unit = np.append(self.supply_cost_per_unit, ls.supply_cost_per_unit)
        self.capacity_decay_rate = np.append(self.capacity_decay_rate, ls.capacity_decay_rate)
        self.churn_rate = np.append(self.churn_rate, ls.churn_rate)
        self.economics_model = ls.economics_model


@dataclass
class BatchTickResult:
    """Result of vectorized batch location ticking."""

    total_revenue: float = 0.0
    total_costs: float = 0.0
    total_profit: float = 0.0
    total_reorder_cost: float = 0.0
    events: list[str] = field(default_factory=list)


class SimNode(BaseModel):
    id: str
    type: str
    label: str
    category: NodeCategory
    spawned_at: int = 0
    active: bool = True

    annual_cost: float = 0.0
    cost_modifiers: dict[str, float] = Field(default_factory=dict)
    revenue_modifiers: dict[str, float] = Field(default_factory=dict)

    location_state: LocationState | None = None

    @property
    def daily_cost(self) -> float:
        return self.annual_cost / 365.0


class SimEdge(BaseModel):
    source: str
    target: str
    relationship: str  # reports_to, supplies, manages, serves


class CompanyState(BaseModel):
    name: str = "Company A"
    tick: int = 0
    stage: int = 1
    cash: float = 10_000.0
    total_employees: int = 3
    locations_opened_this_year: int = 0

    nodes: dict[str, SimNode] = Field(default_factory=dict)
    edges: list[SimEdge] = Field(default_factory=list)


class LocationTickResult(BaseModel):
    revenue: float = 0.0
    costs: float = 0.0
    profit: float = 0.0
    customers_served: int = 0
    events: list[str] = Field(default_factory=list)


class TickResult(BaseModel):
    tick: int
    stage: int
    status: str
    metrics: dict[str, float]
    events: list[str]
    graph: GraphSnapshot


class GraphSnapshot(BaseModel):
    nodes: list[NodeSnapshot]
    edges: list[SimEdge]


class NodeSnapshot(BaseModel):
    """Lightweight node data sent to the frontend each tick."""

    id: str
    type: str
    label: str
    category: str
    spawned_at: int
    metrics: dict[str, float] = Field(default_factory=dict)
