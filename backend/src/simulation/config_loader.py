"""Industry config loader — reads YAML files and validates with Pydantic.

Each industry is defined by a single YAML file in the industries/ directory.
The engine receives an IndustrySpec at construction time and reads all
node types, triggers, bridge mappings, and constants from it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ── Pydantic models for YAML validation ──


class IndustryMeta(BaseModel):
    slug: str
    name: str
    description: str
    icon: str
    playable: bool = False
    total_nodes: int = 0
    growth_stages: int = 0
    key_metrics: list[str] = Field(default_factory=list)
    example_nodes: list[str] = Field(default_factory=list)
    categories: dict[str, int] = Field(default_factory=dict)


class IndustryRoles(BaseModel):
    location_type: str
    founder_type: str
    supplier_types: list[str] = Field(default_factory=list)
    numbered_labels: dict[str, str] = Field(default_factory=dict)


class NodeDef(BaseModel):
    label: str
    category: str  # "location" | "corporate" | "external" | "revenue"
    stage: int
    annual_cost: float = 0.0
    cost_modifiers: dict[str, float] = Field(default_factory=dict)
    revenue_modifiers: dict[str, float] = Field(default_factory=dict)
    enabled: bool = True


class TriggerDef(BaseModel):
    node_type: str
    label: str
    max_instances: int = 1
    cooldown_ticks: int = 0
    is_location_expansion: bool = False
    condition: dict


class BridgeDef(BaseModel):
    marketing_baseline: float = 5.0
    marketing_per_location: float = 1.0
    sustainable_utilization: float = 0.85
    marketing_contributions: dict[str, float] = Field(default_factory=dict)
    quality_modifier_keys: list[str] = Field(default_factory=list)
    infrastructure_multipliers: dict[str, float] = Field(default_factory=dict)
    marketing_boost_neutral: float = 0.5
    marketing_boost_multiplier: float = 20.0


class StageDef(BaseModel):
    min_locations: int
    stage: int


class LocationDefaults(BaseModel):
    economics_model: str = "physical"  # "physical" | "subscription" | "service"
    supply_unit_name: str = "units"  # e.g. "lbs chicken", "licenses", "billable hours"
    location_label: str = "Location"  # e.g. "Restaurant", "Office", "Data Center"
    inventory: float = 80.0
    customers: float = 30.0
    satisfaction: float = 0.7
    price: float = 14.0
    max_capacity: int = 80
    variable_cost_per_unit: float = 1.50
    daily_fixed_costs: float = 300.0
    replenish_threshold: float = 30.0
    replenish_amount: float = 100.0
    supply_cost_per_unit: float = 3.50
    capacity_decay_rate: float = 0.05
    word_of_mouth_rate: float = 0.02
    max_local_customers: float = 120.0
    unified_replenish_amount: float = 200.0
    unified_replenish_threshold: float = 80.0
    # Subscription/service model fields
    churn_rate: float = 0.0  # monthly customer loss rate (subscription model)
    acquisition_cost: float = 0.0  # cost to acquire a new customer
    scaling_cost_per_unit: float = 0.0  # cost to add capacity (SaaS infra, hiring)
    # Simulation dynamics (per-industry tuning)
    satisfaction_penalty_rate: float = 0.02  # satisfaction drop per unserved/demand ratio
    satisfaction_recovery_rate: float = 0.005  # daily satisfaction recovery when no stockouts
    customer_convergence_rate: float = 0.05  # how fast customers converge to allocated demand
    demand_cap_ratio: float = 0.85  # max demand as fraction of capacity
    demand_noise_low: float = 0.9  # daily demand variance lower bound
    demand_noise_high: float = 1.1  # daily demand variance upper bound
    subscription_scaling_threshold: float = 0.8  # capacity utilization that triggers scaling
    subscription_scaling_increment: float = 0.1  # fraction of capacity added per scale event
    new_customer_ratio: float = 0.95  # fraction of customers assumed existing (for CAC calc)

    def to_location_state(self, **overrides: object) -> dict:
        """Dump fields for LocationState init in unified mode."""
        data = self.model_dump(
            exclude={"unified_replenish_amount", "unified_replenish_threshold",
                     "location_label", "supply_unit_name"}
        )
        data["replenish_amount"] = self.unified_replenish_amount
        data["replenish_threshold"] = self.unified_replenish_threshold
        data.update(overrides)
        return data


class ConstantsDef(BaseModel):
    location_open_cost: float = 50000.0
    employees_per_location: int = 15
    starting_cash: float = 50000.0
    new_location_starting_customers: float = 20.0
    new_location_starting_satisfaction: float = 0.5
    volume_discounts: list[list[float]] = Field(
        default_factory=lambda: [[1, 1.0]]
    )
    days_per_month: int = 30
    ticks_per_year: int = 365
    variable_cost_modifier_key: str = "food_cost"
    # Randomized start mode ranges
    random_start_cash_low: float = 30_000.0
    random_start_cash_high: float = 80_000.0
    random_start_satisfaction_low: float = 0.55
    random_start_satisfaction_high: float = 0.85
    random_start_customers_low: float = 20.0
    random_start_customers_high: float = 45.0
    # Engine tuning
    min_spawn_probability: float = 0.02
    staggered_initial_count: int = 2
    default_avg_price: float = 14.0


class DisplayConfig(BaseModel):
    """Frontend display labels and configuration."""

    stage_labels: dict[int, str] = Field(default_factory=lambda: {
        1: "Single Location", 2: "Multi-Location",
        3: "Regional Chain", 4: "National Chain",
    })
    event_noise_filters: list[str] = Field(default_factory=lambda: [
        "spoiled", "Ordered", "Turned away", "Cannot reorder",
    ])
    duration_options: list[int] = Field(default_factory=lambda: [5, 10, 20])


class CeoConfig(BaseModel):
    """Per-industry CEO agent configuration."""

    interval_ticks: int = 182  # ~6 months
    price_min: float = 10.0
    price_max: float = 22.0
    price_default: float = 14.0
    cost_min: float = 1.00
    cost_max: float = 2.50
    cost_default: float = 1.50
    max_locations_per_year_cap: int = 12
    price_unit: str = "per plate"
    cost_unit: str = "food cost per plate"
    expansion_overrides: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "aggressive": {"cooldown_ticks": 45, "cash_threshold": 60_000},
            "normal": {"cooldown_ticks": 90, "cash_threshold": 80_000},
            "conservative": {"cooldown_ticks": 180, "cash_threshold": 120_000},
        }
    )


class IndustrySpec(BaseModel):
    """Complete industry specification loaded from YAML."""

    meta: IndustryMeta
    roles: IndustryRoles
    nodes: dict[str, NodeDef]
    triggers: list[TriggerDef]
    bridge: BridgeDef
    constants: ConstantsDef
    stages: list[StageDef]
    location_defaults: LocationDefaults = LocationDefaults()
    ceo: CeoConfig = CeoConfig()
    display: DisplayConfig = DisplayConfig()


# ── Loader ──

_INDUSTRY_DIR = Path(__file__).parent / "industries"
_cache: dict[str, IndustrySpec] = {}


def load_industry(slug: str) -> IndustrySpec:
    """Load and validate an industry config from YAML. Cached after first load."""
    if slug in _cache:
        return _cache[slug]

    path = _INDUSTRY_DIR / f"{slug}.yaml"
    if not path.exists():
        raise ValueError(f"Industry config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    spec = IndustrySpec(**raw)

    # Cross-reference validation
    if spec.roles.location_type not in spec.nodes:
        raise ValueError(
            f"roles.location_type '{spec.roles.location_type}' not found in nodes"
        )
    if spec.roles.founder_type not in spec.nodes:
        raise ValueError(
            f"roles.founder_type '{spec.roles.founder_type}' not found in nodes"
        )
    for st in spec.roles.supplier_types:
        if st not in spec.nodes:
            raise ValueError(f"roles.supplier_types entry '{st}' not found in nodes")
    for trigger in spec.triggers:
        if trigger.node_type not in spec.nodes:
            raise ValueError(
                f"trigger node_type '{trigger.node_type}' not found in nodes"
            )

    _cache[slug] = spec
    return spec


def clear_cache() -> None:
    """Clear the industry config cache (useful for tests)."""
    _cache.clear()


def list_industry_specs() -> list[IndustrySpec]:
    """List all available industry configs from YAML files."""
    specs = []
    for path in sorted(_INDUSTRY_DIR.glob("*.yaml")):
        specs.append(load_industry(path.stem))
    return specs
