"""Industry config loader — reads YAML files and validates with Pydantic.

Each industry is defined by a single YAML file in the industries/ directory.
The engine receives an IndustrySpec at construction time and reads all
node types, triggers, bridge mappings, and constants from it.
"""

from __future__ import annotations

import os
import tempfile
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
    inventory: float = 0.0
    customers: float = 30.0
    satisfaction: float = 0.7
    # Required economic fields — must be set per-industry, no restaurant defaults.
    price: float
    max_capacity: int
    variable_cost_per_unit: float
    daily_fixed_costs: float
    replenish_threshold: float = 30.0
    replenish_amount: float = 100.0
    supply_cost_per_unit: float = 0.0
    capacity_decay_rate: float = 0.0
    word_of_mouth_rate: float = 0.02
    max_local_customers: float = 120.0
    unified_replenish_amount: float = 200.0
    unified_replenish_threshold: float = 80.0
    # Modifier role lists (auto-discovered post-load if empty).
    cost_modifier_keys: list[str] = Field(default_factory=list)
    revenue_modifier_keys: list[str] = Field(default_factory=list)
    satisfaction_modifier_keys: list[str] = Field(default_factory=list)
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
    # Canonical key under which the volume discount is folded; empty disables.
    variable_cost_modifier_key: str = ""
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
    duration_options: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    speed_presets: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 50, 100, 500])
    company_names: list[str] = Field(default_factory=lambda: [
        "Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co",
        "Epsilon Group", "Zeta Corp", "Eta Inc", "Theta Ltd",
        "Iota Co", "Kappa Group", "Lambda Corp", "Mu Inc",
        "Nu Ltd", "Xi Co", "Omicron Group", "Pi Corp",
        "Rho Inc", "Sigma Ltd", "Tau Co", "Upsilon Group",
    ])
    ceo_strategies: list[str] = Field(default_factory=lambda: [
        "aggressive_growth", "quality_focus", "cost_leader",
        "balanced", "market_dominator", "survivor",
    ])
    max_companies: int = 20
    min_description_words: int = 20


class MathConfig(BaseModel):
    """Mathematical model selection — defaults preserve pre-integration behavior."""

    competition_model: str = "multinomial_logit"  # "lotka_volterra" | "multinomial_logit"
    production_model: str = "linear"  # "cobb_douglas" | "linear"
    growth_model: str = "linear_convergence"  # "logistic_ode" | "linear_convergence"
    production_alpha: float = 0.3  # Cobb-Douglas capital exponent
    production_beta: float = 0.7  # Cobb-Douglas labor exponent
    base_competition: float = 0.5  # Lotka-Volterra off-diagonal alpha_ij
    growth_rate: float = 0.1  # Logistic ODE base growth rate

    # L-V fitness scaling — convert raw bridge attributes into competitive
    # fitness terms used in step_competition.
    # marketing_fitness_scale divides the bridge's marketing value before it
    # multiplies into the per-company growth rate (avoids letting unbounded
    # marketing dominate quality). Empirically tuned to ~10 so quality and
    # marketing exert comparable pressure.
    marketing_fitness_scale: float = 10.0
    # tam_capacity_fraction is the share of TAM each company can occupy at
    # full attractiveness (q^β · m^α). 0.01 → ~1% of TAM per unit of
    # attractiveness, which keeps total carrying-capacity sums bounded
    # within the modeled market.
    tam_capacity_fraction: float = 0.01


class CeoConfig(BaseModel):
    """Per-industry CEO agent configuration."""

    model: str = "claude-sonnet-4-6"
    interval_ticks: int = 182  # ~6 months (used as average for probabilistic mode)
    # Probabilistic activation (replaces fixed interval when enabled)
    base_activation_probability: float = 0.0055  # ~1/182 per tick on average
    crisis_multiplier: float = 3.0  # fires more often when cash is critical
    use_probabilistic_activation: bool = True
    price_min: float = 10.0
    price_max: float = 22.0
    price_default: float = 14.0
    cost_min: float = 1.00
    cost_max: float = 2.50
    cost_default: float = 1.50
    max_locations_per_year_cap: int = 12
    price_unit: str = "per unit"
    cost_unit: str = "unit cost"
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
    location_defaults: LocationDefaults
    math: MathConfig = MathConfig()
    ceo: CeoConfig = CeoConfig()
    display: DisplayConfig = DisplayConfig()


# ── Loader ──

INDUSTRY_DIR = Path(__file__).parent / "industries"
_cache: dict[str, IndustrySpec] = {}


def load_industry(slug: str) -> IndustrySpec:
    """Load and validate an industry config from YAML. Cached after first load."""
    if slug in _cache:
        return _cache[slug]

    path = INDUSTRY_DIR / f"{slug}.yaml"
    if not path.exists():
        raise ValueError(f"Industry config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    spec = IndustrySpec(**raw)

    # Auto-discover modifier keys when not explicitly declared. Any cost_modifier
    # key that appears on a node is added to cost_modifier_keys; same for revenue.
    # Explicit YAML lists win — only fill the empty case so curated YAMLs (e.g.,
    # restaurant declaring satisfaction_baseline as a satisfaction modifier rather
    # than a revenue boost) keep their semantics.
    ld = spec.location_defaults
    if not ld.cost_modifier_keys:
        ld.cost_modifier_keys = sorted({
            k for n in spec.nodes.values() for k in n.cost_modifiers
        })
    if not ld.revenue_modifier_keys and not ld.satisfaction_modifier_keys:
        ld.revenue_modifier_keys = sorted({
            k for n in spec.nodes.values() for k in n.revenue_modifiers
        })

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
    for path in sorted(INDUSTRY_DIR.glob("*.yaml")):
        specs.append(load_industry(path.stem))
    return specs


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write a YAML file atomically.

    Writes to a tempfile in the same directory then `os.replace`s into
    position. Avoids readers (e.g. registry refresh) seeing a half-written
    file when two adaptive sims race to commit.
    """
    fd, tmp_str = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".yaml.tmp"
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
