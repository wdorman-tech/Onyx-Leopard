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


class StageDef(BaseModel):
    min_locations: int
    stage: int


class LocationDefaults(BaseModel):
    inventory: float = 80.0
    customers: float = 30.0
    satisfaction: float = 0.7
    price: float = 14.0
    max_capacity: int = 80
    food_cost_per_plate: float = 1.50
    daily_fixed_costs: float = 300.0
    reorder_point: float = 30.0
    reorder_qty: float = 100.0
    chicken_cost_per_lb: float = 3.50
    spoilage_rate: float = 0.05
    word_of_mouth_rate: float = 0.02
    max_local_customers: float = 120.0
    unified_reorder_qty: float = 200.0
    unified_reorder_point: float = 80.0


class ConstantsDef(BaseModel):
    location_open_cost: float = 50000.0
    employees_per_location: int = 15
    starting_cash: float = 50000.0
    new_location_starting_customers: float = 20.0
    new_location_starting_satisfaction: float = 0.5
    volume_discounts: list[list[float]] = Field(
        default_factory=lambda: [[1, 1.0]]
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
