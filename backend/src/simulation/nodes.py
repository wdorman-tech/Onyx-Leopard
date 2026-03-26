"""Node type registry — defines costs, modifiers, and metadata for all 50 node types.

Architecture note (future direction):
    The node classification system is designed to be industry-agnostic. Right now
    NODE_REGISTRY contains grilled-chicken-restaurant nodes, but the engine itself
    (GrowthEngine) doesn't know or care about the specific node types — it reads
    from the registry and trigger list at runtime.

    To model a different company type (tech startup, logistics co, retail chain,
    etc.), you define a new NodeType enum, a new NODE_REGISTRY mapping, and a new
    TRIGGER_REGISTRY. The engine, location tick, modifier composition, and graph
    snapshot system all work unchanged. Each industry gets its own node
    classification taxonomy and growth triggers while sharing the same simulation
    infrastructure.

    This means the platform can eventually let users pick a company archetype and
    watch it grow with industry-specific nodes appearing on the graph — different
    departments, infrastructure, partnerships, and revenue streams for each type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.simulation.models import NodeCategory, NodeType


@dataclass
class NodeConfig:
    label: str
    category: NodeCategory
    stage: int  # minimum growth stage to appear
    annual_cost: float = 0.0
    cost_modifiers: dict[str, float] = field(default_factory=dict)
    revenue_modifiers: dict[str, float] = field(default_factory=dict)
    enabled: bool = True  # False = deferred to v2


NODE_REGISTRY: dict[NodeType, NodeConfig] = {
    # ── Locations (v1) ──
    NodeType.RESTAURANT: NodeConfig(
        label="Restaurant",
        category=NodeCategory.LOCATION,
        stage=1,
        annual_cost=109_500,  # ~$300/day fixed
    ),
    NodeType.COMMISSARY: NodeConfig(
        label="Commissary Kitchen",
        category=NodeCategory.LOCATION,
        stage=2,
        annual_cost=180_000,
        cost_modifiers={"labor": -0.15},
    ),
    NodeType.DISTRIBUTION_CENTER: NodeConfig(
        label="Distribution Center",
        category=NodeCategory.LOCATION,
        stage=3,
        annual_cost=300_000,
        cost_modifiers={"food_delivery": -0.10},
    ),
    # ── Locations (v2) ──
    NodeType.DRIVE_THRU: NodeConfig(
        label="Drive-Thru Unit", category=NodeCategory.LOCATION, stage=3,
        annual_cost=120_000, enabled=False,
    ),
    NodeType.GHOST_KITCHEN: NodeConfig(
        label="Ghost Kitchen", category=NodeCategory.LOCATION, stage=3,
        annual_cost=72_000, enabled=False,
    ),
    NodeType.FOOD_COURT: NodeConfig(
        label="Food Court Location", category=NodeCategory.LOCATION, stage=3,
        annual_cost=96_000, enabled=False,
    ),
    NodeType.FRANCHISE_LOCATION: NodeConfig(
        label="Franchise Location", category=NodeCategory.LOCATION, stage=4,
        annual_cost=0, enabled=False,
    ),
    NodeType.TEST_LOCATION: NodeConfig(
        label="Test Location", category=NodeCategory.LOCATION, stage=3,
        annual_cost=150_000, enabled=False,
    ),
    # ── Corporate (v1) ──
    NodeType.OWNER_OPERATOR: NodeConfig(
        label="Owner / Operator",
        category=NodeCategory.CORPORATE,
        stage=1,
        annual_cost=0,
    ),
    NodeType.GENERAL_MANAGER: NodeConfig(
        label="General Manager",
        category=NodeCategory.CORPORATE,
        stage=1,
        annual_cost=54_000,
    ),
    NodeType.BOOKKEEPER: NodeConfig(
        label="Bookkeeper",
        category=NodeCategory.CORPORATE,
        stage=1,
        annual_cost=36_000,
        cost_modifiers={"waste": -0.03},
    ),
    NodeType.MARKETING: NodeConfig(
        label="Marketing Manager",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=66_000,
        revenue_modifiers={"customer_growth": 0.40},
    ),
    NodeType.HR: NodeConfig(
        label="HR Department",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=60_000,
        cost_modifiers={"turnover": -0.15},
    ),
    NodeType.TRAINING: NodeConfig(
        label="Training Program",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=48_000,
        revenue_modifiers={"new_location_satisfaction": 0.05},
    ),
    NodeType.AREA_MANAGER: NodeConfig(
        label="Area Manager",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=90_000,
    ),
    NodeType.QUALITY_ASSURANCE: NodeConfig(
        label="Quality Assurance",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=60_000,
        revenue_modifiers={"satisfaction_baseline": 0.03},
    ),
    NodeType.PROCUREMENT: NodeConfig(
        label="Procurement / Supply Chain",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=72_000,
        cost_modifiers={"food_cost": -0.08},
    ),
    NodeType.IT_SUPPORT: NodeConfig(
        label="IT Support",
        category=NodeCategory.CORPORATE,
        stage=2,
        annual_cost=60_000,
        cost_modifiers={"operating": -0.02},
    ),
    NodeType.REAL_ESTATE: NodeConfig(
        label="Real Estate Department",
        category=NodeCategory.CORPORATE,
        stage=3,
        annual_cost=120_000,
    ),
    NodeType.CONSTRUCTION: NodeConfig(
        label="Construction & Facilities",
        category=NodeCategory.CORPORATE,
        stage=3,
        annual_cost=96_000,
        cost_modifiers={"build_cost": -0.20},
    ),
    NodeType.RND_MENU: NodeConfig(
        label="R&D / Menu Development",
        category=NodeCategory.CORPORATE,
        stage=3,
        annual_cost=96_000,
        revenue_modifiers={"menu_innovation": 0.05},
    ),
    NodeType.LEGAL: NodeConfig(
        label="Legal Department",
        category=NodeCategory.CORPORATE,
        stage=3,
        annual_cost=96_000,
    ),
    NodeType.FINANCE_FPA: NodeConfig(
        label="Finance & FP&A",
        category=NodeCategory.CORPORATE,
        stage=3,
        annual_cost=84_000,
        cost_modifiers={"financial_controls": -0.02},
    ),
    # ── Corporate (v2) ──
    NodeType.CORPORATE_HQ: NodeConfig(
        label="Corporate HQ", category=NodeCategory.CORPORATE, stage=3,
        annual_cost=200_000, enabled=False,
    ),
    NodeType.CEO: NodeConfig(
        label="CEO", category=NodeCategory.CORPORATE, stage=3,
        annual_cost=250_000, enabled=False,
    ),
    NodeType.COO: NodeConfig(
        label="COO", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=200_000, enabled=False,
    ),
    NodeType.CFO: NodeConfig(
        label="CFO", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=200_000, enabled=False,
    ),
    NodeType.CMO: NodeConfig(
        label="CMO", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=180_000, enabled=False,
    ),
    NodeType.CUSTOMER_SERVICE: NodeConfig(
        label="Customer Service", category=NodeCategory.CORPORATE, stage=2,
        annual_cost=48_000, enabled=False,
    ),
    NodeType.FRANCHISE_DEV: NodeConfig(
        label="Franchise Development", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=120_000, enabled=False,
    ),
    NodeType.FRANCHISE_ADVISORY: NodeConfig(
        label="Franchise Advisory Council", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=24_000, enabled=False,
    ),
    NodeType.REGIONAL_VP: NodeConfig(
        label="Regional VP", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=150_000, enabled=False,
    ),
    # ── External (v1) ──
    NodeType.CHICKEN_SUPPLIER: NodeConfig(
        label="Chicken Supplier",
        category=NodeCategory.EXTERNAL,
        stage=1,
        annual_cost=0,
    ),
    NodeType.PRODUCE_SUPPLIER: NodeConfig(
        label="Produce & Packaging Supplier",
        category=NodeCategory.EXTERNAL,
        stage=1,
        annual_cost=0,
    ),
    # ── External (v2) ──
    NodeType.BEVERAGE_SUPPLIER: NodeConfig(
        label="Beverage Supplier", category=NodeCategory.EXTERNAL, stage=2,
        annual_cost=0, enabled=False,
    ),
    NodeType.LANDLORD: NodeConfig(
        label="Landlord", category=NodeCategory.EXTERNAL, stage=1,
        annual_cost=0, enabled=False,
    ),
    NodeType.FRANCHISE_OWNER: NodeConfig(
        label="Franchise Owner", category=NodeCategory.EXTERNAL, stage=4,
        annual_cost=0, enabled=False,
    ),
    NodeType.AREA_DEVELOPER: NodeConfig(
        label="Area Developer", category=NodeCategory.EXTERNAL, stage=4,
        annual_cost=0, enabled=False,
    ),
    # ── Revenue (v1) ──
    NodeType.CATERING: NodeConfig(
        label="Catering Division",
        category=NodeCategory.REVENUE,
        stage=2,
        annual_cost=48_000,
        revenue_modifiers={"catering_revenue": 0.15},
    ),
    NodeType.DELIVERY_PARTNERSHIP: NodeConfig(
        label="Delivery Partnerships",
        category=NodeCategory.REVENUE,
        stage=2,
        annual_cost=24_000,
        revenue_modifiers={"customer_reach": 0.20},
    ),
    # ── Revenue (v2) ──
    NodeType.LOYALTY_PROGRAM: NodeConfig(
        label="Loyalty Program", category=NodeCategory.REVENUE, stage=4,
        annual_cost=60_000, enabled=False,
    ),
    NodeType.MERCHANDISE: NodeConfig(
        label="Merchandise / Retail", category=NodeCategory.REVENUE, stage=4,
        annual_cost=36_000, enabled=False,
    ),
    NodeType.LICENSING_IP: NodeConfig(
        label="Licensing & IP", category=NodeCategory.REVENUE, stage=5,
        annual_cost=48_000, enabled=False,
    ),
    NodeType.GIFT_CARD: NodeConfig(
        label="Gift Card Program", category=NodeCategory.REVENUE, stage=4,
        annual_cost=24_000, enabled=False,
    ),
    # ── Specialized (v2) ──
    NodeType.FOOD_SAFETY_LAB: NodeConfig(
        label="Food Safety Lab", category=NodeCategory.CORPORATE, stage=4,
        annual_cost=120_000, enabled=False,
    ),
    NodeType.SECRET_SHOPPER: NodeConfig(
        label="Secret Shopper Program", category=NodeCategory.CORPORATE, stage=3,
        annual_cost=36_000, enabled=False,
    ),
    NodeType.SUSTAINABILITY: NodeConfig(
        label="Sustainability Team", category=NodeCategory.CORPORATE, stage=5,
        annual_cost=96_000, enabled=False,
    ),
    NodeType.INVESTOR_RELATIONS: NodeConfig(
        label="Investor Relations", category=NodeCategory.CORPORATE, stage=5,
        annual_cost=120_000, enabled=False,
    ),
    NodeType.CAPTIVE_INSURANCE: NodeConfig(
        label="Captive Insurance Entity", category=NodeCategory.CORPORATE, stage=5,
        annual_cost=60_000, enabled=False,
    ),
}
