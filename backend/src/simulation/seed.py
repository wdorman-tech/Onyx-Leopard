"""Company seed: everything that describes a company at t=0.

`CompanySeed` holds only fields the engine reads. Seeds come from three
places: archetype sampling (`sample_seed`), `.oleo` company profiles
(`seed_from_oleo`), or a client-supplied JSON body. Node references are
cross-validated against the node library in `library.py`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from src.simulation.library import NodeLibrary

EconomicsModel = Literal["physical", "subscription", "service"]
Archetype = Literal["solo_founder", "small_team", "venture_funded", "enterprise"]

ARCHETYPES: tuple[Archetype, ...] = ("solo_founder", "small_team", "venture_funded", "enterprise")

#: Cobb-Douglas returns-to-scale ceiling: α + β may exceed 1 (increasing
#: returns) but not past this.
MAX_RETURNS_TO_SCALE: float = 1.5


class CompanySeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    niche: str = Field(..., min_length=1, max_length=400)

    economics_model: EconomicsModel
    starting_price: float = Field(..., gt=0.0, le=10_000_000.0)
    base_unit_cost: float = Field(..., ge=0.0, le=10_000_000.0)
    daily_fixed_costs: float = Field(..., ge=0.0, le=1_000_000_000.0)
    starting_cash: float = Field(..., ge=0.0, le=1e12)
    starting_employees: int = Field(..., ge=1, le=10_000_000)
    #: Customers (units) per day one location can serve. Fractional is fine for
    #: high-ticket businesses (a consultancy closes ~0.05 engagements a day).
    base_capacity_per_location: float = Field(..., gt=0.0, le=1e8)
    initial_locations: int = Field(default=1, ge=1, le=1_000)

    #: Annual total addressable market, in dollars of customer spend.
    tam: float = Field(..., gt=0.0, le=1e15)
    #: Annual TAM growth rate (0.05 = 5%/yr).
    market_growth_rate: float = Field(..., ge=-0.5, le=2.0)
    #: Category price elasticity of demand (positive magnitude): how total
    #: customer spend responds to the market's average price. Below 1, an
    #: industry-wide price cut shrinks total spend; above 1, it grows it. The
    #: default of 1 keeps spend independent of price.
    market_price_elasticity: float = Field(default=1.0, gt=0.0, le=5.0)

    initial_supplier_types: list[str] = Field(..., min_length=1, max_length=3)
    initial_revenue_streams: list[str] = Field(..., min_length=1, max_length=3)
    initial_cost_centers: list[str] = Field(..., min_length=1, max_length=3)

    cobb_douglas_alpha: float = Field(..., gt=0.0, le=1.0)
    cobb_douglas_beta: float = Field(..., gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _economically_coherent(self) -> CompanySeed:
        if self.base_unit_cost > self.starting_price:
            raise ValueError("base_unit_cost cannot exceed starting_price (negative gross margin)")
        if self.cobb_douglas_alpha + self.cobb_douglas_beta > MAX_RETURNS_TO_SCALE:
            raise ValueError(f"cobb_douglas_alpha + beta must be <= {MAX_RETURNS_TO_SCALE}")
        return self


# ─── Defaults per economics model ───────────────────────────────────────────

_DEFAULT_REFS: dict[EconomicsModel, dict[str, list[str]]] = {
    "physical": {
        "initial_supplier_types": ["produce_supplier", "payment_processor"],
        "initial_revenue_streams": ["delivery_partnership", "loyalty_program"],
        "initial_cost_centers": ["owner_operator", "training_program"],
    },
    "subscription": {
        "initial_supplier_types": ["cloud_provider", "payment_processor"],
        "initial_revenue_streams": ["subscription_revenue", "usage_based_billing"],
        "initial_cost_centers": ["it_support", "founder_engineer"],
    },
    "service": {
        "initial_supplier_types": ["ai_tooling_stack", "ai_platform_partner"],
        "initial_revenue_streams": ["professional_services", "retainer_contracts"],
        "initial_cost_centers": ["it_support", "training_program"],
    },
}

_DEFAULT_NICHE: dict[EconomicsModel, str] = {
    "physical": "Regional fast-casual chain serving a single signature menu category.",
    "subscription": "B2B SaaS platform sold per-seat to mid-market operations teams.",
    "service": "Boutique consultancy delivering AI automation to industrial SMBs.",
}

_NAME_PREFIXES = (
    "Onyx", "Apex", "Crest", "Nimbus", "Quill", "Helix", "Pivot", "Forge", "Lattice",
    "Echo", "Harbor", "Northwind", "Bright", "Anvil", "Sable", "Cinder", "Verge", "Mosaic",
)  # fmt: skip
_NAME_SUFFIXES: dict[EconomicsModel, tuple[str, ...]] = {
    "physical": ("Eats", "Kitchen", "Foods", "Co.", "Goods", "Market"),
    "subscription": ("Cloud", "Labs", "AI", "Data", "Platform", "Stack"),
    "service": ("Group", "Partners", "Advisory", "Consulting", "Studio", "Works"),
}

# ─── Archetype sampling ─────────────────────────────────────────────────────
#
# Sampled seeds are internally coherent: a revenue target (headcount × revenue
# per employee) fixes capacity, overhead, cash and market size, so a sampled
# company starts near profitable in a contested market at any scale.

_ARCHETYPE_BANDS: dict[Archetype, dict[str, tuple[float, float]]] = {
    "solo_founder": {
        "starting_employees": (1, 3),
        "revenue_per_employee": (100_000.0, 250_000.0),
        "cash_to_revenue": (0.05, 0.30),
        "initial_locations": (1, 1),
        "alpha": (0.10, 0.25),
        "beta": (0.55, 0.80),
    },
    "small_team": {
        "starting_employees": (3, 15),
        "revenue_per_employee": (120_000.0, 300_000.0),
        "cash_to_revenue": (0.05, 0.30),
        "initial_locations": (1, 3),
        "alpha": (0.25, 0.45),
        "beta": (0.45, 0.65),
    },
    "venture_funded": {
        "starting_employees": (5, 50),
        "revenue_per_employee": (150_000.0, 400_000.0),
        "cash_to_revenue": (0.50, 3.00),
        "initial_locations": (1, 5),
        "alpha": (0.20, 0.40),
        "beta": (0.50, 0.75),
    },
    "enterprise": {
        "starting_employees": (200, 10_000),
        "revenue_per_employee": (200_000.0, 600_000.0),
        "cash_to_revenue": (0.10, 0.30),
        "initial_locations": (5, 50),
        "alpha": (0.35, 0.55),
        "beta": (0.40, 0.60),
    },
}

#: `market_elasticity` is the category (not single-firm) price elasticity.
#: Category demand is usually inelastic: food away from home ≈ 0.8 (Andreyeva
#: et al. 2010), business software and professional services lower still, as
#: firms buy them for a job that has to get done.
_ECONOMICS_BANDS: dict[EconomicsModel, dict[str, tuple[float, float]]] = {
    "physical": {"price": (5.0, 60.0), "cost_ratio": (0.20, 0.55), "market_elasticity": (0.6, 1.2)},
    "subscription": {
        "price": (15.0, 1_500.0),
        "cost_ratio": (0.05, 0.30),
        "market_elasticity": (0.4, 0.9),
    },
    "service": {
        "price": (3_000.0, 80_000.0),
        "cost_ratio": (0.15, 0.45),
        "market_elasticity": (0.3, 0.8),
    },
}

_ECONOMICS_WEIGHTS: dict[Archetype, dict[EconomicsModel, float]] = {
    "solo_founder": {"physical": 0.15, "subscription": 0.20, "service": 0.65},
    "small_team": {"physical": 0.45, "subscription": 0.30, "service": 0.25},
    "venture_funded": {"physical": 0.15, "subscription": 0.70, "service": 0.15},
    "enterprise": {"physical": 0.40, "subscription": 0.40, "service": 0.20},
}

#: Physical capacity relative to the revenue target (room to grow into).
CAPACITY_HEADROOM: float = 1.2
#: Operating margin (before marketing) at target revenue on day 0.
STARTING_OPERATING_MARGIN_BAND: tuple[float, float] = (0.08, 0.20)
#: Target revenue as a share of TAM. Several rivals at this share make a
#: market where capacity and demand are both in play.
INITIAL_MARKET_SHARE_BAND: tuple[float, float] = (0.10, 0.25)
#: Annual market growth band.
MARKET_GROWTH_BAND: tuple[float, float] = (-0.05, 0.40)


def company_name(economics: EconomicsModel, rng: random.Random) -> str:
    return f"{rng.choice(_NAME_PREFIXES)} {rng.choice(_NAME_SUFFIXES[economics])}"


def sample_seed(
    archetype: Archetype,
    rng: random.Random,
    library: NodeLibrary,
    economics_model: EconomicsModel | None = None,
) -> CompanySeed:
    """Sample a coherent seed: day-0 revenue covers the library cost of its starting nodes."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown seed archetype {archetype!r}; expected one of {ARCHETYPES}")
    bands = _ARCHETYPE_BANDS[archetype]
    weights = _ECONOMICS_WEIGHTS[archetype]
    economics: EconomicsModel = (
        economics_model or rng.choices(list(weights), weights=list(weights.values()))[0]
    )
    econ = _ECONOMICS_BANDS[economics]
    refs = _DEFAULT_REFS[economics]

    price = round(rng.uniform(*econ["price"]), 2)
    cost_ratio = rng.uniform(*econ["cost_ratio"])
    employees = rng.randint(*map(int, bands["starting_employees"]))
    locations = rng.randint(*map(int, bands["initial_locations"]))
    # Fixed costs must leave the sampled operating margin at target revenue:
    # R * (1 - cost_ratio - margin) = node costs + overhead.
    fixed_share = 1.0 - cost_ratio - rng.uniform(*STARTING_OPERATING_MARGIN_BAND)
    node_burn = library.daily_cost(
        library.starting_nodes([k for ks in refs.values() for k in ks], economics, locations)
    )
    target = employees * rng.uniform(*bands["revenue_per_employee"]) / 365.0
    daily_revenue = max(target, node_burn / fixed_share)
    annual_revenue = daily_revenue * 365.0

    return CompanySeed(
        name=company_name(economics, rng),
        niche=_DEFAULT_NICHE[economics],
        economics_model=economics,
        starting_price=price,
        base_unit_cost=round(price * cost_ratio, 2),
        daily_fixed_costs=round(daily_revenue * fixed_share - node_burn, 2),
        starting_cash=round(annual_revenue * rng.uniform(*bands["cash_to_revenue"]), 2),
        starting_employees=employees,
        base_capacity_per_location=max(
            1e-4, round(daily_revenue / price * CAPACITY_HEADROOM / locations, 4)
        ),
        initial_locations=locations,
        tam=round(annual_revenue / rng.uniform(*INITIAL_MARKET_SHARE_BAND), 2),
        market_growth_rate=round(rng.uniform(*MARKET_GROWTH_BAND), 3),
        cobb_douglas_alpha=round(rng.uniform(*bands["alpha"]), 3),
        cobb_douglas_beta=round(rng.uniform(*bands["beta"]), 3),
        market_price_elasticity=round(rng.uniform(*econ["market_elasticity"]), 2),
        **refs,
    )


# ─── .oleo company profiles ─────────────────────────────────────────────────

#: Representative unit price per economics model when a profile has none.
_OLEO_UNIT_PRICE: dict[EconomicsModel, float] = {
    "physical": 50.0,
    "subscription": 120.0,
    "service": 25_000.0,
}
#: Fraction of annual revenue held as starting cash when the profile omits cash.
_OLEO_CASH_FRACTION_OF_REVENUE: float = 0.15
#: Share of non-COGS daily revenue treated as fixed overhead.
_OLEO_FIXED_COST_SHARE: float = 0.35
#: Headcount per location used to infer a location count from org size.
_OLEO_EMPLOYEES_PER_LOCATION: int = 5_000
#: Bounds on an imported `market.price_elasticity_estimate` (magnitude).
_OLEO_ELASTICITY_BOUNDS: tuple[float, float] = (0.1, 5.0)


def _num(value: Any, fallback: float) -> float:
    return float(value) if isinstance(value, int | float) and value == value else fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _oleo_economics(identity: dict[str, Any]) -> EconomicsModel:
    text = f"{identity.get('business_model', '')} {identity.get('industry', '')}".lower()
    if any(word in text for word in ("saas", "software", "cloud")):
        return "subscription"
    if "service" in text or "consulting" in text:
        return "service"
    return "physical"


def seed_from_oleo(data: dict[str, Any]) -> CompanySeed:
    """Convert an `.oleo` company profile export into a `CompanySeed`."""
    profile = data.get("profile", {})
    identity = profile.get("identity", {})
    org = profile.get("organization", {})
    fin = profile.get("financials", {})
    market = profile.get("market", {})

    name = str(identity.get("name") or "Imported Company")
    economics = _oleo_economics(identity)
    price = _OLEO_UNIT_PRICE[economics]
    headcount = int(_clamp(_num(org.get("total_headcount"), 10), 1, 10_000_000))
    revenue = _num(fin.get("annual_revenue"), 1_000_000.0)
    margin = _clamp(_num(fin.get("gross_margin"), 0.55), 0.05, 0.85)
    share = _clamp(_num(market.get("market_share"), 0.05), 0.001, 0.95)
    daily_revenue = revenue / 365.0
    cash = _num(fin.get("cash"), revenue * _OLEO_CASH_FRACTION_OF_REVENUE)
    locations = int(_clamp(-(-headcount // _OLEO_EMPLOYEES_PER_LOCATION), 1, 1_000))

    return CompanySeed(
        name=name,
        niche=str(identity.get("description") or f"{name} in {identity.get('industry', '')}")[:400],
        economics_model=economics,
        starting_price=price,
        base_unit_cost=round(price * (1.0 - margin), 2),
        daily_fixed_costs=round(
            _clamp(daily_revenue * (1.0 - margin) * _OLEO_FIXED_COST_SHARE, 1, 1e9), 2
        ),
        starting_cash=_clamp(cash, 1_000, 1e12),
        starting_employees=headcount,
        base_capacity_per_location=round(
            _clamp(daily_revenue / price * CAPACITY_HEADROOM / locations, 1e-4, 1e8), 4
        ),
        initial_locations=locations,
        tam=_clamp(_num(market.get("tam"), revenue / share), 1, 1e15),
        market_growth_rate=_clamp(
            _num(fin.get("revenue_growth_rate"), _num(market.get("market_growth_rate"), 0.08)),
            -0.5,
            2.0,
        ),
        cobb_douglas_alpha=0.45 if economics == "physical" else 0.35,
        cobb_douglas_beta=0.55 if economics == "physical" else 0.65,
        market_price_elasticity=_clamp(
            abs(_num(market.get("price_elasticity_estimate"), 1.0)), *_OLEO_ELASTICITY_BOUNDS
        ),
        **_DEFAULT_REFS[economics],
    )
