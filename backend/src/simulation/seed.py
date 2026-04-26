"""Business seed model for Onyx Leopard v2.

`CompanySeed` is the ~30-field input that fully describes a company at t=0
for the v2 stance-driven orchestrator. It replaces the per-industry YAML
`IndustrySpec` from v1. Cross-validation against `node_library.yaml` (i.e.
that `initial_supplier_types` etc. reference real node keys) happens in
`library_loader.py` to avoid a circular import.

Archetype samplers (`sample_seed_for_archetype`) draw realistic-range values
for Monte Carlo sweeps. Priors are anchored to the v1 industry YAMLs
(restaurant, saas_startup, adaptive_*) so a sampled `small_team` seed lands in
the same value space as the hand-tuned restaurant config.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EconomicsModel = Literal["physical", "subscription", "service"]
Archetype = Literal["solo_founder", "small_team", "venture_funded", "enterprise"]

ARCHETYPES: tuple[Archetype, ...] = (
    "solo_founder",
    "small_team",
    "venture_funded",
    "enterprise",
)

# Default node-library keys used by the archetype samplers. These are
# placeholders; library_loader.py validates them against the real
# node_library.yaml at runtime. Kept here as a contract: the seed knows
# what kinds of slots it wants filled, even if the library hasn't loaded yet.
_DEFAULT_SUPPLIERS: dict[EconomicsModel, list[str]] = {
    "physical": ["primary_goods_supplier", "packaging_supplier"],
    "subscription": ["cloud_provider", "payment_processor"],
    "service": ["ai_tooling_stack", "subcontractor_network"],
}
_DEFAULT_REVENUE_STREAMS: dict[EconomicsModel, list[str]] = {
    "physical": ["storefront_sales", "delivery_partnership"],
    "subscription": ["subscription_revenue", "usage_based_billing"],
    "service": ["engagement_revenue", "retainer_revenue"],
}
_DEFAULT_COST_CENTERS: dict[EconomicsModel, list[str]] = {
    "physical": ["cogs", "labor", "facilities"],
    "subscription": ["hosting_costs", "engineering_payroll", "sales_commissions"],
    "service": ["delivery_payroll", "tooling_costs", "subcontractor_fees"],
}

_LOCATION_LABELS: dict[EconomicsModel, str] = {
    "physical": "Location",
    "subscription": "Product",
    "service": "Engagement",
}
_CUSTOMER_UNITS: dict[EconomicsModel, str] = {
    "physical": "diners",
    "subscription": "subscribers",
    "service": "engagements",
}


class CompanySeed(BaseModel):
    """The 30-field company seed for v2 simulations.

    Numeric fields are bounded for realism; out-of-range values are rejected
    at construction time. Reference fields (`initial_supplier_types`,
    `initial_revenue_streams`, `initial_cost_centers`) are typed as
    `list[str]` here — cross-validation against `node_library.yaml` is the
    job of `library_loader.validate_seed_against_library()`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ── Identity (5) ──
    name: str = Field(..., min_length=1, max_length=120)
    niche: str = Field(..., min_length=1, max_length=400)
    archetype: Archetype
    industry_keywords: list[str] = Field(..., min_length=1, max_length=12)
    location_label: str = Field(..., min_length=1, max_length=40)

    # ── Economics (8) ──
    # Note: spec §2 lists 8 economics fields but enumerates only 7 named
    # entries (economics_model, base_price, base_unit_cost, daily_fixed_costs,
    # starting_cash, starting_employees, base_capacity_per_location,
    # margin_target). The 8th is `revenue_per_employee_target` — added so the
    # CEO orchestrator has a productivity yardstick when evaluating hires.
    economics_model: EconomicsModel
    base_price: float = Field(..., gt=0.0, le=10_000_000.0)
    base_unit_cost: float = Field(..., ge=0.0, le=10_000_000.0)
    daily_fixed_costs: float = Field(..., ge=0.0, le=1_000_000.0)
    starting_cash: float = Field(..., ge=0.0, le=1_000_000_000.0)
    starting_employees: int = Field(..., ge=1, le=100_000)
    base_capacity_per_location: int = Field(..., ge=1, le=1_000_000)
    margin_target: float = Field(..., ge=0.0, le=1.0)
    revenue_per_employee_target: float = Field(..., gt=0.0, le=10_000_000.0)

    # ── Market (5) ──
    tam: float = Field(..., gt=0.0, le=1e15)
    competitor_density: int = Field(..., ge=0, le=10)
    market_growth_rate: float = Field(..., ge=-0.5, le=2.0)
    customer_unit_label: str = Field(..., min_length=1, max_length=40)
    seasonality_amplitude: float = Field(..., ge=0.0, le=1.0)

    # ── Initial Setup (12) ──
    # Spec §2 calls for 12 initial-setup fields. The three list fields are
    # the primary content; the remaining 9 wire in the engine-knobs the
    # orchestrator uses on tick 0 (so the company isn't defined solely by
    # node-library refs — it has explicit starting positions too).
    initial_supplier_types: list[str] = Field(..., min_length=1, max_length=3)
    initial_revenue_streams: list[str] = Field(..., min_length=1, max_length=3)
    initial_cost_centers: list[str] = Field(..., min_length=1, max_length=3)
    initial_locations: int = Field(..., ge=1, le=1_000)
    initial_marketing_intensity: float = Field(..., ge=0.0, le=1.0)
    initial_quality_target: float = Field(..., ge=0.0, le=1.0)
    initial_price_position: Literal["discount", "mid", "premium"] = "mid"
    initial_capital_runway_months: float = Field(..., gt=0.0, le=120.0)
    initial_hiring_pace: Literal["frozen", "slow", "steady", "aggressive"] = "steady"
    initial_geographic_scope: Literal["local", "regional", "national", "global"] = "local"
    initial_revenue_concentration: float = Field(..., ge=0.0, le=1.0)
    initial_customer_acquisition_channel: Literal[
        "word_of_mouth", "outbound_sales", "paid_ads", "content", "partnerships"
    ] = "word_of_mouth"

    # ── Validators ──
    @field_validator(
        "initial_supplier_types",
        "initial_revenue_streams",
        "initial_cost_centers",
        "industry_keywords",
    )
    @classmethod
    def _no_blank_entries(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("list entries must be non-empty strings")
        return cleaned

    @field_validator("base_unit_cost")
    @classmethod
    def _cost_below_price(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        # base_unit_cost == base_price means zero gross margin, which is
        # legal for a loss-leader stance but a strict "cost > price" seed is
        # never economically coherent — reject it.
        price = info.data.get("base_price")
        if price is not None and v > price:
            raise ValueError(
                f"base_unit_cost ({v}) cannot exceed base_price ({price}) — "
                "negative gross margin is not a valid starting position"
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Archetype sampling — Monte Carlo entry point
# ─────────────────────────────────────────────────────────────────────────────


# Per-archetype value bands. Tuples are (low, high) ranges sampled uniformly
# unless noted. Anchored to v1 YAML values:
#   solo_founder    - consultancy YAMLs (28k-94k cash, $12k-$28k engagements)
#   small_team      - restaurant.yaml ($50k cash, $14 plate)
#   venture_funded  - saas_startup.yaml + analytics ($50k-$80k cash, $49-$299 sub)
#   enterprise      - multi-product/multi-location mature businesses
_ARCHETYPE_BANDS: dict[Archetype, dict[str, tuple[float, float]]] = {
    "solo_founder": {
        "starting_cash": (15_000.0, 100_000.0),
        "starting_employees_int": (1, 3),
        "daily_fixed_costs": (40.0, 250.0),
        "tam": (1e6, 5e7),
        "initial_capital_runway_months": (3.0, 12.0),
        "revenue_per_employee_target": (100_000.0, 250_000.0),
    },
    "small_team": {
        "starting_cash": (40_000.0, 200_000.0),
        "starting_employees_int": (3, 15),
        "daily_fixed_costs": (150.0, 800.0),
        "tam": (1e7, 5e8),
        "initial_capital_runway_months": (6.0, 18.0),
        "revenue_per_employee_target": (120_000.0, 300_000.0),
    },
    "venture_funded": {
        "starting_cash": (500_000.0, 15_000_000.0),
        "starting_employees_int": (5, 50),
        "daily_fixed_costs": (1_000.0, 12_000.0),
        "tam": (1e8, 1e10),
        "initial_capital_runway_months": (12.0, 36.0),
        "revenue_per_employee_target": (150_000.0, 400_000.0),
    },
    "enterprise": {
        "starting_cash": (10_000_000.0, 500_000_000.0),
        "starting_employees_int": (200, 10_000),
        "daily_fixed_costs": (15_000.0, 500_000.0),
        "tam": (1e9, 1e12),
        "initial_capital_runway_months": (18.0, 60.0),
        "revenue_per_employee_target": (200_000.0, 600_000.0),
    },
}

# Per-economics-model price/cost/capacity bands - pulled from the v1 YAMLs:
#   physical:     restaurant ($14 plate, $1.50 cost, 80-cap)
#   subscription: saas/analytics ($49-$850 sub, $2-$55 cost, 200-500 seats)
#   service:      consulting ($12k-$28k engagement, $1.8k-$4.2k cost, 4-6 cap)
_ECONOMICS_BANDS: dict[EconomicsModel, dict[str, tuple[float, float]]] = {
    "physical": {
        "base_price": (5.0, 60.0),
        "cost_ratio": (0.20, 0.55),  # base_unit_cost = base_price * ratio
        "base_capacity_per_location_int": (40, 250),
    },
    "subscription": {
        "base_price": (15.0, 1_500.0),
        "cost_ratio": (0.05, 0.30),
        "base_capacity_per_location_int": (100, 5_000),
    },
    "service": {
        "base_price": (3_000.0, 80_000.0),
        "cost_ratio": (0.15, 0.45),
        "base_capacity_per_location_int": (3, 30),
    },
}

# Probability weights for economics_model per archetype. Solo founders skew
# service (consultancies); venture-funded skews subscription (SaaS); small_team
# is balanced; enterprise can be anything.
_ECONOMICS_WEIGHTS: dict[Archetype, dict[EconomicsModel, float]] = {
    "solo_founder": {"physical": 0.15, "subscription": 0.20, "service": 0.65},
    "small_team": {"physical": 0.45, "subscription": 0.30, "service": 0.25},
    "venture_funded": {"physical": 0.15, "subscription": 0.70, "service": 0.15},
    "enterprise": {"physical": 0.40, "subscription": 0.40, "service": 0.20},
}


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = list(weights.values())
    return rng.choices(keys, weights=vals, k=1)[0]


def _sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _sample_int(rng: random.Random, lo: float, hi: float) -> int:
    return rng.randint(int(lo), int(hi))


def sample_seed_for_archetype(
    archetype: Archetype,
    *,
    rng: random.Random | None = None,
    name: str | None = None,
    niche: str | None = None,
    industry_keywords: list[str] | None = None,
    economics_model: EconomicsModel | None = None,
) -> CompanySeed:
    """Sample a realistic `CompanySeed` for the given archetype.

    Pass `rng` (a seeded `random.Random`) for reproducibility. Optional
    overrides let callers pin specific fields while randomizing the rest —
    useful for "same niche, vary capital" Monte Carlo sweeps.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}; expected one of {ARCHETYPES}")
    rng = rng or random.Random()

    bands = _ARCHETYPE_BANDS[archetype]
    chosen_economics: EconomicsModel = (
        economics_model
        if economics_model is not None
        else _weighted_choice(rng, _ECONOMICS_WEIGHTS[archetype])  # type: ignore[arg-type]
    )
    econ_bands = _ECONOMICS_BANDS[chosen_economics]

    base_price = round(_sample_uniform(rng, *econ_bands["base_price"]), 2)
    cost_ratio = _sample_uniform(rng, *econ_bands["cost_ratio"])
    base_unit_cost = round(base_price * cost_ratio, 2)
    margin_target = round(1.0 - cost_ratio - rng.uniform(0.0, 0.10), 3)
    margin_target = max(0.05, min(0.85, margin_target))

    starting_cash = round(_sample_uniform(rng, *bands["starting_cash"]), 2)
    starting_employees = _sample_int(rng, *bands["starting_employees_int"])
    daily_fixed = round(_sample_uniform(rng, *bands["daily_fixed_costs"]), 2)
    capacity = _sample_int(rng, *econ_bands["base_capacity_per_location_int"])

    runway = round(_sample_uniform(rng, *bands["initial_capital_runway_months"]), 1)
    revenue_per_emp = round(_sample_uniform(rng, *bands["revenue_per_employee_target"]), 2)

    tam = round(_sample_uniform(rng, *bands["tam"]), 2)
    competitor_density = rng.randint(0, 10)
    market_growth_rate = round(rng.uniform(-0.05, 0.40), 3)
    seasonality_amplitude = round(rng.uniform(0.0, 0.40), 3)

    initial_locations = _initial_locations_for(archetype, rng)
    marketing_intensity = round(rng.uniform(0.10, 0.80), 3)
    quality_target = round(rng.uniform(0.50, 0.95), 3)
    price_position = rng.choice(["discount", "mid", "mid", "premium"])  # mid-biased
    hiring_pace = _hiring_pace_for(archetype, rng)
    geographic_scope = _geographic_scope_for(archetype, rng)
    revenue_concentration = round(rng.uniform(0.10, 0.90), 3)
    acquisition_channel = _acquisition_channel_for(chosen_economics, rng)

    fallback_name = name or _default_name_for(archetype, chosen_economics, rng)
    fallback_niche = niche or _default_niche_for(chosen_economics)
    fallback_keywords = industry_keywords or _default_keywords_for(chosen_economics)

    return CompanySeed(
        name=fallback_name,
        niche=fallback_niche,
        archetype=archetype,
        industry_keywords=fallback_keywords,
        location_label=_LOCATION_LABELS[chosen_economics],
        economics_model=chosen_economics,
        base_price=base_price,
        base_unit_cost=base_unit_cost,
        daily_fixed_costs=daily_fixed,
        starting_cash=starting_cash,
        starting_employees=starting_employees,
        base_capacity_per_location=capacity,
        margin_target=margin_target,
        revenue_per_employee_target=revenue_per_emp,
        tam=tam,
        competitor_density=competitor_density,
        market_growth_rate=market_growth_rate,
        customer_unit_label=_CUSTOMER_UNITS[chosen_economics],
        seasonality_amplitude=seasonality_amplitude,
        initial_supplier_types=list(_DEFAULT_SUPPLIERS[chosen_economics]),
        initial_revenue_streams=list(_DEFAULT_REVENUE_STREAMS[chosen_economics]),
        initial_cost_centers=list(_DEFAULT_COST_CENTERS[chosen_economics]),
        initial_locations=initial_locations,
        initial_marketing_intensity=marketing_intensity,
        initial_quality_target=quality_target,
        initial_price_position=price_position,
        initial_capital_runway_months=runway,
        initial_hiring_pace=hiring_pace,
        initial_geographic_scope=geographic_scope,
        initial_revenue_concentration=revenue_concentration,
        initial_customer_acquisition_channel=acquisition_channel,
    )


def _initial_locations_for(archetype: Archetype, rng: random.Random) -> int:
    if archetype == "solo_founder":
        return 1
    if archetype == "small_team":
        return rng.randint(1, 3)
    if archetype == "venture_funded":
        return rng.randint(1, 5)
    return rng.randint(5, 50)  # enterprise


def _hiring_pace_for(
    archetype: Archetype, rng: random.Random
) -> Literal["frozen", "slow", "steady", "aggressive"]:
    weights: dict[Archetype, dict[str, float]] = {
        "solo_founder": {"frozen": 0.2, "slow": 0.5, "steady": 0.25, "aggressive": 0.05},
        "small_team": {"frozen": 0.05, "slow": 0.30, "steady": 0.50, "aggressive": 0.15},
        "venture_funded": {"frozen": 0.05, "slow": 0.10, "steady": 0.35, "aggressive": 0.50},
        "enterprise": {"frozen": 0.10, "slow": 0.30, "steady": 0.45, "aggressive": 0.15},
    }
    return _weighted_choice(rng, weights[archetype])  # type: ignore[return-value]


def _geographic_scope_for(
    archetype: Archetype, rng: random.Random
) -> Literal["local", "regional", "national", "global"]:
    weights: dict[Archetype, dict[str, float]] = {
        "solo_founder": {"local": 0.65, "regional": 0.30, "national": 0.05, "global": 0.0},
        "small_team": {"local": 0.45, "regional": 0.40, "national": 0.13, "global": 0.02},
        "venture_funded": {"local": 0.05, "regional": 0.20, "national": 0.50, "global": 0.25},
        "enterprise": {"local": 0.0, "regional": 0.10, "national": 0.45, "global": 0.45},
    }
    return _weighted_choice(rng, weights[archetype])  # type: ignore[return-value]


def _acquisition_channel_for(
    economics: EconomicsModel, rng: random.Random
) -> Literal["word_of_mouth", "outbound_sales", "paid_ads", "content", "partnerships"]:
    weights: dict[EconomicsModel, dict[str, float]] = {
        "physical": {
            "word_of_mouth": 0.40,
            "outbound_sales": 0.05,
            "paid_ads": 0.30,
            "content": 0.10,
            "partnerships": 0.15,
        },
        "subscription": {
            "word_of_mouth": 0.10,
            "outbound_sales": 0.25,
            "paid_ads": 0.30,
            "content": 0.25,
            "partnerships": 0.10,
        },
        "service": {
            "word_of_mouth": 0.30,
            "outbound_sales": 0.35,
            "paid_ads": 0.05,
            "content": 0.10,
            "partnerships": 0.20,
        },
    }
    return _weighted_choice(rng, weights[economics])  # type: ignore[return-value]


_NAME_PREFIXES = (
    "Onyx",
    "Apex",
    "Crest",
    "Nimbus",
    "Quill",
    "Helix",
    "Pivot",
    "Forge",
    "Lattice",
    "Echo",
    "Harbor",
    "Northwind",
    "Bright",
    "Anvil",
    "Sable",
    "Cinder",
    "Verge",
    "Mosaic",
)
_NAME_SUFFIXES_BY_ECON: dict[EconomicsModel, tuple[str, ...]] = {
    "physical": ("Eats", "Kitchen", "Foods", "Co.", "Goods", "Market"),
    "subscription": ("Cloud", "Labs", "AI", "Data", "Platform", "Stack"),
    "service": ("Group", "Partners", "Advisory", "Consulting", "Studio", "Works"),
}


def _default_name_for(archetype: Archetype, economics: EconomicsModel, rng: random.Random) -> str:
    return f"{rng.choice(_NAME_PREFIXES)} {rng.choice(_NAME_SUFFIXES_BY_ECON[economics])}"


_DEFAULT_NICHE_BY_ECON: dict[EconomicsModel, str] = {
    "physical": "Regional fast-casual chain serving a single signature menu category.",
    "subscription": "B2B SaaS platform sold per-seat to mid-market operations teams.",
    "service": "Boutique consultancy delivering AI automation to industrial SMBs.",
}
_DEFAULT_KEYWORDS_BY_ECON: dict[EconomicsModel, list[str]] = {
    "physical": ["restaurant", "fast-casual", "food-service"],
    "subscription": ["saas", "b2b", "platform"],
    "service": ["consulting", "ai", "professional-services"],
}


def _default_niche_for(economics: EconomicsModel) -> str:
    return _DEFAULT_NICHE_BY_ECON[economics]


def _default_keywords_for(economics: EconomicsModel) -> list[str]:
    return list(_DEFAULT_KEYWORDS_BY_ECON[economics])
