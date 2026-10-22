"""Shocks: discrete events that perturb the environment, arriving as Poisson processes.

Each shock type is data (`SHOCK_TYPES`): a scope, a description, and impact
ranges for a *minor* event. Severity scales both how far an impact deviates
from neutral and how long it lasts.

  * scope "market"  — scheduled once per sim, hits every company (crashes,
                      recessions, viral demand).
  * scope "company" — scheduled per company (departures, competitor entry).

Impact keys ending in `_mult` are multiplicative around 1.0 and compose by
product; all other keys are additive and compose by sum.
"""

from __future__ import annotations

import math
import random
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["minor", "moderate", "severe"]
Scope = Literal["market", "company"]

#: How far an impact deviates from neutral, relative to the minor-severity range.
SEVERITY_INTENSITY: Final[dict[Severity, float]] = {"minor": 1.0, "moderate": 2.0, "severe": 3.5}
#: Duration range in ticks (days) per severity.
SEVERITY_DURATION_DAYS: Final[dict[Severity, tuple[int, int]]] = {
    "minor": (14, 35),
    "moderate": (45, 90),
    "severe": (90, 180),
}
#: Probability of each severity when a shock arrives.
SEVERITY_WEIGHTS: Final[dict[Severity, float]] = {"minor": 0.60, "moderate": 0.30, "severe": 0.10}
#: Default arrival rate per type: ~1 event per type every 3 sim-years.
DEFAULT_LAMBDA_PER_TICK: Final[float] = 1.0 / 1095.0

MULT_SUFFIX: Final[str] = "_mult"

#: `impact` maps key → (low, high) deviation for a minor shock. For `_mult`
#: keys the drawn value is a fractional deviation from 1.0; otherwise a raw
#: additive magnitude. `duration_scale` stretches the severity duration.
SHOCK_TYPES: Final[dict[str, dict]] = {
    "market_crash": {
        "scope": "market",
        "description": "Broad demand contraction; external financing tightens.",
        "impact": {
            "market_demand_mult": (-0.20, -0.10),
            "tam_mult": (-0.15, -0.08),
            "financing_availability_mult": (-0.30, -0.15),
        },
    },
    "economic_recession": {
        "scope": "market",
        "description": "Sustained demand depression, tighter wallets, harder to raise.",
        "duration_scale": 1.5,
        "impact": {
            "market_demand_mult": (-0.15, -0.07),
            "consumer_spending_mult": (-0.12, -0.05),
            "financing_availability_mult": (-0.20, -0.10),
        },
    },
    "viral_growth_event": {
        "scope": "market",
        "description": "Niche trending: demand surges and acquisition gets cheaper, briefly.",
        "impact": {
            "market_demand_mult": (0.15, 0.40),
            "tam_mult": (0.10, 0.30),
            "acquisition_cost_mult": (-0.20, -0.10),
        },
    },
    "talent_war": {
        "scope": "market",
        "description": "Hiring costs spike, wage pressure, attrition risk elevated.",
        "impact": {
            "hire_cost_mult": (0.15, 0.35),
            "wage_inflation_mult": (0.05, 0.18),
            "attrition_mult": (0.10, 0.25),
        },
    },
    "regulatory_change": {
        "scope": "market",
        "description": "New compliance regime adds fixed cost and may cap capacity.",
        "impact": {
            "fixed_cost_mult": (0.05, 0.15),
            "capacity_cap_mult": (-0.10, -0.03),
            "compliance_cost_add": (1_500.0, 8_000.0),
        },
    },
    "new_competitor_entry": {
        "scope": "company",
        "description": "A well-funded entrant prices aggressively in our segment.",
        "impact": {
            "market_share_mult": (-0.12, -0.05),
            "price_ceiling_mult": (-0.10, -0.04),
        },
    },
    "key_employee_departure": {
        "scope": "company",
        "description": "A senior non-founder hire resigned; capacity and morale dip.",
        "impact": {
            "team_capacity_mult": (-0.12, -0.06),
            "satisfaction_mult": (-0.08, -0.03),
            "replacement_cost_add": (5_000.0, 20_000.0),
            "retire_random_employee": (1.0, 1.0),
        },
    },
    "supply_chain_disruption": {
        "scope": "company",
        "description": "Supplier costs jumped and replenishment is slower than usual.",
        "impact": {
            "supply_cost_mult": (0.10, 0.25),
            "inventory_throughput_mult": (-0.15, -0.07),
        },
    },
}

#: One-shot impact keys: applied once on arrival, not every active tick.
ONE_SHOT_KEYS: Final[frozenset[str]] = frozenset({"replacement_cost_add", "retire_random_employee"})


class Shock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    scope: Scope
    severity: Severity
    duration_ticks: int = Field(gt=0)
    impact: dict[str, float]
    description: str
    tick_started: int

    def active_at(self, tick: int) -> bool:
        return self.tick_started <= tick < self.tick_started + self.duration_ticks


def make_shock(name: str, severity: Severity, tick: int, rng: random.Random) -> Shock:
    spec = SHOCK_TYPES[name]
    intensity = SEVERITY_INTENSITY[severity]
    impact: dict[str, float] = {}
    for key, (lo, hi) in spec["impact"].items():
        if key.endswith(MULT_SUFFIX):
            impact[key] = 1.0 + rng.uniform(lo, hi) * intensity
        else:
            impact[key] = rng.uniform(lo, hi) * intensity
    lo_d, hi_d = SEVERITY_DURATION_DAYS[severity]
    return Shock(
        name=name,
        scope=spec["scope"],
        severity=severity,
        duration_ticks=int(rng.randint(lo_d, hi_d) * spec.get("duration_scale", 1.0)),
        impact=impact,
        description=f"{name.replace('_', ' ').title()} ({severity}): {spec['description']}",
        tick_started=tick,
    )


class ShockScheduler:
    """Poisson arrivals for every shock type in one scope. Deterministic per seed."""

    def __init__(self, scope: Scope, rng_seed: int, rate_multiplier: float = 1.0) -> None:
        if rate_multiplier < 0 or not math.isfinite(rate_multiplier):
            raise ValueError(f"rate_multiplier must be finite and >= 0; got {rate_multiplier}")
        self._rng = random.Random(rng_seed)
        self._types = sorted(n for n, s in SHOCK_TYPES.items() if s["scope"] == scope)
        self._lambda = DEFAULT_LAMBDA_PER_TICK * rate_multiplier

    def arrivals(self, tick: int) -> list[Shock]:
        if self._lambda == 0:
            return []
        out: list[Shock] = []
        for name in self._types:
            for _ in range(self._poisson(self._lambda)):
                severity = self._rng.choices(
                    list(SEVERITY_WEIGHTS), weights=list(SEVERITY_WEIGHTS.values())
                )[0]
                out.append(make_shock(name, severity, tick, self._rng))
        return out

    def _poisson(self, lam: float) -> int:
        """Knuth's method; λ ≪ 1 per tick so this terminates in ~1 draw."""
        threshold, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self._rng.random()
            if p <= threshold:
                return k
            k += 1


def environment(active: list[Shock]) -> dict[str, float]:
    """Compose continuous impacts of active shocks. Missing keys mean neutral."""
    env: dict[str, float] = {}
    for shock in active:
        for key, value in shock.impact.items():
            if key in ONE_SHOT_KEYS:
                continue
            if key.endswith(MULT_SUFFIX):
                env[key] = env.get(key, 1.0) * value
            else:
                env[key] = env.get(key, 0.0) + value
    return env
