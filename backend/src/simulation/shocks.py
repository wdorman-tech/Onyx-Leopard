"""Shock library and Poisson-arrival scheduler for the v2 simulation engine.

Shocks make simulations dynamic. Without them, every CEO stance just optimizes
its happy-path. With them, you observe how stance interacts with adversity —
the actual research output of the platform.

This module is standalone:
    - `Shock`            — pydantic v2 model, the data contract.
    - `make_*`           — 8 factory functions, one per built-in shock type.
    - `ShockScheduler`   — per-sim Poisson arrival schedule, deterministic
                           given an RNG seed. Tracks active shocks and prunes
                           them when their duration expires.
    - `apply_active_shocks` — composes multiplicative + additive impacts from
                              the active set into a derived state dict.

Determinism: every random draw goes through a `random.Random` instance that
the caller seeds. No `time.time()`, no `os.urandom()`, no module-level RNG.
"""

from __future__ import annotations

import math
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Severity = Literal["minor", "moderate", "severe"]

#: Scales the *intensity* of an impact by severity. Intensity is "how far the
#: multiplier deviates from neutral (1.0)" or, for additive impacts, the raw
#: magnitude. Tuned so severe > moderate > minor with meaningful spread.
SEVERITY_INTENSITY: dict[Severity, float] = {
    "minor": 1.0,
    "moderate": 2.0,
    "severe": 3.5,
}

#: Severity-scaled duration multiplier. A "minor" event lingers about a month;
#: a "severe" one lingers about a quarter-plus. Tick = 1 sim day.
SEVERITY_DURATION_DAYS: dict[Severity, tuple[int, int]] = {
    "minor": (14, 35),
    "moderate": (45, 90),
    "severe": (90, 180),
}

#: Suffix convention for impact keys:
#:   - `*_mult`  → multiplicative around 1.0; composed by product
#:   - everything else → additive; composed by sum
MULT_SUFFIX = "_mult"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Shock(BaseModel):
    """A discrete event that perturbs the simulation environment.

    Instances are produced by the `make_*` factories, scheduled by
    `ShockScheduler`, and consumed by `apply_active_shocks` and (later) the
    CEO orchestrator.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str
    severity: Severity
    duration_ticks: int = Field(gt=0, description="How long the shock stays active (ticks).")
    impact: dict[str, float] = Field(
        default_factory=dict,
        description="Per-key environmental perturbation. Keys ending in '_mult' are "
        "multiplicative around 1.0; others are additive.",
    )
    description: str = Field(default="", description="Human-readable, fed to CEO context.")
    tick_started: int | None = Field(
        default=None,
        description="Set by the scheduler when the shock activates. None = not yet injected.",
    )

    def is_active(self, current_tick: int) -> bool:
        """True iff the shock has been injected and has not yet expired."""
        if self.tick_started is None:
            return False
        return current_tick < self.tick_started + self.duration_ticks


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _scaled_mult(rng: random.Random, severity: Severity, base_min: float, base_max: float) -> float:
    """Draw a multiplicative impact whose deviation from 1.0 scales with severity.

    `base_min` / `base_max` describe the *minor*-severity deviation range from 1.0.
    A `base_min=-0.10, base_max=-0.05` for severity="severe" yields ~3.5x deeper cuts.
    """
    intensity = SEVERITY_INTENSITY[severity]
    raw_dev = rng.uniform(base_min, base_max)
    return 1.0 + raw_dev * intensity


def _scaled_additive(
    rng: random.Random, severity: Severity, base_min: float, base_max: float
) -> float:
    """Draw an additive impact whose magnitude scales linearly with severity."""
    intensity = SEVERITY_INTENSITY[severity]
    return rng.uniform(base_min, base_max) * intensity


def _duration_for(rng: random.Random, severity: Severity) -> int:
    lo, hi = SEVERITY_DURATION_DAYS[severity]
    return rng.randint(lo, hi)


def _validate_severity(severity: str) -> Severity:
    if severity not in SEVERITY_INTENSITY:
        raise ValueError(
            f"Unknown severity '{severity}'. Expected one of {list(SEVERITY_INTENSITY)}."
        )
    return severity  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Built-in shock factories (8)
# ---------------------------------------------------------------------------


def make_market_crash(rng: random.Random, severity: str = "moderate") -> Shock:
    """TAM contraction + cash-tightening (credit/financing dries up)."""
    sev = _validate_severity(severity)
    return Shock(
        name="market_crash",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "market_demand_mult": _scaled_mult(rng, sev, -0.20, -0.10),
            "tam_mult": _scaled_mult(rng, sev, -0.15, -0.08),
            "financing_availability_mult": _scaled_mult(rng, sev, -0.30, -0.15),
        },
        description=(
            f"Market crash ({sev}): broad demand contraction, TAM compresses, "
            "external financing tightens."
        ),
    )


def make_new_competitor_entry(rng: random.Random, severity: str = "moderate") -> Shock:
    """A funded competitor enters with predatory pricing — share + price pressure."""
    sev = _validate_severity(severity)
    return Shock(
        name="new_competitor_entry",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "competitor_pressure_mult": _scaled_mult(rng, sev, 0.05, 0.15),
            "market_share_mult": _scaled_mult(rng, sev, -0.12, -0.05),
            "price_ceiling_mult": _scaled_mult(rng, sev, -0.10, -0.04),
        },
        description=(
            f"New competitor entry ({sev}): well-funded entrant pricing aggressively "
            "in our segment."
        ),
    )


def make_key_employee_departure(rng: random.Random, severity: str = "moderate") -> Shock:
    """Critical hire leaves — capacity dip, satisfaction wobble, replacement cost."""
    sev = _validate_severity(severity)
    return Shock(
        name="key_employee_departure",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "team_capacity_mult": _scaled_mult(rng, sev, -0.12, -0.06),
            "satisfaction_mult": _scaled_mult(rng, sev, -0.08, -0.03),
            "replacement_cost_add": _scaled_additive(rng, sev, 5_000.0, 20_000.0),
            "retire_random_employee": 1.0,  # signal to orchestrator
        },
        description=(
            f"Key employee departure ({sev}): a senior non-founder hire resigned. "
            "Pipeline knowledge and morale take a hit."
        ),
    )


def make_supply_chain_disruption(rng: random.Random, severity: str = "moderate") -> Shock:
    """Supplier costs spike + lead-time delay reduces effective inventory turn."""
    sev = _validate_severity(severity)
    return Shock(
        name="supply_chain_disruption",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "supply_cost_mult": _scaled_mult(rng, sev, 0.10, 0.25),
            "lead_time_mult": _scaled_mult(rng, sev, 0.20, 0.50),
            "inventory_throughput_mult": _scaled_mult(rng, sev, -0.15, -0.07),
        },
        description=(
            f"Supply chain disruption ({sev}): supplier costs jumped and replenishment "
            "is slower than usual."
        ),
    )


def make_regulatory_change(rng: random.Random, severity: str = "moderate") -> Shock:
    """New rules: higher fixed costs (compliance) + a possible capacity cap."""
    sev = _validate_severity(severity)
    return Shock(
        name="regulatory_change",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "fixed_cost_mult": _scaled_mult(rng, sev, 0.05, 0.15),
            "capacity_cap_mult": _scaled_mult(rng, sev, -0.10, -0.03),
            "compliance_cost_add": _scaled_additive(rng, sev, 1_500.0, 8_000.0),
        },
        description=(
            f"Regulatory change ({sev}): new compliance regime adds ongoing fixed cost "
            "and may cap effective capacity."
        ),
    )


def make_viral_growth_event(rng: random.Random, severity: str = "moderate") -> Shock:
    """Sudden TAM surge in our niche keywords — limited window, high upside."""
    sev = _validate_severity(severity)
    return Shock(
        name="viral_growth_event",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "market_demand_mult": _scaled_mult(rng, sev, 0.15, 0.40),
            "tam_mult": _scaled_mult(rng, sev, 0.10, 0.30),
            "acquisition_cost_mult": _scaled_mult(rng, sev, -0.20, -0.10),
        },
        description=(
            f"Viral growth event ({sev}): niche keywords trending. Demand surges, "
            "CAC drops, but the window is finite."
        ),
    )


def make_economic_recession(rng: random.Random, severity: str = "moderate") -> Shock:
    """Multi-quarter demand depression — longer than a market_crash, less acute."""
    sev = _validate_severity(severity)
    # Recessions linger — bias duration upward by 50%.
    base_duration = _duration_for(rng, sev)
    return Shock(
        name="economic_recession",
        severity=sev,
        duration_ticks=int(base_duration * 1.5),
        impact={
            "market_demand_mult": _scaled_mult(rng, sev, -0.15, -0.07),
            "consumer_spending_mult": _scaled_mult(rng, sev, -0.12, -0.05),
            "financing_availability_mult": _scaled_mult(rng, sev, -0.20, -0.10),
        },
        description=(
            f"Economic recession ({sev}): sustained demand depression, tighter "
            "consumer wallets, harder to raise."
        ),
    )


def make_talent_war(rng: random.Random, severity: str = "moderate") -> Shock:
    """Hire costs spike across all node types; existing employees harder to retain."""
    sev = _validate_severity(severity)
    return Shock(
        name="talent_war",
        severity=sev,
        duration_ticks=_duration_for(rng, sev),
        impact={
            "hire_cost_mult": _scaled_mult(rng, sev, 0.15, 0.35),
            "wage_inflation_mult": _scaled_mult(rng, sev, 0.05, 0.18),
            "attrition_mult": _scaled_mult(rng, sev, 0.10, 0.25),
        },
        description=(
            f"Talent war ({sev}): hiring costs spike, wage pressure across the org, "
            "attrition risk elevated."
        ),
    )


#: Registry of all built-in shock factories, keyed by `Shock.name`.
SHOCK_FACTORIES: dict[str, callable] = {
    "market_crash": make_market_crash,
    "new_competitor_entry": make_new_competitor_entry,
    "key_employee_departure": make_key_employee_departure,
    "supply_chain_disruption": make_supply_chain_disruption,
    "regulatory_change": make_regulatory_change,
    "viral_growth_event": make_viral_growth_event,
    "economic_recession": make_economic_recession,
    "talent_war": make_talent_war,
}


# ---------------------------------------------------------------------------
# Severity sampler
# ---------------------------------------------------------------------------

#: Default per-tick distribution over severity buckets when the scheduler
#: rolls a shock arrival. Most shocks are minor; severe events are rare.
DEFAULT_SEVERITY_WEIGHTS: dict[Severity, float] = {
    "minor": 0.60,
    "moderate": 0.30,
    "severe": 0.10,
}


def _draw_severity(rng: random.Random, weights: dict[Severity, float]) -> Severity:
    severities = list(weights.keys())
    cum_weights = [weights[s] for s in severities]
    return rng.choices(severities, weights=cum_weights, k=1)[0]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ShockScheduler:
    """Poisson-process arrival schedule for shocks, per shock type.

    Each shock type has its own λ (mean arrivals per tick). Per tick, for each
    type, draws an arrival count from Poisson(λ) and instantiates that many
    shocks via the registered factory. Active shocks are tracked and pruned
    when their `duration_ticks` window expires.

    Determinism: a single `random.Random` seeded by the caller drives every
    draw. Two schedulers with the same seed and same `lambdas` produce
    byte-identical arrival sequences.
    """

    def __init__(
        self,
        rng_seed: int,
        lambdas: dict[str, float] | None = None,
        severity_weights: dict[Severity, float] | None = None,
        factories: dict[str, callable] | None = None,
    ) -> None:
        self._rng = random.Random(rng_seed)
        self._factories = factories if factories is not None else dict(SHOCK_FACTORIES)
        self._severity_weights = (
            severity_weights if severity_weights is not None else dict(DEFAULT_SEVERITY_WEIGHTS)
        )
        # Default lambda per type: ~1 expected arrival per ~3 sim years
        # (1095 ticks) -> lambda ~= 9.1e-4. Tuned so a 5-year sim sees ~1-2
        # events per type on average, never an avalanche. Caller can override.
        default_lambda = 1.0 / 1095.0
        self._lambdas: dict[str, float] = {name: default_lambda for name in self._factories}
        if lambdas:
            unknown = set(lambdas) - set(self._factories)
            if unknown:
                raise ValueError(
                    f"Unknown shock types in `lambdas`: {sorted(unknown)}. "
                    f"Known: {sorted(self._factories)}."
                )
            self._lambdas.update(lambdas)
        # Validate λ ≥ 0
        for name, lam in self._lambdas.items():
            if lam < 0 or not math.isfinite(lam):
                raise ValueError(f"Lambda for '{name}' must be finite and ≥ 0; got {lam}.")
        self._active: list[Shock] = []

    # -- accessors -----------------------------------------------------------

    @property
    def active(self) -> list[Shock]:
        """Currently-active shocks (read-only view; modify via tick())."""
        return list(self._active)

    @property
    def lambdas(self) -> dict[str, float]:
        return dict(self._lambdas)

    # -- main loop -----------------------------------------------------------

    def tick(self, current_tick: int) -> list[Shock]:
        """Advance the schedule by one tick.

        Returns the list of shocks that *arrived this tick* (already added to
        the active set with `tick_started=current_tick`). Expired shocks are
        pruned from the active set first.
        """
        # 1. Prune expired
        self._active = [s for s in self._active if s.is_active(current_tick)]

        # 2. Roll arrivals per type
        arrivals: list[Shock] = []
        for name in sorted(self._factories):  # sorted → deterministic iteration order
            lam = self._lambdas[name]
            if lam <= 0:
                continue
            count = self._poisson(lam)
            for _ in range(count):
                severity = _draw_severity(self._rng, self._severity_weights)
                shock = self._factories[name](self._rng, severity)
                shock.tick_started = current_tick
                arrivals.append(shock)
                self._active.append(shock)
        return arrivals

    # -- Poisson sampler -----------------------------------------------------

    def _poisson(self, lam: float) -> int:
        """Knuth's algorithm — fast and deterministic for small λ.

        For sim-realistic λ (≪ 1 per tick), this almost always returns 0 or 1
        in a single iteration, so performance is fine without falling back to
        the inverse-CDF method.
        """
        if lam == 0.0:
            return 0
        threshold = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self._rng.random()
            if p <= threshold:
                return k - 1


# ---------------------------------------------------------------------------
# Composition helper
# ---------------------------------------------------------------------------


def apply_active_shocks(active: list[Shock], base_state: dict[str, float]) -> dict[str, float]:
    """Compose all active shock impacts into a derived state dict.

    Composition rules by impact-key suffix:
        - `*_mult` keys are multiplicative around 1.0:
            base_state.get(key, 1.0) * shock1[key] * shock2[key] * ...
          So two demand-cutting shocks (each < 1.0) compound to a deeper cut.
        - All other keys are additive:
            base_state.get(key, 0.0) + shock1[key] + shock2[key] + ...

    The function does **not** mutate `base_state`. It returns a fresh dict
    containing every key referenced by any active shock plus the keys that
    were already in `base_state`.
    """
    # Start from a copy so unrelated keys in base_state pass through unchanged.
    out: dict[str, float] = dict(base_state)

    # First pass: identify all keys referenced and seed neutral defaults.
    for shock in active:
        for key in shock.impact:
            if key in out:
                continue
            out[key] = 1.0 if key.endswith(MULT_SUFFIX) else 0.0

    # Second pass: compose.
    for shock in active:
        for key, value in shock.impact.items():
            if key.endswith(MULT_SUFFIX):
                out[key] *= value
            else:
                out[key] += value
    return out


__all__ = [
    "DEFAULT_SEVERITY_WEIGHTS",
    "MULT_SUFFIX",
    "SEVERITY_DURATION_DAYS",
    "SEVERITY_INTENSITY",
    "SHOCK_FACTORIES",
    "Severity",
    "Shock",
    "ShockScheduler",
    "apply_active_shocks",
    "make_economic_recession",
    "make_key_employee_departure",
    "make_market_crash",
    "make_new_competitor_entry",
    "make_regulatory_change",
    "make_supply_chain_disruption",
    "make_talent_war",
    "make_viral_growth_event",
]
