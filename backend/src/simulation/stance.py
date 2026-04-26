"""CEO stance — locked persona that drives orchestrator decisions.

Per v2 redesign (`V2_REDESIGN_PLAN.md` §3): every company is born with a
`CeoStance` set at creation time and frozen for the life of the simulation.
The stance is injected into every orchestrator LLM call as inception-prompted
system context (CAMEL-AI pattern), so the CEO's reasoning never drifts from
its archetype regardless of what the simulation throws at it.

Two stances given identical seeds and identical shocks should produce
meaningfully different outcomes — that divergence is the entire research
output of the platform.
"""

from __future__ import annotations

import random
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# Canonical archetype keys. Anything outside this set is rejected by the sampler.
ARCHETYPES: Final[tuple[str, ...]] = (
    "founder_operator",
    "venture_growth",
    "bootstrap",
    "consolidator",
    "turnaround",
)

HiringBias = Literal["lean", "balanced", "build_bench"]
TimeHorizon = Literal["quarterly", "annual", "decade"]


class CeoStance(BaseModel):
    """Locked CEO persona — set at company creation, frozen for the entire sim.

    Every field is consumed by `to_system_prompt()` to produce the role-lock
    paragraph injected into orchestrator LLM calls. Frozen via `ConfigDict`
    so accidental mutation in the orchestrator (or anywhere downstream)
    raises `ValidationError` rather than silently drifting the persona.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    archetype: str = Field(
        ...,
        description=(
            "One of: founder_operator | venture_growth | bootstrap | "
            "consolidator | turnaround."
        ),
    )
    risk_tolerance: float = Field(..., ge=0.0, le=1.0, description="0=cautious, 1=YOLO")
    growth_obsession: float = Field(
        ..., ge=0.0, le=1.0, description="0=profit-focused, 1=growth-at-all-costs"
    )
    quality_floor: float = Field(
        ..., ge=0.0, le=1.0, description="0=ship anything, 1=excellence required"
    )
    hiring_bias: HiringBias
    time_horizon: TimeHorizon
    cash_comfort: float = Field(
        ..., gt=0.0, description="Months of runway before this CEO gets nervous."
    )
    signature_moves: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Two or three short phrases describing instinctive plays.",
    )
    voice: str = Field(
        ...,
        min_length=10,
        description="One or two sentences of self-description in first person.",
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


# Per-archetype sampling envelopes. Numeric fields are uniform draws inside
# the (lo, hi) range; categorical fields draw from the listed options.
# These envelopes are deliberately tight where the archetype demands it
# (e.g. bootstrap risk_tolerance must stay low) and loose elsewhere so the
# Monte Carlo runner gets useful variance across runs of the same archetype.
_ARCHETYPE_PROFILES: Final[dict[str, dict[str, object]]] = {
    "founder_operator": {
        "risk_tolerance": (0.45, 0.70),
        "growth_obsession": (0.40, 0.65),
        "quality_floor": (0.55, 0.85),
        "hiring_bias": ("lean", "balanced"),
        "time_horizon": ("annual", "decade"),
        "cash_comfort": (6.0, 12.0),
        "signature_moves": [
            "stay close to the customer",
            "hire only when it hurts",
            "ship product I would buy myself",
            "protect the original vision",
            "founder-led sales until $5M ARR",
        ],
        "voice": (
            "I started this thing in my garage and I am still the one taking the "
            "calls when something breaks. We grow when the product earns it."
        ),
    },
    "venture_growth": {
        "risk_tolerance": (0.75, 0.98),
        "growth_obsession": (0.80, 1.0),
        "quality_floor": (0.30, 0.55),
        "hiring_bias": ("build_bench",),
        "time_horizon": ("quarterly", "annual"),
        "cash_comfort": (3.0, 6.0),
        "signature_moves": [
            "burn capital to win the category",
            "always over-staff sales",
            "expand before profitable",
            "raise the next round on momentum",
            "hire ahead of the curve",
        ],
        "voice": (
            "Speed is the only moat. We will not lose this market because we were "
            "too cheap to staff up — burn the runway, win the category."
        ),
    },
    "bootstrap": {
        "risk_tolerance": (0.05, 0.30),
        "growth_obsession": (0.10, 0.35),
        "quality_floor": (0.55, 0.80),
        "hiring_bias": ("lean",),
        "time_horizon": ("annual", "decade"),
        "cash_comfort": (12.0, 24.0),
        "signature_moves": [
            "profitable from day one",
            "no debt, no dilution",
            "default to lean",
            "hire only against booked revenue",
            "fire fast, hire slow",
        ],
        "voice": (
            "Every dollar in the bank is a dollar of freedom. I will not chase "
            "growth that I cannot pay for out of the operating account."
        ),
    },
    "consolidator": {
        "risk_tolerance": (0.40, 0.65),
        "growth_obsession": (0.50, 0.75),
        "quality_floor": (0.45, 0.70),
        "hiring_bias": ("balanced", "build_bench"),
        "time_horizon": ("annual", "decade"),
        "cash_comfort": (6.0, 14.0),
        "signature_moves": [
            "buy or partner before building",
            "consolidate fragmented suppliers",
            "standardize then scale",
            "roll up adjacent niches",
            "harvest mature lines",
        ],
        "voice": (
            "This market is fragmented and I am here to bring order. We grow by "
            "absorbing the operators who cannot run a tight ship."
        ),
    },
    "turnaround": {
        "risk_tolerance": (0.55, 0.85),
        "growth_obsession": (0.30, 0.55),
        "quality_floor": (0.40, 0.65),
        "hiring_bias": ("lean", "balanced"),
        "time_horizon": ("quarterly", "annual"),
        "cash_comfort": (2.0, 5.0),
        "signature_moves": [
            "stop the bleeding first",
            "cut anything that does not pay rent",
            "renegotiate every supplier",
            "concentrate on the profitable core",
            "rebuild trust with one quick win",
        ],
        "voice": (
            "I take the calls nobody else wants. The patient is on the table; my "
            "job is to keep the lights on long enough for a real plan to take hold."
        ),
    },
}


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Inclusive uniform draw rounded to 3 decimals for stable transcripts."""
    return round(rng.uniform(lo, hi), 3)


def sample_stance(archetype: str, rng: random.Random) -> CeoStance:
    """Produce an internally-consistent stance for the given archetype.

    Numeric trait ranges and categorical choices are drawn from
    `_ARCHETYPE_PROFILES`. Two or three signature moves are picked without
    replacement from the archetype's pool. The voice line is taken verbatim
    from the profile (it is the canonical first-person identity for the role).

    Raises:
        ValueError: if `archetype` is not one of `ARCHETYPES`.
    """

    if archetype not in ARCHETYPES:
        raise ValueError(
            f"Unknown archetype {archetype!r}. Valid: {', '.join(ARCHETYPES)}."
        )

    profile = _ARCHETYPE_PROFILES[archetype]

    risk_lo, risk_hi = profile["risk_tolerance"]  # type: ignore[misc]
    growth_lo, growth_hi = profile["growth_obsession"]  # type: ignore[misc]
    quality_lo, quality_hi = profile["quality_floor"]  # type: ignore[misc]
    cash_lo, cash_hi = profile["cash_comfort"]  # type: ignore[misc]

    moves_pool: list[str] = list(profile["signature_moves"])  # type: ignore[arg-type]
    n_moves = rng.randint(2, 3)
    signature_moves = rng.sample(moves_pool, k=n_moves)

    return CeoStance(
        archetype=archetype,
        risk_tolerance=_uniform(rng, risk_lo, risk_hi),
        growth_obsession=_uniform(rng, growth_lo, growth_hi),
        quality_floor=_uniform(rng, quality_lo, quality_hi),
        hiring_bias=rng.choice(list(profile["hiring_bias"])),  # type: ignore[arg-type]
        time_horizon=rng.choice(list(profile["time_horizon"])),  # type: ignore[arg-type]
        cash_comfort=round(rng.uniform(cash_lo, cash_hi), 1),
        signature_moves=signature_moves,
        voice=str(profile["voice"]),
    )


# ---------------------------------------------------------------------------
# Inception prompt
# ---------------------------------------------------------------------------


def _qualitative(value: float, low: str, mid: str, high: str) -> str:
    """Map a 0..1 trait to a short qualitative phrase for the prompt."""
    if value < 0.34:
        return low
    if value < 0.67:
        return mid
    return high


def to_system_prompt(stance: CeoStance) -> str:
    """Render `stance` as a CAMEL-AI inception-prompted system message.

    The output is a single coherent paragraph the CEO LLM reads once at the
    start of every call. It references every stance attribute so the model
    has no excuse to drift — risk, growth, quality, hiring, horizon, cash
    comfort, signature moves, and voice are all stated explicitly. Kept
    well under 1500 chars to leave headroom for the orchestrator's state
    window and decision history.
    """

    risk = _qualitative(
        stance.risk_tolerance,
        "deeply cautious",
        "measured but willing to bet",
        "aggressive and risk-seeking",
    )
    growth = _qualitative(
        stance.growth_obsession,
        "profit-focused, indifferent to growth for its own sake",
        "balanced between growth and margin",
        "growth-obsessed, willing to sacrifice margin to win share",
    )
    quality = _qualitative(
        stance.quality_floor,
        "comfortable shipping rough work to move fast",
        "holding a working quality bar",
        "uncompromising about quality",
    )

    moves = "; ".join(f'"{m}"' for m in stance.signature_moves)

    return (
        f"You are the CEO of this company and you will remain in this role for "
        f"the entire simulation. Your archetype is {stance.archetype}. "
        f'In your own words: "{stance.voice}" '
        f"You are {risk} (risk_tolerance={stance.risk_tolerance:.2f}). "
        f"You are {growth} (growth_obsession={stance.growth_obsession:.2f}). "
        f"On execution you are {quality} (quality_floor={stance.quality_floor:.2f}). "
        f"Your hiring bias is {stance.hiring_bias} — you build the team accordingly. "
        f"You think on a {stance.time_horizon} time horizon; decisions further out "
        f"than that are noise to you. You start to get nervous when runway drops "
        f"below {stance.cash_comfort:.1f} months of cash. "
        f"Your signature moves are {moves}. "
        f"Every decision you propose must be consistent with this persona. Do not "
        f"drift. When asked to justify a call, cite the specific stance attribute "
        f"that drove it."
    )


__all__ = [
    "ARCHETYPES",
    "CeoStance",
    "HiringBias",
    "TimeHorizon",
    "sample_stance",
    "to_system_prompt",
]
