"""CEO stance: the locked persona behind every CEO decision.

Each company gets a `CeoStance` at creation, frozen for the run. The stance is
rendered into the AI CEO's system prompt (inception prompting) so the persona
cannot drift, and it parameterizes the heuristic CEO's thresholds. Two stances
on identical seeds should produce different outcomes; that divergence is the
point of the simulation.
"""

from __future__ import annotations

import random
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

HiringBias = Literal["lean", "balanced", "build_bench"]
TimeHorizon = Literal["quarterly", "annual", "decade"]


class CeoStance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    archetype: str
    risk_tolerance: float = Field(..., ge=0.0, le=1.0, description="0=cautious, 1=YOLO")
    growth_obsession: float = Field(..., ge=0.0, le=1.0, description="0=profit, 1=growth")
    quality_floor: float = Field(..., ge=0.0, le=1.0, description="0=ship anything, 1=excellence")
    hiring_bias: HiringBias
    time_horizon: TimeHorizon
    cash_comfort: float = Field(..., gt=0.0, description="Months of runway before nerves.")
    signature_moves: list[str] = Field(..., min_length=2, max_length=3)
    voice: str = Field(..., min_length=10)


#: Stance fields an AI CEO may cite in `references_stance`.
STANCE_FIELDS: Final[tuple[str, ...]] = (
    "risk_tolerance",
    "growth_obsession",
    "quality_floor",
    "hiring_bias",
    "time_horizon",
    "cash_comfort",
    "signature_moves",
)

#: Archetype sampling envelopes. Numeric fields draw uniformly from (lo, hi);
#: categorical fields draw from the listed options. `label` and `summary` are
#: shown on the setup screen.
PROFILES: Final[dict[str, dict]] = {
    "founder_operator": {
        "label": "Founder-Operator",
        "summary": "Close to the customer, grows when the product earns it.",
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
        "label": "Venture Growth",
        "summary": "Burns capital to win the category. Raises on momentum.",
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
        "label": "Bootstrapper",
        "summary": "Profitable from day one. No debt, no dilution.",
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
        "label": "Consolidator",
        "summary": "Buys, partners, standardizes, then scales.",
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
        "label": "Turnaround",
        "summary": "Stops the bleeding first, cuts anything that does not pay rent.",
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

ARCHETYPES: Final[tuple[str, ...]] = tuple(PROFILES)

#: Archetypes that access external capital (raises, investor nodes).
RAISE_ALLOWED_ARCHETYPES: Final[frozenset[str]] = frozenset({"venture_growth", "consolidator"})


def sample_stance(archetype: str, rng: random.Random) -> CeoStance:
    if archetype not in PROFILES:
        raise ValueError(f"unknown CEO archetype {archetype!r}; valid: {', '.join(ARCHETYPES)}")
    p = PROFILES[archetype]
    return CeoStance(
        archetype=archetype,
        risk_tolerance=round(rng.uniform(*p["risk_tolerance"]), 3),
        growth_obsession=round(rng.uniform(*p["growth_obsession"]), 3),
        quality_floor=round(rng.uniform(*p["quality_floor"]), 3),
        hiring_bias=rng.choice(p["hiring_bias"]),
        time_horizon=rng.choice(p["time_horizon"]),
        cash_comfort=round(rng.uniform(*p["cash_comfort"]), 1),
        signature_moves=rng.sample(p["signature_moves"], k=rng.randint(2, 3)),
        voice=p["voice"],
    )


def _qualitative(value: float, low: str, mid: str, high: str) -> str:
    return low if value < 0.34 else mid if value < 0.67 else high


def to_system_prompt(stance: CeoStance) -> str:
    """Render the stance as a role-locking persona paragraph."""
    risk = _qualitative(
        stance.risk_tolerance, "deeply cautious", "measured but willing to bet", "risk-seeking"
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
        f"You are the CEO of this company for the entire simulation. Your archetype is "
        f'{stance.archetype}. In your own words: "{stance.voice}" '
        f"You are {risk} (risk_tolerance={stance.risk_tolerance:.2f}). "
        f"You are {growth} (growth_obsession={stance.growth_obsession:.2f}). "
        f"On execution you are {quality} (quality_floor={stance.quality_floor:.2f}). "
        f"Your hiring bias is {stance.hiring_bias}. You think on a {stance.time_horizon} "
        f"horizon. You get nervous below {stance.cash_comfort:.1f} months of runway. "
        f"Your signature moves are {moves}. Every decision must be consistent with this "
        f"persona. Do not drift. Cite the stance attributes that drove each call."
    )
