"""Jev (TypeSafe System One) as the judgment layer of the weekly heuristic CEO.

Python detects; Jev rules. The heuristic CEO still finds every setup
mechanically (cash crisis, capacity or production binding, idle capacity) and
proposes one action. Jev answers a single Noul about that proposal and vetoes
it only when the answer is clearly low for that kind of action.

Design follows the Jev 1.13 guidance proven in alpaca-jev:
  * every number is computed and rounded in Python; Jev never does arithmetic,
  * the state carries only what the question needs, including both sides of
    the call (acting vs holding off) projected over the horizon,
  * the question names the exact fields it refers to and judges the company's
    outcome, not a CEO persona's tastes,
  * the gate pivots on a rolling baseline per company *and* action kind, not
    0.5, because Noul answers sit at a model- and question-specific level and
    the signal is the deviation. A veto needs the answer to fall
    `VETO_Z` standard deviations below that baseline (EWMA mean and variance),
    so repeats of the same proposal, which score alike, are not a coin flip.
  * a cost cut is not optional when the company would run out of cash inside
    the horizon by holding off: such a cut skips Jev and the rule acts.

The AI CEO (OpenRouter) is untouched; Jev only judges the small weekly calls.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from dotenv import dotenv_values

from src.simulation import llm
from src.simulation.engine import DAYS_PER_MONTH

if TYPE_CHECKING:
    from src.simulation.ceo import CeoDecision
    from src.simulation.engine import Company

log = logging.getLogger(__name__)

API_KEY_ENV: Final[str] = "TYPESAFE_API_KEY"
MODEL_ENV: Final[str] = "TYPESAFE_DEFAULT_MODEL"
DEFAULT_MODEL: Final[str] = "jev-latest"
#: TypeSafe bills input tokens only.
INPUT_PRICE_PER_MTOK: Final[float] = 0.042
REQUEST_TIMEOUT_S: Final[float] = 10.0
#: Horizon the proposal is judged over, in days.
HORIZON_DAYS: Final[int] = 90
#: EWMA weight of each new answer in the per-(company, kind) baseline.
BASELINE_ALPHA: Final[float] = 0.12
#: Veto only when the answer is this many baseline standard deviations low:
#: the one-sided 5% level, so a repeat of a proposal Jev rated before, which
#: scores within its usual spread, is vetoed about 1 time in 20.
VETO_Z: Final[float] = 1.645
#: Variance the baseline starts from: that of an answer uniform on [0, 1].
#: Until Jev's own spread is observed, only a far-below-baseline answer vetoes.
PRIOR_VARIANCE: Final[float] = 1.0 / 12.0

QUESTION_NAME: Final[str] = "stronger"
QUESTION: Final[str] = (
    "Taking the action in `proposal.action` now leaves this company more likely to be "
    f"solvent and competitive {HORIZON_DAYS} days from now than holding off. Compare "
    "`if_it_acts` with `if_it_holds`; their projections leave out any revenue the action "
    "itself would win or lose, which `company.binding_constraint` bears on."
)


def jev_key() -> str | None:
    """Key from the environment, else read live from backend/.env."""
    key = os.environ.get(API_KEY_ENV) or dotenv_values(llm.ENV_FILE).get(API_KEY_ENV) or ""
    return key.strip() or None


@dataclass
class JevVerdict:
    approved: bool
    p: float
    baseline: float
    decision: CeoDecision  # what gets recorded: the proposal (approved) or a hold


@dataclass
class JevStats:
    calls: int = 0
    approved: int = 0
    vetoed: int = 0
    #: Survival cuts that skipped Jev (holding off would run out of cash).
    forced: int = 0
    errors: int = 0
    input_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * INPUT_PRICE_PER_MTOK


@dataclass
class JevJudge:
    """One per run. `client` is any object with an async `system_one(state, questions)`."""

    client: Any
    model: str = DEFAULT_MODEL
    stats: JevStats = field(default_factory=JevStats)
    #: (company_id, action kind) → (EWMA mean, EWMA variance) of past answers.
    baselines: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    #: Set after an auth/permission failure; the rule CEO then acts alone.
    disabled_reason: str | None = None

    @classmethod
    def from_env(cls) -> JevJudge:
        from typesafe_sdk import AsyncTypeSafeClient

        model = os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        client = AsyncTypeSafeClient(api_key=jev_key(), model=model, timeout=REQUEST_TIMEOUT_S)
        return cls(client=client, model=model)

    @property
    def available(self) -> bool:
        return self.disabled_reason is None

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.stats.calls,
            "approved": self.stats.approved,
            "vetoed": self.stats.vetoed,
            "forced": self.stats.forced,
            "errors": self.stats.errors,
            "cost_usd": self.stats.cost_usd,
            "error": self.disabled_reason,
        }

    async def review(self, company: Company, proposal: CeoDecision, day: int) -> JevVerdict | None:
        """Ask Jev about one proposal. None means "no verdict": the rule acts alone."""
        from typesafe_sdk import Noul, TypeSafeAuthenticationError, TypeSafePermissionDeniedError

        if not self.available:
            return None
        if is_forced(company, proposal):
            self.stats.forced += 1
            return None
        question = Noul(instructions=QUESTION)
        try:
            response = await self.client.system_one(
                state=build_state(company, proposal), questions={QUESTION_NAME: question}
            )
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as exc:
            self.disabled_reason = f"Jev rejected the TypeSafe key: {exc}"
            company.events.append(f"Day {day}: {self.disabled_reason}. Rules act alone.")
            return None
        except Exception as exc:  # transport/validation: this call only
            self.stats.errors += 1
            log.warning("Jev review failed for %s on day %d: %s", company.name, day, exc)
            return None

        self.stats.calls += 1
        usage = getattr(response, "usage", None)
        self.stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        p = float(response.answers[QUESTION_NAME].noul)

        # Gate on the answers that came before this one; the first answer of a
        # kind is its own pivot (z = 0) and goes ahead.
        key = (company.company_id, action_kind(proposal))
        mean, variance = self.baselines.get(key, (p, PRIOR_VARIANCE))
        z = (p - mean) / math.sqrt(variance) if variance > 0 else 0.0
        approved = z >= -VETO_Z
        self.baselines[key] = ewma_update(mean, variance, p, BASELINE_ALPHA)
        baseline = mean

        verdict = f"Jev p={p:.2f} vs baseline {baseline:.2f} (z={z:+.1f})"
        if approved:
            self.stats.approved += 1
            decision = proposal.model_copy(
                update={
                    "source": "jev",
                    "reasoning": f"{proposal.reasoning} [{verdict}: approved]",
                    "jev_p": p,
                    "jev_approved": True,
                }
            )
        else:
            self.stats.vetoed += 1
            decision = proposal.model_copy(
                update={
                    "source": "jev",
                    "spawn": [],
                    "retire": [],
                    "open_locations": 0,
                    "price": None,
                    "marketing_intensity": None,
                    "raise_amount": None,
                    "reasoning": (
                        f"Held: rule proposed to {describe_action(company, proposal)}. "
                        f"[{verdict}: vetoed]"
                    ),
                    "jev_p": p,
                    "jev_approved": False,
                }
            )
        return JevVerdict(approved=approved, p=p, baseline=baseline, decision=decision)

    async def aclose(self) -> None:
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


# ─── State for the question (all arithmetic done here) ─────────────────────


def _r(x: float) -> float:
    return round(x) if abs(x) >= 100 else round(x, 3)


def describe_action(company: Company, proposal: CeoDecision) -> str:
    lib = company.library
    parts = [f"hire a {lib.nodes[k].label}" for k in proposal.spawn]
    parts += [f"cut the {lib.nodes[k].label}" for k in proposal.retire]
    if proposal.open_locations and company.primary_location:
        parts.append(
            f"open {proposal.open_locations} more {lib.nodes[company.primary_location].label}"
        )
    return " and ".join(parts) or "hold"


def _action_costs(company: Company, proposal: CeoDecision) -> tuple[float, float]:
    """(one-time cost, change in daily fixed cost) of the proposal, at today's cost levels."""
    lib = company.library
    hire_mult = company.env.get("hire_cost_mult", 1.0)
    added = list(proposal.spawn)
    if proposal.open_locations and company.primary_location:
        added += [company.primary_location] * proposal.open_locations
    one_time = sum(lib.nodes[k].hire_cost for k in added) * hire_mult
    daily = sum(lib.nodes[k].daily_fixed_costs for k in added) - sum(
        lib.nodes[k].daily_fixed_costs for k in proposal.retire
    )
    return one_time, daily * company.fixed_cost_mult


def action_kind(proposal: CeoDecision) -> str:
    """Baselines are kept per kind: Jev rates growth moves and cuts on different levels."""
    return "grow" if proposal.spawn or proposal.open_locations else "cut"


def ewma_update(mean: float, variance: float, x: float, alpha: float) -> tuple[float, float]:
    """Exponentially weighted mean and variance after observing `x` (RiskMetrics form)."""
    delta = x - mean
    return mean + alpha * delta, (1.0 - alpha) * (variance + alpha * delta * delta)


def is_forced(company: Company, proposal: CeoDecision) -> bool:
    """A cut is not optional when holding off runs out of cash inside the horizon."""
    _, daily = _action_costs(company, proposal)
    runway = company.runway_months()
    return action_kind(proposal) == "cut" and daily < 0 and runway * DAYS_PER_MONTH < HORIZON_DAYS


def _outlook(cash: float, daily_profit: float) -> dict[str, Any]:
    burn = -daily_profit
    return {
        "cash_now": _r(cash),
        "daily_profit": _r(daily_profit),
        f"cash_in_{HORIZON_DAYS}_days": _r(cash + HORIZON_DAYS * daily_profit),
        "runway_months": (
            "unlimited, profitable" if burn <= 0 else _r(max(0.0, cash) / (burn * DAYS_PER_MONTH))
        ),
    }


def build_state(company: Company, proposal: CeoDecision) -> dict[str, Any]:
    one_time, daily = _action_costs(company, proposal)
    history = list(company.history)
    share_then = history[-4].market_share if len(history) >= 4 else None
    profit = company.daily_revenue - company.daily_costs
    return {
        "company": {
            "market_share": _r(company.market_share),
            "market_share_3_months_ago": None if share_then is None else _r(share_then),
            "binding_constraint": company.binding,
            "capacity_utilization": _r(company.utilization),
            "customer_satisfaction": _r(company.satisfaction),
            "employees": company.employees,
            "locations": company.locations,
        },
        "proposal": {
            "action": describe_action(company, proposal),
            "why_the_rule_fired": proposal.reasoning,
            "one_time_cost": _r(one_time),
            "change_in_daily_costs": _r(daily),
            "one_time_cost_as_share_of_cash": (
                _r(one_time / company.cash) if company.cash > 0 else None
            ),
        },
        "if_it_holds": _outlook(company.cash, profit),
        "if_it_acts": _outlook(company.cash - one_time, profit - daily),
    }
