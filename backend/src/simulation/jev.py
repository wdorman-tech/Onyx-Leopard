"""Jev (TypeSafe System One) as the judgment layer of the weekly heuristic CEO.

Python detects; Jev rules. The heuristic CEO still finds every setup
mechanically (cash crisis, capacity or production binding, idle capacity) and
proposes one action. Jev answers a single Noul about that proposal and the
action goes ahead only if the answer clears the company's own baseline.

Design follows the Jev 1.13 guidance proven in alpaca-jev:
  * every number is computed and rounded in Python; Jev never does arithmetic,
  * the state carries only what the question needs,
  * the question names the exact fields it refers to,
  * the gate pivots on a rolling per-company baseline, not 0.5, because Jev's
    Noul answers sit at a model-specific level and the signal is the deviation.

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
#: EMA weight of each new answer in the per-company baseline.
BASELINE_ALPHA: Final[float] = 0.12
#: Approve when the answer sits at least this far above the updated baseline.
EDGE_THRESHOLD: Final[float] = 0.0

QUESTION_NAME: Final[str] = "stronger"


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
    baselines: dict[str, float] = field(default_factory=dict)
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
            "errors": self.stats.errors,
            "cost_usd": self.stats.cost_usd,
            "error": self.disabled_reason,
        }

    async def review(self, company: Company, proposal: CeoDecision, day: int) -> JevVerdict | None:
        """Ask Jev about one proposal. None means "no verdict": the rule acts alone."""
        from typesafe_sdk import Noul, TypeSafeAuthenticationError, TypeSafePermissionDeniedError

        if not self.available:
            return None
        question = Noul(
            instructions=(
                "Taking the action in `proposal.action` now will leave this company in a "
                f"stronger position {HORIZON_DAYS} days from now than holding off, judged by "
                "the priorities of the CEO described in `ceo`."
            )
        )
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

        # Update the baseline before gating so the first answer has a pivot.
        prev = self.baselines.get(company.company_id, p)
        baseline = (1 - BASELINE_ALPHA) * prev + BASELINE_ALPHA * p
        self.baselines[company.company_id] = baseline
        approved = p - baseline >= EDGE_THRESHOLD

        verdict = f"Jev p={p:.2f} vs baseline {baseline:.2f}"
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
    """(one-time cost, change in daily fixed cost) of the proposal."""
    lib = company.library
    hire_mult = company.env.get("hire_cost_mult", 1.0)
    added = list(proposal.spawn)
    if proposal.open_locations and company.primary_location:
        added += [company.primary_location] * proposal.open_locations
    one_time = sum(lib.nodes[k].hire_cost for k in added) * hire_mult
    daily = sum(lib.nodes[k].daily_fixed_costs for k in added) - sum(
        lib.nodes[k].daily_fixed_costs for k in proposal.retire
    )
    return one_time, daily


def build_state(company: Company, proposal: CeoDecision) -> dict[str, Any]:
    stance = company.stance
    runway = company.runway_months()
    one_time, daily = _action_costs(company, proposal)
    history = list(company.history)
    share_then = history[-4].market_share if len(history) >= 4 else None
    profit = company.daily_revenue - company.daily_costs
    return {
        "ceo": {
            "archetype": stance.archetype.replace("_", " "),
            "risk_tolerance": stance.risk_tolerance,
            "growth_obsession": stance.growth_obsession,
            "quality_floor": stance.quality_floor,
            "hiring_bias": stance.hiring_bias,
            "time_horizon": stance.time_horizon,
            "gets_nervous_below_months_of_runway": stance.cash_comfort,
            "signature_moves": stance.signature_moves,
        },
        "company": {
            "cash": _r(company.cash),
            "daily_profit": _r(profit),
            "runway_months": "unlimited, profitable" if math.isinf(runway) else _r(runway),
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
            "cash_after_one_time_cost": _r(company.cash - one_time),
            "one_time_cost_as_share_of_cash": (
                _r(one_time / company.cash) if company.cash > 0 else None
            ),
        },
    }
