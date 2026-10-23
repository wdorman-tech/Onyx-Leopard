"""CEOs: a free rule-based operator and an LLM strategist, both stance-locked.

  * `HeuristicCeo` runs every `HEURISTIC_CADENCE_DAYS`: cash-crisis cuts,
    expansion when physical capacity binds, hires when production binds,
    layoffs when demand binds. Its reserve threshold comes from the stance.
  * `AiCeo` runs every `AI_CADENCE_DAYS` (and on severe shocks) through
    OpenRouter. The prompt is compact on purpose: cheap models, small bills.

Every decision passes through `sanitize`, which enforces library
prerequisites, caps, and engine bounds before the engine sees it.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.simulation.engine import (
    AI_CADENCE_DAYS,
    DAYS_PER_MONTH,
    HEURISTIC_CADENCE_DAYS,
    INSOLVENT_DAYS_TO_DEATH,
    MARKETING_INTENSITY_ELASTICITY,
    MARKETING_INTENSITY_RANGE,
    MARKETING_SPEND_FRACTION,
    MAX_LOCATIONS_PER_DECISION,
    MAX_SPAWNS_PER_DECISION,
    PRICE_BOUNDS_FACTOR,
    RAISE_CAP_MONTHS_OF_REVENUE,
    SHARE_MARKETING_EXPONENT,
    Company,
)
from src.simulation.library import COST_BUCKET_GAIN, REVENUE_BUCKET_GAIN
from src.simulation.llm import LlmError, OpenRouterClient
from src.simulation.stance import RAISE_ALLOWED_ARCHETYPES, STANCE_FIELDS, to_system_prompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.simulation.shocks import Shock

log = logging.getLogger(__name__)

#: Consecutive heuristic checks a constraint must bind before the CEO acts.
PERSIST_CHECKS: int = 3
#: Utilization below this, with demand binding, is excess capacity.
LOW_UTILIZATION: float = 0.40
#: How many of its own past decisions, and months of history, the AI CEO sees.
AI_DECISION_MEMORY: int = 3
AI_HISTORY_MONTHS: int = 6
#: Characters of past reasoning kept per remembered decision.
AI_REASONING_MEMORY_CHARS: int = 240


class CeoDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: int
    #: "jev" = a heuristic proposal Jev approved (or vetoed into a hold).
    source: Literal["heuristic", "ai", "jev"]
    spawn: list[str] = Field(default_factory=list)
    retire: list[str] = Field(default_factory=list)
    open_locations: int = 0
    price: float | None = None
    marketing_intensity: float | None = None
    raise_amount: float | None = None
    reasoning: str
    references_stance: list[str] = Field(default_factory=list)
    jev_p: float | None = None
    jev_approved: bool | None = None


def sanitize(
    company: Company,
    *,
    day: int,
    source: Literal["heuristic", "ai"],
    reasoning: str,
    spawn: Iterable[str] = (),
    retire: Iterable[str] = (),
    open_locations: int = 0,
    price: float | None = None,
    marketing_intensity: float | None = None,
    raise_amount: float | None = None,
    references_stance: Iterable[str] = (),
) -> CeoDecision:
    """Drop or clamp anything the engine must not apply; note what changed."""
    lib, seed = company.library, company.seed
    notes: list[str] = []
    projected = dict(company.spawned)
    founder = lib.founder()

    kept_retire: list[str] = []
    for key in retire:
        node = lib.get(key)
        if node is None or projected.get(key, 0) <= 0:
            notes.append(f"retire {key}: not spawned")
        elif key == founder:
            notes.append("cannot retire the founder")
        elif key == company.primary_location and projected[key] <= 1:
            notes.append("cannot close the last location")
        else:
            projected[key] -= 1
            kept_retire.append(key)

    kept_spawn: list[str] = []
    for key in list(spawn)[:MAX_SPAWNS_PER_DECISION]:
        node = lib.get(key)
        if node is None or seed.economics_model not in node.applicable_economics:
            notes.append(f"spawn {key}: not available for this business")
        elif not lib.prerequisites_met(key, projected):
            notes.append(f"spawn {key}: needs {node.prerequisites}")
        elif projected.get(key, 0) >= node.category_caps.hard_cap:
            notes.append(f"spawn {key}: at hard cap {node.category_caps.hard_cap}")
        else:
            projected[key] = projected.get(key, 0) + 1
            kept_spawn.append(key)

    locations = max(0, min(MAX_LOCATIONS_PER_DECISION, int(open_locations or 0)))
    if company.primary_location is None:
        locations = 0
    else:
        cap = lib.nodes[company.primary_location].category_caps.hard_cap
        locations = min(locations, max(0, cap - projected.get(company.primary_location, 0)))

    # Adding and closing the same node in one decision is churn: the count ends
    # where it started but every addition still pays its hire cost. Net it out.
    for key in list(kept_retire):
        if key in kept_spawn:
            kept_spawn.remove(key)
        elif key == company.primary_location and locations:
            locations -= 1
        else:
            continue
        kept_retire.remove(key)
        notes.append(f"netted an add against a close of {key}")

    if price is not None:
        lo, hi = (seed.starting_price * f for f in PRICE_BOUNDS_FACTOR)
        # The prompt shows bounds rounded like every other number, so a price
        # copied from it may sit a rounding step outside: that is the bound.
        if math.isfinite(price) and lo - _rounding_step(lo) <= price < lo:
            price = lo
        elif math.isfinite(price) and hi < price <= hi + _rounding_step(hi):
            price = hi
        # Anything else out of range is rejected, not clamped: it is usually a
        # misread lever (0 meant as "keep"), and clamping it would be a 90% cut.
        if not math.isfinite(price) or not lo <= price <= hi:
            notes.append(f"price {price} outside [{lo:.2f}, {hi:.2f}]; kept the current price")
            price = None

    if marketing_intensity is not None:
        if math.isfinite(marketing_intensity):
            lo, hi = MARKETING_INTENSITY_RANGE
            marketing_intensity = max(lo, min(hi, marketing_intensity))
        else:
            marketing_intensity = None

    if raise_amount is not None:
        cap = max(
            seed.starting_cash,
            RAISE_CAP_MONTHS_OF_REVENUE * company.daily_revenue * DAYS_PER_MONTH,
        )
        if company.stance.archetype not in RAISE_ALLOWED_ARCHETYPES:
            notes.append("raises not allowed for this archetype")
            raise_amount = None
        elif not math.isfinite(raise_amount) or raise_amount <= 0:
            raise_amount = None
        elif raise_amount > cap:
            notes.append(f"raise capped at {cap:,.0f}")
            raise_amount = cap

    if notes:
        reasoning = f"{reasoning} [engine: {'; '.join(notes)}]"
    return CeoDecision(
        day=day,
        source=source,
        spawn=kept_spawn,
        retire=kept_retire,
        open_locations=locations,
        price=price,
        marketing_intensity=marketing_intensity,
        raise_amount=raise_amount,
        reasoning=reasoning.strip() or "(no reasoning given)",
        references_stance=[r for r in references_stance if r in STANCE_FIELDS],
    )


# ─── Heuristic CEO ─────────────────────────────────────────────────────────


class HeuristicCeo:
    """Rule-based operator. Free, deterministic, and always on.

    `propose` detects a setup and returns one action without committing to it;
    the caller either `commit`s it (rule or Jev approved) or `defer`s it (Jev
    vetoed), so a veto leaves no side effects behind.
    """

    def __init__(self) -> None:
        self._binding: deque[str] = deque(maxlen=PERSIST_CHECKS)
        self._low_util: deque[bool] = deque(maxlen=PERSIST_CHECKS)
        self._hires: list[str] = []  # ops hires, newest last

    def _persistent(self, constraint: str) -> bool:
        return len(self._binding) == PERSIST_CHECKS and all(b == constraint for b in self._binding)

    def decide(self, company: Company, day: int) -> CeoDecision | None:
        """Propose and commit in one step (no judge)."""
        proposal = self.propose(company, day)
        if proposal is not None:
            self.commit(company, proposal)
        return proposal

    def commit(self, company: Company, decision: CeoDecision) -> None:
        lib = company.library
        if decision.open_locations or decision.spawn:
            self._binding.clear()
        self._hires += [k for k in decision.spawn if lib.nodes[k].category == "ops"]
        for key in decision.retire:
            if key in self._hires:
                self._hires.remove(key)

    def defer(self) -> None:
        """A vetoed setup must persist for a full window again before it is re-proposed."""
        self._binding.clear()

    def propose(self, company: Company, day: int) -> CeoDecision | None:
        if day % HEURISTIC_CADENCE_DAYS:
            return None
        self._binding.append(company.binding)
        self._low_util.append(company.utilization < LOW_UTILIZATION)

        stance = company.stance
        lib = company.library
        # Cash this CEO refuses to dip below: `cash_comfort` months of fixed burn.
        reserve = stance.cash_comfort * DAYS_PER_MONTH * company.fixed_burn
        runway = company.runway_months()

        if runway < stance.cash_comfort:
            target = self._costliest_cuttable(company)
            if target is not None:
                return sanitize(
                    company,
                    day=day,
                    source="heuristic",
                    retire=[target],
                    reasoning=(
                        f"Runway {runway:.1f} months is below my {stance.cash_comfort:.1f}-month "
                        f"comfort. Cutting {lib.nodes[target].label}, the cut that saves the most "
                        "per day net of what it earns."
                    ),
                    references_stance=["cash_comfort"],
                )

        loc = company.primary_location
        if self._persistent("capacity") and loc is not None:
            cost = lib.nodes[loc].hire_cost * company.env.get("hire_cost_mult", 1.0)
            if company.cash - cost > reserve:
                return sanitize(
                    company,
                    day=day,
                    source="heuristic",
                    open_locations=1,
                    reasoning=(
                        f"Demand exceeds physical capacity for {PERSIST_CHECKS} straight checks "
                        f"and cash covers a {lib.nodes[loc].label} above my reserve."
                    ),
                    references_stance=["cash_comfort", "growth_obsession"],
                )

        if self._persistent("production"):
            hire = self._cheapest_ops(company)
            if hire is not None and company.cash - lib.nodes[hire].hire_cost > reserve:
                return sanitize(
                    company,
                    day=day,
                    source="heuristic",
                    spawn=[hire],
                    reasoning=(
                        f"Production capacity is the bottleneck; hiring {lib.nodes[hire].label}, "
                        "the cheapest ops role that adds headcount."
                    ),
                    references_stance=["hiring_bias"],
                )

        hires = [k for k in self._hires if company.spawned.get(k, 0) > 0]
        if self._persistent("demand") and all(self._low_util) and hires:
            key = hires[-1]
            return sanitize(
                company,
                day=day,
                source="heuristic",
                retire=[key],
                reasoning=(
                    f"Utilization under {LOW_UTILIZATION:.0%} with demand binding. Releasing "
                    f"the most recent ops hire ({lib.nodes[key].label})."
                ),
                references_stance=["cash_comfort"],
            )
        return None

    @staticmethod
    def _costliest_cuttable(company: Company) -> str | None:
        """The non-exec node whose removal saves the most per day, if any saves at all.

        Saving = its own daily cost, less the revenue it adds per customer
        (revenue bucket) and the overhead it holds down (cost bucket). A node
        that earns or saves more than it costs is never the one to cut. It is a
        rule of thumb, not a counterfactual: effects through quality, marketing,
        capacity and production are not priced, so a cut can still cost more
        than it saves (the AI CEO and Jev are the judgment layers).
        """
        lib = company.library
        overhead = (
            company._base_fixed_costs()
            * company.env.get("fixed_cost_mult", 1.0)
            * company.env.get("wage_inflation_mult", 1.0)
        )
        candidates = []
        for key, count in company.spawned.items():
            node = lib.nodes[key]
            if count <= 0 or node.category in ("exec", "location") or node.daily_fixed_costs <= 0:
                continue
            effects = lib.effects(key)
            saving = (
                node.daily_fixed_costs * company.fixed_cost_mult
                - REVENUE_BUCKET_GAIN * effects.get("revenue", 0.0) * company.core_revenue
                + COST_BUCKET_GAIN * effects.get("cost", 0.0) * overhead
            )
            if saving > 0:
                candidates.append((saving, key))
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _cheapest_ops(company: Company) -> str | None:
        lib = company.library
        candidates = [
            (n.hire_cost + DAYS_PER_MONTH * n.daily_fixed_costs, k)
            for k, n in lib.nodes.items()
            if n.category == "ops"
            and n.employees_count > 0
            and n.daily_fixed_costs > 0  # free roles are founder slots, not hires
            and company.seed.economics_model in n.applicable_economics
            and lib.prerequisites_met(k, company.spawned)
            and company.spawned.get(k, 0) < n.category_caps.hard_cap
        ]
        return min(candidates)[1] if candidates else None


# ─── AI CEO ────────────────────────────────────────────────────────────────


class _AiOutput(BaseModel):
    """Lenient parse of the model's JSON (prompt-JSON fallback may be sloppy)."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    references_stance: list[str] = Field(default_factory=list)
    spawn: list[str] = Field(default_factory=list)
    retire: list[str] = Field(default_factory=list)
    open_locations: int = 0
    price: float | None = None
    marketing_intensity: float | None = None
    raise_amount: float | None = None


def _num(x: float) -> float:
    return round(x) if abs(x) >= 100 else round(x, 3)


#: Significant figures for prices in the prompt: a price copied back must land
#: where the engine computed it (whole-unit rounding cost up to 1.9% at a kink).
PRICE_SIGNIFICANT_FIGURES: int = 6


def _price(x: float) -> float:
    return float(f"{x:.{PRICE_SIGNIFICANT_FIGURES}g}")


def _rounding_step(x: float) -> float:
    """Largest error `_price` adds to `x`."""
    return abs(x) * 10.0 ** (1 - PRICE_SIGNIFICANT_FIGURES)


def _optional_num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else _num(x)


def _key_array(options: list[str]) -> dict[str, Any]:
    if not options:
        return {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    return {"type": "array", "items": {"type": "string", "enum": options}}


def decision_schema(spawnable: list[str], spawned: list[str], locations: int = 0) -> dict[str, Any]:
    """Strict JSON schema; node enums keep weak models from inventing keys.

    Every lever is a change, not a target, and the descriptions say so: weak
    models otherwise send the location count they want to end up with.
    """
    properties: dict[str, Any] = {
        "reasoning": {"type": "string", "description": "2-4 sentences citing the numbers."},
        "references_stance": {
            "type": "array",
            "items": {"type": "string", "enum": list(STANCE_FIELDS)},
        },
        "spawn": {
            **_key_array(spawnable),
            "description": "Nodes to add now, one entry each. [] = add nothing.",
        },
        "retire": {
            **_key_array(spawned),
            "description": "Nodes to close now, one entry each (repeat a key to close several). "
            "[] = close nothing.",
        },
        "open_locations": {
            "type": "integer",
            "description": f"How many NEW locations to open now, on top of the {locations} you "
            f"run. 0 = open none. At most {MAX_LOCATIONS_PER_DECISION}.",
        },
        "price": {"type": ["number", "null"], "description": "New unit price, null = keep."},
        "marketing_intensity": {
            "type": ["number", "null"],
            "description": "0-2 (1 = baseline), null = keep.",
        },
        "raise_amount": {"type": ["number", "null"], "description": "USD to raise, null = none."},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


class AiCeo:
    """LLM strategist for one company. All AI CEOs in a run share one client."""

    def __init__(self, client: OpenRouterClient, company: Company) -> None:
        self.client = client
        self.system_prompt = self._system_prompt(company)

    @property
    def available(self) -> bool:
        return self.client.available

    async def decide(
        self,
        company: Company,
        day: int,
        companies: list[Company],
        tam: float,
        market_shocks: Sequence[Shock] = (),
        best_response: tuple[float, float] | None = None,
    ) -> CeoDecision | None:
        if not self.client.available:
            return None
        spawnable = self._spawnable(company)
        try:
            raw = await self.client.complete_json(
                system=self.system_prompt,
                user=self._user_prompt(
                    company, day, companies, tam, spawnable, market_shocks, best_response
                ),
                schema_name="ceo_decision",
                schema=decision_schema(spawnable, sorted(company.spawned), company.locations),
            )
            out = _AiOutput.model_validate(raw)
        except (LlmError, ValidationError) as exc:
            log.warning("AI CEO %s failed on day %d: %s", company.name, day, exc)
            # Fatal errors (bad key, no credit) are reported once for the whole run.
            if not (isinstance(exc, LlmError) and exc.fatal):
                company.events.append(f"Day {day}: {company.name}: AI CEO call failed ({exc})")
            return None
        return sanitize(company, day=day, source="ai", **out.model_dump())

    # ── Prompts ──

    @staticmethod
    def _system_prompt(company: Company) -> str:
        loc = company.primary_location
        loc_label = company.library.nodes[loc].label if loc else "location"
        can_raise = company.stance.archetype in RAISE_ALLOWED_ARCHETYPES
        # Share-level effect of marketing intensity: reach exponent × logit exponent.
        pull = 2.0 * MARKETING_INTENSITY_ELASTICITY * SHARE_MARKETING_EXPONENT
        return (
            f"{to_system_prompt(company.stance)}\n\n"
            f"You run {company.name} ({company.seed.niche}) in a daily simulation, competing "
            "with other companies for one shared market. Every rival started from the same "
            "seed as you: same unit cost, same starting price. You decide about every "
            f"{AI_CADENCE_DAYS} days (less often when the AI budget is tight), and early when "
            "a severe shock hits.\n"
            "How the world works:\n"
            "- Total customer spend follows the market's average price on a linear demand "
            "curve. `market.category_price_elasticity` is its elasticity at the average "
            "price customers are settling on, and it rises as prices rise: below 1, prices "
            "falling across the market "
            "shrink total spend; above 1, they grow it. No customer pays `market.choke_price` "
            "or more: an offer priced there wins nobody, whatever rivals charge. Elasticities "
            "are null once the average price has passed the choke.\n"
            "- Customers drift toward the best mix of satisfaction, marketing reach and price. "
            "`you.target_share` is where your current offer settles; `you.market_share` is "
            "where you are now. Customers move over weeks, not overnight. A price cut that "
            "rivals match puts shares back where they were, at lower margins for everyone.\n"
            "- `you.price_elasticity` is how much your unit sales respond to your own price "
            "once customers settle. `you.settled_demand_vs_capacity` is the demand your "
            "current offer settles at, over what you can serve; above 1 you turn customers "
            "away. `you.best_response_price` is the list price that maximizes your profit "
            "once customers settle, if rivals keep today's prices and offers; it already "
            "accounts for your elasticity, your capacity, any price cap and the price "
            "bounds. `you.monthly_gain_at_best_response` is the monthly gain once customers "
            "have settled; the first months after a cut can cost money while customers move. "
            "Rivals react, so treat it as a strong guide, not a promise: when rivals are "
            "short of capacity too and everyone raises toward the choke at once, the market "
            "can empty for a month.\n"
            f"- Your pull on customers scales with ((1 + marketing_intensity) / 2)^{pull:.2f}: "
            f"at 0 you keep about {0.5**pull:.0%} of it, at 2 you gain about "
            f"{1.5**pull - 1:.0%}.\n"
            "- Customers served = min(demand, physical capacity, production capacity). "
            "`binding` tells you which one limits you. Capacity grows with locations "
            f"(`open_locations` opens that many NEW {loc_label}s, each paying its hire_cost; "
            "close one by putting its key in retire) and infrastructure nodes; production "
            "follows Cobb-Douglas in capital and headcount. Headcount raises how much demand "
            "you can serve, not how much demand you get.\n"
            "- `price` is your list price; customers pay `effective_price`, which a shock can "
            "cap at `price_cap`. Revenue = served × revenue_per_customer (effective price plus "
            "revenue-stream uplift). gross_margin = (revenue_per_customer - unit_cost) / "
            "revenue_per_customer; contribution_margin also nets out marketing. Costs = "
            "unit_cost × served + every node's daily cost + marketing "
            f"({MARKETING_SPEND_FRACTION:.0%} of revenue × marketing_intensity). Spawning a "
            "node costs its hire_cost once.\n"
            f"- Negative cash for {INSOLVENT_DAYS_TO_DEATH} straight days is bankruptcy.\n"
            f"- Raising capital is {'allowed' if can_raise else 'NOT allowed'} for you.\n"
            "Pick spawn keys only from can_spawn and retire keys only from spawned. Every "
            "lever is a change from today, not a target: null leaves price, "
            "marketing_intensity and raise_amount alone; [] or 0 leaves spawn, retire and "
            "open_locations alone. `price` is a new absolute list price within "
            "`you.price_bounds`. Put your reasoning first."
        )

    @staticmethod
    def _spawnable(company: Company) -> list[str]:
        lib = company.library
        return sorted(
            k
            for k, n in lib.nodes.items()
            if company.seed.economics_model in n.applicable_economics
            and lib.prerequisites_met(k, company.spawned)
            and company.spawned.get(k, 0) < n.category_caps.hard_cap
        )

    @staticmethod
    def _user_prompt(
        company: Company,
        day: int,
        companies: list[Company],
        tam: float,
        spawnable: list[str],
        market_shocks: Sequence[Shock] = (),
        best_response: tuple[float, float] | None = None,
    ) -> str:
        lib = company.library
        price_bounds = [_price(company.seed.starting_price * f) for f in PRICE_BOUNDS_FACTOR]
        best_price, best_gain = best_response or (None, None)
        runway = company.runway_months()
        recent = [d for d in company.decisions if d.source == "ai"][-AI_DECISION_MEMORY:]
        unit_cost = company.unit_cost()
        cap = company.env.get("price_ceiling_mult", 1.0)
        gross_profit = company.daily_revenue - company.served * unit_cost
        payload = {
            "day": day,
            "you": {
                "cash": _num(company.cash),
                "runway_months": None if math.isinf(runway) else _num(runway),
                "monthly_revenue": _num(company.daily_revenue * DAYS_PER_MONTH),
                "monthly_gross_profit": _num(gross_profit * DAYS_PER_MONTH),
                "monthly_profit": _num(
                    (company.daily_revenue - company.daily_costs) * DAYS_PER_MONTH
                ),
                "price": _price(company.price),
                "effective_price": _price(company.effective_price(company.env)),
                "price_cap": None if cap >= 1.0 else _price(company.seed.starting_price * cap),
                "starting_price": _price(company.seed.starting_price),
                "revenue_per_customer": _price(company.revenue_per_unit()),
                "unit_cost": _price(unit_cost),
                "gross_margin": _num(company.gross_margin()),
                "contribution_margin": _num(company.contribution_margin()),
                "price_elasticity": _optional_num(company.own_price_elasticity),
                "marketing_intensity": _num(company.marketing_intensity),
                "market_share": _num(company.market_share),
                "target_share": _num(company.target_share),
                "satisfaction": _num(company.satisfaction),
                "utilization": _num(company.utilization),
                "settled_demand_vs_capacity": (
                    None
                    if math.isinf(ratio := company.settled_demand_vs_capacity())
                    else _num(ratio)
                ),
                "price_bounds": price_bounds,
                "best_response_price": None if best_price is None else _price(best_price),
                "monthly_gain_at_best_response": _optional_num(
                    None if best_gain is None else best_gain * DAYS_PER_MONTH
                ),
                "binding": company.binding,
                "employees": company.employees,
                "locations": company.locations,
            },
            "monthly_history": [
                {k: _num(v) if isinstance(v, float) else v for k, v in vars(s).items()}
                for s in list(company.history)[-AI_HISTORY_MONTHS:]
            ],
            "market": {
                "annual_tam_at_starting_price": _num(tam),
                "category_price_elasticity": _optional_num(company.category_elasticity),
                "choke_price": _price(
                    company.seed.starting_price * (1.0 + 1.0 / company.seed.market_price_elasticity)
                ),
                "rivals": [
                    {
                        "name": c.name,
                        "alive": c.alive,
                        "share": _num(c.market_share),
                        "price": _price(c.price),
                        "effective_price": _price(c.effective_price(c.env)),
                        "locations": c.locations,
                    }
                    for c in companies
                    if c is not company
                ],
            },
            "shocks": [
                {
                    "name": s.name,
                    "scope": s.scope,
                    "severity": s.severity,
                    "days_left": s.tick_started + s.duration_ticks - day,
                    "about": s.description,
                }
                for s in [*market_shocks, *company.shocks]
            ],
            "your_last_decisions": [
                {"day": d.day, "reasoning": d.reasoning[:AI_REASONING_MEMORY_CHARS]} for d in recent
            ],
            "spawned": dict(sorted(company.spawned.items())),
            "can_spawn_columns": ["key", "category", "hire_cost", "daily_cost", "effects"],
            "can_spawn": [
                [
                    k,
                    lib.nodes[k].category,
                    _num(lib.nodes[k].hire_cost),
                    _num(lib.nodes[k].daily_fixed_costs),
                    " ".join(f"{b}{v:+.2f}" for b, v in lib.effects(k).items()),
                ]
                for k in spawnable
            ],
        }
        return json.dumps(payload, separators=(",", ":"))
