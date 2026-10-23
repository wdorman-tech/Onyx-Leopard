"""Simulation engine: companies competing for one shared market, one tick = one day.

Per tick (`MarketSim.step`):
  1. TAM compounds at the market growth rate; market-wide shocks arrive/expire.
  2. A multinomial logit over satisfaction, marketing reach and price gives
     the share each company's current offer earns. Customer bases move toward
     those shares under N-species Lotka-Volterra competition (customers
     switch over weeks, not overnight); the bases set today's spend shares.
  3. Category spend responds to the market's average price (constant
     elasticity). Each company serves min(demand, physical capacity,
     Cobb-Douglas production), books revenue and costs, and updates cash.
  4. CEOs decide: heuristic every few days, AI CEO monthly (or on a severe
     shock) while the AI budget is on pace. Decisions spawn/retire nodes and
     move price/marketing/capital.

Headcount never buys market share directly: it buys capacity and production,
which decide how much of the earned demand a company can serve.

Everything industry-specific comes from the seed and the node library.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.simulation import dynamics
from src.simulation.library import Modifiers, NodeLibrary
from src.simulation.seed import CompanySeed
from src.simulation.shocks import Shock, ShockScheduler, environment
from src.simulation.stance import RAISE_ALLOWED_ARCHETYPES, CeoStance

if TYPE_CHECKING:
    from src.simulation.ceo import AiCeo, CeoDecision, HeuristicCeo
    from src.simulation.jev import JevJudge
    from src.simulation.llm import OpenRouterClient

log = logging.getLogger(__name__)

DAYS_PER_YEAR: float = 365.0
DAYS_PER_MONTH: float = 30.0

# ─── Satisfaction ───────────────────────────────────────────────────────────
SATISFACTION_START: float = 0.5
#: Steady-state satisfaction with no quality investment.
SATISFACTION_BASELINE: float = 0.5
#: Steady-state satisfaction gained per unit of summed quality modifiers.
QUALITY_TO_SATISFACTION_GAIN: float = 0.5
#: Daily relaxation rate toward the steady state (time constant ≈ 10 days).
SATISFACTION_RELAXATION_RATE: float = 0.1

# ─── Market share (multinomial logit) ──────────────────────────────────────
SHARE_QUALITY_EXPONENT: float = 1.5
#: Share elasticity of marketing reach. Reach sums every sales and marketing
#: node, so this is what stops sales headcount from buying share one-for-one:
#: meta-analyses put selling-effort elasticity at ~0.31 (Albers, Mantrala &
#: Sridhar 2010) and long-run advertising elasticity at ~0.24 (Sethuraman,
#: Tellis & Briesch 2011). 20× the reach buys ~2.5× the pull, not 20×.
SHARE_MARKETING_EXPONENT: float = 0.3
#: Price elasticity of attraction when it cannot be calibrated from the seed
#: (a single company has no rivals to lose share to): 10% higher price → ~14%
#: less pull. With rivals, `MarketSim` calibrates it from the starting margin.
PRICE_ELASTICITY: float = 1.5
#: Floor on the calibrated share elasticity. A very high starting margin would
#: otherwise imply customers ignore price differences altogether.
MIN_SHARE_PRICE_ELASTICITY: float = 0.25
#: Diminishing returns of marketing intensity on reach.
MARKETING_INTENSITY_ELASTICITY: float = 0.3
#: Daily marketing spend as a fraction of trailing daily revenue at intensity 1.0.
MARKETING_SPEND_FRACTION: float = 0.05
MARKETING_INTENSITY_RANGE: tuple[float, float] = (0.0, 2.0)

# ─── Capacity and production ───────────────────────────────────────────────
INFRA_TO_CAPACITY_GAIN: float = 0.20
#: Saturation scale for infra → capacity: multiplier is 1 + gain * tanh(sum / K).
INFRA_SATURATION_K: float = 5.0
#: Cobb-Douglas TFP is calibrated so day-0 production = this × physical capacity.
PRODUCTION_HEADROOM: float = 1.25
#: TFP gain per unit of summed quality modifiers (better process, better people).
COBB_DOUGLAS_QUALITY_GAIN: float = 0.05

# ─── Node-bucket effects ────────────────────────────────────────────────────
COST_BUCKET_GAIN: float = 1.0
COST_MULT_RANGE: tuple[float, float] = (0.5, 1.5)
REVENUE_BUCKET_GAIN: float = 1.0
REVENUE_MULT_RANGE: tuple[float, float] = (0.5, 2.0)
#: Daily financing inflow per unit of |capital modifiers|, as a fraction of
#: starting cash (investor nodes give standing access to capital).
CAPITAL_BUCKET_DAILY_RATE: float = 0.001

# ─── Lifecycle ──────────────────────────────────────────────────────────────
INSOLVENT_DAYS_TO_DEATH: int = 30
#: Daily chance to lose one employee-bearing node per unit of attrition_mult
#: above neutral (talent war).
ATTRITION_DAILY_PROBABILITY: float = 0.005
#: History snapshots for the AI CEO prompt: one per month, last N months.
HISTORY_EVERY_DAYS: int = 30
HISTORY_LEN: int = 12
#: Whole-run time series for the run report: at most this many samples per
#: company (daily for runs up to this length, evenly spaced beyond it).
MAX_SERIES_POINTS: int = 1000

# ─── Lotka-Volterra customer bases ─────────────────────────────────────────
#: Each company's customer base (a fraction of the market's customers) follows
#: N-species LV competition. Carrying capacities are reset daily so the LV rest
#: point is exactly the logit share of today's offers (`dynamics.
#: lv_capacities_for_equilibrium`): bases drift toward what each offer earns,
#: rivals' bases slow each other on the way, and nobody is excluded at rest.
#: Intrinsic growth rate; a mid-share base closes ~63% of a gap in ~2 months.
LV_GROWTH_RATE: float = 0.05
#: Mean off-diagonal competition coefficient (customer overlap between rivals).
LV_COMPETITION_BASE: float = 0.3
LV_COMPETITION_PERTURBATION: float = 0.1
#: A base never falls below this share of customers: zero is absorbing in LV,
#: and a company with any offer at all keeps a sliver of the market.
CUSTOMER_BASE_FLOOR: float = 1e-4

# ─── CEO decision bounds ────────────────────────────────────────────────────
PRICE_BOUNDS_FACTOR: tuple[float, float] = (0.1, 10.0)
MAX_LOCATIONS_PER_DECISION: int = 5
MAX_SPAWNS_PER_DECISION: int = 5
#: A single raise is capped at max(starting cash, this many months of revenue).
RAISE_CAP_MONTHS_OF_REVENUE: float = 12.0

#: Heuristic CEO cadence (days) and AI CEO cadence (days).
HEURISTIC_CADENCE_DAYS: int = 7
AI_CADENCE_DAYS: int = 30


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


@dataclass
class Snapshot:
    day: int
    cash: float
    monthly_revenue: float
    monthly_profit: float
    price: float
    gross_margin: float
    market_share: float
    satisfaction: float
    utilization: float
    employees: int
    locations: int


class Company:
    """One company: seed + stance + spawned nodes + running state."""

    def __init__(
        self,
        *,
        company_id: str,
        name: str,
        seed: CompanySeed,
        stance: CeoStance,
        library: NodeLibrary,
        rng: random.Random,
        shock_scheduler: ShockScheduler,
    ) -> None:
        library.validate_seed(seed)
        self.company_id = company_id
        self.name = name
        self.seed = seed
        self.stance = stance
        self.library = library
        self.rng = rng
        self.shock_scheduler = shock_scheduler

        self.alive = True
        self.cash = float(seed.starting_cash)
        self.price = float(seed.starting_price)
        self.marketing_intensity = 1.0
        self.satisfaction = SATISFACTION_START
        self.insolvent_days = 0
        self.capex = 0.0  # cumulative hire/build-out spend since day 0

        # Last-tick metrics.
        self.daily_revenue = 0.0
        self.core_revenue = 0.0  # served × price, before revenue-stream uplift
        self.daily_costs = 0.0
        self.fixed_burn = 0.0
        #: Today's multiplier on node daily costs (cost bucket × shocks).
        self.fixed_cost_mult = 1.0
        self.market_share = 0.0  # share of category spend served by our customer base
        self.target_share = 0.0  # share today's offer earns once customers settle
        self.unit_share = 0.0  # share of category units (sets our pull on the price index)
        self.own_price_elasticity = 0.0
        #: LV customer base; MarketSim seeds it at the first day's target share.
        self.customer_base: float | None = None
        self.demand = 0.0
        self.capacity = 0.0
        self.production = 0.0
        self.served = 0.0
        self.utilization = 0.0
        self.binding = "demand"
        self.env: dict[str, float] = {}

        # Whole-run record for the run report.
        self.total_revenue = 0.0
        self.total_costs = 0.0
        self.peak_cash = self.cash
        self.died_on: int | None = None
        self.series: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []  # structured shocks and bankruptcy

        self.shocks: list[Shock] = []  # active company-scope shocks
        self.history: deque[Snapshot] = deque(maxlen=HISTORY_LEN)
        self.decisions: list[CeoDecision] = []  # full run log, for the post-mortem
        # Drained into each tick payload.
        self.new_decisions: list[CeoDecision] = []
        self.events: list[str] = []

        self.primary_location = library.primary_location(seed.economics_model)
        self.spawned = library.starting_nodes(
            [
                *seed.initial_supplier_types,
                *seed.initial_revenue_streams,
                *seed.initial_cost_centers,
            ],
            seed.economics_model,
            seed.initial_locations,
        )

        # Calibrate TFP so day-0 production sits PRODUCTION_HEADROOM above capacity;
        # afterwards production moves only with capital, headcount and quality.
        mods = library.modifiers(self.spawned)
        base = dynamics.cobb_douglas(
            1.0 + COBB_DOUGLAS_QUALITY_GAIN * mods.quality,
            self._capital_stock(mods),
            float(self.employees),
            seed.cobb_douglas_alpha,
            seed.cobb_douglas_beta,
        )
        self.tfp0 = PRODUCTION_HEADROOM * self._physical_capacity(mods) / base

    # ── Derived quantities ──

    @property
    def employees(self) -> int:
        return self.seed.starting_employees + sum(
            self.library.nodes[k].employees_count * c for k, c in self.spawned.items()
        )

    @property
    def locations(self) -> int:
        return sum(
            c
            for k, c in self.spawned.items()
            if self.library.nodes[k].category == "location"
            and self.library.nodes[k].capacity_contribution > 0
        )

    def _physical_capacity(self, mods: Modifiers) -> float:
        infra = 1.0 + INFRA_TO_CAPACITY_GAIN * math.tanh(mods.infrastructure / INFRA_SATURATION_K)
        return self.seed.base_capacity_per_location * max(1, self.locations) * infra

    def _capital_stock(self, mods: Modifiers) -> float:
        return self.seed.starting_cash * (1.0 + abs(mods.capital)) + self.capex

    def runway_months(self) -> float:
        net_burn = self.daily_costs - self.daily_revenue
        if net_burn <= 0:
            return math.inf
        return max(0.0, self.cash) / (net_burn * DAYS_PER_MONTH)

    def unit_cost(self) -> float:
        """Variable cost per customer served today, after supply shocks."""
        return self.seed.base_unit_cost * self.env.get("supply_cost_mult", 1.0)

    def gross_margin(self) -> float:
        """(price - unit cost) / price at today's effective price."""
        price = self.effective_price(self.env)
        return (price - self.unit_cost()) / price

    # ── Tick phases ──

    def attraction_inputs(self) -> tuple[float, float, float, float]:
        """(quality, marketing reach, effective price, extra weight) for the logit."""
        mods = self.library.modifiers(self.spawned)
        env = environment(self.shocks)
        reach = (1.0 + mods.marketing) * max(
            self.marketing_intensity, 1e-6
        ) ** MARKETING_INTENSITY_ELASTICITY
        return (
            self.satisfaction,
            reach,
            self.effective_price(env),
            env.get("market_share_mult", 1.0),
        )

    def effective_price(self, env: dict[str, float]) -> float:
        ceiling = env.get("price_ceiling_mult", 1.0)
        if ceiling < 1.0:
            return min(self.price, self.seed.starting_price * ceiling)
        return self.price

    def receive_shocks(self, tick: int) -> None:
        for shock in self.shock_scheduler.arrivals(tick, self.shocks):
            self.shocks.append(shock)
            self.events.append(f"Day {tick}: {self.name}: {shock.description}")
            self.log.append(shock_record(shock, self.company_id))
            self._apply_one_shot(shock)
        self.shocks = [s for s in self.shocks if s.active_at(tick)]

    def operate(
        self, tick: int, market_spend: float, share: float, market_shocks: list[Shock]
    ) -> None:
        """Serve customers for one day and settle cash."""
        env = self.env = environment(market_shocks + self.shocks)
        mods = self.library.modifiers(self.spawned)

        # Satisfaction relaxes toward its quality-driven steady state; shocks
        # move the steady state, not the level, so they cannot compound daily.
        target = SATISFACTION_BASELINE + QUALITY_TO_SATISFACTION_GAIN * mods.quality
        target = min(1.0, max(0.0, target * env.get("satisfaction_mult", 1.0)))
        self.satisfaction = dynamics.relax(self.satisfaction, target, SATISFACTION_RELAXATION_RATE)

        price = self.effective_price(env)
        self.market_share = share
        self.demand = market_spend * share / price
        self.demand *= env.get("inventory_throughput_mult", 1.0)
        self.capacity = (
            self._physical_capacity(mods)
            * env.get("team_capacity_mult", 1.0)
            * env.get("capacity_cap_mult", 1.0)
        )
        tfp = self.tfp0 * (1.0 + COBB_DOUGLAS_QUALITY_GAIN * mods.quality)
        self.production = dynamics.cobb_douglas(
            tfp,
            self._capital_stock(mods),
            float(max(1, self.employees)),
            self.seed.cobb_douglas_alpha,
            self.seed.cobb_douglas_beta,
        )
        self.served, self.binding = min(
            (self.demand, "demand"), (self.capacity, "capacity"), (self.production, "production")
        )
        self.utilization = self.served / self.capacity if self.capacity > 0 else 0.0

        revenue_mult = _clamp(1.0 + REVENUE_BUCKET_GAIN * mods.revenue, REVENUE_MULT_RANGE)
        cost_mult = _clamp(1.0 + COST_BUCKET_GAIN * mods.cost, COST_MULT_RANGE)
        self.fixed_cost_mult = (
            cost_mult * env.get("fixed_cost_mult", 1.0) * env.get("wage_inflation_mult", 1.0)
        )
        self.fixed_burn = self._base_fixed_costs() * self.fixed_cost_mult
        marketing_spend = (
            MARKETING_SPEND_FRACTION
            * self.marketing_intensity
            * self.daily_revenue  # trailing day
            * env.get("acquisition_cost_mult", 1.0)
        )
        variable = self.served * self.unit_cost()

        self.core_revenue = self.served * price
        self.daily_revenue = self.core_revenue * revenue_mult
        self.daily_costs = variable + self.fixed_burn + marketing_spend
        self.cash += self.daily_revenue - self.daily_costs
        self.total_revenue += self.daily_revenue
        self.total_costs += self.daily_costs

        financing = max(0.0, env.get("financing_availability_mult", 1.0))
        if self.stance.archetype in RAISE_ALLOWED_ARCHETYPES and mods.capital:
            self.cash += (
                abs(mods.capital) * CAPITAL_BUCKET_DAILY_RATE * self.seed.starting_cash * financing
            )

        attrition = env.get("attrition_mult", 1.0)
        if attrition > 1.0 and self.rng.random() < ATTRITION_DAILY_PROBABILITY * attrition:
            self._retire_random_employee()

        self.insolvent_days = self.insolvent_days + 1 if self.cash < 0 else 0
        if self.insolvent_days >= INSOLVENT_DAYS_TO_DEATH:
            self.alive = False
            self.died_on = tick
            self.events.append(f"Day {tick}: {self.name} went bankrupt.")
            self.log.append({"day": tick, "kind": "bankruptcy", "company_id": self.company_id})

        if tick % HISTORY_EVERY_DAYS == 0:
            self.history.append(
                Snapshot(
                    day=tick,
                    cash=self.cash,
                    monthly_revenue=self.daily_revenue * DAYS_PER_MONTH,
                    monthly_profit=(self.daily_revenue - self.daily_costs) * DAYS_PER_MONTH,
                    price=price,
                    gross_margin=self.gross_margin(),
                    market_share=self.market_share,
                    satisfaction=self.satisfaction,
                    utilization=self.utilization,
                    employees=self.employees,
                    locations=self.locations,
                )
            )

    def _base_fixed_costs(self) -> float:
        """Daily overhead plus every spawned node's daily cost, before multipliers."""
        return self.seed.daily_fixed_costs + self.library.daily_cost(self.spawned)

    def _apply_one_shot(self, shock: Shock) -> None:
        # Costs scale with the company: days of one average employee's share
        # of fixed costs, so a departure weighs the same at any headcount.
        per_employee = self._base_fixed_costs() * self.fixed_cost_mult / max(1, self.employees)
        self.cash -= shock.impact.get("replacement_cost_employee_days", 0.0) * per_employee
        if shock.impact.get("retire_random_employee", 0.0) > 0:
            self._retire_random_employee()

    def _retire_random_employee(self) -> None:
        candidates = [
            k
            for k, c in self.spawned.items()
            if c > 0
            and self.library.nodes[k].category not in ("exec", "location")
            and self.library.nodes[k].employees_count > 0
        ]
        if candidates:
            self._retire(self.rng.choice(sorted(candidates)))

    # ── Decisions ──

    def apply(self, decision: CeoDecision) -> None:
        hire_mult = self.env.get("hire_cost_mult", 1.0)
        for key in decision.spawn:
            self._spawn(key, hire_mult)
        for key in decision.retire:
            self._retire(key)
        if self.primary_location is not None:
            for _ in range(decision.open_locations):
                self._spawn(self.primary_location, hire_mult)
        if decision.price is not None:
            self.price = decision.price
        if decision.marketing_intensity is not None:
            self.marketing_intensity = decision.marketing_intensity
        if decision.raise_amount:
            financing = max(0.0, self.env.get("financing_availability_mult", 1.0))
            self.cash += decision.raise_amount * financing
        self.decisions.append(decision)
        self.new_decisions.append(decision)

    def _spawn(self, key: str, hire_mult: float) -> None:
        node = self.library.nodes[key]
        cost = node.hire_cost * hire_mult
        self.cash -= cost
        self.capex += cost
        self.spawned[key] = self.spawned.get(key, 0) + 1

    def _retire(self, key: str) -> None:
        if self.spawned.get(key, 0) <= 0:
            return
        self.spawned[key] -= 1
        if self.spawned[key] == 0:
            del self.spawned[key]

    # ── Output ──

    def close_day(self, tick: int, sample: bool) -> None:
        """End-of-day bookkeeping after decisions; appends a series row when sampled."""
        self.peak_cash = max(self.peak_cash, self.cash)
        if not (sample or self.died_on == tick):
            return
        runway = self.runway_months()
        self.series.append(
            {
                "day": tick,
                "alive": self.alive,
                "cash": self.cash,
                "revenue": self.daily_revenue if self.alive else 0.0,
                "core_revenue": self.core_revenue if self.alive else 0.0,
                "costs": self.daily_costs if self.alive else 0.0,
                "profit": (self.daily_revenue - self.daily_costs) if self.alive else 0.0,
                "market_share": self.market_share if self.alive else 0.0,
                "price": self.price,
                "marketing_intensity": self.marketing_intensity,
                "satisfaction": self.satisfaction,
                "utilization": self.utilization,
                "employees": self.employees,
                "locations": self.locations,
                "runway_months": None if math.isinf(runway) else runway,
                "binding": self.binding,
            }
        )

    def graph(self) -> dict[str, Any]:
        nodes = []
        edges = []
        for key, count in sorted(self.spawned.items()):
            node = self.library.nodes[key]
            nodes.append(
                {
                    "id": key,
                    "type": key,
                    "label": f"{node.label} (×{count})" if count > 1 else node.label,
                    "category": node.category,
                    "spawned_at": 0,
                    "metrics": {
                        "count": float(count),
                        "daily_cost": node.daily_fixed_costs * count,
                        "employees": float(node.employees_count * count),
                    },
                }
            )
            edges.extend(
                {"source": key, "target": p, "relationship": "depends_on"}
                for p in node.prerequisites
                if p in self.spawned
            )
        return {"nodes": nodes, "edges": edges}

    def payload(self) -> dict[str, Any]:
        decisions = [d.model_dump() for d in self.new_decisions]
        self.new_decisions = []
        return {
            "company_id": self.company_id,
            "name": self.name,
            "archetype": self.stance.archetype,
            "alive": self.alive,
            "cash": self.cash,
            "daily_revenue": self.daily_revenue if self.alive else 0.0,
            "core_revenue": self.core_revenue if self.alive else 0.0,
            "daily_costs": self.daily_costs if self.alive else 0.0,
            "market_share": self.market_share if self.alive else 0.0,
            "price": self.price,
            "marketing_intensity": self.marketing_intensity,
            "satisfaction": self.satisfaction,
            "utilization": self.utilization,
            "binding": self.binding,
            "employees": self.employees,
            "locations": self.locations,
            "runway_months": None if math.isinf(r := self.runway_months()) else r,
            "shocks": [s.name for s in self.shocks],
            "decisions": decisions,
            "graph": self.graph(),
        }


@dataclass
class MarketSim:
    """All companies in one shared market. One instance per simulation run."""

    sim_id: str
    companies: list[Company]
    max_ticks: int
    tam: float  # annual
    market_growth_rate: float  # annual
    market_shocks: ShockScheduler
    heuristic: dict[str, HeuristicCeo]
    ai: dict[str, AiCeo] = field(default_factory=dict)
    llm: OpenRouterClient | None = None
    jev: JevJudge | None = None
    #: Run settings echoed into the saved run (model, budget, toggles).
    settings: dict[str, Any] = field(default_factory=dict)
    lv_seed: int = 0
    tick: int = 0
    active_market_shocks: list[Shock] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lv_alpha = dynamics.competition_matrix(
            len(self.companies),
            LV_COMPETITION_BASE,
            LV_COMPETITION_PERTURBATION,
            np.random.default_rng(self.lv_seed),
        )
        # Every company in a run shares one seed, so the market is described by it.
        seed = self.companies[0].seed
        self.reference_price = seed.starting_price
        self.market_price_elasticity = seed.market_price_elasticity
        self.share_price_elasticity = calibrated_share_elasticity(seed, len(self.companies))
        self.daily_spend = 0.0
        self._severe_seen: set[tuple[str, str, int]] = set()
        self._ai_stop_reported = False
        self._ai_pacing_reported = False
        self.ai_rounds_paced = 0  # monthly AI rounds skipped to stretch the budget
        self.sample_every = max(1, math.ceil(self.max_ticks / MAX_SERIES_POINTS))
        self.market_series: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []  # structured market shocks

    @property
    def alive(self) -> list[Company]:
        return [c for c in self.companies if c.alive]

    @property
    def is_complete(self) -> bool:
        return self.tick >= self.max_ticks or not self.alive

    def ai_due(self, tick: int) -> bool:
        """True when `tick` will trigger the monthly AI CEO round."""
        return (
            tick % AI_CADENCE_DAYS == 0
            and self.llm is not None
            and self.llm.available
            and self.llm.budget.on_pace(tick, self.max_ticks)
            and any(c.company_id in self.ai for c in self.alive)
        )

    async def step(self) -> None:
        """Advance one day. Events and decisions accumulate until `payload()`."""
        self.tick += 1
        tick = self.tick
        self.tam *= (1.0 + self.market_growth_rate) ** (1.0 / DAYS_PER_YEAR)

        for shock in self.market_shocks.arrivals(tick, self.active_market_shocks):
            self.active_market_shocks.append(shock)
            self.events.append(f"Day {tick}: Market: {shock.description}")
            self.log.append(shock_record(shock, None))
        self.active_market_shocks = [s for s in self.active_market_shocks if s.active_at(tick)]
        market_env = environment(self.active_market_shocks)

        alive = self.alive
        if alive:
            self._serve(tick, alive, market_env)
        await self._decide(tick, self.alive)

        sample = tick % self.sample_every == 0 or self.is_complete
        for company in alive:
            company.close_day(tick, sample)
        if sample:
            self.market_series.append(
                {
                    "day": tick,
                    "tam": self.tam,
                    "spend": self.daily_spend * DAYS_PER_YEAR,
                    "alive": len(self.alive),
                    "shocks": [s.name for s in self.active_market_shocks],
                }
            )

    def _serve(self, tick: int, alive: list[Company], market_env: dict[str, float]) -> None:
        """Clear the market for one day: shares, category spend, then each company operates."""
        for company in alive:
            company.receive_shocks(tick)
        inputs = [c.attraction_inputs() for c in alive]
        targets = dynamics.logit_shares(
            [i[0] for i in inputs],
            [i[1] for i in inputs],
            [i[2] for i in inputs],
            [i[3] for i in inputs],
            quality_exponent=SHARE_QUALITY_EXPONENT,
            marketing_exponent=SHARE_MARKETING_EXPONENT,
            price_elasticity=self.share_price_elasticity,
        )
        shares = self._customer_shares(alive, targets)

        # Average price per unit sold across the market (spend-weighted harmonic
        # mean); category spend moves with it at the seed's price elasticity.
        prices = [i[2] for i in inputs]
        price_index = 1.0 / sum(s / p for s, p in zip(shares, prices, strict=True))
        self.daily_spend = dynamics.constant_elasticity_spend(
            self.tam / DAYS_PER_YEAR,
            price_index,
            self.reference_price,
            self.market_price_elasticity,
        ) * market_env.get("market_demand_mult", 1.0)
        for company, share, target, price in zip(alive, shares, targets, prices, strict=True):
            company.target_share = target
            company.unit_share = share / price * price_index
            company.own_price_elasticity = dynamics.own_price_elasticity(
                self.share_price_elasticity,
                target,
                self.market_price_elasticity,
                company.unit_share,
            )
            company.operate(tick, self.daily_spend, share, self.active_market_shocks)

    def _customer_shares(self, alive: list[Company], targets: list[float]) -> list[float]:
        """Advance every customer base one day toward its target; return spend shares."""
        if len(alive) == 1:
            alive[0].customer_base = 1.0
            return [1.0]
        target = np.array(targets)
        base = np.array(
            [
                t if c.customer_base is None else c.customer_base
                for c, t in zip(alive, targets, strict=True)
            ]
        )
        idx = [self.companies.index(c) for c in alive]
        alpha = self._lv_alpha[np.ix_(idx, idx)]
        base = dynamics.lotka_volterra_step(
            base,
            np.full(len(alive), LV_GROWTH_RATE),
            dynamics.lv_capacities_for_equilibrium(alpha, target),
            alpha,
        )
        base = np.maximum(base, CUSTOMER_BASE_FLOOR)
        for company, b in zip(alive, base, strict=True):
            company.customer_base = float(b)
        return [float(b) for b in base / base.sum()]

    async def _decide(self, tick: int, companies: list[Company]) -> None:
        # Weekly rule CEO: Python proposes; Jev, when on, approves or vetoes.
        proposals = []
        for company in companies:
            ceo = self.heuristic[company.company_id]
            proposal = ceo.propose(company, tick)
            if proposal is not None:
                proposals.append((company, ceo, proposal))
        if proposals and self.jev is not None and self.jev.available:
            verdicts = await asyncio.gather(
                *(self.jev.review(c, proposal, tick) for c, _, proposal in proposals)
            )
        else:
            verdicts = [None] * len(proposals)
        for (company, ceo, proposal), verdict in zip(proposals, verdicts, strict=True):
            if verdict is None or verdict.approved:
                ceo.commit(company, proposal)
            else:
                ceo.defer()
            company.apply(proposal if verdict is None else verdict.decision)

        # AI CEOs wake monthly, and early on the first tick of a severe shock
        # that hits them (market-wide or their own). Monthly rounds run only
        # while spend is on pace to last the whole run; severe shocks may
        # borrow ahead, which pushes the next monthly round back.
        market_severe = self._new_severe("market", self.active_market_shocks)
        monthly = tick % AI_CADENCE_DAYS == 0 and self._ai_on_pace(tick)
        due = []
        for company in companies:
            own_severe = self._new_severe(company.company_id, company.shocks)
            if company.company_id in self.ai and (monthly or market_severe or own_severe):
                due.append(company)
        if due:
            results = await asyncio.gather(
                *(
                    self.ai[c.company_id].decide(
                        c, tick, self.companies, self.tam, self.active_market_shocks
                    )
                    for c in due
                )
            )
            for company, decision in zip(due, results, strict=True):
                if decision is not None:
                    company.apply(decision)

        if self.llm is not None and not self.llm.available and not self._ai_stop_reported:
            self._ai_stop_reported = True
            budget = self.llm.budget
            reason = self.llm.disabled_reason or (
                f"AI budget of ${budget.ceiling_usd:.2f} reached "
                f"(${budget.ceiling_usd - budget.spent_usd:.2f} kept for the CEO post-mortems)"
            )
            self.events.append(f"Day {tick}: {reason}. CEOs continue on heuristics.")

    def _ai_on_pace(self, tick: int) -> bool:
        if self.llm is None or not self.llm.available or not self.ai:
            return True
        if self.llm.budget.on_pace(tick, self.max_ticks):
            return True
        self.ai_rounds_paced += 1
        if not self._ai_pacing_reported:
            self._ai_pacing_reported = True
            budget = self.llm.budget
            self.events.append(
                f"Day {tick}: AI spend (${budget.spent_usd:.3f} of ${budget.ceiling_usd:.2f}) "
                "is ahead of pace, so AI CEOs skip some monthly rounds to last the whole run. "
                "Heuristics cover the gaps."
            )
        return False

    def _new_severe(self, scope: str, shocks: list[Shock]) -> bool:
        keys = {(scope, s.name, s.tick_started) for s in shocks if s.severity == "severe"}
        fresh = keys - self._severe_seen
        self._severe_seen |= fresh
        return bool(fresh)

    def ai_status(self) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        return {
            "model": self.llm.model.id,
            "spent_usd": self.llm.budget.spent_usd,
            "budget_usd": self.llm.budget.ceiling_usd,
            "calls": self.llm.budget.calls,
            "paced_rounds": self.ai_rounds_paced,
            "error": self.llm.disabled_reason,
        }

    def payload(self) -> dict[str, Any]:
        """Snapshot for the stream; drains events and decisions since the last call."""
        events = self.events + [e for c in self.companies for e in c.events]
        self.events = []
        for c in self.companies:
            c.events = []
        return {
            "type": "tick",
            "tick": self.tick,
            "max_ticks": self.max_ticks,
            "tam": self.tam,
            "alive": len(self.alive),
            "market_shocks": [s.name for s in self.active_market_shocks],
            "events": events,
            "ai": self.ai_status(),
            "jev": self.jev.status() if self.jev is not None else None,
            "companies": [c.payload() for c in self.companies],
        }


def calibrated_share_elasticity(seed: CompanySeed, companies: int) -> float:
    """Logit price elasticity at which the seed's starting price is the Nash price.

    All companies start from the same seed with equal shares, so the observed
    starting margin pins down how price-sensitive switching must be for that
    price to be an equilibrium (`dynamics.nash_share_elasticity`). Undercutting
    then pays only as far as the margin the market actually supports.
    """
    if companies < 2:
        return PRICE_ELASTICITY
    margin = (seed.starting_price - seed.base_unit_cost) / seed.starting_price
    elasticity = dynamics.nash_share_elasticity(
        margin, 1.0 / companies, seed.market_price_elasticity
    )
    return max(MIN_SHARE_PRICE_ELASTICITY, elasticity)


def shock_record(shock: Shock, company_id: str | None) -> dict[str, Any]:
    return {
        "day": shock.tick_started,
        "kind": "shock",
        "company_id": company_id,
        "name": shock.name,
        "scope": shock.scope,
        "severity": shock.severity,
        "duration_days": shock.duration_ticks,
        "description": shock.description,
    }
