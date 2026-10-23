"""Simulation engine: companies competing for one shared market, one tick = one day.

Per tick (`MarketSim.step`):
  1. TAM compounds at the market growth rate; market-wide shocks arrive/expire.
  2. A multinomial logit over satisfaction, marketing reach and price gives
     the share each company's current offer earns. Customer bases move toward
     those shares under N-species Lotka-Volterra competition (customers
     switch over weeks, not overnight); the bases set today's spend shares.
  3. Category spend responds to the market's average price (linear demand
     calibrated to the seed's elasticity). Each company serves min(demand, physical capacity,
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
    from collections.abc import Callable, Iterable

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
#: Floor on the calibrated share elasticity. The Lerner inversion needs
#: ε_s ≤ 0 (no switching at all) when 1/margin - 1 < (ε_m - 1)·share: a margin
#: near 1 (unit cost a few percent of price), or an elastic category (ε_m > 1)
#: where the industry gains by cutting price together. Sampled seeds stay above
#: it (lowest ≈ 0.064, high-margin SaaS at 10 rivals); such custom or imported
#: seeds are clamped here and start only near, not at, the Nash price.
MIN_SHARE_PRICE_ELASTICITY: float = 0.05
#: Ceiling on the calibrated share elasticity: near-perfect substitutes. A thin
#: or zero starting margin would otherwise imply ε → ∞ (any price gap takes the
#: whole market). Measured brand-level price elasticities average -2.6 and
#: rarely pass -6 (Bijmolt, van Heerde & Pieters 2005 meta-analysis).
MAX_SHARE_PRICE_ELASTICITY: float = 10.0
#: Elasticity of reach to marketing intensity at the baseline (intensity 1).
#: Reach uses the (1 + intensity) transform (Hanssens, Parsons & Schultz 2001):
#: a pure power law sends reach to zero at zero spend, which no market shows,
#: since brand and word of mouth remain. ((1 + x) / 2)^(2e) has elasticity e
#: at x = 1, so intensity 0 keeps (1/2)^(2e) of baseline reach (≈ 66%).
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
# Cost and revenue multipliers live on `Modifiers` (library.py).
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

#: Best-response price search: log-spaced grid over the price bounds, then
#: golden-section refinement around the best grid point.
BEST_RESPONSE_GRID_POINTS: int = 241
BEST_RESPONSE_REFINE_STEPS: int = 40
#: Relative contribution gap below which today's list price counts as best.
BEST_RESPONSE_TOLERANCE: float = 1e-9
#: The search stops this far (relative) below the choke price: further than the
#: prompt's price rounding (6 significant figures), so a copied price can never
#: round up onto the choke, where nobody buys.
BEST_RESPONSE_CHOKE_MARGIN: float = 1e-4

#: Heuristic CEO cadence (days) and AI CEO cadence (days).
HEURISTIC_CADENCE_DAYS: int = 7
AI_CADENCE_DAYS: int = 30


def fixed_cost_multiplier(mods: Modifiers, env: dict[str, float]) -> float:
    """Today's multiplier on daily node and overhead costs: cost bucket × shocks."""
    return (
        mods.cost_multiplier()
        * env.get("fixed_cost_mult", 1.0)
        * env.get("wage_inflation_mult", 1.0)
    )


@dataclass
class Snapshot:
    day: int
    cash: float
    monthly_revenue: float
    monthly_profit: float
    price: float  # list price
    effective_price: float  # what customers paid (after any price cap)
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
        #: Multiplier on node daily costs (cost bucket × shocks) as of the last
        #: operate(); seeded from the starting nodes below.
        self.fixed_cost_mult = 1.0
        #: Revenue per unit served, relative to price (revenue-stream uplift).
        self.revenue_mult = 1.0
        self.market_share = 0.0  # share of category spend served by our customer base
        self.target_share = 0.0  # share today's offer earns once customers settle
        self.own_price_elasticity = 0.0
        #: Category demand elasticity at today's settled average price.
        self.category_elasticity = seed.market_price_elasticity
        #: LV customer base; MarketSim seeds it at the first day's target share.
        self.customer_base: float | None = None
        self.demand = 0.0
        #: Demand once customers settle on today's offers (target share).
        self.settled_demand = 0.0
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
        self.fixed_cost_mult = mods.cost_multiplier()
        self.revenue_mult = mods.revenue_multiplier()

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

    def capacity_and_production(self, env: dict[str, float]) -> tuple[float, float]:
        """(physical capacity, Cobb-Douglas production) for today's nodes under `env`."""
        mods = self.library.modifiers(self.spawned)
        capacity = (
            self._physical_capacity(mods)
            * env.get("team_capacity_mult", 1.0)
            * env.get("capacity_cap_mult", 1.0)
        )
        tfp = self.tfp0 * (1.0 + COBB_DOUGLAS_QUALITY_GAIN * mods.quality)
        production = dynamics.cobb_douglas(
            tfp,
            self._capital_stock(mods),
            float(max(1, self.employees)),
            self.seed.cobb_douglas_alpha,
            self.seed.cobb_douglas_beta,
        )
        return capacity, production

    def _physical_capacity(self, mods: Modifiers) -> float:
        infra = 1.0 + INFRA_TO_CAPACITY_GAIN * math.tanh(mods.infrastructure / INFRA_SATURATION_K)
        return self.seed.base_capacity_per_location * max(1, self.locations) * infra

    def _capital_stock(self, mods: Modifiers) -> float:
        return self.seed.starting_cash * (1.0 + abs(mods.capital)) + self.capex

    def settled_demand_vs_capacity(self) -> float:
        """Settled demand over what the company can serve; above 1 it turns customers away."""
        servable = min(self.capacity, self.production)
        return self.settled_demand / servable if servable > 0 else math.inf

    def runway_months(self) -> float:
        net_burn = self.daily_costs - self.daily_revenue
        if net_burn <= 0:
            return math.inf
        return max(0.0, self.cash) / (net_burn * DAYS_PER_MONTH)

    def unit_cost(self) -> float:
        """Variable cost per customer served today, after supply shocks."""
        return self.seed.base_unit_cost * self.env.get("supply_cost_mult", 1.0)

    def revenue_per_unit(self) -> float:
        """What one customer served brings in: effective price × revenue-stream uplift."""
        return self.effective_price(self.env) * self.revenue_mult

    def gross_margin(self) -> float:
        """(revenue per unit - unit cost) / revenue per unit."""
        revenue = self.revenue_per_unit()
        return (revenue - self.unit_cost()) / revenue

    def contribution_margin(self) -> float:
        """Gross margin after marketing too, which scales with revenue.

        This is the margin the Lerner rule prices against: every cost here
        moves with each unit sold.
        """
        # Marketing is at most a few percent of revenue (intensity ≤ 2), so net > 0.
        net = self.revenue_per_unit() * (1.0 - self._marketing_share())
        return (net - self.unit_cost()) / net

    def _marketing_share(self) -> float:
        """Marketing spend as a share of revenue (it scales with revenue)."""
        return (
            MARKETING_SPEND_FRACTION
            * self.marketing_intensity
            * self.env.get("acquisition_cost_mult", 1.0)
        )

    # ── Tick phases ──

    def attraction_inputs(
        self, shocks: list[Shock] | None = None
    ) -> tuple[float, float, float, float]:
        """(quality, marketing reach, effective price, extra weight) for the logit.

        `shocks` replaces the company's own shocks (for what-if pricing).
        """
        mods = self.library.modifiers(self.spawned)
        env = environment(self.shocks if shocks is None else shocks)
        reach = (1.0 + mods.marketing) * ((1.0 + self.marketing_intensity) / 2.0) ** (
            2.0 * MARKETING_INTENSITY_ELASTICITY
        )
        return (
            self.satisfaction,
            reach,
            self.effective_price(env),
            env.get("market_share_mult", 1.0),
        )

    def effective_price(self, env: dict[str, float]) -> float:
        return self.capped_price(self.price, env)

    def capped_price(self, price: float, env: dict[str, float]) -> float:
        """What customers pay at list `price`: a price-ceiling shock caps it."""
        ceiling = env.get("price_ceiling_mult", 1.0)
        if ceiling < 1.0:
            return min(price, self.seed.starting_price * ceiling)
        return price

    def receive_shocks(self, tick: int, market_shocks: list[Shock] | None = None) -> None:
        arrived = self.shock_scheduler.arrivals(tick, self.shocks)
        self.shocks = [s for s in self.shocks + arrived if s.active_at(tick)]
        if not arrived:
            return
        # One-shot costs are priced at today's nodes and shocks, not yesterday's.
        env = environment([*(market_shocks or []), *self.shocks])
        for shock in arrived:
            self.events.append(f"Day {tick}: {self.name}: {shock.description}")
            self.log.append(shock_record(shock, self.company_id))
            self._apply_one_shot(shock, env)

    def operate(
        self,
        tick: int,
        market_spend: float,
        share: float,
        market_shocks: list[Shock],
        choke_price: float = math.inf,
    ) -> None:
        """Serve customers for one day and settle cash.

        No customer pays `choke_price` or more, so an offer priced there sells
        nothing, whatever its customer base.
        """
        env = self.env = environment(market_shocks + self.shocks)
        mods = self.library.modifiers(self.spawned)

        # Satisfaction relaxes toward its quality-driven steady state; shocks
        # move the steady state, not the level, so they cannot compound daily.
        target = SATISFACTION_BASELINE + QUALITY_TO_SATISFACTION_GAIN * mods.quality
        target = min(1.0, max(0.0, target * env.get("satisfaction_mult", 1.0)))
        self.satisfaction = dynamics.relax(self.satisfaction, target, SATISFACTION_RELAXATION_RATE)

        price = self.effective_price(env)
        self.market_share = share
        throughput = env.get("inventory_throughput_mult", 1.0) if price < choke_price else 0.0
        self.demand = market_spend * share / price * throughput
        self.settled_demand = market_spend * self.target_share / price * throughput
        self.capacity, self.production = self.capacity_and_production(env)
        self.served, self.binding = min(
            (self.demand, "demand"), (self.capacity, "capacity"), (self.production, "production")
        )
        self.utilization = self.served / self.capacity if self.capacity > 0 else 0.0

        self.revenue_mult = mods.revenue_multiplier()
        self.fixed_cost_mult = fixed_cost_multiplier(mods, env)
        self.fixed_burn = self._base_fixed_costs() * self.fixed_cost_mult
        marketing_spend = (
            MARKETING_SPEND_FRACTION
            * self.marketing_intensity
            * self.daily_revenue  # trailing day
            * env.get("acquisition_cost_mult", 1.0)
        )
        variable = self.served * self.unit_cost()

        self.core_revenue = self.served * price
        self.daily_revenue = self.core_revenue * self.revenue_mult
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
                    price=self.price,
                    effective_price=price,
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

    def _apply_one_shot(self, shock: Shock, env: dict[str, float]) -> None:
        # Costs scale with the company: days of one average employee's share
        # of fixed costs, so a departure weighs the same at any headcount.
        mult = fixed_cost_multiplier(self.library.modifiers(self.spawned), env)
        per_employee = self._base_fixed_costs() * mult / max(1, self.employees)
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
        self.share_price_elasticity = calibrated_share_elasticity(
            self.companies[0], len(self.companies)
        )
        #: Linear category demand ends here: no customer pays this much.
        self.choke_price = seed.starting_price * (1.0 + 1.0 / seed.market_price_elasticity)
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
            company.receive_shocks(tick, self.active_market_shocks)
        inputs = self._reserve(c.attraction_inputs() for c in alive)
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
        # mean); category spend moves with it along the seed's linear demand curve.
        prices = [i[2] for i in inputs]
        price_index = 1.0 / sum(s / p for s, p in zip(shares, prices, strict=True))
        self.daily_spend = dynamics.linear_demand_spend(
            self.tam / DAYS_PER_YEAR,
            price_index,
            self.reference_price,
            self.market_price_elasticity,
        ) * market_env.get("market_demand_mult", 1.0)
        # Elasticities are long-run, so use settled (target) shares throughout.
        settled_index = 1.0 / sum(t / p for t, p in zip(targets, prices, strict=True))
        # Category demand is linear, so its elasticity depends on the price level.
        category = dynamics.linear_demand_elasticity(
            settled_index, self.reference_price, self.market_price_elasticity
        )
        for company, share, target, price in zip(alive, shares, targets, prices, strict=True):
            company.target_share = target
            company.category_elasticity = category
            company.own_price_elasticity = (
                dynamics.own_price_elasticity(
                    self.share_price_elasticity,
                    target,
                    category,
                    target / price * settled_index,
                )
                if math.isfinite(category)
                else math.inf  # priced past the choke: nobody buys at all
            )
            company.operate(
                tick, self.daily_spend, share, self.active_market_shocks, self.choke_price
            )

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
        # A bankrupt company's customers choose again by today's offers (logit
        # IIA), not in proportion to whichever rivals happened to be big.
        freed = 0.0
        for company in self.companies:
            if not company.alive and company.customer_base:
                freed += company.customer_base
                company.customer_base = 0.0
        base = base + freed * target
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

    def _reserve(
        self, inputs: Iterable[tuple[float, float, float, float]]
    ) -> list[tuple[float, float, float, float]]:
        """Logit inputs with no pull for an offer priced at or past the choke price.

        The choke is the category's reservation price: nobody pays that much,
        whatever a rival charges, so such an offer wins no customers.
        """
        return [(q, m, p, w if p < self.choke_price else 0.0) for q, m, p, w in inputs]

    def best_response_price(self, company: Company) -> tuple[float, float] | None:
        """List price that maximizes the company's settled daily contribution.

        Rivals hold today's prices and offers; the company keeps its current
        nodes (capacity, production, costs) and shocks. Customer bases are
        taken as settled (logit target shares, floored like the engine's), so
        the gain is where customers end up, not this month's change. Searches
        the engine's price bounds below the choke price, so the answer is a
        price the engine accepts and customers can pay. Today's list price is
        kept when it earns within a relative BEST_RESPONSE_TOLERANCE of the
        best. Under a price cap every list price at or above it earns the same,
        so the answer aims at the best price for when the cap lifts.

        "Rivals hold" is the assumption that can fail: when every firm has far
        more demand than capacity, each one's answer is near the choke, and if
        all reprice there on the same day the market empties until they come
        back down (a Bertrand-Edgeworth cycle, real for capacity-bound rivals).
        Returns
        (price, settled daily contribution gain over today's list price), or
        None if the company is not trading.
        """
        alive = self.alive
        if company not in alive:
            return None
        idx = alive.index(company)
        rivals = [c.attraction_inputs() for c in alive]
        lo, hi = (company.seed.starting_price * f for f in PRICE_BOUNDS_FACTOR)
        hi = min(hi, self.choke_price * (1.0 - BEST_RESPONSE_CHOKE_MARGIN))
        # Today's nodes, not the morning cache: a decision may have changed them.
        mods = company.library.modifiers(company.spawned)

        def world(own_shocks: list[Shock]) -> Callable[[float], float]:
            """Settled daily contribution at a list price, under these company shocks."""
            env = environment([*self.active_market_shocks, *own_shocks])
            inputs = list(rivals)
            inputs[idx] = company.attraction_inputs(own_shocks)
            servable = min(company.capacity_and_production(env))
            throughput = env.get("inventory_throughput_mult", 1.0)
            net_share = mods.revenue_multiplier() * (1.0 - company._marketing_share())
            unit_cost = company.seed.base_unit_cost * env.get("supply_cost_mult", 1.0)

            def contribution(list_price: float) -> float:
                price = company.capped_price(list_price, env)
                rows = self._reserve(
                    (q, m, price if i == idx else p, w) for i, (q, m, p, w) in enumerate(inputs)
                )
                shares = dynamics.logit_shares(
                    [r[0] for r in rows],
                    [r[1] for r in rows],
                    [r[2] for r in rows],
                    [r[3] for r in rows],
                    quality_exponent=SHARE_QUALITY_EXPONENT,
                    marketing_exponent=SHARE_MARKETING_EXPONENT,
                    price_elasticity=self.share_price_elasticity,
                )
                # The engine never lets a customer base fall below its floor.
                floored = [max(s, CUSTOMER_BASE_FLOOR) for s in shares]
                shares = [s / sum(floored) for s in floored]
                index = 1.0 / sum(s / r[2] for s, r in zip(shares, rows, strict=True))
                spend = dynamics.linear_demand_spend(
                    self.tam / DAYS_PER_YEAR,
                    index,
                    self.reference_price,
                    self.market_price_elasticity,
                ) * env.get("market_demand_mult", 1.0)
                served = min(spend * shares[idx] / price * throughput, servable)
                return served * (price * net_share - unit_cost)

            return contribution

        def search(contribution: Callable[[float], float]) -> float:
            grid = np.geomspace(lo, hi, BEST_RESPONSE_GRID_POINTS)
            values = [contribution(float(p)) for p in grid]
            best = int(np.argmax(values))
            a, b = float(grid[max(best - 1, 0)]), float(grid[min(best + 1, len(grid) - 1)])
            ratio = (math.sqrt(5.0) - 1.0) / 2.0
            for _ in range(BEST_RESPONSE_REFINE_STEPS):
                left, right = b - ratio * (b - a), a + ratio * (b - a)
                if contribution(left) >= contribution(right):
                    b = right
                else:
                    a = left
            refined = (a + b) / 2.0
            return refined if contribution(refined) >= values[best] else float(grid[best])

        def close(x: float, best: float) -> bool:
            return x >= best - BEST_RESPONSE_TOLERANCE * abs(best)

        now = world(company.shocks)
        price = search(now)
        current = now(company.price)
        if company.capped_price(hi, company.env) < hi:
            # Customers pay the cap whatever the list price above it. Aim the
            # list price at the best one for when the capping shock is over.
            uncapped = [s for s in company.shocks if "price_ceiling_mult" not in s.impact]
            after = search(world(uncapped))
            if close(now(after), now(price)):
                return after, now(after) - current
        if close(current, now(price)):
            return company.price, 0.0  # flat curve: no pointless move
        return price, now(price) - current

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
        ai_live = self.llm is not None and self.llm.available
        monthly = ai_live and tick % AI_CADENCE_DAYS == 0 and self._ai_on_pace(tick)
        due = []
        for company in companies:
            own_severe = self._new_severe(company.company_id, company.shocks)
            if (
                ai_live
                and company.company_id in self.ai
                and (monthly or market_severe or own_severe)
            ):
                due.append(company)
        if due:
            results = await asyncio.gather(
                *(
                    self.ai[c.company_id].decide(
                        c,
                        tick,
                        self.companies,
                        self.tam,
                        self.active_market_shocks,
                        self.best_response_price(c),
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


def calibrated_share_elasticity(company: Company, companies: int) -> float:
    """Logit price elasticity at which the seed's starting price is the Nash price.

    All companies start from the same seed and nodes with equal shares, so the
    observed starting contribution margin (revenue-stream uplift and marketing
    included) pins down how price-sensitive switching must be for that price
    to be an equilibrium while demand binds (`dynamics.nash_share_elasticity`).
    Undercutting then pays only as far as the margin the market supports.

    Two cases fall outside that claim, by design: a company that starts
    capacity-bound gains by raising price until settled demand meets capacity
    (a real excess-demand signal), and a seed clamped to MIN or
    MAX_SHARE_PRICE_ELASTICITY (see those constants) starts near Nash, not at it.
    """
    if companies < 2:
        return PRICE_ELASTICITY
    margin = company.contribution_margin()
    if margin <= 0:
        return MAX_SHARE_PRICE_ELASTICITY  # no markup at all: perfect competition
    elasticity = dynamics.nash_share_elasticity(
        margin, 1.0 / companies, company.seed.market_price_elasticity
    )
    return min(MAX_SHARE_PRICE_ELASTICITY, max(MIN_SHARE_PRICE_ELASTICITY, elasticity))


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
