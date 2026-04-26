"""V2 unified simulation engine — seed/stance/library/orchestrator-driven.

Replaces the v1 `unified.py` `CompanyAgent` + `UnifiedEngine` pair. The v2
engine is industry-agnostic: it consumes a `CompanySeed` (~31 fields), a
locked `CeoStance`, a universal `NodeLibrary`, and an `OrchestratorBundle`
(heuristic + Haiku tactical + Sonnet strategic) that decides what nodes to
spawn and retire each tick.

Key differences from v1:

  * No `IndustrySpec`. No per-industry YAML triggers/conditions DSL.
  * Spawned nodes tracked as `dict[node_key, count]`, not as a SimNode graph.
  * Bridge modifiers auto-derived via `bridge.derive_bridge_aggregate(...)`
    every tick from the actual spawned set — no static modifier-key
    declarations, eliminating the empty-modifier-key bug class by
    construction (V2 plan §6).
  * Single-source-of-truth shock injection through `ShockScheduler`.
  * Revenue/cost math is analytic and library-driven (no `LocationArrays`
    SoA layer in this first pass — can be re-introduced for perf later).

This file is intentionally smaller than v1 — v1's caching infrastructure,
volume-discount machinery, and IndustrySpec coupling have been removed
rather than ported. v2 sims are designed to grow back any optimization that
benchmarks demand, but the baseline is "obvious math, derive everything."

Live in parallel with `unified.py` until wave 5 deletes the legacy file.
"""

from __future__ import annotations

import logging
import math
import random as _random_module
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.simulation.bridge import BridgeAggregate, derive_bridge_aggregate
from src.simulation.library_loader import NodeLibrary
from src.simulation.orchestrator import (
    CeoDecision,
    CompanyState,
    HeuristicOrchestrator,
    OrchestratorBundle,
    StrategicOrchestrator,
    TacticalOrchestrator,
)
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import CompanySeed
from src.simulation.shocks import Shock, ShockScheduler, apply_active_shocks
from src.simulation.stance import CeoStance

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: How many tick-snapshots each rolling history carries. The orchestrator
#: spec calls for "last 10 ticks" of state context; we keep 100 to give the
#: critic agent and the CEO history window headroom.
HISTORY_LEN: int = 100

#: Days per month (consistent with v1 and `orchestrator.DAYS_PER_MONTH`).
DAYS_PER_MONTH: float = 30.0

#: Target satisfaction convergence per tick toward the steady state. Linear
#: convergence rather than full ODE — "obvious math" baseline; biosim's
#: RK45 logistic ODE replaces this in Phase 3.
SATISFACTION_CONVERGENCE_RATE: float = 0.1

#: Satisfaction floor / ceiling enforcement.
SATISFACTION_MIN: float = 0.0
SATISFACTION_MAX: float = 1.0

#: Default capacity per location node when a node has no explicit
#: `capacity_contribution`. The seed's `base_capacity_per_location` is the
#: per-location seat/subscriber count; we multiply it by the count of nodes
#: in the `location` category to get total capacity.
LOCATION_CAPACITY_MULTIPLIER_DEFAULT: float = 1.0

#: Base satisfaction multiplier — mapped from `BridgeAggregate.quality`.
#: A unit increase in the quality bucket bumps satisfaction by this much.
QUALITY_TO_SATISFACTION_GAIN: float = 0.5

#: Marketing pressure scaling — mapped from `BridgeAggregate.marketing`.
#: Used as a multiplier on customer acquisition rate.
MARKETING_TO_DEMAND_GAIN: float = 0.25

#: Infrastructure efficiency — mapped from `BridgeAggregate.infrastructure`.
#: Used as a multiplier on effective capacity.
INFRA_TO_CAPACITY_GAIN: float = 0.20

#: Saturation constant for infra → capacity gain. Without it,
#: stacking infra nodes produces an unbounded multiplier (P-MAG-5). The
#: saturation curve is `1 + INFRA_TO_CAPACITY_GAIN * tanh(sum / K)` so
#: the cap-out value is `1 + INFRA_TO_CAPACITY_GAIN`.
INFRA_SATURATION_K: float = 5.0

#: Solo-mode demand fraction: in absence of multi-company allocation,
#: a solo company captures this fraction of (TAM / competitor_density+1)
#: at full satisfaction. The previous `/100.0` literal was a bug that
#: suppressed solo demand by 100x (P-MAG-1).
SOLO_DEMAND_SATURATION: float = 0.05

#: Multi-tick insolvency window: a company stays alive while cash is
#: negative for at most this many *consecutive* ticks, then dies.
#: Replaces v1's single-tick check (P-ENG-2). Aligned with CLAUDE.md's
#: multi-tick insolvency mandate.
INSOLVENT_TICKS_TO_DEATH: int = 30

#: When `key_employee_departure` shock fires, randomly retire one
#: non-founder/non-exec node (the `retire_random_employee` signal).
#: This constant gates the cooldown so the same shock doesn't keep firing.
ATTRITION_BASE_RATE: float = 0.0001

#: Per-replenish_supplier signal, charge this many days of supplier
#: fixed cost as a one-shot cash debit (inventory restock).
REPLENISH_INVENTORY_DAYS: float = 7.0

#: Bounds on price adjustments the CEO may apply via `adjust_params`.
#: Prices outside `[seed.starting_price * MIN, seed.starting_price * MAX]`
#: are rejected (logged + ignored). Phase 2 will move tighter validation
#: into the orchestrator; this is the engine-side last-line check.
PRICE_ADJUST_MIN_FACTOR: float = 0.1
PRICE_ADJUST_MAX_FACTOR: float = 10.0

#: Bounds on marketing intensity multiplier. `marketing_intensity` from
#: the CEO is clamped into this band before it hits the math.
MARKETING_INTENSITY_MIN: float = 0.0
MARKETING_INTENSITY_MAX: float = 2.0

#: Stance archetypes that may apply `raise_amount` (capital raise).
#: Bootstrap, founder_operator, and turnaround archetypes do not access
#: external capital markets; venture_growth and consolidator can raise.
RAISE_ALLOWED_ARCHETYPES: frozenset[str] = frozenset({"venture_growth", "consolidator"})

# ─── Bridge bucket gains (Phase 1.4 — P-ENG-6) ─────────────────────────
#
# `bridge.cost`, `bridge.revenue`, `bridge.capital` are signed sums of
# matching modifier_keys across spawned nodes. Author convention:
#   - cost bucket: negative value = savings, positive = added cost
#   - revenue bucket: positive value = revenue uplift fraction
#   - capital bucket: magnitude = daily access-to-capital signal
# These gain constants scale the bucket sum into a multiplier; the
# clamps below cap how much the bridge can move the math in one tick.

#: Per unit of summed `bridge.cost`, fraction added to daily_burn.
COST_BUCKET_GAIN: float = 1.0
#: Min/max clamp on the cost-side multiplier (1.0 + GAIN * sum).
COST_BUCKET_MULT_MIN: float = 0.5
COST_BUCKET_MULT_MAX: float = 1.5

#: Per unit of summed `bridge.revenue`, fraction added to daily_revenue.
REVENUE_BUCKET_GAIN: float = 1.0
REVENUE_BUCKET_MULT_MIN: float = 0.5
REVENUE_BUCKET_MULT_MAX: float = 2.0

#: Per unit of summed |`bridge.capital`|, daily cash inflow if the
#: stance archetype is permitted to raise external capital. Models the
#: "always-on financing access" investor nodes provide rather than the
#: lumpy `raise_amount` tactical action.
CAPITAL_BUCKET_DAILY_GAIN: float = 25.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class TickResult:
    """Per-tick result emitted by `CompanyAgentV2.step()`.

    Carries the minimum the SSE stream and tests need to consume. A separate
    `to_graph_snapshot()` call produces the full graph payload when the
    frontend requests it.

    `critic_scores` carries `critic.CriticScore` objects produced by the
    optional `CriticAgent` (Phase 2.3). Empty list when no critic is
    wired or no strategic decision fired this tick. Frontend (Phase 5.3)
    surfaces these in the Critic Scores panel.
    """

    tick: int
    cash: float
    daily_revenue: float
    daily_costs: float
    monthly_revenue: float
    capacity_utilization: float
    avg_satisfaction: float
    employee_count: int
    spawned_nodes: dict[str, int]
    decisions: list[CeoDecision]
    arriving_shocks: list[Shock]
    active_shocks: list[Shock]
    bankrupt: bool
    critic_scores: list[Any] = field(default_factory=list)


@dataclass
class _Histories:
    """Bounded rolling histories for the orchestrator's state window."""

    revenue: deque[float] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    cash: deque[float] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    satisfaction: deque[float] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    capacity_utilization: deque[float] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    spawned_snapshots: deque[dict[str, int]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )
    decisions: deque[CeoDecision] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )


# ---------------------------------------------------------------------------
# CompanyAgentV2
# ---------------------------------------------------------------------------


class CompanyAgentV2:
    """Single-company simulation agent for v2.

    Owns:
      * The locked `seed` and `stance` (frozen for the run).
      * The `NodeLibrary` (read-only reference; loaded once per process).
      * The `OrchestratorBundle` (heuristic + tactical + strategic CEO tiers).
      * The `ShockScheduler` (per-sim Poisson; can be a no-shock scheduler).
      * Mutable simulation state: `tick`, `cash`, `spawned_nodes`, histories,
        active shocks, the running daily/monthly revenue/costs.

    The `step()` method runs one tick and returns a `TickResult`. Multi-
    company competition lives in `MultiCompanySimV2` (below).
    """

    def __init__(
        self,
        *,
        seed: CompanySeed,
        stance: CeoStance,
        library: NodeLibrary,
        sim_id: str,
        company_id: str,
        rng: _random_module.Random,
        transcript: Transcript | None = None,
        cost_tracker: CostTracker | None = None,
        shock_scheduler: ShockScheduler | None = None,
        orchestrator: OrchestratorBundle | None = None,
    ) -> None:
        self.seed = seed
        self.stance = stance
        self.library = library
        self.sim_id = sim_id
        self.company_id = company_id
        self.rng = rng
        self.transcript = transcript
        self.cost_tracker = cost_tracker or CostTracker(ceiling_usd=50.0)

        self.tick: int = 0
        self.cash: float = float(seed.starting_cash)
        self.alive: bool = True
        self.daily_revenue: float = 0.0
        self.daily_costs: float = 0.0
        self.monthly_revenue: float = 0.0
        self.satisfaction: float = 0.5
        self.capacity_utilization: float = 0.0
        self.allocated_demand: float = 0.0  # set by MultiCompanySimV2 each tick
        self.market_share: float = 0.0      # set by MultiCompanySimV2 each tick

        # Mutable runtime state — defaults from seed, mutated by CEO via
        # adjust_params (Decision 2A). The seed remains frozen=True; these
        # fields exist on the agent so the immutable-vs-runtime distinction
        # is visible at every read site.
        self.current_price: float = float(seed.starting_price)
        self.marketing_multiplier: float = 1.0

        # Multi-tick insolvency counter (P-ENG-2). Increments each tick
        # cash < 0; resets when cash recovers; death at INSOLVENT_TICKS_TO_DEATH.
        self.consecutive_insolvent: int = 0

        # One-time shock-cost ledger: each shock instance can charge a
        # one-shot cost (e.g. replacement_cost_add) exactly once, the
        # first tick we see it active. Identified by Python id() since
        # Shock is a pydantic model and not hashable by default.
        self._charged_one_time_shocks: set[int] = set()

        # Last-applied environment dict — needed by `_spawn_node` so it
        # can read `hire_cost_mult` without changing the helper signature.
        # Defaults to neutral (no shocks) until step() runs.
        self._last_env: dict[str, float] = {}

        # Spawned nodes: dict[node_key -> count]. Initialized from seed refs +
        # founder. Founder is a contractual must-have; library should declare it.
        self.spawned_nodes: dict[str, int] = {}
        self._initialize_spawned_nodes()

        # Histories
        self._hist = _Histories()

        # Shock plumbing — default to a scheduler with no arrivals if caller
        # didn't provide one. Tests pass a real scheduler to inject events.
        if shock_scheduler is None:
            shock_scheduler = ShockScheduler(
                rng_seed=rng.randint(0, 2**31 - 1), lambdas={}
            )
        self.shock_scheduler = shock_scheduler
        self.active_shocks: list[Shock] = []

        # Orchestrator — build the bundle if caller didn't pass one.
        if orchestrator is None:
            library_dict = self._library_as_dict()
            heuristic = HeuristicOrchestrator(
                seed=seed, stance=stance, library=library_dict
            )
            tactical = TacticalOrchestrator(
                seed=seed,
                stance=stance,
                library=library,
                transcript=self.transcript,
                cost_tracker=self.cost_tracker,
            )
            strategic = StrategicOrchestrator(
                seed=seed,
                stance=stance,
                library=library,
                transcript=self.transcript,
                cost_tracker=self.cost_tracker,
            )
            orchestrator = OrchestratorBundle(
                heuristic=heuristic,
                tactical=tactical,
                strategic=strategic,
            )
        self.orchestrator = orchestrator

        # Cross-validate seed refs against library. Hard-fail per V2 spec.
        library.validate_seed(seed)

    # ── Initialization helpers ──

    def _initialize_spawned_nodes(self) -> None:
        """Populate `spawned_nodes` from seed initial refs + founder.

        Each ref in `initial_supplier_types`, `initial_revenue_streams`,
        `initial_cost_centers` starts at count 1. The founder node is also
        spawned at t=0 if present in the library (looks for `founder` first,
        then any node with category `exec`).
        """
        for ref_list in (
            self.seed.initial_supplier_types,
            self.seed.initial_revenue_streams,
            self.seed.initial_cost_centers,
        ):
            for node_key in ref_list:
                self.spawned_nodes[node_key] = self.spawned_nodes.get(node_key, 0) + 1

        # Founder: spawn one if any founder-style node exists. Tries the
        # canonical names first, then falls back to any exec-category node
        # with no prerequisites.
        founder_key = self._first_founder_node()
        if founder_key is not None:
            self.spawned_nodes[founder_key] = self.spawned_nodes.get(founder_key, 0) + 1

        # First location is implicit in v2's economics — at least one location
        # node spawned to give the company a place to operate. If the seed
        # didn't declare it as initial_revenue_stream, spawn one matching
        # `economics_model` from the library.
        if not self._has_node_in_category("location"):
            loc_key = self._first_location_for_economics(self.seed.economics_model)
            if loc_key is not None:
                self.spawned_nodes[loc_key] = (
                    self.spawned_nodes.get(loc_key, 0) + 1
                )

    def _has_node_in_category(self, category: str) -> bool:
        for node_key, count in self.spawned_nodes.items():
            if count <= 0:
                continue
            node_def = self.library.nodes.get(node_key)
            if node_def and node_def.category == category:
                return True
        return False

    def _first_founder_node(self) -> str | None:
        """Return the canonical founder node_key, or fallback to first
        no-prereq exec-category node."""
        for canonical in ("founder", "founder_ceo", "ceo"):
            if canonical in self.library.nodes:
                return canonical
        for key, node in sorted(self.library.nodes.items()):
            if node.category == "exec" and not node.prerequisites:
                return key
        return None

    def _first_location_for_economics(self, economics_model: str) -> str | None:
        """Find the first library node in `location` category that supports
        the given economics model. Returns None if none exists."""
        for node_key, node_def in sorted(self.library.nodes.items()):
            if node_def.category != "location":
                continue
            if economics_model in node_def.applicable_economics:
                return node_key
        return None

    def _library_as_dict(self) -> dict[str, dict]:
        """Convert NodeLibrary to the dict-of-dicts shape HeuristicOrchestrator
        expects (it deliberately doesn't import library_loader)."""
        return {
            key: node.model_dump()
            for key, node in self.library.nodes.items()
        }

    # ── Tick loop ──

    async def step(self) -> TickResult:
        """Advance the simulation by one tick.

        Sequence:
          1. Increment tick counter.
          2. Pull shock arrivals and prune expired ones.
          3. Compute current capacity, daily burn, modifier aggregates.
          4. Apply shocks to a base environment dict.
          5. Compute revenue / costs / satisfaction for this tick.
          6. Update cash.
          7. Build `CompanyState` and call orchestrator.
          8. Apply orchestrator decisions (spawn / retire / adjust_params).
          9. Record histories.
         10. Return `TickResult`.
        """
        if not self.alive:
            return self._build_dead_tick_result()

        self.tick += 1

        # 2. Shocks
        arrivals = self.shock_scheduler.tick(self.tick)
        for shock in arrivals:
            shock.tick_started = self.tick
            self.active_shocks.append(shock)
        # Prune expired shocks (those whose duration_ticks have passed).
        self.active_shocks = [
            s for s in self.active_shocks
            if s.tick_started is not None
            and (self.tick - s.tick_started) < s.duration_ticks
        ]

        # 3. Modifier aggregation from spawned nodes.
        bridge_agg = derive_bridge_aggregate(self.library, self.spawned_nodes)

        # Capacity from spawned location/capacity_contribution nodes
        location_count = sum(
            count
            for key, count in self.spawned_nodes.items()
            if key in self.library.nodes
            and self.library.nodes[key].category == "location"
        )
        # Effective capacity = base * location count, with saturating infra
        # multiplier (P-MAG-5). `tanh` keeps the multiplier in
        # [1.0, 1.0 + INFRA_TO_CAPACITY_GAIN] so stacking dozens of infra
        # nodes can no longer produce an unbounded capacity inflation.
        infra_sum = sum(bridge_agg.infrastructure.values())
        infra_mult = 1.0 + INFRA_TO_CAPACITY_GAIN * math.tanh(
            infra_sum / INFRA_SATURATION_K
        )
        total_capacity = (
            float(self.seed.base_capacity_per_location)
            * max(1, location_count)
            * infra_mult
        )

        # Daily burn from all spawned nodes' daily_fixed_costs + seed base.
        # Then apply `bridge.cost` bucket as a signed multiplier
        # (Phase 1.4 — P-ENG-6). Per author convention, negative bucket
        # values are savings (lower burn); positive add cost.
        daily_burn = self.seed.daily_fixed_costs
        for key, count in self.spawned_nodes.items():
            node = self.library.nodes.get(key)
            if node is None:
                continue
            daily_burn += node.daily_fixed_costs * count
        cost_bucket_sum = sum(bridge_agg.cost.values())
        cost_mult = max(
            COST_BUCKET_MULT_MIN,
            min(COST_BUCKET_MULT_MAX, 1.0 + COST_BUCKET_GAIN * cost_bucket_sum),
        )
        daily_burn *= cost_mult

        # 4. Apply shocks to environment. Every shock impact key declared
        #    by any factory in `shocks.py` is now consumed by the math
        #    below (P-ENG-5). `apply_active_shocks` seeds neutral defaults
        #    (1.0 for *_mult, 0.0 for additive) for any key referenced.
        base_env: dict[str, float] = {
            # Multiplicative impacts
            "market_demand_mult": 1.0,
            "tam_mult": 1.0,
            "financing_availability_mult": 1.0,
            "competitor_pressure_mult": 1.0,
            "market_share_mult": 1.0,
            "price_ceiling_mult": 1.0,
            "team_capacity_mult": 1.0,
            "satisfaction_mult": 1.0,
            "supply_cost_mult": 1.0,
            "lead_time_mult": 1.0,
            "inventory_throughput_mult": 1.0,
            "fixed_cost_mult": 1.0,
            "capacity_cap_mult": 1.0,
            "acquisition_cost_mult": 1.0,
            "consumer_spending_mult": 1.0,
            "hire_cost_mult": 1.0,
            "wage_inflation_mult": 1.0,
            "attrition_mult": 1.0,
            # Additive impacts
            "compliance_cost_add": 0.0,
            "replacement_cost_add": 0.0,
        }
        env = apply_active_shocks(self.active_shocks, base_env)
        self._last_env = env  # exposed to _spawn_node for hire_cost_mult

        # 4a. One-time shock effects — fire exactly once per shock arrival.
        #     `replacement_cost_add` is a one-shot cash debit (e.g. paying a
        #     recruiter to backfill a key departure). `retire_random_employee`
        #     is a signal that triggers an actual node retirement.
        for shock in self.active_shocks:
            shock_id = id(shock)
            if shock_id in self._charged_one_time_shocks:
                continue
            self._charged_one_time_shocks.add(shock_id)
            if "replacement_cost_add" in shock.impact:
                self.cash -= shock.impact["replacement_cost_add"]
            if shock.impact.get("retire_random_employee", 0.0) > 0:
                self._retire_random_employee_node()

        # 4b. Continuous additive cost-side shocks (compliance) — applied
        #     each tick the shock is active.
        daily_burn += float(env.get("compliance_cost_add", 0.0))
        # Wage inflation — multiplicative on burn, applied alongside fixed_cost_mult.
        daily_burn *= float(env.get("wage_inflation_mult", 1.0))

        # 5. Satisfaction — converge linearly toward (quality-driven) target,
        #    then attenuate by satisfaction_mult shock (e.g. key departure).
        quality_boost = QUALITY_TO_SATISFACTION_GAIN * sum(
            bridge_agg.quality.values()
        )
        target_satisfaction = min(
            SATISFACTION_MAX,
            max(SATISFACTION_MIN, 0.5 + quality_boost),
        )
        self.satisfaction += (
            target_satisfaction - self.satisfaction
        ) * SATISFACTION_CONVERGENCE_RATE
        self.satisfaction *= float(env.get("satisfaction_mult", 1.0))
        self.satisfaction = max(SATISFACTION_MIN, min(SATISFACTION_MAX, self.satisfaction))

        # 5a. Marketing pressure: bridge.marketing bucket × CEO multiplier.
        #     `acquisition_cost_mult < 1.0` means cheaper CAC, so the same
        #     marketing budget produces more pressure: divide by the mult.
        marketing_pressure = 1.0 + MARKETING_TO_DEMAND_GAIN * sum(
            bridge_agg.marketing.values()
        )
        marketing_pressure *= self.marketing_multiplier
        acq_cost_mult = float(env.get("acquisition_cost_mult", 1.0))
        if acq_cost_mult > 0:
            marketing_pressure /= acq_cost_mult

        # 5b. Demand: market-allocated (multi-company) or solo-mode fallback.
        #     Layer in shock impacts: tam, demand, consumer spending,
        #     lead-time / throughput friction (supply chain).
        demand_mult = (
            float(env.get("market_demand_mult", 1.0))
            * float(env.get("tam_mult", 1.0))
            * float(env.get("consumer_spending_mult", 1.0))
        )
        throughput_mult = (
            float(env.get("inventory_throughput_mult", 1.0))
            / max(0.01, float(env.get("lead_time_mult", 1.0)))
        )
        if self.allocated_demand > 0:
            effective_demand = self.allocated_demand * marketing_pressure * demand_mult
        else:
            # Solo-mode: TAM / (competitor_density + 1), saturated by
            # SOLO_DEMAND_SATURATION (P-MAG-1 fix — was bugged `/100.0`).
            base = self.seed.tam / max(1, self.seed.competitor_density + 1)
            effective_demand = (
                base * marketing_pressure * demand_mult * self.satisfaction
                * SOLO_DEMAND_SATURATION
            )
        effective_demand *= throughput_mult

        # 5c. Apply capacity reductions (key-departure team_capacity, regulatory cap).
        capacity_after_shock = total_capacity * (
            float(env.get("team_capacity_mult", 1.0))
            * float(env.get("capacity_cap_mult", 1.0))
        )

        # Served customers bounded by post-shock capacity
        served = min(effective_demand, capacity_after_shock)
        self.capacity_utilization = (
            served / capacity_after_shock if capacity_after_shock > 0 else 0.0
        )

        # 6. Revenue = served * effective_price. Effective price respects the
        #    competitor-entry price ceiling (price_ceiling_mult applies to the
        #    seed's starting_price as the reference list price).
        #    Cost side: served * base_unit_cost + daily_burn (scaled by shocks).
        supply_cost_mult = float(env.get("supply_cost_mult", 1.0))
        fixed_cost_mult = float(env.get("fixed_cost_mult", 1.0))
        price_ceiling_mult = float(env.get("price_ceiling_mult", 1.0))
        # Revenue uplift from `bridge.revenue` bucket — signed sum scaled
        # into a multiplier on top of base served*price.
        revenue_bucket_sum = sum(bridge_agg.revenue.values())
        revenue_uplift = max(
            REVENUE_BUCKET_MULT_MIN,
            min(
                REVENUE_BUCKET_MULT_MAX,
                1.0 + REVENUE_BUCKET_GAIN * revenue_bucket_sum,
            ),
        )
        # `price_ceiling_mult` only caps when an active shock asserts a
        # ceiling (mult < 1.0). When neutral (1.0) or above, the CEO's
        # `current_price` flows through unchanged.
        if price_ceiling_mult < 1.0:
            effective_price = min(
                self.current_price,
                self.seed.starting_price * price_ceiling_mult,
            )
        else:
            effective_price = self.current_price
        self.daily_revenue = served * effective_price * revenue_uplift
        self.daily_costs = (
            served * self.seed.base_unit_cost * supply_cost_mult
            + daily_burn * fixed_cost_mult
        )

        # 6a. Capital availability (`bridge.capital` bucket) — investor /
        #     funding nodes provide standing access to capital. Magnitude
        #     of the signed bucket sum becomes a small daily cash inflow,
        #     gated on whether the stance archetype raises external capital
        #     at all. Bootstrap and turnaround stances ignore this signal.
        #
        #     `financing_availability_mult` (market_crash, recession) scales
        #     the standing inflow — when credit dries up, capital nodes still
        #     exist but their daily access shrinks. Clamped at zero so a
        #     deeply negative shock can't invert the sign.
        capital_signal = abs(sum(bridge_agg.capital.values()))
        financing_mult = max(0.0, float(env.get("financing_availability_mult", 1.0)))
        if (
            capital_signal > 0
            and self.stance.archetype in RAISE_ALLOWED_ARCHETYPES
        ):
            self.cash += capital_signal * CAPITAL_BUCKET_DAILY_GAIN * financing_mult

        # 7. Update cash
        self.cash += self.daily_revenue - self.daily_costs
        self.monthly_revenue = self.daily_revenue * DAYS_PER_MONTH

        # 7a. Multi-tick insolvency check (P-ENG-2 + CLAUDE.md mandate).
        #     Single-tick negative cash no longer kills — companies survive
        #     short cash dips. Death only fires after INSOLVENT_TICKS_TO_DEATH
        #     consecutive negative-cash ticks.
        if self.cash < 0:
            self.consecutive_insolvent += 1
        else:
            self.consecutive_insolvent = 0
        if self.consecutive_insolvent >= INSOLVENT_TICKS_TO_DEATH:
            self.alive = False
            return self._build_dead_tick_result()

        # 7b. Probabilistic attrition — talent_war elevates attrition_mult.
        attrition_mult = float(env.get("attrition_mult", 1.0))
        if attrition_mult > 1.0:
            p_retire = ATTRITION_BASE_RATE * attrition_mult
            if self.rng.random() < p_retire:
                self._retire_random_employee_node()

        # 7. Build CompanyState for orchestrator
        employee_count = self._compute_employee_count()
        state = CompanyState(
            tick=self.tick,
            cash=self.cash,
            daily_burn=daily_burn,
            monthly_revenue=self.monthly_revenue,
            spawned_nodes=dict(self.spawned_nodes),
            capacity_utilization=min(2.0, self.capacity_utilization),
            avg_satisfaction=self.satisfaction,
            employee_count=employee_count,
            active_shocks=list(self.active_shocks),
            recent_decisions=list(self._hist.decisions)[-3:],
        )

        # 8. Orchestrator
        decisions, critic_scores = await self.orchestrator.tick(state)
        for decision in decisions:
            self._apply_decision(decision)
            self._hist.decisions.append(decision)

        # 9. Histories
        self._hist.revenue.append(self.daily_revenue)
        self._hist.cash.append(self.cash)
        self._hist.satisfaction.append(self.satisfaction)
        self._hist.capacity_utilization.append(self.capacity_utilization)
        self._hist.spawned_snapshots.append(dict(self.spawned_nodes))

        # 10. Result
        return TickResult(
            tick=self.tick,
            cash=self.cash,
            daily_revenue=self.daily_revenue,
            daily_costs=self.daily_costs,
            monthly_revenue=self.monthly_revenue,
            capacity_utilization=self.capacity_utilization,
            avg_satisfaction=self.satisfaction,
            employee_count=employee_count,
            spawned_nodes=dict(self.spawned_nodes),
            decisions=list(decisions),
            critic_scores=list(critic_scores),
            arriving_shocks=list(arrivals),
            active_shocks=list(self.active_shocks),
            bankrupt=False,
        )

    def _build_dead_tick_result(self) -> TickResult:
        return TickResult(
            tick=self.tick,
            cash=self.cash,
            daily_revenue=0.0,
            daily_costs=0.0,
            monthly_revenue=0.0,
            capacity_utilization=0.0,
            avg_satisfaction=0.0,
            employee_count=0,
            spawned_nodes=dict(self.spawned_nodes),
            decisions=[],
            arriving_shocks=[],
            active_shocks=list(self.active_shocks),
            bankrupt=True,
        )

    def _compute_employee_count(self) -> int:
        total = self.seed.starting_employees
        for key, count in self.spawned_nodes.items():
            node = self.library.nodes.get(key)
            if node:
                total += node.employees_count * count
        return int(total)

    # ── Decision application ──

    def _apply_decision(self, decision: CeoDecision) -> None:
        """Apply a CeoDecision to internal state.

        Defensive: orchestrator should have already filtered out unmet
        prerequisites and hard-cap breaches, but we re-check here so a buggy
        orchestrator can't corrupt simulation state silently.
        """
        # Spawn
        for node_key in decision.spawn_nodes:
            self._spawn_node(node_key)

        # Retire
        for node_key in decision.retire_nodes:
            self._retire_node(node_key)

        # adjust_params — wired into math (P-ENG-1, Decision 2A). Each key
        # is bounds-checked at the engine boundary so an invalid LLM emission
        # is logged and ignored rather than corrupting state silently.
        if decision.adjust_params:
            self._apply_param_adjustments(decision.adjust_params)

        # open_locations: spawn the seed's preferred location node N times.
        if decision.open_locations > 0:
            loc_key = self._first_location_for_economics(self.seed.economics_model)
            if loc_key is not None:
                for _ in range(decision.open_locations):
                    self._spawn_node(loc_key)

    def _apply_param_adjustments(self, params: dict[str, float]) -> None:
        """Apply CEO `adjust_params` to runtime state.

        Recognized keys:
          - `price`: new value for `self.current_price`. Bounded by
            `[seed.starting_price * PRICE_ADJUST_MIN_FACTOR,
              seed.starting_price * PRICE_ADJUST_MAX_FACTOR]`.
          - `marketing_intensity`: new value for `self.marketing_multiplier`,
            clamped to `[MARKETING_INTENSITY_MIN, MARKETING_INTENSITY_MAX]`.
          - `raise_amount`: cash injection. Only takes effect when the
            stance archetype is in `RAISE_ALLOWED_ARCHETYPES`. Negative
            raises are rejected. Phase 2 will tighten validation in the
            orchestrator; this is the engine-side last-line check.
          - `replenish_supplier`: signal (any positive value) — incurs
            REPLENISH_INVENTORY_DAYS days of supplier-node fixed costs as
            a one-shot cash debit. Models routine inventory restocking.

        Unknown keys are logged and ignored. The engine refuses to silently
        accept inputs it does not understand.
        """
        for key, raw_value in params.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                log.warning(
                    "Tick %d: adjust_params[%s]=%r not coercible to float; ignoring",
                    self.tick, key, raw_value,
                )
                continue

            if key == "price":
                lo = self.seed.starting_price * PRICE_ADJUST_MIN_FACTOR
                hi = self.seed.starting_price * PRICE_ADJUST_MAX_FACTOR
                if value <= 0 or value < lo or value > hi:
                    log.warning(
                        "Tick %d: rejected price=%.4f (allowed [%.4f, %.4f])",
                        self.tick, value, lo, hi,
                    )
                    continue
                self.current_price = value

            elif key == "marketing_intensity":
                self.marketing_multiplier = max(
                    MARKETING_INTENSITY_MIN,
                    min(MARKETING_INTENSITY_MAX, value),
                )

            elif key == "raise_amount":
                if value < 0:
                    log.warning(
                        "Tick %d: rejected raise_amount=%.2f (must be ≥ 0)",
                        self.tick, value,
                    )
                    continue
                if self.stance.archetype not in RAISE_ALLOWED_ARCHETYPES:
                    log.info(
                        "Tick %d: ignored raise_amount=%.2f — stance %r does "
                        "not permit external capital raises",
                        self.tick, value, self.stance.archetype,
                    )
                    continue
                # Active credit-tightening shock haircuts the actual inflow.
                # `financing_availability_mult < 1.0` during market_crash /
                # recession means investors close their wallets — the CEO
                # may still try to raise, but receives less.
                financing_mult = max(
                    0.0, float(self._last_env.get("financing_availability_mult", 1.0))
                )
                self.cash += value * financing_mult

            elif key == "replenish_supplier":
                if value <= 0:
                    continue
                self._replenish_supplier_inventory()

            else:
                log.warning(
                    "Tick %d: unknown adjust_params key %r=%r; ignoring",
                    self.tick, key, value,
                )

    def _replenish_supplier_inventory(self) -> None:
        """Charge a one-shot cash debit equal to REPLENISH_INVENTORY_DAYS
        days of supplier-node fixed costs. Models routine restocking.
        """
        supplier_burn = 0.0
        for node_key, count in self.spawned_nodes.items():
            node = self.library.nodes.get(node_key)
            if node is None or node.category != "supplier":
                continue
            supplier_burn += node.daily_fixed_costs * count
        if supplier_burn > 0:
            self.cash -= supplier_burn * REPLENISH_INVENTORY_DAYS

    def _retire_random_employee_node(self) -> None:
        """Retire one random non-founder/non-exec spawned node.

        Triggered by `key_employee_departure` shock's `retire_random_employee`
        signal and by probabilistic attrition under elevated `attrition_mult`
        (talent_war shock). Excludes the founder so the company doesn't
        lose its one mandatory exec.
        """
        candidates: list[str] = []
        for node_key, count in self.spawned_nodes.items():
            if count <= 0:
                continue
            node = self.library.nodes.get(node_key)
            if node is None:
                continue
            if node.category == "exec":
                continue  # never retire the founder via attrition
            if node.employees_count <= 0:
                continue  # only employee-bearing nodes can attrit
            candidates.append(node_key)
        if not candidates:
            return
        victim = self.rng.choice(candidates)
        self._retire_node(victim)

    def _spawn_node(self, node_key: str) -> None:
        node = self.library.nodes.get(node_key)
        if node is None:
            log.warning(
                "Tick %d: orchestrator tried to spawn unknown node_key=%s "
                "(filtered, skipping)", self.tick, node_key,
            )
            return

        # Prerequisite check
        for prereq in node.prerequisites:
            if self.spawned_nodes.get(prereq, 0) <= 0:
                log.warning(
                    "Tick %d: spawn %s blocked — missing prerequisite %s",
                    self.tick, node_key, prereq,
                )
                return

        # Hard-cap check
        current = self.spawned_nodes.get(node_key, 0)
        if current + 1 > node.category_caps.hard_cap:
            log.warning(
                "Tick %d: spawn %s blocked — hard_cap %d already at %d",
                self.tick, node_key, node.category_caps.hard_cap, current,
            )
            return

        # Pay hire cost — scaled by talent_war's `hire_cost_mult` shock.
        hire_cost_mult = float(self._last_env.get("hire_cost_mult", 1.0))
        self.cash -= node.hire_cost * hire_cost_mult
        self.spawned_nodes[node_key] = current + 1

    def _retire_node(self, node_key: str) -> None:
        current = self.spawned_nodes.get(node_key, 0)
        if current <= 0:
            log.warning(
                "Tick %d: retire %s blocked — no instances spawned",
                self.tick, node_key,
            )
            return
        self.spawned_nodes[node_key] = current - 1
        if self.spawned_nodes[node_key] == 0:
            del self.spawned_nodes[node_key]

    # ── Snapshot / introspection ──

    def to_graph_snapshot(self) -> dict[str, Any]:
        """Frontend-compatible graph snapshot.

        Output shape matches v1's `_build_result` graphs payload — one node
        entry per spawned node (deduplicated by node_key with `count`
        property). Edges derived from prerequisite chain.
        """
        nodes_out: list[dict[str, Any]] = []
        for node_key, count in sorted(self.spawned_nodes.items()):
            if count <= 0:
                continue
            node_def = self.library.nodes.get(node_key)
            if node_def is None:
                continue
            nodes_out.append({
                "id": node_key,
                "type": node_key,
                "label": f"{node_def.label} (×{count})" if count > 1 else node_def.label,
                "category": node_def.category,
                "spawned_at": 0,  # v2 doesn't track per-instance spawn ticks
                "metrics": {"count": float(count)},
            })

        edges_out: list[dict[str, str]] = []
        for node_key in self.spawned_nodes:
            node_def = self.library.nodes.get(node_key)
            if node_def is None:
                continue
            for prereq in node_def.prerequisites:
                if prereq in self.spawned_nodes:
                    edges_out.append({
                        "source": node_key,
                        "target": prereq,
                        "relationship": "depends_on",
                    })

        return {
            "nodes": nodes_out,
            "edges": edges_out,
        }


# ---------------------------------------------------------------------------
# MultiCompanySimV2
# ---------------------------------------------------------------------------


#: Default daily TAM growth rate. Equivalent to ~3.7%/year at 365 ticks.
#: Per-sim values come from the seed (`market_growth_rate` field) where
#: present; this constant is the floor when no seed value is given.
DEFAULT_MARKET_GROWTH_RATE: float = 0.0001

#: Default multinomial-logit exponents on the share-attraction formula
#: `s_i ∝ q_i^β * m_i^α`. v2 should derive these from a `MarketParams`
#: model alongside the seed; for now these are owner-tunable defaults
#: pinned to industry-agnostic values per CLAUDE.md.
DEFAULT_SHARE_ALPHA: float = 1.0  # marketing exponent
DEFAULT_SHARE_BETA: float = 1.5   # quality exponent

#: Per-company shock multipliers that affect cross-company share
#: allocation. Computed in `MultiCompanySimV2.step` so that one
#: company's `new_competitor_entry` shock pulls share away from itself.
COMPETITOR_PRESSURE_NEUTRAL: float = 1.0
MARKET_SHARE_NEUTRAL: float = 1.0


class MultiCompanySimV2:
    """V2 multi-company simulation runner with shared market.

    Holds a list of `CompanyAgentV2` instances and ticks them in lockstep.
    Each tick:
      1. Update TAM (linear growth).
      2. Compute each company's quality/marketing attractiveness, modulated
         by per-company shock impacts (`competitor_pressure_mult`,
         `market_share_mult`).
      3. Compute market shares via multinomial logit.
      4. Allocate revenue ceiling per company.
      5. Set each company's `allocated_demand`; tick them.
      6. Death check (already handled inside `CompanyAgentV2.step`).
      7. Emit `all_dead=True` exactly once when the last company dies.
    """

    def __init__(
        self,
        *,
        sim_id: str,
        companies: list[CompanyAgentV2],
        max_ticks: int,
        tam_initial: float,
        market_growth_rate: float = DEFAULT_MARKET_GROWTH_RATE,
        share_alpha: float = DEFAULT_SHARE_ALPHA,
        share_beta: float = DEFAULT_SHARE_BETA,
    ) -> None:
        self.sim_id = sim_id
        self.companies = companies
        self.max_ticks = max_ticks
        self.tam = tam_initial
        self.market_growth_rate = market_growth_rate
        self.share_alpha = share_alpha
        self.share_beta = share_beta
        self.tick: int = 0
        # `is_complete` semantics (P-ENG-4): when all companies die before
        # `max_ticks`, the next call to `step` emits an `all_dead` payload
        # and `is_complete` flips True. Tracks whether we already emitted.
        self._all_dead_emitted: bool = False

    @property
    def alive_companies(self) -> list[CompanyAgentV2]:
        return [c for c in self.companies if c.alive]

    @property
    def is_complete(self) -> bool:
        if self.tick >= self.max_ticks:
            return True
        # Sim ends when there are no living companies AND we have already
        # emitted the closing `all_dead` payload (so the SSE stream gets a
        # graceful end-state event before the loop exits).
        return not self.alive_companies and self._all_dead_emitted

    async def step(self) -> dict[str, Any]:
        """Run one shared-market tick across all companies.

        Returns one of three payload shapes:
          * normal tick — `{tick, tam, alive, results, shares}`
          * all-dead closing tick (emitted exactly once) —
            `{tick, tam, alive: 0, results: [], all_dead: True}`
          * subsequent calls after all-dead — same as all-dead but
            `is_complete` is already True so callers should have stopped.
        """
        self.tick += 1
        self.tam *= 1.0 + self.market_growth_rate

        alive = self.alive_companies
        if not alive:
            payload: dict[str, Any] = {
                "tick": self.tick,
                "tam": self.tam,
                "alive": 0,
                "results": [],
            }
            if not self._all_dead_emitted:
                payload["all_dead"] = True
                self._all_dead_emitted = True
            return payload

        # Compute per-company "quality" and "marketing" magnitudes from the
        # bridge aggregate, then attenuate by shock impacts so a competitor
        # entry pulls share away from the affected company specifically.
        attractions: list[float] = []
        qualities: list[float] = []
        marketings: list[float] = []
        for c in alive:
            bridge_agg = derive_bridge_aggregate(c.library, c.spawned_nodes)
            quality = max(0.01, 1.0 + sum(bridge_agg.quality.values()))
            marketing = max(0.01, 1.0 + sum(bridge_agg.marketing.values()))
            qualities.append(quality)
            marketings.append(marketing)

            # Per-company share modulation from active shocks: lower
            # market_share_mult or higher competitor_pressure_mult both
            # reduce this company's share of attraction.
            company_env = apply_active_shocks(
                c.active_shocks,
                {
                    "market_share_mult": MARKET_SHARE_NEUTRAL,
                    "competitor_pressure_mult": COMPETITOR_PRESSURE_NEUTRAL,
                },
            )
            share_mult = float(company_env.get("market_share_mult", 1.0))
            comp_pressure = max(
                0.01, float(company_env.get("competitor_pressure_mult", 1.0))
            )
            attraction = (
                math.pow(quality, self.share_beta)
                * math.pow(marketing, self.share_alpha)
                * share_mult
                / comp_pressure
            )
            attractions.append(attraction)

        total_attraction = sum(attractions)
        if total_attraction <= 0:
            shares = [1.0 / len(alive)] * len(alive)
        else:
            shares = [a / total_attraction for a in attractions]

        # Revenue ceiling per company = TAM * share, in customers (divide by
        # the company's own starting_price as the unit-economics anchor).
        for c, share in zip(alive, shares, strict=True):
            ceiling_revenue = self.tam * share
            ceiling_customers = (
                ceiling_revenue / c.seed.starting_price
                if c.seed.starting_price > 0
                else 0.0
            )
            c.allocated_demand = ceiling_customers
            c.market_share = share

        # Tick each company
        results: list[TickResult] = []
        for c in alive:
            tr = await c.step()
            results.append(tr)

        # If all companies died as a result of this tick, set the flag so
        # the next caller-visible payload emits `all_dead`. We avoid emitting
        # it on the same tick that already carries real `results`.
        return {
            "tick": self.tick,
            "tam": self.tam,
            "alive": len(self.alive_companies),
            "results": results,
            "shares": shares,
        }

    async def run(self) -> list[dict[str, Any]]:
        """Run to completion, returning tick-by-tick result dicts."""
        out: list[dict[str, Any]] = []
        while not self.is_complete:
            out.append(await self.step())
        return out


__all__ = [
    "ATTRITION_BASE_RATE",
    "CAPITAL_BUCKET_DAILY_GAIN",
    "COST_BUCKET_GAIN",
    "COST_BUCKET_MULT_MAX",
    "COST_BUCKET_MULT_MIN",
    "DAYS_PER_MONTH",
    "DEFAULT_MARKET_GROWTH_RATE",
    "DEFAULT_SHARE_ALPHA",
    "DEFAULT_SHARE_BETA",
    "HISTORY_LEN",
    "INFRA_SATURATION_K",
    "INFRA_TO_CAPACITY_GAIN",
    "INSOLVENT_TICKS_TO_DEATH",
    "MARKETING_INTENSITY_MAX",
    "MARKETING_INTENSITY_MIN",
    "MARKETING_TO_DEMAND_GAIN",
    "PRICE_ADJUST_MAX_FACTOR",
    "PRICE_ADJUST_MIN_FACTOR",
    "QUALITY_TO_SATISFACTION_GAIN",
    "RAISE_ALLOWED_ARCHETYPES",
    "REPLENISH_INVENTORY_DAYS",
    "REVENUE_BUCKET_GAIN",
    "REVENUE_BUCKET_MULT_MAX",
    "REVENUE_BUCKET_MULT_MIN",
    "SATISFACTION_CONVERGENCE_RATE",
    "SOLO_DEMAND_SATURATION",
    "CompanyAgentV2",
    "MultiCompanySimV2",
    "TickResult",
]
