"""Unified simulation engine — company node graphs competing in a shared market.

Combines the growth engine's trigger-based org-chart expansion with the market
engine's competitive share attraction math. Each company grows its node graph
autonomously, while the market determines how much revenue each company earns
based on its derived quality, marketing, and capacity.

Tick cycle:
 1. Update TAM
 2. Derive q_i, m_i, K_i from each company's node graph  (bridge)
 3. Compute share attraction                              (market math)
 4. Compute revenue ceiling per firm                      (market math)
 5. Allocate demand to each firm's locations               (bridge)
 6. Tick each location with allocated demand               (location economics)
 7. Per-company: corporate overhead, cash, stage           (growth logic)
 8. Per-company: check growth triggers, spawn nodes        (growth logic)
 9. Death check                                            (market logic)
10. New entrant spawn check                                (market logic)
"""

from __future__ import annotations

import random as _random_module
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.simulation.bridge import derive_competitive_attributes
from src.simulation.config_loader import IndustrySpec, load_industry
from src.simulation.location import tick_locations_batch
from src.simulation.market.colors import agent_color
from src.simulation.market.engine import (
    AGENT_NAMES,
    compute_hhi,
    compute_spawn_probability,
)
from src.simulation.models import (
    CompanyState,
    GraphSnapshot,
    LocationArrays,
    LocationConfig,
    LocationState,
    NodeCategory,
    NodeSnapshot,
    SimEdge,
    SimNode,
)
from src.simulation.triggers import build_triggers
from src.simulation.unified_models import UnifiedStartConfig

_MAX_THREAD_WORKERS = 8
_THREADING_LOC_THRESHOLD = 100


class CompanyAgent:
    """A single company in the unified simulation.

    Manages its own node graph and trigger state, reusing the same patterns
    as GrowthEngine but without an independent tick cycle.
    """

    def __init__(
        self,
        name: str,
        index: int,
        spec: IndustrySpec,
        cash: float = 50_000.0,
        rng: _random_module.Random | None = None,
    ) -> None:
        self.spec = spec
        self.state = CompanyState(name=name, cash=cash)
        self.color = agent_color(index)
        self.index = index
        self._node_counter = 0

        # Each company gets its own trigger instances (they track fired state)
        self.triggers = build_triggers(spec)

        # Market state (set by UnifiedEngine each tick)
        self.alive = True
        self.death_counter = 0
        self.share = 0.0
        self.prev_share = 0.0

        # Derived competitive attributes (set each tick by bridge)
        self.quality = 0.0
        self.marketing = 0.0
        self.capacity = 0.0

        # Per-tick financials (computed during tick)
        self.daily_revenue = 0.0
        self.daily_costs = 0.0

        # CEO agent state
        self.strategy: str | None = None
        self._marketing_boost: float = 0.5  # 0.5 = neutral (no effect)
        self._expansion_pace: str = "normal"
        self._max_locations_per_year: int = 999  # uncapped by default
        self._ceo_decision_history: list[dict] = []

        self._rng = rng or _random_module.Random()
        np_seed = self._rng.randint(0, 2**32 - 1)
        self._np_rng = np.random.default_rng(np_seed)

        # Per-industry location simulation constants (for batch tick)
        ld = spec.location_defaults
        self._loc_config = LocationConfig(
            satisfaction_penalty_rate=ld.satisfaction_penalty_rate,
            satisfaction_recovery_rate=ld.satisfaction_recovery_rate,
            customer_convergence_rate=ld.customer_convergence_rate,
            demand_cap_ratio=ld.demand_cap_ratio,
            demand_noise_low=ld.demand_noise_low,
            demand_noise_high=ld.demand_noise_high,
            subscription_scaling_threshold=ld.subscription_scaling_threshold,
            subscription_scaling_increment=ld.subscription_scaling_increment,
            new_customer_ratio=ld.new_customer_ratio,
            days_per_month=spec.constants.days_per_month,
            variable_cost_modifier_key=spec.constants.variable_cost_modifier_key,
        )

        # Struct-of-arrays for vectorized location ticking
        self._loc_arrays = LocationArrays()

        # Per-tick caches (refreshed by refresh_caches())
        self._cached_location_count: int = 0
        self._cached_active_node_count: int = 0
        self._cached_node_type_counts: dict[str, int] = {}
        self._cached_total_daily_revenue: float = 0.0
        self._cached_avg_location_margin: float = 0.0
        self._cached_avg_satisfaction: float = 0.0
        self._cached_location_nodes: list[SimNode] = []
        self._cached_corporate_overhead: float = 0.0
        self._cached_cost_mods: dict[str, float] = {}
        self._cached_revenue_mods: dict[str, float] = {}
        self._graph_dirty: bool = True
        self._cached_graph: GraphSnapshot | None = None

        # Seed initial nodes from config
        loc_type = spec.roles.location_type
        self._add_node(spec.roles.founder_type)
        loc_id = self._add_node(loc_type)
        self.state.nodes[loc_id].location_state = LocationState(
            **spec.location_defaults.to_location_state()
        )
        self._supplier_ids: list[str] = []
        for supplier_type in spec.roles.supplier_types:
            sid = self._add_node(supplier_type, edges_to=[loc_id], rel="supplies")
            self._supplier_ids.append(sid)
        self.state.total_employees = spec.constants.employees_per_location
        self._rebuild_loc_arrays()
        self.refresh_caches()

    # ── Node management ──

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}-{self._node_counter}"

    def _add_node(
        self,
        node_type: str,
        edges_to: list[str] | None = None,
        rel: str = "manages",
    ) -> str:
        config = self.spec.nodes[node_type]
        node_id = self._next_id(node_type)
        count = sum(1 for n in self.state.nodes.values() if n.type == node_type)

        label = config.label
        if node_type in self.spec.roles.numbered_labels:
            prefix = self.spec.roles.numbered_labels[node_type]
            label = f"{prefix} #{count + 1}"

        node = SimNode(
            id=node_id,
            type=node_type,
            label=label,
            category=NodeCategory(config.category),
            spawned_at=self.state.tick,
            annual_cost=config.annual_cost,
            cost_modifiers=dict(config.cost_modifiers),
            revenue_modifiers=dict(config.revenue_modifiers),
        )
        self.state.nodes[node_id] = node
        self._graph_dirty = True
        if node_type == self.spec.roles.location_type:
            self._cached_location_count += 1

        if edges_to:
            for target_id in edges_to:
                self.state.edges.append(
                    SimEdge(source=node_id, target=target_id, relationship=rel)
                )

        return node_id

    # ── SoA management ──

    def _rebuild_loc_arrays(self) -> None:
        """Full rebuild of LocationArrays from SimNode data."""
        loc_type = self.spec.roles.location_type
        arrays = LocationArrays()
        for node in self.state.nodes.values():
            if node.type == loc_type and node.active and node.location_state is not None:
                arrays.append_location(node.id, node.label, node.location_state)
        self._loc_arrays = arrays

    def _sync_arrays_to_nodes(self) -> None:
        """Write mutable array fields back to SimNode.location_state.

        Uses __dict__ to bypass Pydantic __setattr__ validation in the hot path.
        """
        customers = self._loc_arrays.customers
        inventory = self._loc_arrays.inventory
        satisfaction = self._loc_arrays.satisfaction
        nodes = self.state.nodes
        for i, node_id in enumerate(self._loc_arrays.node_ids):
            ls = nodes[node_id].location_state
            if ls is not None:
                ls.__dict__["customers"] = float(customers[i])
                ls.__dict__["inventory"] = float(inventory[i])
                ls.__dict__["satisfaction"] = float(satisfaction[i])

    def refresh_caches(self) -> None:
        """Refresh all per-tick caches in a single pass over nodes."""
        loc_type = self.spec.roles.location_type
        location_count = 0
        active_node_count = 0
        node_type_counts: dict[str, int] = {}
        total_daily_revenue = 0.0
        location_margins: list[float] = []
        satisfaction_sum = 0.0
        location_nodes: list[SimNode] = []
        corporate_overhead = 0.0
        cost_mods: dict[str, float] = {}
        revenue_mods: dict[str, float] = {}

        for node in self.state.nodes.values():
            if not node.active:
                continue
            active_node_count += 1
            nt = node.type
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

            # Aggregate modifiers
            for key, value in node.cost_modifiers.items():
                cost_mods[key] = cost_mods.get(key, 1.0) * (1.0 + value)
            for key, value in node.revenue_modifiers.items():
                revenue_mods[key] = revenue_mods.get(key, 0.0) + value

            # Corporate overhead
            if (
                (node.category == NodeCategory.CORPORATE and node.annual_cost > 0)
                or node.category == NodeCategory.REVENUE
                or (node.category == NodeCategory.LOCATION and nt != loc_type)
            ):
                corporate_overhead += node.daily_cost

            # Location-specific aggregates
            if nt == loc_type and node.location_state is not None:
                location_count += 1
                ls = node.location_state
                location_nodes.append(node)
                satisfaction_sum += ls.satisfaction
                served = min(ls.customers, ls.max_capacity, ls.inventory)
                total_daily_revenue += served * ls.price
                rev = ls.customers * ls.price
                costs = ls.daily_fixed_costs + ls.customers * ls.variable_cost_per_unit
                if rev > 0:
                    location_margins.append((rev - costs) / rev)

        self._cached_location_count = location_count
        self._cached_active_node_count = active_node_count
        self._cached_node_type_counts = node_type_counts
        self._cached_total_daily_revenue = total_daily_revenue
        self._cached_avg_location_margin = (
            sum(location_margins) / len(location_margins) if location_margins else 0.0
        )
        self._cached_avg_satisfaction = (
            satisfaction_sum / location_count if location_count > 0 else 0.0
        )
        self._cached_location_nodes = location_nodes
        self._cached_corporate_overhead = corporate_overhead
        self._cached_cost_mods = cost_mods
        self._cached_revenue_mods = revenue_mods

    def location_count(self) -> int:
        return self._cached_location_count

    def active_node_count(self) -> int:
        return self._cached_active_node_count

    def node_type_count(self, nt: str) -> int:
        return self._cached_node_type_counts.get(nt, 0)

    def has_node_type(self, nt: str) -> bool:
        return self._cached_node_type_counts.get(nt, 0) > 0

    def active_locations(self) -> list[SimNode]:
        return self._cached_location_nodes

    def aggregate_modifiers(self) -> dict[str, float]:
        """Return aggregated node modifiers from cache, with volume discount."""
        mods = dict(self._cached_cost_mods)
        mods.update(self._cached_revenue_mods)

        loc_count = self.location_count()
        volume_mult = 1.0
        for threshold, mult in self.spec.constants.volume_discounts:
            if loc_count >= threshold:
                volume_mult = mult
                break
        key = self.spec.constants.variable_cost_modifier_key
        mods[key] = mods.get(key, 1.0) * volume_mult

        return mods

    def _compute_trigger_metrics(self) -> tuple[dict[str, float], dict[str, int]]:
        """Build metrics dict for trigger evaluation from cached values."""
        metrics: dict[str, float] = {
            "location_count": self._cached_location_count,
            "monthly_revenue": self._cached_total_daily_revenue * self.spec.constants.days_per_month,
            "cash": self.state.cash,
            "total_employees": self.state.total_employees,
            "avg_satisfaction": self._cached_avg_satisfaction,
            "avg_location_margin": self._cached_avg_location_margin,
            "locations_opened_this_year": self.state.locations_opened_this_year,
        }
        return metrics, dict(self._cached_node_type_counts)

    def check_triggers(self) -> list[str]:
        """Check all growth triggers and spawn nodes. Returns event messages."""
        events: list[str] = []
        metrics, node_type_counts = self._compute_trigger_metrics()
        consts = self.spec.constants
        loc_type = self.spec.roles.location_type
        expansion_overrides = self.spec.ceo.expansion_overrides

        for trigger in self.triggers:
            if not trigger.can_fire(metrics, node_type_counts, self.state.tick):
                continue

            if trigger.is_location_expansion:
                # CEO expansion overrides
                if self.strategy is not None:
                    overrides = expansion_overrides.get(
                        self._expansion_pace,
                        expansion_overrides["normal"],
                    )
                    cash_threshold = overrides["cash_threshold"]
                    trigger.cooldown_ticks = int(overrides["cooldown_ticks"])
                    if self.state.locations_opened_this_year >= self._max_locations_per_year:
                        continue
                else:
                    cash_threshold = consts.location_open_cost

                if self.state.cash < cash_threshold:
                    continue
                loc_id = self._add_node(loc_type)
                self.state.nodes[loc_id].location_state = LocationState(
                    **self.spec.location_defaults.to_location_state(
                        customers=consts.new_location_starting_customers,
                        satisfaction=consts.new_location_starting_satisfaction,
                    )
                )
                self.state.cash -= consts.location_open_cost
                self.state.total_employees += consts.employees_per_location
                self.state.locations_opened_this_year += 1
                for sid in self._supplier_ids:
                    self.state.edges.append(
                        SimEdge(source=sid, target=loc_id, relationship="supplies")
                    )
                # Update metrics for subsequent triggers
                node_type_counts[loc_type] = node_type_counts.get(loc_type, 0) + 1
                metrics["location_count"] = node_type_counts[loc_type]
                metrics["cash"] = self.state.cash
            else:
                owner_ids = [
                    n.id for n in self.state.nodes.values()
                    if n.type == self.spec.roles.founder_type
                ]
                self._add_node(
                    trigger.node_type,
                    edges_to=owner_ids,
                    rel="reports_to",
                )
                node_type_counts[trigger.node_type] = (
                    node_type_counts.get(trigger.node_type, 0) + 1
                )

            trigger.mark_fired(self.state.tick)
            events.append(f"{self.state.name}: {trigger.label}")

        return events

    def compute_corporate_overhead(self) -> float:
        """Sum daily costs of all non-location active nodes (cached)."""
        return self._cached_corporate_overhead

    def update_stage(self) -> None:
        loc_count = self.location_count()
        self.state.stage = 1
        for stage_def in self.spec.stages:
            if loc_count >= stage_def.min_locations:
                self.state.stage = stage_def.stage

    def avg_satisfaction(self) -> float:
        return self._cached_avg_satisfaction

    def build_graph_snapshot(self) -> GraphSnapshot:
        if not self._graph_dirty and self._cached_graph is not None:
            # Structure unchanged — just update location metrics in-place
            for ns in self._cached_graph.nodes:
                node = self.state.nodes.get(ns.id)
                if node and node.location_state:
                    ns.metrics = {
                        "customers": node.location_state.customers,
                        "satisfaction": round(node.location_state.satisfaction, 2),
                        "inventory": round(node.location_state.inventory, 1),
                    }
            return self._cached_graph

        node_snapshots = []
        for n in self.state.nodes.values():
            if not n.active:
                continue
            metrics: dict[str, float] = {}
            if n.location_state:
                metrics = {
                    "customers": n.location_state.customers,
                    "satisfaction": round(n.location_state.satisfaction, 2),
                    "inventory": round(n.location_state.inventory, 1),
                }
            node_snapshots.append(
                NodeSnapshot(
                    id=n.id,
                    type=n.type,
                    label=n.label,
                    category=n.category.value,
                    spawned_at=n.spawned_at,
                    metrics=metrics,
                )
            )
        self._cached_graph = GraphSnapshot(
            nodes=node_snapshots,
            edges=list(self.state.edges),
        )
        self._graph_dirty = False
        return self._cached_graph


class UnifiedEngine:
    """Multi-company simulation with shared market competition.

    Each company is a CompanyAgent with its own node graph. The market layer
    determines revenue allocation via share attraction. The growth layer
    manages each company's internal economics and trigger-based expansion.
    """

    def __init__(
        self,
        config: UnifiedStartConfig | None = None,
        seed: int | None = None,
    ) -> None:
        cfg = config or UnifiedStartConfig()
        self.params = cfg.params
        self.max_ticks = cfg.max_ticks
        self.rng = _random_module.Random(seed)

        # Load industry spec
        self.spec = load_industry(cfg.industry)

        self.tick_num: int = 0
        self.tam: float = self.params.tam_0
        self.companies: list[CompanyAgent] = []
        self._next_company_idx: int = 0
        self._status: str = "operating"
        self.focused_company_id: str = ""

        # Spawn initial companies based on start mode
        initial_count = cfg.num_companies
        if cfg.start_mode == "staggered":
            initial_count = min(self.spec.constants.staggered_initial_count, cfg.num_companies)

        self._target_companies = cfg.num_companies
        self._start_mode = cfg.start_mode

        # AI CEO agent settings
        self.ai_ceo_enabled: bool = cfg.ai_ceo_enabled
        self._company_strategies: dict[int, str] = dict(cfg.company_strategies)
        self._pending_ceo_calls: bool = False

        for i in range(initial_count):
            company = self._spawn_company(cfg.start_mode)
            if self.ai_ceo_enabled:
                company.strategy = self._company_strategies.get(i, "balanced")

        self._executor: ThreadPoolExecutor | None = None
        self._use_threading = cfg.num_companies > 2
        if self._use_threading:
            self._executor = ThreadPoolExecutor(
                max_workers=min(_MAX_THREAD_WORKERS, cfg.num_companies),
            )

        if self.companies:
            self.focused_company_id = self.companies[0].state.name

    @property
    def is_complete(self) -> bool:
        if self._status == "collapsed":
            return True
        return self.max_ticks > 0 and self.tick_num >= self.max_ticks

    def _spawn_company(self, start_mode: str = "identical") -> CompanyAgent:
        idx = self._next_company_idx
        self._next_company_idx += 1

        name = AGENT_NAMES[idx % len(AGENT_NAMES)]
        if idx >= len(AGENT_NAMES):
            name = f"{name} {idx // len(AGENT_NAMES) + 1}"

        consts = self.spec.constants
        cash = self.params.starting_cash
        if start_mode == "randomized":
            cash = self.rng.uniform(consts.random_start_cash_low, consts.random_start_cash_high)

        agent = CompanyAgent(name=name, index=idx, spec=self.spec, cash=cash, rng=self.rng)

        # Assign CEO strategy for new market entrants
        if self.ai_ceo_enabled and agent.strategy is None:
            agent.strategy = "balanced"

        if start_mode == "randomized":
            # Add variance to first location's starting conditions
            for node in agent.state.nodes.values():
                if node.location_state is not None:
                    node.location_state.satisfaction = self.rng.uniform(
                        consts.random_start_satisfaction_low, consts.random_start_satisfaction_high
                    )
                    node.location_state.customers = self.rng.uniform(
                        consts.random_start_customers_low, consts.random_start_customers_high
                    )
            agent._rebuild_loc_arrays()

        self.companies.append(agent)
        return agent

    def _alive_companies(self) -> list[CompanyAgent]:
        return [c for c in self.companies if c.alive]

    def tick(self) -> dict:
        """Run one unified simulation step. Returns tick data dict for SSE."""
        self.tick_num += 1
        events: list[str] = []
        alive = self._alive_companies()

        if not alive:
            self._status = "collapsed"
            return self._build_result(events)

        # Advance each company's tick counter and refresh caches
        for company in alive:
            company.state.tick = self.tick_num
            company.refresh_caches()
            if self.tick_num % self.spec.constants.ticks_per_year == 0:
                company.state.locations_opened_this_year = 0

        # ── Step 1: Update TAM ──
        self.tam *= 1.0 + self.params.g_market

        # ── Step 2: Derive competitive attributes from node graphs ──
        for company in alive:
            q, m, k = derive_competitive_attributes(
                company.state, self.spec, marketing_boost=company._marketing_boost,
            )
            company.quality = q
            company.marketing = m
            company.capacity = k

        # ── Step 3: Compute share attraction ──
        qualities = [c.quality if c.alive else 0.0 for c in self.companies]
        marketings = [c.marketing if c.alive else 0.0 for c in self.companies]
        alive_flags = [c.alive for c in self.companies]

        shares = _compute_shares(
            qualities, marketings, alive_flags,
            self.params.alpha, self.params.beta,
        )
        for company, share in zip(self.companies, shares, strict=False):
            company.share = share

        # ── Step 4: Revenue ceiling per firm ──
        ceilings: dict[int, float] = {}
        for company in alive:
            ceiling = min(self.tam * company.share, company.capacity)
            ceilings[company.index] = ceiling

        # ── Steps 5-8: Per-company work (parallel when threading enabled) ──
        def _tick_company(company: CompanyAgent, ceiling: float) -> list[str]:
            """Per-company tick: demand alloc, location tick, overhead, triggers."""
            company_events: list[str] = []
            loc_arrays = company._loc_arrays
            n = loc_arrays.size
            if n > 0:
                # Step 5: Allocate demand using arrays directly
                prices = loc_arrays.price
                avg_price = float(sum(prices)) / n if n > 0 else self.spec.constants.default_avg_price
                ceiling_customers = ceiling / avg_price if avg_price > 0 else 0.0
                scores = loc_arrays.satisfaction * loc_arrays.max_capacity
                total_score = float(sum(scores))
                if total_score > 0:
                    allocated_arr = ceiling_customers * (scores / total_score)
                else:
                    allocated_arr = np.full(n, ceiling_customers / n)

                # Step 6: Vectorized batch location tick
                mods = company.aggregate_modifiers()
                cash_snapshot = company.state.cash
                batch = tick_locations_batch(
                    loc_arrays, mods, cash_snapshot, allocated_arr,
                    company._np_rng, company.state.name, loc_arrays.labels,
                    supply_unit_name=self.spec.location_defaults.supply_unit_name,
                    config=company._loc_config,
                )
                company._sync_arrays_to_nodes()
                company_events.extend(batch.events)

                company.state.cash = cash_snapshot + batch.total_profit - batch.total_reorder_cost

                # Step 7: Corporate overhead, cash, stage
                overhead = company.compute_corporate_overhead()
                company.state.cash -= overhead

                company.daily_revenue = batch.total_revenue
                company.daily_costs = batch.total_costs + overhead
                company.update_stage()

            # Step 8: Check growth triggers
            trigger_events = company.check_triggers()
            if trigger_events:
                company.refresh_caches()
                company._rebuild_loc_arrays()
            company_events.extend(trigger_events)
            return company_events

        # Thread only when total locations across all companies justify the overhead.
        total_locs = sum(c._loc_arrays.size for c in alive)
        use_threads = (
            self._use_threading
            and self._executor is not None
            and len(alive) > 2
            and total_locs >= _THREADING_LOC_THRESHOLD
        )
        if use_threads:
            futures = [
                self._executor.submit(_tick_company, company, ceilings[company.index])
                for company in alive
            ]
            for f in futures:
                events.extend(f.result())
        else:
            for company in alive:
                events.extend(_tick_company(company, ceilings[company.index]))

        # ── Step 8.5: Flag CEO agent decisions ──
        if (
            self.ai_ceo_enabled
            and self.tick_num > 0
            and self.tick_num % self.spec.ceo.interval_ticks == 0
        ):
            self._pending_ceo_calls = True

        # ── Step 9: Death check ──
        for company in self.companies:
            if not company.alive:
                continue

            if company.state.cash < self.params.b_death:
                company.death_counter += 1
            else:
                company.death_counter = 0

            if company.death_counter >= self.params.t_death:
                company.alive = False
                company.death_counter = 0
                company.daily_revenue = 0.0
                company.daily_costs = 0.0
                company._graph_dirty = True
                events.append(f"{company.state.name}: BANKRUPT")

        # ── Step 10: New entrant spawn check ──
        alive_now = self._alive_companies()
        if alive_now:
            alive_shares = [c.share for c in alive_now]
            hhi = compute_hhi(alive_shares)
            captured = sum(c.daily_revenue for c in alive_now)
            unserved = max(0.0, 1.0 - captured / self.tam) if self.tam > 0 else 0.0

            p_spawn = compute_spawn_probability(
                hhi, self.params.lambda_entry, self.params.g_market,
                self.params.g_ref, unserved,
            )

            if self._start_mode == "staggered":
                alive_count = len(alive_now)
                if alive_count < self._target_companies:
                    p_spawn = max(p_spawn, self.spec.constants.min_spawn_probability)

            if self.rng.random() < p_spawn:
                new_company = self._spawn_company(self._start_mode)
                events.append(f"{new_company.state.name}: Entered the market")

        if not self._alive_companies():
            self._status = "collapsed"

        return self._build_result(events)

    def _build_result(self, events: list[str]) -> dict:
        """Build the tick result dict for SSE streaming."""
        alive = self._alive_companies()
        alive_shares = [c.share for c in alive]
        hhi = compute_hhi(alive_shares) if alive else 1.0
        captured = sum(c.daily_revenue for c in alive)

        agent_snapshots: list[dict] = []
        for c in self.companies:
            tam_share_ceiling = self.tam * c.share if c.share > 0 else 0.0
            util = c.daily_revenue / c.capacity if c.capacity > 0 else 0.0
            agent_snapshots.append({
                "id": c.state.name,
                "name": c.state.name,
                "alive": c.alive,
                "color": c.color,
                "quality": round(c.quality, 4),
                "marketing": round(c.marketing, 2),
                "capacity": round(c.capacity, 2),
                "share": round(c.share, 6),
                "utilization": round(min(1.0, util), 4),
                "binding_constraint": "capacity" if c.capacity <= tam_share_ceiling else "demand",
                "cash": round(c.state.cash, 2),
                "daily_revenue": round(c.daily_revenue, 2),
                "daily_costs": round(c.daily_costs, 2),
                "stage": c.state.stage,
                "location_count": c.location_count(),
                "node_count": c.active_node_count(),
                "avg_satisfaction": round(c.avg_satisfaction(), 3),
                "total_employees": c.state.total_employees,
                "strategy": c.strategy,
            })

        # Build graphs for all companies (alive + dead for ghost rendering)
        graphs: dict[str, dict] = {}
        for c in self.companies:
            snap = c.build_graph_snapshot()
            graphs[c.state.name] = {
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.type,
                        "label": n.label,
                        "category": n.category,
                        "spawned_at": n.spawned_at,
                        "metrics": n.metrics,
                    }
                    for n in snap.nodes
                ],
                "edges": [
                    {"source": e.source, "target": e.target, "relationship": e.relationship}
                    for e in snap.edges
                ],
            }

        return {
            "tick": self.tick_num,
            "status": self._status,
            "mode": "unified",
            "tam": round(self.tam, 2),
            "captured": round(captured, 2),
            "hhi": round(hhi, 6),
            "agent_count": len(alive),
            "agents": agent_snapshots,
            "focused_company_id": self.focused_company_id,
            "graphs": graphs,
            "events": events,
        }


    async def run_ceo_agents(self) -> list[dict]:
        """Call Claude API for each alive company with a strategy. Returns decision events."""
        from src.simulation.ceo_agent import (
            apply_decision,
            build_ceo_system_prompt,
            build_ceo_user_prompt,
            call_ceo_agent,
        )

        decisions: list[dict] = []
        alive = self._alive_companies()

        for company in alive:
            if company.strategy is None:
                continue

            system_prompt = build_ceo_system_prompt(company.strategy, spec=self.spec)
            user_prompt = build_ceo_user_prompt(
                company, self.companies, self.tick_num, self.tam,
            )

            decision = await call_ceo_agent(
                company.state.name, system_prompt, user_prompt,
            )
            apply_decision(company, decision, self.tick_num)

            decisions.append({
                "company_name": company.state.name,
                "tick": self.tick_num,
                "sim_year": round(self.tick_num / self.spec.constants.ticks_per_year, 1),
                "strategy": company.strategy,
                "decision": decision.model_dump(),
            })

        self._pending_ceo_calls = False
        return decisions

    async def generate_reports(self) -> list[dict]:
        """Generate end-of-simulation reports for all companies with a strategy."""
        from src.simulation.ceo_agent import (
            build_report_system_prompt,
            build_report_user_prompt,
            call_report_agent,
        )

        reports: list[dict] = []
        system_prompt = build_report_system_prompt(spec=self.spec)

        for company in self.companies:
            if company.strategy is None:
                continue

            user_prompt = build_report_user_prompt(
                company, self.companies, self.tick_num, self.tam,
            )
            report = await call_report_agent(
                company.state.name, system_prompt, user_prompt,
            )
            reports.append(report.model_dump())

        return reports


def _compute_shares(
    qualities: list[float],
    marketings: list[float],
    alive_flags: list[bool],
    alpha: float,
    beta: float,
) -> list[float]:
    """Compute market shares using the multinomial logit share attraction formula."""
    q = np.array(qualities, dtype=np.float64)
    m = np.array(marketings, dtype=np.float64)
    alive = np.array(alive_flags, dtype=np.bool_)

    q_safe = np.maximum(q, 1e-12)
    m_safe = np.maximum(m, 0.0)
    attractions = np.where(alive, np.power(q_safe, beta) * np.power(m_safe, alpha), 0.0)
    total = float(attractions.sum())

    if total <= 0:
        alive_count = int(alive.sum())
        if alive_count == 0:
            return [0.0] * len(qualities)
        equal = 1.0 / alive_count
        return [equal if f else 0.0 for f in alive_flags]

    return (attractions / total).tolist()
