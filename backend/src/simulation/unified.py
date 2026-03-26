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

import copy
import math
import random as _random_module

from src.simulation.bridge import (
    allocate_demand_to_locations,
    derive_competitive_attributes,
)
from src.simulation.growth import (
    EMPLOYEES_PER_LOCATION,
    LOCATION_OPEN_COST,
    NEW_LOCATION_STARTING_CUSTOMERS,
    NEW_LOCATION_STARTING_SATISFACTION,
    VOLUME_DISCOUNTS,
)

# Unified mode inventory calibration: higher throughput needs bigger orders.
# At 68 customers/day (~85% of 80 capacity), consumption is ~71 lbs/day
# (including spoilage). With reorder_qty=200 and reorder_point=80, the
# location reorders every ~3 days with no stockouts.
UNIFIED_REORDER_QTY = 200.0
UNIFIED_REORDER_POINT = 80.0
from src.simulation.location import tick_location
from src.simulation.market.colors import agent_color
from src.simulation.market.engine import (
    AGENT_NAMES,
    compute_hhi,
    compute_share_attraction,
    compute_spawn_probability,
)
from src.simulation.models import (
    CompanyState,
    GraphSnapshot,
    LocationState,
    NodeCategory,
    NodeSnapshot,
    NodeType,
    SimEdge,
    SimNode,
)
from src.simulation.nodes import NODE_REGISTRY
from src.simulation.triggers import TRIGGER_REGISTRY
from src.simulation.unified_models import (
    UnifiedAgentSnapshot,
    UnifiedParams,
    UnifiedStartConfig,
    UnifiedTickData,
)


class CompanyAgent:
    """A single company in the unified simulation.

    Manages its own node graph and trigger state, reusing the same patterns
    as GrowthEngine but without an independent tick cycle.
    """

    def __init__(
        self,
        name: str,
        index: int,
        cash: float = 50_000.0,
        rng: _random_module.Random | None = None,
    ) -> None:
        self.state = CompanyState(name=name, cash=cash)
        self.color = agent_color(index)
        self.index = index
        self._node_counter = 0

        # Each company gets its own trigger instances (they track fired state)
        self.triggers = copy.deepcopy(TRIGGER_REGISTRY)

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

        self._rng = rng or _random_module.Random()

        # Seed initial nodes
        self._add_node(NodeType.OWNER_OPERATOR)
        loc_id = self._add_node(NodeType.RESTAURANT)
        self.state.nodes[loc_id].location_state = LocationState(
            reorder_qty=UNIFIED_REORDER_QTY,
            reorder_point=UNIFIED_REORDER_POINT,
        )
        self._add_node(NodeType.CHICKEN_SUPPLIER, edges_to=[loc_id], rel="supplies")
        self._add_node(NodeType.PRODUCE_SUPPLIER, edges_to=[loc_id], rel="supplies")
        self.state.total_employees = EMPLOYEES_PER_LOCATION

    # ── Node management (same patterns as GrowthEngine) ──

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}-{self._node_counter}"

    def _add_node(
        self,
        node_type: NodeType,
        edges_to: list[str] | None = None,
        rel: str = "manages",
    ) -> str:
        config = NODE_REGISTRY[node_type]
        node_id = self._next_id(node_type.value)
        count = sum(1 for n in self.state.nodes.values() if n.type == node_type)
        label = config.label
        if node_type == NodeType.RESTAURANT:
            label = f"Location #{count + 1}"
        elif node_type == NodeType.AREA_MANAGER:
            label = f"Area Manager #{count + 1}"

        node = SimNode(
            id=node_id,
            type=node_type,
            label=label,
            category=config.category,
            spawned_at=self.state.tick,
            annual_cost=config.annual_cost,
            cost_modifiers=dict(config.cost_modifiers),
            revenue_modifiers=dict(config.revenue_modifiers),
        )
        self.state.nodes[node_id] = node

        if edges_to:
            for target_id in edges_to:
                self.state.edges.append(
                    SimEdge(source=node_id, target=target_id, relationship=rel)
                )

        return node_id

    def location_count(self) -> int:
        return sum(
            1 for n in self.state.nodes.values()
            if n.type == NodeType.RESTAURANT and n.active
        )

    def active_locations(self) -> list[SimNode]:
        return [
            n for n in self.state.nodes.values()
            if n.type == NodeType.RESTAURANT and n.active and n.location_state is not None
        ]

    def aggregate_modifiers(self) -> dict[str, float]:
        """Aggregate all node modifiers (same logic as GrowthEngine)."""
        mods: dict[str, float] = {}

        for node in self.state.nodes.values():
            if not node.active:
                continue
            for key, value in node.cost_modifiers.items():
                mods[key] = mods.get(key, 1.0) * (1.0 + value)
            for key, value in node.revenue_modifiers.items():
                mods[key] = mods.get(key, 0.0) + value

        loc_count = self.location_count()
        volume_mult = 1.0
        for threshold, mult in VOLUME_DISCOUNTS:
            if loc_count >= threshold:
                volume_mult = mult
                break
        mods["food_cost"] = mods.get("food_cost", 1.0) * volume_mult

        return mods

    def check_triggers(self) -> list[str]:
        """Check all growth triggers and spawn nodes. Returns event messages."""
        events: list[str] = []

        for trigger in self.triggers:
            if not trigger.can_fire(self):
                continue

            if trigger.node_type == NodeType.RESTAURANT:
                if self.state.cash < LOCATION_OPEN_COST:
                    continue
                loc_id = self._add_node(NodeType.RESTAURANT)
                self.state.nodes[loc_id].location_state = LocationState(
                    customers=NEW_LOCATION_STARTING_CUSTOMERS,
                    satisfaction=NEW_LOCATION_STARTING_SATISFACTION,
                    reorder_qty=UNIFIED_REORDER_QTY,
                    reorder_point=UNIFIED_REORDER_POINT,
                )
                self.state.cash -= LOCATION_OPEN_COST
                self.state.total_employees += EMPLOYEES_PER_LOCATION
                self.state.locations_opened_this_year += 1
                for n in self.state.nodes.values():
                    if n.type in (NodeType.CHICKEN_SUPPLIER, NodeType.PRODUCE_SUPPLIER):
                        self.state.edges.append(
                            SimEdge(source=n.id, target=loc_id, relationship="supplies")
                        )
            else:
                owner_ids = [
                    n.id for n in self.state.nodes.values()
                    if n.type == NodeType.OWNER_OPERATOR
                ]
                self._add_node(
                    trigger.node_type,
                    edges_to=owner_ids,
                    rel="reports_to",
                )

            trigger.mark_fired(self.state.tick)
            events.append(f"{self.state.name}: {trigger.label}")

        return events

    def compute_corporate_overhead(self) -> float:
        """Sum daily costs of all non-restaurant active nodes."""
        overhead = sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active and n.category == NodeCategory.CORPORATE and n.annual_cost > 0
        )
        overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active and n.category == NodeCategory.REVENUE
        )
        overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active
            and n.category == NodeCategory.LOCATION
            and n.type != NodeType.RESTAURANT
        )
        return overhead

    def update_stage(self) -> None:
        loc_count = self.location_count()
        if loc_count >= 51:
            self.state.stage = 4
        elif loc_count >= 11:
            self.state.stage = 3
        elif loc_count >= 2:
            self.state.stage = 2
        else:
            self.state.stage = 1

    def avg_satisfaction(self) -> float:
        locations = self.active_locations()
        if not locations:
            return 0.0
        return sum(
            loc.location_state.satisfaction  # type: ignore[union-attr]
            for loc in locations
        ) / len(locations)

    def build_graph_snapshot(self) -> GraphSnapshot:
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
                    type=n.type.value,
                    label=n.label,
                    category=n.category.value,
                    spawned_at=n.spawned_at,
                    metrics=metrics,
                )
            )
        return GraphSnapshot(
            nodes=node_snapshots,
            edges=list(self.state.edges),
        )


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

        self.tick_num: int = 0
        self.tam: float = self.params.tam_0
        self.companies: list[CompanyAgent] = []
        self._next_company_idx: int = 0
        self._status: str = "operating"
        self.focused_company_id: str = ""

        # Spawn initial companies based on start mode
        initial_count = cfg.num_companies
        if cfg.start_mode == "staggered":
            initial_count = min(2, cfg.num_companies)

        self._target_companies = cfg.num_companies
        self._start_mode = cfg.start_mode

        for _ in range(initial_count):
            self._spawn_company(cfg.start_mode)

        if self.companies:
            self.focused_company_id = self.companies[0].state.name

    @property
    def is_complete(self) -> bool:
        if self._status == "collapsed":
            return True
        if self.max_ticks > 0 and self.tick_num >= self.max_ticks:
            return True
        return False

    def _spawn_company(self, start_mode: str = "identical") -> CompanyAgent:
        idx = self._next_company_idx
        self._next_company_idx += 1

        name = AGENT_NAMES[idx % len(AGENT_NAMES)]
        if idx >= len(AGENT_NAMES):
            name = f"{name} {idx // len(AGENT_NAMES) + 1}"

        cash = self.params.starting_cash
        if start_mode == "randomized":
            cash = self.rng.uniform(30_000.0, 80_000.0)

        agent = CompanyAgent(name=name, index=idx, cash=cash, rng=self.rng)

        if start_mode == "randomized":
            # Add variance to first location's starting conditions
            for node in agent.state.nodes.values():
                if node.location_state is not None:
                    node.location_state.satisfaction = self.rng.uniform(0.55, 0.85)
                    node.location_state.customers = self.rng.uniform(20.0, 45.0)

        self.companies.append(agent)
        return agent

    def _alive_companies(self) -> list[CompanyAgent]:
        return [c for c in self.companies if c.alive]

    def tick(self) -> dict:  # noqa: C901
        """Run one unified simulation step. Returns tick data dict for SSE."""
        self.tick_num += 1
        events: list[str] = []
        alive = self._alive_companies()

        if not alive:
            self._status = "collapsed"
            return self._build_result(events)

        # Advance each company's tick counter
        for company in alive:
            company.state.tick = self.tick_num
            if self.tick_num % 365 == 0:
                company.state.locations_opened_this_year = 0

        # ── Step 1: Update TAM ──
        self.tam *= 1.0 + self.params.g_market

        # ── Step 2: Derive competitive attributes from node graphs ──
        for company in alive:
            q, m, k = derive_competitive_attributes(company.state)
            company.quality = q
            company.marketing = m
            company.capacity = k

        # ── Step 3: Compute share attraction ──
        # Build temporary AgentState-like objects for the share formula
        qualities = [c.quality if c.alive else 0.0 for c in self.companies]
        marketings = [c.marketing if c.alive else 0.0 for c in self.companies]
        alive_flags = [c.alive for c in self.companies]

        shares = _compute_shares(
            qualities, marketings, alive_flags,
            self.params.alpha, self.params.beta,
        )
        for company, share in zip(self.companies, shares):
            company.share = share

        # ── Step 4: Revenue ceiling per firm ──
        ceilings: dict[int, float] = {}
        for company in alive:
            ceiling = min(self.tam * company.share, company.capacity)
            ceilings[company.index] = ceiling

        # ── Step 5: Allocate demand to locations ──
        # Convert revenue ceiling ($/day) to customer count, then distribute
        for company in alive:
            locations = company.active_locations()
            if not locations:
                continue

            avg_price = sum(
                loc.location_state.price  # type: ignore[union-attr]
                for loc in locations
            ) / len(locations)

            ceiling_customers = ceilings[company.index] / avg_price if avg_price > 0 else 0.0
            demand_alloc = allocate_demand_to_locations(ceiling_customers, locations)

            # ── Step 6: Tick each location with allocated demand ──
            mods = company.aggregate_modifiers()
            total_revenue = 0.0
            total_costs = 0.0

            for node in locations:
                allocated = demand_alloc.get(node.id, 0.0)
                result, reorder_cost = tick_location(
                    node.location_state,  # type: ignore[arg-type]
                    mods,
                    company.state.cash,
                    allocated_demand=allocated,
                )
                company.state.cash -= reorder_cost
                company.state.cash += result.profit
                total_revenue += result.revenue
                total_costs += result.costs

                for evt in result.events:
                    events.append(f"{company.state.name} {node.label}: {evt}")

            # ── Step 7: Corporate overhead, cash, stage ──
            overhead = company.compute_corporate_overhead()
            company.state.cash -= overhead
            total_costs += overhead

            company.daily_revenue = total_revenue
            company.daily_costs = total_costs
            company.update_stage()

        # ── Step 8: Check growth triggers per firm ──
        for company in alive:
            trigger_events = company.check_triggers()
            events.extend(trigger_events)

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

            # In staggered mode, boost entry until target company count reached
            if self._start_mode == "staggered":
                alive_count = len(alive_now)
                if alive_count < self._target_companies:
                    p_spawn = max(p_spawn, 0.02)  # floor 2% per tick

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
            loc_count = c.location_count()
            node_count = sum(1 for n in c.state.nodes.values() if n.active)

            tam_share_ceiling = self.tam * c.share if c.share > 0 else 0.0
            util = c.daily_revenue / c.capacity if c.capacity > 0 else 0.0
            constraint = "capacity" if c.capacity <= tam_share_ceiling else "demand"

            snap = UnifiedAgentSnapshot(
                id=c.state.name,
                name=c.state.name,
                alive=c.alive,
                color=c.color,
                quality=round(c.quality, 4),
                marketing=round(c.marketing, 2),
                capacity=round(c.capacity, 2),
                share=round(c.share, 6),
                utilization=round(min(1.0, util), 4),
                binding_constraint=constraint,
                cash=round(c.state.cash, 2),
                daily_revenue=round(c.daily_revenue, 2),
                daily_costs=round(c.daily_costs, 2),
                stage=c.state.stage,
                location_count=loc_count,
                node_count=node_count,
                avg_satisfaction=round(c.avg_satisfaction(), 3),
                total_employees=c.state.total_employees,
            )
            agent_snapshots.append(snap.model_dump())

        # Build graph for the focused company
        focused = next(
            (c for c in self.companies if c.state.name == self.focused_company_id),
            self.companies[0] if self.companies else None,
        )
        graph = focused.build_graph_snapshot() if focused else GraphSnapshot(nodes=[], edges=[])

        result = UnifiedTickData(
            tick=self.tick_num,
            status=self._status,
            tam=round(self.tam, 2),
            captured=round(captured, 2),
            hhi=round(hhi, 6),
            agent_count=len(alive),
            agents=agent_snapshots,
            focused_company_id=self.focused_company_id,
            graph=graph,
            events=events,
        )
        return result.model_dump()


def _compute_shares(
    qualities: list[float],
    marketings: list[float],
    alive_flags: list[bool],
    alpha: float,
    beta: float,
) -> list[float]:
    """Compute market shares using the multinomial logit share attraction formula.

    Same math as market engine's compute_share_attraction, but works with
    raw lists instead of AgentState objects.
    """
    attractions: list[float] = []
    total = 0.0
    for q, m, is_alive in zip(qualities, marketings, alive_flags):
        if is_alive:
            q_safe = max(q, 1e-12)
            m_safe = max(m, 0.0)
            a = (q_safe ** beta) * (m_safe ** alpha)
        else:
            a = 0.0
        attractions.append(a)
        total += a

    if total <= 0:
        alive_count = sum(1 for f in alive_flags if f)
        if alive_count == 0:
            return [0.0] * len(qualities)
        equal = 1.0 / alive_count
        return [equal if f else 0.0 for f in alive_flags]

    return [a / total for a in attractions]
