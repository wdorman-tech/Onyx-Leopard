"""Growth engine — manages a company graph that grows over time."""

from __future__ import annotations

from src.simulation.config_loader import IndustrySpec, load_industry
from src.simulation.location import tick_location
from src.simulation.models import (
    CompanyState,
    GraphSnapshot,
    LocationState,
    NodeCategory,
    NodeSnapshot,
    SimEdge,
    SimNode,
)
from src.simulation.triggers import build_triggers


class GrowthEngine:
    def __init__(
        self,
        max_ticks: int = 0,
        industry: str = "restaurant",
        spec: IndustrySpec | None = None,
    ) -> None:
        self.spec = spec or load_industry(industry)
        self.max_ticks = max_ticks
        self.state = CompanyState(cash=self.spec.constants.starting_cash)
        self._status = "operating"
        self._node_counter = 0
        self._triggers = build_triggers(self.spec)

        # Seed initial nodes from config
        self._add_node(self.spec.roles.founder_type, edges_to=[])
        loc_id = self._add_node(self.spec.roles.location_type, edges_to=[])
        self.state.nodes[loc_id].location_state = LocationState(
            **self.spec.location_defaults.model_dump(
                exclude={"unified_reorder_qty", "unified_reorder_point"}
            )
        )
        self._supplier_ids: list[str] = []
        for supplier_type in self.spec.roles.supplier_types:
            sid = self._add_node(supplier_type, edges_to=[loc_id], rel="supplies")
            self._supplier_ids.append(sid)

        self.state.total_employees = self.spec.constants.employees_per_location

    @property
    def is_complete(self) -> bool:
        if self._status == "bankrupt":
            return True
        if self.max_ticks > 0 and self.state.tick >= self.max_ticks:
            return True
        return False

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

        if edges_to:
            for target_id in edges_to:
                self.state.edges.append(SimEdge(source=node_id, target=target_id, relationship=rel))

        return node_id

    def _location_count(self) -> int:
        loc_type = self.spec.roles.location_type
        return sum(
            1 for n in self.state.nodes.values()
            if n.type == loc_type and n.active
        )

    def _aggregate_modifiers(self) -> dict[str, float]:
        """Aggregate all node modifiers multiplicatively."""
        mods: dict[str, float] = {}

        for node in self.state.nodes.values():
            if not node.active:
                continue
            for key, value in node.cost_modifiers.items():
                mods[key] = mods.get(key, 1.0) * (1.0 + value)
            for key, value in node.revenue_modifiers.items():
                mods[key] = mods.get(key, 0.0) + value

        # Volume discount on food cost
        loc_count = self._location_count()
        volume_mult = 1.0
        for threshold, mult in self.spec.constants.volume_discounts:
            if loc_count >= threshold:
                volume_mult = mult
                break
        mods["food_cost"] = mods.get("food_cost", 1.0) * volume_mult

        return mods

    def _compute_metrics(self) -> tuple[dict[str, float], dict[str, int]]:
        """Compute metrics and node type counts for trigger evaluation."""
        loc_type = self.spec.roles.location_type
        node_type_counts: dict[str, int] = {}
        total_daily_revenue = 0.0
        location_margins: list[float] = []
        satisfaction_sum = 0.0
        location_count = 0

        for node in self.state.nodes.values():
            if not node.active:
                continue
            node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1

            if node.type == loc_type and node.location_state is not None:
                location_count += 1
                ls = node.location_state
                satisfaction_sum += ls.satisfaction
                served = min(ls.customers, ls.max_capacity, ls.inventory)
                total_daily_revenue += served * ls.price
                rev = ls.customers * ls.price
                costs = ls.daily_fixed_costs + ls.customers * ls.food_cost_per_plate
                if rev > 0:
                    location_margins.append((rev - costs) / rev)

        metrics: dict[str, float] = {
            "location_count": location_count,
            "monthly_revenue": total_daily_revenue * 30,
            "cash": self.state.cash,
            "total_employees": self.state.total_employees,
            "avg_satisfaction": (
                satisfaction_sum / location_count if location_count > 0 else 0.0
            ),
            "avg_location_margin": (
                sum(location_margins) / len(location_margins)
                if location_margins else 0.0
            ),
            "locations_opened_this_year": self.state.locations_opened_this_year,
        }
        return metrics, node_type_counts

    def _check_triggers(self) -> list[str]:
        """Check all growth triggers and spawn nodes. Returns event messages."""
        events: list[str] = []
        metrics, node_type_counts = self._compute_metrics()
        consts = self.spec.constants
        loc_type = self.spec.roles.location_type

        for trigger in self._triggers:
            if not trigger.can_fire(metrics, node_type_counts, self.state.tick):
                continue

            if trigger.is_location_expansion:
                if self.state.cash < consts.location_open_cost:
                    continue
                loc_id = self._add_node(loc_type)
                self.state.nodes[loc_id].location_state = LocationState(
                    customers=consts.new_location_starting_customers,
                    satisfaction=consts.new_location_starting_satisfaction,
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
            events.append(trigger.label)

        return events

    def tick(self) -> dict:
        self.state.tick += 1

        # Reset yearly counter
        if self.state.tick % 365 == 0:
            self.state.locations_opened_this_year = 0

        events: list[str] = []
        loc_type = self.spec.roles.location_type

        # 1. Check growth triggers
        trigger_events = self._check_triggers()
        events.extend(trigger_events)

        # 2. Aggregate modifiers
        mods = self._aggregate_modifiers()

        # 3. Tick each location
        total_revenue = 0.0
        total_costs = 0.0
        total_customers = 0
        satisfaction_sum = 0.0
        loc_count = 0

        for node in list(self.state.nodes.values()):
            if node.type != loc_type or not node.active or not node.location_state:
                continue

            result, reorder_cost = tick_location(node.location_state, mods, self.state.cash)
            self.state.cash -= reorder_cost
            self.state.cash += result.profit

            total_revenue += result.revenue
            total_costs += result.costs
            total_customers += result.customers_served
            satisfaction_sum += node.location_state.satisfaction
            loc_count += 1

            for evt in result.events:
                events.append(f"{node.label}: {evt}")

        # 4. Corporate overhead
        corporate_overhead = sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active and n.category == NodeCategory.CORPORATE and n.annual_cost > 0
        )
        corporate_overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active and n.category == NodeCategory.REVENUE
        )
        corporate_overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active
            and n.category == NodeCategory.LOCATION
            and n.type != loc_type
        )
        self.state.cash -= corporate_overhead

        total_costs += corporate_overhead
        total_profit = total_revenue - total_costs
        avg_satisfaction = satisfaction_sum / loc_count if loc_count > 0 else 0.0

        # 5. Update stage
        self._update_stage(loc_count)

        # 6. Bankruptcy check
        if self.state.cash <= 0:
            self._status = "bankrupt"
            self.state.cash = 0
            events.append("BANKRUPT — out of cash")

        # 7. Build graph snapshot for frontend
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
            node_snapshots.append(NodeSnapshot(
                id=n.id,
                type=n.type,
                label=n.label,
                category=n.category.value,
                spawned_at=n.spawned_at,
                metrics=metrics,
            ))

        graph = GraphSnapshot(
            nodes=node_snapshots,
            edges=[e for e in self.state.edges],
        )

        return {
            "tick": self.state.tick,
            "stage": self.state.stage,
            "status": self._status,
            "metrics": {
                "cash": round(self.state.cash, 2),
                "daily_revenue": round(total_revenue, 2),
                "daily_costs": round(total_costs, 2),
                "daily_profit": round(total_profit, 2),
                "total_customers": total_customers,
                "total_locations": loc_count,
                "avg_satisfaction": round(avg_satisfaction, 3),
                "total_employees": self.state.total_employees,
                "corporate_overhead": round(corporate_overhead, 2),
            },
            "events": events,
            "graph": graph.model_dump(),
        }

    def _update_stage(self, loc_count: int) -> None:
        self.state.stage = 1
        for stage_def in self.spec.stages:
            if loc_count >= stage_def.min_locations:
                self.state.stage = stage_def.stage
