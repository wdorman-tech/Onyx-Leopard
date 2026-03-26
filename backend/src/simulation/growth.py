"""Growth engine — manages a company graph that grows over time."""

from __future__ import annotations

from src.simulation.location import tick_location
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

# Volume-based food cost discount (from real chain research)
VOLUME_DISCOUNTS = [
    (21, 0.72),
    (11, 0.80),
    (6, 0.87),
    (3, 0.93),
    (1, 1.00),
]

LOCATION_OPEN_COST = 50_000.0
EMPLOYEES_PER_LOCATION = 15
NEW_LOCATION_STARTING_CUSTOMERS = 20.0
NEW_LOCATION_STARTING_SATISFACTION = 0.5


class GrowthEngine:
    def __init__(self, max_ticks: int = 0, industry: str = "restaurant") -> None:
        self.max_ticks = max_ticks
        self.industry = industry
        self.state = CompanyState(cash=50_000.0)
        self._status = "operating"
        self._node_counter = 0

        # Seed initial nodes
        self._add_node(NodeType.OWNER_OPERATOR, edges_to=[])
        loc_id = self._add_node(NodeType.RESTAURANT, edges_to=[])
        # Give first location the default starting state
        self.state.nodes[loc_id].location_state = LocationState()
        self._add_node(NodeType.CHICKEN_SUPPLIER, edges_to=[loc_id], rel="supplies")
        self._add_node(NodeType.PRODUCE_SUPPLIER, edges_to=[loc_id], rel="supplies")

        self.state.total_employees = EMPLOYEES_PER_LOCATION

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
                self.state.edges.append(SimEdge(source=node_id, target=target_id, relationship=rel))

        return node_id

    def _location_count(self) -> int:
        return sum(
            1 for n in self.state.nodes.values()
            if n.type == NodeType.RESTAURANT and n.active
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
        for threshold, mult in VOLUME_DISCOUNTS:
            if loc_count >= threshold:
                volume_mult = mult
                break
        mods["food_cost"] = mods.get("food_cost", 1.0) * volume_mult

        return mods

    def _check_triggers(self) -> list[str]:
        """Check all growth triggers and spawn nodes. Returns event messages."""
        events: list[str] = []

        for trigger in TRIGGER_REGISTRY:
            if not trigger.can_fire(self):
                continue

            # Special handling for new locations
            if trigger.node_type == NodeType.RESTAURANT:
                if self.state.cash < LOCATION_OPEN_COST:
                    continue
                loc_id = self._add_node(NodeType.RESTAURANT)
                self.state.nodes[loc_id].location_state = LocationState(
                    customers=NEW_LOCATION_STARTING_CUSTOMERS,
                    satisfaction=NEW_LOCATION_STARTING_SATISFACTION,
                )
                self.state.cash -= LOCATION_OPEN_COST
                self.state.total_employees += EMPLOYEES_PER_LOCATION
                self.state.locations_opened_this_year += 1
                # Connect suppliers to new location
                for n in self.state.nodes.values():
                    if n.type in (NodeType.CHICKEN_SUPPLIER, NodeType.PRODUCE_SUPPLIER):
                        self.state.edges.append(
                            SimEdge(source=n.id, target=loc_id, relationship="supplies")
                        )
            else:
                # Connect corporate nodes to the owner
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
            events.append(trigger.label)

        return events

    def tick(self) -> dict:
        self.state.tick += 1

        # Reset yearly counter
        if self.state.tick % 365 == 0:
            self.state.locations_opened_this_year = 0

        events: list[str] = []

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
            if node.type != NodeType.RESTAURANT or not node.active or not node.location_state:
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
        # Revenue-adjacent node costs too
        corporate_overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active and n.category == NodeCategory.REVENUE
        )
        # Infrastructure costs (commissary, DC)
        corporate_overhead += sum(
            n.daily_cost for n in self.state.nodes.values()
            if n.active
            and n.category == NodeCategory.LOCATION
            and n.type != NodeType.RESTAURANT
        )
        self.state.cash -= corporate_overhead

        total_costs += corporate_overhead
        total_profit = total_revenue - total_costs
        avg_satisfaction = satisfaction_sum / loc_count if loc_count > 0 else 0.0

        # 5. Update stage
        if loc_count >= 51:
            self.state.stage = 4
        elif loc_count >= 11:
            self.state.stage = 3
        elif loc_count >= 2:
            self.state.stage = 2
        else:
            self.state.stage = 1

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
                type=n.type.value,
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
