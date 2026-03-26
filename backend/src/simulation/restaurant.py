"""Single-restaurant simulation engine. One item (grilled chicken), pure math, no AI."""

from __future__ import annotations

import math
import random


class RestaurantEngine:
    def __init__(self, max_ticks: int = 365) -> None:
        self.max_ticks = max_ticks
        self._tick = 0

        # State
        self.cash = 10_000.0
        self.inventory = 50.0  # lbs of chicken
        self.customers = 30.0
        self.satisfaction = 0.7  # 0–1

        # Parameters
        self.price = 12.00
        self.max_capacity = 60
        self.food_cost_per_plate = 1.50  # cooking costs (gas, oil, seasoning)
        self.daily_fixed_costs = 368.00
        self.reorder_point = 20.0
        self.reorder_qty = 80.0  # enough to cover max_capacity
        self.chicken_cost_per_lb = 3.50
        self.spoilage_rate = 0.05
        self.word_of_mouth_rate = 0.02
        self.max_local_customers = 120.0

        self._status = "operating"

    @property
    def is_complete(self) -> bool:
        return self._tick >= self.max_ticks or self._status == "bankrupt"

    def tick(self) -> dict:
        self._tick += 1
        events: list[str] = []

        # 1. Demand — logistic growth with satisfaction modifier + noise
        growth = self.word_of_mouth_rate * self.satisfaction * (1 - self.customers / self.max_local_customers)
        self.customers *= 1 + growth
        actual_demand = self.customers * random.uniform(0.9, 1.1)
        actual_demand = max(0, math.floor(actual_demand))

        # 2. Supply constraint
        servable = min(actual_demand, self.max_capacity, int(self.inventory))
        unserved = max(0, actual_demand - servable)

        if unserved > 0 and actual_demand > self.max_capacity:
            events.append(f"Turned away {unserved} customers (at capacity)")
        elif unserved > 0 and self.inventory < actual_demand:
            events.append(f"Turned away {unserved} customers (out of chicken)")

        # 3. Revenue
        daily_revenue = servable * self.price

        # 4. Inventory consumption
        self.inventory -= servable

        # 5. Spoilage
        spoiled = self.inventory * self.spoilage_rate
        if spoiled >= 1:
            events.append(f"{spoiled:.0f} lbs chicken spoiled")
        self.inventory -= spoiled
        self.inventory = max(0, self.inventory)

        # 6. Auto-reorder
        if self.inventory < self.reorder_point:
            order_cost = self.reorder_qty * self.chicken_cost_per_lb
            if self.cash >= order_cost:
                self.inventory += self.reorder_qty
                self.cash -= order_cost
                events.append(f"Ordered {self.reorder_qty:.0f} lbs chicken (${order_cost:.0f})")
            else:
                events.append("Cannot reorder — insufficient cash")

        # 7. Costs
        daily_variable_costs = servable * self.food_cost_per_plate
        daily_costs = self.daily_fixed_costs + daily_variable_costs

        # 8. Cash update
        daily_profit = daily_revenue - daily_costs
        self.cash += daily_profit

        # 9. Satisfaction update
        if unserved > 0 and actual_demand > 0:
            self.satisfaction = max(0, self.satisfaction - 0.02 * (unserved / actual_demand))
        else:
            self.satisfaction = min(1.0, self.satisfaction + 0.005)

        # 10. Bankruptcy check
        if self.cash <= 0:
            self._status = "bankrupt"
            self.cash = 0
            events.append("BANKRUPT — out of cash")

        return {
            "tick": self._tick,
            "metrics": {
                "cash": round(self.cash, 2),
                "daily_revenue": round(daily_revenue, 2),
                "daily_costs": round(daily_costs, 2),
                "daily_profit": round(daily_profit, 2),
                "customers": actual_demand,
                "inventory": round(self.inventory, 1),
                "satisfaction": round(self.satisfaction, 3),
            },
            "status": self._status,
            "events": events,
        }
