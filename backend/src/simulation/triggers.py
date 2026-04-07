"""Data-driven growth trigger system.

Replaces the 19 hardcoded trigger classes with a single DataDrivenTrigger
that evaluates condition dicts from YAML config.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from src.simulation.conditions import evaluate_condition
from src.simulation.config_loader import IndustrySpec


@dataclass
class DataDrivenTrigger:
    """A single growth trigger loaded from industry YAML config."""

    node_type: str
    label: str
    max_instances: int = 1
    cooldown_ticks: int = 0
    is_location_expansion: bool = False
    condition: dict = field(default_factory=dict)
    _last_fired: int = -999_999

    def can_fire(
        self,
        metrics: dict[str, float],
        node_type_counts: dict[str, int],
        tick: int,
    ) -> bool:
        if tick - self._last_fired < self.cooldown_ticks:
            return False
        if node_type_counts.get(self.node_type, 0) >= self.max_instances:
            return False
        return evaluate_condition(self.condition, metrics, node_type_counts)

    def mark_fired(self, tick: int) -> None:
        self._last_fired = tick


def build_triggers(spec: IndustrySpec) -> list[DataDrivenTrigger]:
    """Create a list of DataDrivenTrigger instances from an IndustrySpec.

    Returns a fresh list each call (triggers track fired state, so each
    company needs its own copy).
    """
    return [
        DataDrivenTrigger(
            node_type=t.node_type,
            label=t.label,
            max_instances=t.max_instances,
            cooldown_ticks=t.cooldown_ticks,
            is_location_expansion=t.is_location_expansion,
            condition=copy.deepcopy(t.condition),
        )
        for t in spec.triggers
    ]
