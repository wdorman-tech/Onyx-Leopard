"""Growth trigger registry — conditions for spawning new nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.simulation.models import NodeType

if TYPE_CHECKING:
    from src.simulation.growth import GrowthEngine


@dataclass
class GrowthTrigger:
    node_type: NodeType
    label: str  # event log message on spawn
    max_instances: int = 1
    cooldown_ticks: int = 0
    _last_fired: int = -999_999

    def should_fire(self, engine: GrowthEngine) -> bool:
        raise NotImplementedError

    def can_fire(self, engine: GrowthEngine) -> bool:
        if engine.state.tick - self._last_fired < self.cooldown_ticks:
            return False
        count = sum(
            1 for n in engine.state.nodes.values()
            if n.type == self.node_type and n.active
        )
        if count >= self.max_instances:
            return False
        return self.should_fire(engine)

    def mark_fired(self, tick: int) -> None:
        self._last_fired = tick


# ── Helper functions ──

def _location_count(engine: GrowthEngine) -> int:
    return sum(
        1 for n in engine.state.nodes.values()
        if n.type == NodeType.RESTAURANT and n.active
    )


def _has_node(engine: GrowthEngine, node_type: NodeType) -> bool:
    return any(
        n.type == node_type and n.active for n in engine.state.nodes.values()
    )


def _avg_location_margin(engine: GrowthEngine) -> float:
    """Average daily profit margin across locations."""
    margins = []
    for n in engine.state.nodes.values():
        if n.type == NodeType.RESTAURANT and n.active and n.location_state:
            ls = n.location_state
            rev = ls.customers * ls.price
            costs = ls.daily_fixed_costs + ls.customers * ls.food_cost_per_plate
            if rev > 0:
                margins.append((rev - costs) / rev)
    return sum(margins) / len(margins) if margins else 0.0


def _total_daily_revenue(engine: GrowthEngine) -> float:
    total = 0.0
    for n in engine.state.nodes.values():
        if n.type == NodeType.RESTAURANT and n.active and n.location_state:
            ls = n.location_state
            served = min(ls.customers, ls.max_capacity, ls.inventory)
            total += served * ls.price
    return total


def _avg_satisfaction(engine: GrowthEngine) -> float:
    sats = [
        n.location_state.satisfaction
        for n in engine.state.nodes.values()
        if n.type == NodeType.RESTAURANT and n.active and n.location_state
    ]
    return sum(sats) / len(sats) if sats else 0.0


# ── Trigger definitions ──

class GeneralManagerTrigger(GrowthTrigger):
    """Hire when any location's monthly revenue exceeds $25K."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.GENERAL_MANAGER,
            label="Hired General Manager",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _total_daily_revenue(engine) * 30 > 15_000


class BookkeeperTrigger(GrowthTrigger):
    """Hire when locations > 1 or monthly revenue > $65K."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.BOOKKEEPER,
            label="Hired Bookkeeper",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return (
            _location_count(engine) > 1
            or _total_daily_revenue(engine) * 30 > 25_000
        )


class MarketingTrigger(GrowthTrigger):
    """Hire when locations >= 3."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.MARKETING,
            label="Hired Marketing Manager",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 3


class HRTrigger(GrowthTrigger):
    """Hire when employees > 50 or locations >= 4."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.HR,
            label="Established HR Department",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return engine.state.total_employees > 50 or _location_count(engine) >= 4


class TrainingTrigger(GrowthTrigger):
    """Establish when locations >= 3."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.TRAINING,
            label="Established Training Program",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 3


class AreaManagerTrigger(GrowthTrigger):
    """Hire one per 6 locations."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.AREA_MANAGER,
            label="Hired Area Manager",
            max_instances=999,
            cooldown_ticks=30,
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        locs = _location_count(engine)
        current_ams = sum(
            1 for n in engine.state.nodes.values()
            if n.type == NodeType.AREA_MANAGER and n.active
        )
        return locs >= 6 and locs > current_ams * 6


class QualityAssuranceTrigger(GrowthTrigger):
    """Hire when locations >= 3."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.QUALITY_ASSURANCE,
            label="Hired Quality Assurance Manager",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 3


class ProcurementTrigger(GrowthTrigger):
    """Hire when locations >= 5."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.PROCUREMENT,
            label="Hired Procurement Manager",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 5


class ITSupportTrigger(GrowthTrigger):
    """Hire when locations >= 5."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.IT_SUPPORT,
            label="Hired IT Support",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 5


class RealEstateTrigger(GrowthTrigger):
    """Hire when opening 3+ locations per year."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.REAL_ESTATE,
            label="Established Real Estate Department",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return engine.state.locations_opened_this_year >= 3


class ConstructionTrigger(GrowthTrigger):
    """Hire when locations >= 8."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.CONSTRUCTION,
            label="Established Construction & Facilities",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 8


class RNDMenuTrigger(GrowthTrigger):
    """Hire when locations >= 10 and monthly revenue > $200K."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.RND_MENU,
            label="Established R&D / Menu Development",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return (
            _location_count(engine) >= 10
            and _total_daily_revenue(engine) * 30 > 200_000
        )


class LegalTrigger(GrowthTrigger):
    """Hire when locations >= 10."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.LEGAL,
            label="Established Legal Department",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 10


class FinanceFPATrigger(GrowthTrigger):
    """Hire when monthly revenue > $500K."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.FINANCE_FPA,
            label="Established Finance & FP&A",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _total_daily_revenue(engine) * 30 > 500_000


class CommissaryTrigger(GrowthTrigger):
    """Build when locations >= 5."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.COMMISSARY,
            label="Built Commissary Kitchen",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 5


class DistributionCenterTrigger(GrowthTrigger):
    """Build when locations >= 20."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.DISTRIBUTION_CENTER,
            label="Built Distribution Center",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 20


class CateringTrigger(GrowthTrigger):
    """Launch when locations >= 5 and avg satisfaction > 0.7."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.CATERING,
            label="Launched Catering Division",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 5 and _avg_satisfaction(engine) > 0.7


class DeliveryPartnershipTrigger(GrowthTrigger):
    """Partner when locations >= 3."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.DELIVERY_PARTNERSHIP,
            label="Established Delivery Partnerships",
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return _location_count(engine) >= 3


class NewLocationTrigger(GrowthTrigger):
    """Open a new restaurant when financially ready."""

    def __init__(self):
        super().__init__(
            node_type=NodeType.RESTAURANT,
            label="Opened new restaurant location",
            max_instances=999,
            cooldown_ticks=90,  # 3 months between openings
        )

    def should_fire(self, engine: GrowthEngine) -> bool:
        return (
            engine.state.cash > 80_000
            and _avg_location_margin(engine) >= 0.10
            and _has_node(engine, NodeType.GENERAL_MANAGER)
        )


# All triggers in evaluation order
TRIGGER_REGISTRY: list[GrowthTrigger] = [
    GeneralManagerTrigger(),
    BookkeeperTrigger(),
    NewLocationTrigger(),
    MarketingTrigger(),
    DeliveryPartnershipTrigger(),
    TrainingTrigger(),
    QualityAssuranceTrigger(),
    HRTrigger(),
    AreaManagerTrigger(),
    ProcurementTrigger(),
    ITSupportTrigger(),
    CommissaryTrigger(),
    CateringTrigger(),
    RealEstateTrigger(),
    ConstructionTrigger(),
    RNDMenuTrigger(),
    LegalTrigger(),
    FinanceFPATrigger(),
    DistributionCenterTrigger(),
]
