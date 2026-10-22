"""Universal node library: loader, validation, and modifier buckets.

`node_library.yaml` is the single, industry-agnostic catalog of every node a
company can spawn. The loader hard-fails on any structural defect (schema,
unknown category/economics, dangling prerequisite, prerequisite cycle) so the
engine never sees a half-valid library.

Modifier buckets: each node's `modifier_keys` are classified by name suffix
into the buckets the engine reads. A key K lands in bucket B when K equals one
of B's tags or ends with "_<tag>"; the first bucket in `BUCKET_TAGS` order
wins. Keys matching no tag are ignored by the engine.

    quality        satisfaction / retention / churn_reduction / quality
    marketing      pipeline_strength / lead_gen / awareness / marketing / brand
    infrastructure capacity_uplift / infrastructure / throughput / efficiency
    revenue        revenue_boost / monetization / expansion / upsell / premium / revenue
    capital        runway_extension / runway / funding / capital
    cost           cost_reduction / overhead / savings / waste / cost
                   (signed: negative = savings, positive = added cost)
"""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.simulation.seed import CompanySeed

DEFAULT_LIBRARY_PATH: Final[Path] = Path(__file__).resolve().parent / "node_library.yaml"

CATEGORIES: Final[frozenset[str]] = frozenset(
    {"location", "supplier", "ops", "sales", "marketing", "finance", "exec", "external", "revenue"}
)
ECONOMICS: Final[frozenset[str]] = frozenset({"physical", "subscription", "service"})
DAYS_PER_MONTH: Final[float] = 30.0

#: Bucket → tags, in priority order. Multi-token tags precede their substrings.
BUCKET_TAGS: Final[dict[str, tuple[str, ...]]] = {
    "quality": ("churn_reduction", "satisfaction", "retention", "quality"),
    "marketing": ("pipeline_strength", "lead_gen", "awareness", "marketing", "brand"),
    "infrastructure": ("capacity_uplift", "infrastructure", "throughput", "efficiency"),
    "revenue": ("revenue_boost", "monetization", "expansion", "upsell", "premium", "revenue"),
    "capital": ("runway_extension", "runway", "funding", "capital"),
    "cost": ("cost_reduction", "overhead", "savings", "waste", "cost"),
}


class LibraryValidationError(Exception):
    """`node_library.yaml` (or a seed checked against it) is structurally invalid."""


class CategoryCaps(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_cap: int = Field(..., ge=1)
    hard_cap: int = Field(..., ge=1)


class NodeDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    category: str
    label: str = Field(..., min_length=1)
    commentary: str = ""
    hire_cost: float = Field(..., ge=0.0)
    daily_fixed_costs: float = Field(..., ge=0.0)
    employees_count: int = Field(..., ge=0)
    capacity_contribution: int = Field(..., ge=0)
    modifier_keys: dict[str, float] = Field(default_factory=dict)
    prerequisites: list[str] = Field(default_factory=list)
    category_caps: CategoryCaps
    applicable_economics: list[str] = Field(..., min_length=1)


@dataclass(frozen=True)
class Modifiers:
    """Summed modifier magnitudes per bucket for a company's spawned set."""

    quality: float = 0.0
    marketing: float = 0.0
    infrastructure: float = 0.0
    revenue: float = 0.0
    capital: float = 0.0
    cost: float = 0.0


def classify_modifier(key: str) -> str | None:
    for bucket, tags in BUCKET_TAGS.items():
        if any(key == tag or key.endswith("_" + tag) for tag in tags):
            return bucket
    return None


class NodeLibrary:
    """Validated, read-only view of the node library. Shared across sims."""

    def __init__(self, nodes: dict[str, NodeDef]) -> None:
        self.nodes = nodes
        # Classify every modifier key once so per-tick aggregation is a sum.
        self._buckets: dict[str, dict[str, float]] = {}
        for key, node in nodes.items():
            sums: dict[str, float] = {}
            for mod_key, value in node.modifier_keys.items():
                bucket = classify_modifier(mod_key)
                if bucket is not None:
                    sums[bucket] = sums.get(bucket, 0.0) + value
            self._buckets[key] = sums

    def __contains__(self, key: object) -> bool:
        return key in self.nodes

    def get(self, key: str) -> NodeDef | None:
        return self.nodes.get(key)

    def effects(self, key: str) -> dict[str, float]:
        """Per-bucket modifier contribution of one instance of `key`."""
        return dict(self._buckets[key])

    def modifiers(self, spawned: dict[str, int]) -> Modifiers:
        totals: dict[str, float] = {}
        for key, count in spawned.items():
            for bucket, value in self._buckets[key].items():
                totals[bucket] = totals.get(bucket, 0.0) + value * count
        return Modifiers(**totals)

    def prerequisites_met(self, key: str, spawned: dict[str, int]) -> bool:
        node = self.nodes.get(key)
        return node is not None and all(spawned.get(p, 0) > 0 for p in node.prerequisites)

    def primary_location(self, economics_model: str) -> str | None:
        """Cheapest capacity-bearing, prerequisite-free, paid location for an economics."""
        candidates = [
            (n.hire_cost + DAYS_PER_MONTH * n.daily_fixed_costs, key)
            for key, n in self.nodes.items()
            if n.category == "location"
            and n.capacity_contribution > 0
            and n.hire_cost > 0
            and not n.prerequisites
            and economics_model in n.applicable_economics
        ]
        return min(candidates)[1] if candidates else None

    def founder(self) -> str | None:
        for key in ("founder", "founder_ceo", "ceo"):
            if key in self.nodes:
                return key
        for key, node in sorted(self.nodes.items()):
            if node.category == "exec" and not node.prerequisites:
                return key
        return None

    def starting_nodes(
        self, refs: Iterable[str], economics_model: str, locations: int
    ) -> dict[str, int]:
        """Day-0 node set: seed references + founder + `locations` primary locations."""
        spawned: dict[str, int] = {}
        keys = [*refs, self.founder()] + [self.primary_location(economics_model)] * locations
        for key in keys:
            if key is not None:
                spawned[key] = spawned.get(key, 0) + 1
        return spawned

    def daily_cost(self, spawned: dict[str, int]) -> float:
        return sum(self.nodes[k].daily_fixed_costs * c for k, c in spawned.items())

    def validate_seed(self, seed: CompanySeed) -> None:
        errors: list[str] = []
        for field in ("initial_supplier_types", "initial_revenue_streams", "initial_cost_centers"):
            for ref in getattr(seed, field):
                node = self.nodes.get(ref)
                if node is None:
                    errors.append(f"{field}: unknown node '{ref}'")
                elif seed.economics_model not in node.applicable_economics:
                    errors.append(
                        f"{field}: '{ref}' does not apply to economics_model "
                        f"'{seed.economics_model}' (supports {node.applicable_economics})"
                    )
        if errors:
            raise LibraryValidationError("seed failed library validation: " + "; ".join(errors))


def load_library(path: Path = DEFAULT_LIBRARY_PATH) -> NodeLibrary:
    if not path.exists():
        raise LibraryValidationError(f"node library not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LibraryValidationError(f"node library is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise LibraryValidationError("node library must be a non-empty mapping of key → node")

    nodes: dict[str, NodeDef] = {}
    errors: list[str] = []
    for key, fields in raw.items():
        try:
            nodes[key] = NodeDef(key=key, **fields)
        except (TypeError, ValidationError) as exc:
            errors.append(f"node '{key}': {exc}")
            continue
        node = nodes[key]
        if node.category not in CATEGORIES:
            errors.append(f"node '{key}': unknown category '{node.category}'")
        bad_econ = set(node.applicable_economics) - ECONOMICS
        if bad_econ:
            errors.append(f"node '{key}': unknown economics {sorted(bad_econ)}")
        if node.category_caps.soft_cap > node.category_caps.hard_cap:
            errors.append(f"node '{key}': soft_cap > hard_cap")
        missing = [p for p in node.prerequisites if p not in raw]
        if missing:
            errors.append(f"node '{key}': unknown prerequisites {missing}")
    if errors:
        raise LibraryValidationError("node library invalid:\n  - " + "\n  - ".join(errors))

    try:
        TopologicalSorter({k: n.prerequisites for k, n in nodes.items()}).prepare()
    except CycleError as exc:
        raise LibraryValidationError(f"prerequisite cycle: {exc.args[1]}") from exc
    return NodeLibrary(nodes)


_library: NodeLibrary | None = None


def get_library() -> NodeLibrary:
    """Process-wide library, loaded once. Read-only, so safe to share across sims."""
    global _library
    if _library is None:
        _library = load_library()
    return _library
