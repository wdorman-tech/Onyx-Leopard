"""Universal node library loader for Onyx Leopard v2.

Replaces v1's per-industry `config_loader.py`. v2 has exactly one library:
`backend/src/simulation/node_library.yaml`, an industry-agnostic catalog of
every node type the engine knows. The CEO orchestrator picks from this
catalog when it spawns nodes; what changes between simulations is not the
library but the `CompanySeed` and `CeoStance` driving spawns.

Per V2 plan §1, the loader hard-fails on any structural defect — schema
violation, prerequisite cycle, dangling prereq reference, unknown category,
unknown economics tag — by raising `LibraryValidationError` at load time.
A half-validated library is never returned.

Cross-validation against `CompanySeed` (the seed's `initial_supplier_types`
etc. must reference real library keys whose `applicable_economics` includes
the seed's `economics_model`) lives here rather than in `seed.py` to avoid a
circular import — `seed.py` is the contract; this module is the enforcer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import networkx as nx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.simulation.seed import CompanySeed

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LIBRARY_PATH: Final[Path] = Path(__file__).resolve().parent / "node_library.yaml"

# Allowed categories. Per V2 plan §1 the canonical set is 8; `revenue` is
# added because the source YAMLs all model revenue streams as first-class
# nodes (catering, retainers, marketplace, etc.). Kept consistent with
# `tests/test_node_library.py::ALLOWED_CATEGORIES`.
ALLOWED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "location",
        "supplier",
        "ops",
        "sales",
        "marketing",
        "finance",
        "exec",
        "external",
        "revenue",
    }
)

ALLOWED_ECONOMICS: Final[frozenset[str]] = frozenset({"physical", "subscription", "service"})


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class LibraryValidationError(Exception):
    """Raised when `node_library.yaml` fails any structural invariant.

    Covers: missing file, malformed YAML, Pydantic schema violation, unknown
    category, unknown economics tag, dangling prerequisite reference, cycle
    in the prerequisite graph, duplicate node key.

    Always raised at load time so the engine never sees a half-validated
    library (V2 plan: hard-fail validation).
    """


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


class CategoryCaps(BaseModel):
    """Per-node spawn caps. soft_cap = diminishing returns; hard_cap = ceiling."""

    model_config = ConfigDict(extra="forbid")

    soft_cap: int = Field(..., ge=1)
    hard_cap: int = Field(..., ge=1)


class NodeDef(BaseModel):
    """One entry in the universal node library.

    `key` is added by the loader (the YAML stores it as the dict key, not as
    a field on the entry). The remaining fields mirror the YAML schema 1:1.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    category: str
    label: str = Field(..., min_length=1)
    #: Optional human-readable note (typical deal sizes, sourcing rationale).
    #: Industry-agnostic engine never reads this; surfaced in tooltips and
    #: in the orchestrator prompt as supplementary context.
    commentary: str = Field(default="")
    hire_cost: float = Field(..., ge=0.0)
    daily_fixed_costs: float = Field(..., ge=0.0)
    employees_count: int = Field(..., ge=0)
    capacity_contribution: int = Field(..., ge=0)
    modifier_keys: dict[str, float] = Field(default_factory=dict)
    prerequisites: list[str] = Field(default_factory=list)
    category_caps: CategoryCaps
    applicable_economics: list[str] = Field(..., min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Library
# ─────────────────────────────────────────────────────────────────────────────


class NodeLibrary:
    """In-memory, validated view of the universal node library.

    Constructed only via `load_library()` — direct instantiation is allowed
    but skips validation, so it should be reserved for tests that construct
    a library from already-validated `NodeDef` objects.
    """

    __slots__ = ("_by_category", "nodes")

    def __init__(self, nodes: dict[str, NodeDef]) -> None:
        self.nodes: dict[str, NodeDef] = nodes
        # Pre-bucket by category — bridge / orchestrator iterate this often.
        buckets: dict[str, list[NodeDef]] = {}
        for node in nodes.values():
            buckets.setdefault(node.category, []).append(node)
        self._by_category: dict[str, list[NodeDef]] = buckets

    # ── Lookup ──

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, key: str) -> NodeDef:
        return self.nodes[key]

    def get(self, key: str) -> NodeDef | None:
        return self.nodes.get(key)

    def by_category(self) -> dict[str, list[NodeDef]]:
        """Return a fresh shallow-copy mapping of category → nodes.

        Returned dict is safe to mutate (it's a new dict each call); the
        inner lists are also fresh copies so callers can sort/filter without
        disturbing internal state.
        """
        return {cat: list(nodes) for cat, nodes in self._by_category.items()}

    # ── Spawn-time checks ──

    def prerequisites_satisfied(self, spawned: set[str], target_key: str) -> bool:
        """True iff every prereq of `target_key` is in `spawned`.

        Returns False if `target_key` is not in the library — the orchestrator
        should never ask about an unknown node, and silently returning True
        would let invalid spawns through.
        """
        node = self.nodes.get(target_key)
        if node is None:
            return False
        return all(prereq in spawned for prereq in node.prerequisites)

    # ── Seed cross-validation ──

    def validate_seed(self, seed: CompanySeed) -> None:
        """Validate a `CompanySeed` against this library.

        Raises `LibraryValidationError` if any of the seed's reference lists
        (`initial_supplier_types`, `initial_revenue_streams`,
        `initial_cost_centers`) names a node_key not in the library, or if
        any referenced node's `applicable_economics` does not include the
        seed's `economics_model`.
        """
        errors: list[str] = []
        ref_lists: tuple[tuple[str, list[str]], ...] = (
            ("initial_supplier_types", seed.initial_supplier_types),
            ("initial_revenue_streams", seed.initial_revenue_streams),
            ("initial_cost_centers", seed.initial_cost_centers),
        )
        for field_name, refs in ref_lists:
            for ref in refs:
                node = self.nodes.get(ref)
                if node is None:
                    errors.append(
                        f"{field_name} references unknown node key '{ref}' "
                        "(not present in node_library.yaml)"
                    )
                    continue
                if seed.economics_model not in node.applicable_economics:
                    errors.append(
                        f"{field_name} entry '{ref}' is not applicable to "
                        f"economics_model='{seed.economics_model}' "
                        f"(node supports: {node.applicable_economics})"
                    )
        if errors:
            raise LibraryValidationError(
                "CompanySeed failed library cross-validation:\n  - "
                + "\n  - ".join(errors)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def _read_yaml_dict(path: Path) -> dict[str, dict]:
    """Read `path` as a YAML mapping of node_key → node fields."""
    if not path.exists():
        raise LibraryValidationError(f"node library not found at {path}")
    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise LibraryValidationError(f"node library is not valid YAML: {exc}") from exc
    if raw is None:
        raise LibraryValidationError(f"node library at {path} is empty")
    if not isinstance(raw, dict):
        raise LibraryValidationError(
            f"node library must parse to a mapping; got {type(raw).__name__}"
        )
    # Reject duplicate / non-string keys defensively. PyYAML deduplicates
    # silently (last wins), so we also flag any non-string keys here.
    bad_keys = [repr(k) for k in raw if not isinstance(k, str) or not k]
    if bad_keys:
        raise LibraryValidationError(
            f"node library has non-string or empty keys: {bad_keys}"
        )
    return raw


def _build_node_defs(raw: dict[str, dict]) -> dict[str, NodeDef]:
    """Build `NodeDef` objects from raw YAML, collecting all schema errors."""
    seen_keys: set[str] = set()
    nodes: dict[str, NodeDef] = {}
    schema_errors: list[str] = []

    for key, fields in raw.items():
        if key in seen_keys:
            schema_errors.append(f"duplicate node key '{key}'")
            continue
        seen_keys.add(key)

        if not isinstance(fields, dict):
            schema_errors.append(
                f"node '{key}' must be a mapping; got {type(fields).__name__}"
            )
            continue

        # Deep-copy to avoid mutating the YAML payload, then inject the key.
        payload = dict(fields)
        payload["key"] = key
        try:
            nodes[key] = NodeDef(**payload)
        except ValidationError as exc:
            schema_errors.append(f"node '{key}' failed schema validation: {exc}")

    if schema_errors:
        raise LibraryValidationError(
            "node library schema errors:\n  - " + "\n  - ".join(schema_errors)
        )
    return nodes


def _validate_categories(nodes: dict[str, NodeDef]) -> None:
    bad = [(k, n.category) for k, n in nodes.items() if n.category not in ALLOWED_CATEGORIES]
    if bad:
        raise LibraryValidationError(
            f"unknown categories (allowed: {sorted(ALLOWED_CATEGORIES)}): {bad}"
        )


def _validate_economics(nodes: dict[str, NodeDef]) -> None:
    bad: list[tuple[str, list[str]]] = []
    for key, node in nodes.items():
        invalid = [e for e in node.applicable_economics if e not in ALLOWED_ECONOMICS]
        if invalid:
            bad.append((key, invalid))
    if bad:
        raise LibraryValidationError(
            f"unknown applicable_economics values (allowed: "
            f"{sorted(ALLOWED_ECONOMICS)}): {bad}"
        )


def _validate_caps(nodes: dict[str, NodeDef]) -> None:
    bad = [
        (k, n.category_caps.soft_cap, n.category_caps.hard_cap)
        for k, n in nodes.items()
        if n.category_caps.soft_cap > n.category_caps.hard_cap
    ]
    if bad:
        raise LibraryValidationError(
            f"category_caps with soft_cap > hard_cap (need soft <= hard): {bad}"
        )


def _validate_prerequisites(nodes: dict[str, NodeDef]) -> None:
    """Check (a) prereqs reference existing keys, (b) the graph is a DAG."""
    keys = set(nodes.keys())
    dangling: list[tuple[str, str]] = []
    for key, node in nodes.items():
        for prereq in node.prerequisites:
            if prereq not in keys:
                dangling.append((key, prereq))
    if dangling:
        raise LibraryValidationError(
            f"prerequisites reference nonexistent nodes: {dangling}"
        )

    g: nx.DiGraph = nx.DiGraph()
    for key, node in nodes.items():
        g.add_node(key)
        for prereq in node.prerequisites:
            # Edge prereq -> dependent. A cycle = the orchestrator could
            # never reach a valid spawn order.
            g.add_edge(prereq, key)
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise LibraryValidationError(
            f"prerequisite graph contains cycles: {cycles}"
        )


def load_library(path: Path | None = None) -> NodeLibrary:
    """Load and fully validate the node library.

    Defaults to `backend/src/simulation/node_library.yaml`. On any structural
    defect, raises `LibraryValidationError` — never returns a partially
    validated library.
    """
    target = path or DEFAULT_LIBRARY_PATH
    raw = _read_yaml_dict(target)
    nodes = _build_node_defs(raw)
    _validate_categories(nodes)
    _validate_economics(nodes)
    _validate_caps(nodes)
    _validate_prerequisites(nodes)
    return NodeLibrary(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton cache
# ─────────────────────────────────────────────────────────────────────────────

_cached_library: NodeLibrary | None = None


def get_library() -> NodeLibrary:
    """Return the process-wide cached library, loading once on first call."""
    global _cached_library
    if _cached_library is None:
        _cached_library = load_library()
    return _cached_library


def _reset_library_cache() -> None:
    """Clear the cached library. Test-only hook."""
    global _cached_library
    _cached_library = None
