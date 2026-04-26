"""Validation tests for the universal node library.

The node library (`backend/src/simulation/node_library.yaml`) is the foundational
artifact for v2: industry-agnostic node taxonomy that the orchestrator spawns
from. These tests guard the schema and structural invariants:

  1. Library loads without error
  2. All required fields present on every entry
  3. Prerequisite graph is a DAG (no cycles)
  4. No prerequisite references a non-existent node_key
  5. `applicable_economics` values are from the allowed set
  6. `modifier_keys` values are floats in a reasonable range
  7. Categories are from the allowed set
  8. Auxiliary: snake_case keys; soft_cap <= hard_cap; non-negative costs
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import yaml

# ── Constants ──

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "src" / "simulation" / "node_library.yaml"

# Per V2 plan §1, the canonical category set is 8. We add `revenue` because the
# 7 source YAMLs all model revenue streams as first-class nodes (catering,
# retainers, marketplace, etc.), and they don't fit cleanly into the 8.
ALLOWED_CATEGORIES: set[str] = {
    "location", "supplier", "ops", "sales", "marketing",
    "finance", "exec", "external", "revenue",
}

ALLOWED_ECONOMICS: set[str] = {"physical", "subscription", "service"}

REQUIRED_FIELDS: tuple[str, ...] = (
    "category",
    "label",
    "hire_cost",
    "daily_fixed_costs",
    "employees_count",
    "capacity_contribution",
    "modifier_keys",
    "prerequisites",
    "category_caps",
    "applicable_economics",
)

REQUIRED_CAP_FIELDS: tuple[str, ...] = ("soft_cap", "hard_cap")

# Plausible bounds for modifier magnitudes. The existing YAMLs use values in
# roughly [-0.30, 0.40] for the same concepts; we widen to ±1.0 to allow
# headroom without permitting absurd values like 50.0.
MOD_KEY_MIN: float = -1.0
MOD_KEY_MAX: float = 1.0

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ── Fixtures ──


@pytest.fixture(scope="module")
def library() -> dict[str, dict[str, Any]]:
    """Load the node library once per module."""
    with LIBRARY_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "node_library.yaml must parse to a dict"
    return data


# ── Test 1: Loads without error ──


def test_library_loads_without_error() -> None:
    """The YAML file exists, parses, and is non-empty."""
    assert LIBRARY_PATH.exists(), f"Library not found at {LIBRARY_PATH}"
    with LIBRARY_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert len(data) > 0, "Library is empty"


# ── Test 2: All required fields present ──


def test_all_required_fields_present(library: dict[str, dict[str, Any]]) -> None:
    """Every entry has the full schema."""
    missing: list[tuple[str, str]] = []
    for key, entry in library.items():
        if not isinstance(entry, dict):
            pytest.fail(f"Entry '{key}' is not a dict; got {type(entry).__name__}")
        for field in REQUIRED_FIELDS:
            if field not in entry:
                missing.append((key, field))
        caps = entry.get("category_caps", {})
        if isinstance(caps, dict):
            for cap_field in REQUIRED_CAP_FIELDS:
                if cap_field not in caps:
                    missing.append((key, f"category_caps.{cap_field}"))
    assert not missing, f"Missing required fields: {missing}"


# ── Test 3: Prerequisite graph is a DAG ──


def test_prerequisite_graph_is_acyclic(library: dict[str, dict[str, Any]]) -> None:
    """No cycles in the prerequisite chain (orchestrator must be able to
    topologically order spawns)."""
    g: nx.DiGraph[str] = nx.DiGraph()
    for key, entry in library.items():
        g.add_node(key)
        for prereq in entry.get("prerequisites", []):
            # Edge: prereq -> dependent. Cycle here means a node depends on itself
            # (transitively).
            g.add_edge(prereq, key)
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        pytest.fail(f"Prerequisite graph contains cycles: {cycles}")


# ── Test 4: All prerequisites exist ──


def test_prerequisites_reference_existing_nodes(library: dict[str, dict[str, Any]]) -> None:
    """Every prereq must be a node_key in the library."""
    keys = set(library.keys())
    invalid: list[tuple[str, str]] = []
    for key, entry in library.items():
        for prereq in entry.get("prerequisites", []):
            if prereq not in keys:
                invalid.append((key, prereq))
    assert not invalid, f"Prerequisites referencing missing nodes: {invalid}"


# ── Test 5: applicable_economics values are valid ──


def test_applicable_economics_valid(library: dict[str, dict[str, Any]]) -> None:
    """Each entry's economics list must be a non-empty subset of the allowed set."""
    bad: list[tuple[str, list[Any]]] = []
    for key, entry in library.items():
        econ = entry.get("applicable_economics", [])
        if not isinstance(econ, list) or not econ:
            bad.append((key, econ))
            continue
        for value in econ:
            if value not in ALLOWED_ECONOMICS:
                bad.append((key, econ))
                break
    assert not bad, (
        f"Invalid applicable_economics (must be non-empty subset of "
        f"{sorted(ALLOWED_ECONOMICS)}): {bad}"
    )


# ── Test 6: modifier_keys values are floats in a reasonable range ──


def test_modifier_keys_are_floats_in_range(library: dict[str, dict[str, Any]]) -> None:
    """modifier_keys must be {str: number} with values in [MOD_KEY_MIN, MOD_KEY_MAX]."""
    out_of_range: list[tuple[str, str, Any]] = []
    bad_type: list[tuple[str, str, Any]] = []
    for key, entry in library.items():
        mods = entry.get("modifier_keys", {})
        if not isinstance(mods, dict):
            pytest.fail(f"'{key}'.modifier_keys must be a dict; got {type(mods).__name__}")
        for mod_key, mod_val in mods.items():
            if not isinstance(mod_key, str) or not mod_key:
                bad_type.append((key, str(mod_key), mod_val))
                continue
            if not isinstance(mod_val, (int, float)) or isinstance(mod_val, bool):
                bad_type.append((key, mod_key, mod_val))
                continue
            if not (MOD_KEY_MIN <= float(mod_val) <= MOD_KEY_MAX):
                out_of_range.append((key, mod_key, mod_val))
    assert not bad_type, f"modifier_keys with invalid types: {bad_type}"
    assert not out_of_range, (
        f"modifier_keys outside [{MOD_KEY_MIN}, {MOD_KEY_MAX}]: {out_of_range}"
    )


# ── Test 7: categories from allowed set ──


def test_categories_from_allowed_set(library: dict[str, dict[str, Any]]) -> None:
    bad: list[tuple[str, str]] = []
    for key, entry in library.items():
        cat = entry.get("category", "")
        if cat not in ALLOWED_CATEGORIES:
            bad.append((key, cat))
    assert not bad, (
        f"Invalid categories (must be in {sorted(ALLOWED_CATEGORIES)}): {bad}"
    )


# ── Auxiliary integrity checks ──


def test_keys_are_snake_case(library: dict[str, dict[str, Any]]) -> None:
    """All node_keys are snake_case for consistent referencing."""
    bad = [k for k in library.keys() if not SNAKE_CASE_RE.match(k)]
    assert not bad, f"Non-snake_case keys: {bad}"


def test_caps_consistent(library: dict[str, dict[str, Any]]) -> None:
    """soft_cap <= hard_cap and both >= 1."""
    bad: list[tuple[str, int, int]] = []
    for key, entry in library.items():
        caps = entry.get("category_caps", {})
        soft = caps.get("soft_cap")
        hard = caps.get("hard_cap")
        if not isinstance(soft, int) or not isinstance(hard, int):
            bad.append((key, soft, hard))
            continue
        if soft < 1 or hard < 1 or soft > hard:
            bad.append((key, soft, hard))
    assert not bad, f"Inconsistent category_caps (need 1 <= soft_cap <= hard_cap): {bad}"


def test_costs_are_non_negative(library: dict[str, dict[str, Any]]) -> None:
    """hire_cost, daily_fixed_costs, employees_count, capacity_contribution >= 0."""
    bad: list[tuple[str, str, Any]] = []
    for key, entry in library.items():
        for field in (
            "hire_cost", "daily_fixed_costs", "employees_count", "capacity_contribution",
        ):
            value = entry.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                bad.append((key, field, value))
                continue
            if value < 0:
                bad.append((key, field, value))
    assert not bad, f"Negative or non-numeric cost fields: {bad}"


def test_at_least_one_location_node(library: dict[str, dict[str, Any]]) -> None:
    """The library must contain at least one node per category that's used by
    seeds (location is the most critical — companies can't run without one)."""
    cats_present = {entry.get("category") for entry in library.values()}
    assert "location" in cats_present, "Library must contain at least one location node"


def test_modifier_keys_non_empty_for_revenue_and_cost_modifiers(
    library: dict[str, dict[str, Any]],
) -> None:
    """Sanity: at least 5 nodes register *some* modifier (otherwise the auto-
    derived bridge would always be empty — defeats the v2 purpose)."""
    nodes_with_mods = sum(1 for e in library.values() if e.get("modifier_keys"))
    assert nodes_with_mods >= 5, (
        f"Only {nodes_with_mods} nodes register modifier keys; "
        "expected the library to include broad modifier coverage so the "
        "bridge has something to compute."
    )
