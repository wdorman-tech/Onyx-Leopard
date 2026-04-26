"""Tests for `library_loader.py` (Onyx Leopard v2).

Validates the load-time invariants of the universal node library and the
cross-validation contract between `CompanySeed` and `NodeLibrary`. Per V2
plan, every structural defect is a hard load-time failure — not a warning.

Real `node_library.yaml` is exercised via `load_library()` for the happy
path. Adversarial cases (cycles, dangling prereqs, bad categories) are
constructed in `tmp_path` so we never mutate the real library on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.simulation.library_loader import (
    DEFAULT_LIBRARY_PATH,
    LibraryValidationError,
    NodeDef,
    NodeLibrary,
    _reset_library_cache,
    get_library,
    load_library,
)
from src.simulation.seed import CompanySeed

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — minimal valid node entry & seed builders
# ─────────────────────────────────────────────────────────────────────────────


_UNSET = object()


def _node_entry(
    *,
    category: str = "ops",
    label: str = "Test Node",
    prerequisites: list[str] | None = None,
    applicable_economics: Any = _UNSET,
    modifier_keys: dict[str, float] | None = None,
    soft_cap: int = 1,
    hard_cap: int = 1,
) -> dict[str, Any]:
    # Sentinel-default for applicable_economics so callers can pass [] to
    # exercise the empty-list rejection path (a `None` default with `or`
    # would silently turn [] into the default, masking the test).
    if applicable_economics is _UNSET:
        applicable_economics = ["subscription"]
    return {
        "category": category,
        "label": label,
        "hire_cost": 0.0,
        "daily_fixed_costs": 0.0,
        "employees_count": 0,
        "capacity_contribution": 0,
        "modifier_keys": modifier_keys or {},
        "prerequisites": prerequisites or [],
        "category_caps": {"soft_cap": soft_cap, "hard_cap": hard_cap},
        "applicable_economics": applicable_economics,
    }


def _write_yaml(path: Path, data: dict[str, dict[str, Any]]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _subscription_seed_kwargs() -> dict[str, Any]:
    """Subscription-economics seed using node keys that exist in the real library."""
    return {
        "name": "Test SaaS Co.",
        "niche": "B2B SaaS for unit tests",
        "archetype": "venture_funded",
        "industry_keywords": ["saas", "test"],
        "location_label": "Product",
        "economics_model": "subscription",
        "starting_price": 99.0,
        "base_unit_cost": 12.0,
        "daily_fixed_costs": 1500.0,
        "starting_cash": 2_000_000.0,
        "starting_employees": 8,
        "base_capacity_per_location": 500,
        "margin_target": 0.7,
        "revenue_per_employee_target": 250_000.0,
        "tam": 5e9,
        "competitor_density": 5,
        "market_growth_rate": 0.20,
        "customer_unit_label": "subscribers",
        "seasonality_amplitude": 0.05,
        "initial_supplier_types": ["cloud_provider", "payment_processor"],
        "initial_revenue_streams": ["subscription_revenue"],
        "initial_cost_centers": ["it_support"],
        "initial_locations": 1,
        "initial_marketing_intensity": 0.4,
        "initial_quality_target": 0.8,
        "initial_price_position": "mid",
        "initial_capital_runway_months": 18.0,
        "initial_hiring_pace": "steady",
        "initial_geographic_scope": "national",
        "initial_revenue_concentration": 0.3,
        "initial_customer_acquisition_channel": "content",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_singleton() -> None:
    """Each test starts with no cached library (the test fixture is also
    cleared after, to avoid leaking state into unrelated suites)."""
    _reset_library_cache()
    yield
    _reset_library_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Real library — happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_load_real_library_succeeds_with_122_nodes() -> None:
    """Real `node_library.yaml` loads, validates, and exposes 122 nodes."""
    lib = load_library()
    assert isinstance(lib, NodeLibrary)
    assert len(lib) == 122, f"expected 122 nodes in library, got {len(lib)}"
    # Every entry should be a `NodeDef` with its `key` populated from the
    # YAML mapping key (NodeDef itself wouldn't store the key without us
    # injecting it).
    for key, node in lib.nodes.items():
        assert isinstance(node, NodeDef)
        assert node.key == key


def test_default_library_path_is_node_library_yaml() -> None:
    assert DEFAULT_LIBRARY_PATH.name == "node_library.yaml"
    assert DEFAULT_LIBRARY_PATH.exists()


def test_by_category_buckets_real_library() -> None:
    lib = load_library()
    buckets = lib.by_category()
    # Every node ends up in exactly one bucket; bucket count >= the size of
    # ALLOWED_CATEGORIES intersected with present categories.
    assert sum(len(v) for v in buckets.values()) == len(lib)
    assert "location" in buckets and len(buckets["location"]) >= 1
    # Mutating the returned dict must not affect the library.
    buckets["location"].clear()
    assert len(lib.by_category()["location"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Seed cross-validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_seed_accepts_seed_with_real_node_keys() -> None:
    lib = load_library()
    seed = CompanySeed(**_subscription_seed_kwargs())
    # Should not raise.
    lib.validate_seed(seed)


def test_validate_seed_rejects_unknown_supplier_reference() -> None:
    lib = load_library()
    kwargs = _subscription_seed_kwargs()
    kwargs["initial_supplier_types"] = ["definitely_not_a_real_node"]
    seed = CompanySeed(**kwargs)
    with pytest.raises(LibraryValidationError, match="initial_supplier_types"):
        lib.validate_seed(seed)


def test_validate_seed_rejects_economics_mismatch() -> None:
    """Seed asks for `subscription` economics but cites a supplier that's
    only `physical` — must be rejected."""
    lib = load_library()
    kwargs = _subscription_seed_kwargs()
    # `chicken_supplier` is a real library node but applicable_economics is
    # ["physical"] only — it cannot be a supplier for a subscription seed.
    kwargs["initial_supplier_types"] = ["chicken_supplier"]
    seed = CompanySeed(**kwargs)
    with pytest.raises(LibraryValidationError, match="not applicable to economics_model"):
        lib.validate_seed(seed)


def test_validate_seed_collects_all_errors_in_one_message() -> None:
    """The error message should enumerate every violation in one shot —
    callers shouldn't have to re-run to find the next error."""
    lib = load_library()
    kwargs = _subscription_seed_kwargs()
    kwargs["initial_supplier_types"] = ["nope_one", "nope_two"]
    seed = CompanySeed(**kwargs)
    with pytest.raises(LibraryValidationError) as excinfo:
        lib.validate_seed(seed)
    assert "nope_one" in str(excinfo.value)
    assert "nope_two" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial libraries (constructed in tmp_path)
# ─────────────────────────────────────────────────────────────────────────────


def test_cycle_in_prerequisite_graph_raises(tmp_path: Path) -> None:
    """a -> b -> c -> a is rejected at load time."""
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {
            "a": _node_entry(prerequisites=["c"]),
            "b": _node_entry(prerequisites=["a"]),
            "c": _node_entry(prerequisites=["b"]),
        },
    )
    with pytest.raises(LibraryValidationError, match="cycle"):
        load_library(lib_path)


def test_self_loop_is_a_cycle(tmp_path: Path) -> None:
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(prerequisites=["a"])},
    )
    with pytest.raises(LibraryValidationError, match="cycle"):
        load_library(lib_path)


def test_missing_prerequisite_raises(tmp_path: Path) -> None:
    """`a -> nonexistent` is rejected."""
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(prerequisites=["nonexistent"])},
    )
    with pytest.raises(LibraryValidationError, match="nonexistent"):
        load_library(lib_path)


def test_bad_category_raises(tmp_path: Path) -> None:
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(category="foo")},
    )
    with pytest.raises(LibraryValidationError, match=r"(?i)categor"):
        load_library(lib_path)


def test_bad_economics_value_raises(tmp_path: Path) -> None:
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(applicable_economics=["mythical"])},
    )
    with pytest.raises(LibraryValidationError, match=r"(?i)economics"):
        load_library(lib_path)


def test_empty_economics_list_raises_via_schema(tmp_path: Path) -> None:
    """`applicable_economics: []` violates the Pydantic min_length=1 constraint."""
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(applicable_economics=[])},
    )
    with pytest.raises(LibraryValidationError, match="schema"):
        load_library(lib_path)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    payload = _node_entry()
    payload.pop("hire_cost")
    lib_path = _write_yaml(tmp_path / "lib.yaml", {"a": payload})
    with pytest.raises(LibraryValidationError, match="schema"):
        load_library(lib_path)


def test_extra_field_raises(tmp_path: Path) -> None:
    """`extra='forbid'` — a typo'd field should not be silently accepted."""
    payload = _node_entry()
    payload["unexpected_field"] = 42
    lib_path = _write_yaml(tmp_path / "lib.yaml", {"a": payload})
    with pytest.raises(LibraryValidationError, match="schema"):
        load_library(lib_path)


def test_soft_cap_greater_than_hard_cap_raises(tmp_path: Path) -> None:
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {"a": _node_entry(soft_cap=10, hard_cap=2)},
    )
    with pytest.raises(LibraryValidationError, match=r"(?i)cap"):
        load_library(lib_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(LibraryValidationError, match=r"(?i)not found"):
        load_library(tmp_path / "does_not_exist.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(LibraryValidationError, match=r"(?i)empty"):
        load_library(p)


def test_non_dict_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(LibraryValidationError, match=r"(?i)mapping"):
        load_library(p)


# ─────────────────────────────────────────────────────────────────────────────
# Spawn-time helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_prerequisites_satisfied_no_prereqs_returns_true(tmp_path: Path) -> None:
    """Empty `spawned` set + node with no prereqs → True."""
    lib_path = _write_yaml(tmp_path / "lib.yaml", {"a": _node_entry()})
    lib = load_library(lib_path)
    assert lib.prerequisites_satisfied(set(), "a") is True


def test_prerequisites_satisfied_missing_prereq_returns_false(tmp_path: Path) -> None:
    lib_path = _write_yaml(
        tmp_path / "lib.yaml",
        {
            "root": _node_entry(),
            "child": _node_entry(prerequisites=["root"]),
        },
    )
    lib = load_library(lib_path)
    assert lib.prerequisites_satisfied(set(), "child") is False
    assert lib.prerequisites_satisfied({"root"}, "child") is True


def test_prerequisites_satisfied_unknown_target_returns_false(tmp_path: Path) -> None:
    """Asking about a node not in the library is itself a bug — fail closed."""
    lib_path = _write_yaml(tmp_path / "lib.yaml", {"a": _node_entry()})
    lib = load_library(lib_path)
    assert lib.prerequisites_satisfied({"a"}, "ghost") is False


# ─────────────────────────────────────────────────────────────────────────────
# Singleton cache
# ─────────────────────────────────────────────────────────────────────────────


def test_get_library_returns_memoized_singleton() -> None:
    a = get_library()
    b = get_library()
    assert a is b


def test_reset_cache_forces_reload() -> None:
    a = get_library()
    _reset_library_cache()
    b = get_library()
    assert a is not b
    # But the contents are still equivalent.
    assert len(a) == len(b)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5 P-MAG-6 — seed defaults must reference real library node_keys
# ─────────────────────────────────────────────────────────────────────────────


def test_seed_default_refs_all_exist_in_production_library() -> None:
    """Renaming a node in node_library.yaml without updating
    `_DEFAULT_SUPPLIERS / _DEFAULT_REVENUE_STREAMS / _DEFAULT_COST_CENTERS`
    in seed.py used to fail silently at sample time. This test makes that
    drift loud at CI time.
    """
    from src.simulation.seed import (
        _DEFAULT_COST_CENTERS,
        _DEFAULT_REVENUE_STREAMS,
        _DEFAULT_SUPPLIERS,
    )

    _reset_library_cache()
    lib = get_library()
    library_keys = set(lib.nodes.keys())

    missing: dict[str, list[str]] = {}
    for econ, refs in _DEFAULT_SUPPLIERS.items():
        for ref in refs:
            if ref not in library_keys:
                missing.setdefault(f"suppliers/{econ}", []).append(ref)
    for econ, refs in _DEFAULT_REVENUE_STREAMS.items():
        for ref in refs:
            if ref not in library_keys:
                missing.setdefault(f"revenue/{econ}", []).append(ref)
    for econ, refs in _DEFAULT_COST_CENTERS.items():
        for ref in refs:
            if ref not in library_keys:
                missing.setdefault(f"cost/{econ}", []).append(ref)

    assert not missing, (
        "seed.py default refs reference unknown node_keys: "
        f"{missing}. Either rename the keys in node_library.yaml back, "
        "or update the default tables in seed.py."
    )
