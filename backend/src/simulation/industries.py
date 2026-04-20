"""Industry registry — dynamically loaded from YAML config files.

Replaces the old hardcoded INDUSTRY_REGISTRY dict. Each *.yaml file in
the industries/ directory defines a complete industry specification.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from src.simulation.config_loader import (
    INDUSTRY_DIR,
    atomic_write_yaml,
    clear_cache,
    list_industry_specs,
)


@dataclass(frozen=True)
class IndustryConfig:
    """Lightweight industry metadata for the /api/simulate/industries endpoint."""

    slug: str
    name: str
    description: str
    icon: str
    playable: bool
    total_nodes: int
    growth_stages: int
    key_metrics: tuple[str, ...]
    example_nodes: tuple[str, ...]
    categories: dict[str, int]


def _build_registry() -> dict[str, IndustryConfig]:
    """Load all industry YAML files and build the registry."""
    registry: dict[str, IndustryConfig] = {}
    for spec in list_industry_specs():
        meta = spec.meta
        registry[meta.slug] = IndustryConfig(
            slug=meta.slug,
            name=meta.name,
            description=meta.description,
            icon=meta.icon,
            playable=meta.playable,
            total_nodes=meta.total_nodes,
            growth_stages=meta.growth_stages,
            key_metrics=tuple(meta.key_metrics),
            example_nodes=tuple(meta.example_nodes),
            categories=dict(meta.categories),
        )
    return registry


INDUSTRY_REGISTRY: dict[str, IndustryConfig] = _build_registry()

# Serializes commit_industry calls so concurrent adaptive sims can't race on
# the global registry / cache mutation. Acquired around the file write and the
# subsequent registry rebuild.
industry_registry_lock = asyncio.Lock()


def refresh_registry() -> None:
    """Rebuild the registry from disk. Call after saving a new industry YAML."""
    global INDUSTRY_REGISTRY
    clear_cache()
    INDUSTRY_REGISTRY = _build_registry()


async def commit_industry(slug: str, spec_dict: dict) -> Path:
    """Atomically write a generated industry YAML and refresh the registry.

    Single entry point for adding a generated industry at runtime. Holds the
    registry lock for the entire write+rebuild so other coroutines see either
    the pre-state or the fully-committed post-state — never a half-written
    file or a partially-cleared cache.
    """
    path = INDUSTRY_DIR / f"{slug}.yaml"
    async with industry_registry_lock:
        atomic_write_yaml(path, spec_dict)
        refresh_registry()
    return path
