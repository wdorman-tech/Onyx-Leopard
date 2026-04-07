"""Industry registry — dynamically loaded from YAML config files.

Replaces the old hardcoded INDUSTRY_REGISTRY dict. Each *.yaml file in
the industries/ directory defines a complete industry specification.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation.config_loader import list_industry_specs


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
