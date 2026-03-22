from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import CompanyGraph

logger = logging.getLogger(__name__)


def hill_function(signal: float, K: float, n: float) -> float:
    """Core Hill function: signal^n / (K^n + signal^n).

    Produces sigmoidal response: gradual for n=1, switch-like for n>=4.
    """
    if K <= 0 or n <= 0:
        return 0.0
    s_n = abs(signal) ** n
    k_n = K ** n
    return s_n / (k_n + s_n) if (k_n + s_n) > 0 else 0.0


def hill_coefficient_for_structure(structure_type: str | None) -> float:
    """Map org structure type to Hill coefficient n.

    flat -> n=1 (gradual consensus)
    matrix -> n=2 (moderate)
    hierarchical -> n=4 (command-and-control switch)
    """
    mapping = {
        "flat": 1.0,
        "matrix": 2.0,
        "functional": 2.0,
        "divisional": 3.0,
        "hierarchical": 4.0,
    }
    return mapping.get(structure_type or "", 2.0)


def propagate_signal(
    graph: CompanyGraph,
    event_strength: float,
    source_node_id: str | None,
    hill_n: float = 2.0,
    hill_K: float = 0.5,
) -> dict[str, float]:
    """Propagate a signal event through the org hierarchy via reports_to edges.

    Each hop through an edge applies the Hill function, which can amplify
    (n>1) or attenuate the signal. Returns activation level per node.
    """
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relationship == "reports_to":
            adjacency.setdefault(edge.target, []).append(edge.source)
            adjacency.setdefault(edge.source, []).append(edge.target)

    activations: dict[str, float] = {}
    node_ids = {n.id for n in graph.nodes}

    if source_node_id and source_node_id in node_ids:
        activations[source_node_id] = min(1.0, event_strength)
    else:
        for nid in node_ids:
            activations[nid] = min(1.0, event_strength * 0.5)
        return activations

    visited: set[str] = {source_node_id}
    frontier = [source_node_id]

    while frontier:
        next_frontier: list[str] = []
        for nid in frontier:
            parent_activation = activations[nid]
            for neighbor in adjacency.get(nid, []):
                if neighbor in visited:
                    continue
                child_activation = hill_function(parent_activation, hill_K, hill_n)
                activations[neighbor] = child_activation
                visited.add(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier

    for nid in node_ids:
        if nid not in activations:
            activations[nid] = 0.0

    return activations
