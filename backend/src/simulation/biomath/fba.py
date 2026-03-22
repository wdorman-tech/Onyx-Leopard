from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

from src.simulation.biomath.models import FluxSolution

if TYPE_CHECKING:
    from src.schemas import CompanyGraph

logger = logging.getLogger(__name__)


def build_stoichiometry_matrix(
    graph: CompanyGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Build stoichiometry matrix S from edge structure.

    Resource flow edges (funds, supplies) define fluxes.
    S * v = 0 enforces conservation: cash in = cash out per node.

    Returns:
        S: stoichiometry matrix (n_nodes x n_edges)
        v_min: lower bounds per flux
        v_max: upper bounds per flux
        node_ids: ordered node ID list (rows)
        edge_keys: ordered edge key list (columns)
    """
    resource_edges = [
        e for e in graph.edges if e.relationship in ("funds", "supplies")
    ]

    if not resource_edges:
        return np.zeros((0, 0)), np.array([]), np.array([]), [], []

    node_ids = [n.id for n in graph.nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    n_edges = len(resource_edges)

    S = np.zeros((n_nodes, n_edges))
    v_min = np.zeros(n_edges)
    v_max = np.zeros(n_edges)
    edge_keys: list[str] = []

    for j, edge in enumerate(resource_edges):
        key = f"{edge.source}->{edge.target}"
        edge_keys.append(key)

        src_idx = node_idx.get(edge.source)
        tgt_idx = node_idx.get(edge.target)

        if src_idx is not None:
            S[src_idx, j] = -1.0  # outflow from source
        if tgt_idx is not None:
            S[tgt_idx, j] = 1.0  # inflow to target

        src_node = next((n for n in graph.nodes if n.id == edge.source), None)
        if src_node:
            max_flow = src_node.metrics.get("revenue", 0) or src_node.metrics.get("budget", 0)
            v_max[j] = max(max_flow, 0.0)
        else:
            v_max[j] = 1e9

    return S, v_min, v_max, node_ids, edge_keys


def solve_resource_allocation(
    S: np.ndarray,
    c: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
) -> FluxSolution:
    """Solve the FBA LP: minimize c^T * v subject to S * v = 0, v_min <= v <= v_max.

    We use equality constraints (conservation) and bound constraints.
    """
    if S.size == 0 or len(c) == 0:
        return FluxSolution(feasible=True)

    n_nodes, n_edges = S.shape
    bounds = list(zip(v_min.tolist(), v_max.tolist()))

    result = linprog(
        c=-c,  # negate because linprog minimizes, we want to maximize
        A_eq=S,
        b_eq=np.zeros(n_nodes),
        bounds=bounds,
        method="highs",
    )

    if result.success:
        fluxes = {str(i): float(result.x[i]) for i in range(n_edges)}

        shadow_prices: dict[str, float] = {}
        if hasattr(result, "eqlin") and result.eqlin is not None:
            marginals = result.eqlin.marginals if hasattr(result.eqlin, "marginals") else []
            for i, val in enumerate(marginals):
                shadow_prices[str(i)] = float(val)

        return FluxSolution(
            fluxes=fluxes,
            shadow_prices=shadow_prices,
            feasible=True,
            objective_value=float(-result.fun),
        )

    logger.warning("FBA LP infeasible, relaxing conservation constraints")
    return FluxSolution(feasible=False)


def run_fba(
    graph: CompanyGraph,
    objective: str = "growth",
) -> tuple[FluxSolution, list[str], list[str]]:
    """Run full FBA pipeline: build matrix, set objective, solve LP.

    Args:
        graph: current company graph
        objective: "growth" (maximize headcount flow) or "profitability" (maximize cash flow)

    Returns:
        (solution, node_ids, edge_keys)
    """
    S, v_min, v_max, node_ids, edge_keys = build_stoichiometry_matrix(graph)

    if S.size == 0:
        return FluxSolution(feasible=True), node_ids, edge_keys

    n_edges = S.shape[1]
    c = np.ones(n_edges)  # default: maximize total flow

    if objective == "profitability":
        for j, key in enumerate(edge_keys):
            if "funds" in key or "revenue" in key:
                c[j] = 2.0
            else:
                c[j] = 0.5

    solution = solve_resource_allocation(S, c, v_min, v_max)
    return solution, node_ids, edge_keys
