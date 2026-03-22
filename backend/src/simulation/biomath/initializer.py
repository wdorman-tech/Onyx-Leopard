from __future__ import annotations

from src.schemas import CompanyGraph, NodeData, SimulationParameters
from src.simulation.biomath.models import (
    ApoptosisState,
    BioConfig,
    BioParams,
    BioState,
    CellCycleState,
)


def _derive_carrying_capacity(
    node: NodeData,
    sim_params: SimulationParameters,
    multiplier: float,
) -> float:
    """Derive carrying capacity K from node type and metrics."""
    match node.type:
        case "department" | "team":
            headcount = node.metrics.get("headcount", 1)
            budget = node.metrics.get("budget", 0)
            if budget > 0 and headcount > 0:
                # K = budget * multiplier / avg_cost_per_head
                avg_cost = budget / headcount
                K = budget * multiplier / avg_cost
            else:
                K = max(headcount * multiplier, 1.0)
            return max(K, 1.0)

        case "revenue_stream":
            revenue = node.metrics.get("revenue", 0)
            return max(revenue * multiplier, 1.0)

        case "cost_center":
            budget = node.metrics.get("budget", 0)
            return max(budget * multiplier, 1.0)

        case "external":
            revenue = node.metrics.get("revenue", 0)
            return max(revenue * multiplier, 1.0)

        case _:
            headcount = node.metrics.get("headcount", 1)
            return max(headcount * multiplier, 1.0)


def _derive_growth_rate(
    node: NodeData,
    sim_params: SimulationParameters,
) -> float:
    """Derive intrinsic growth rate r from node data."""
    if node.type == "revenue_stream":
        growth = node.metrics.get("growth_rate", 0)
        if growth > 0:
            return growth / 52.0  # annual -> weekly
    return sim_params.logistic_growth_rate


def _derive_population(node: NodeData) -> float:
    """Get the 'population' variable for this node."""
    if node.type == "revenue_stream":
        return node.metrics.get("revenue", 0.0)
    return node.metrics.get("headcount", 0.0)


def _derive_cash(node: NodeData) -> float:
    """Get the cash/liquidity variable."""
    if node.type == "revenue_stream":
        revenue = node.metrics.get("revenue", 0)
        margin = node.metrics.get("margin", 0.5)
        return revenue * margin
    return node.metrics.get("budget", 0.0)


def _derive_capital(node: NodeData) -> float:
    """Get capital for Cobb-Douglas."""
    return node.metrics.get("budget", 0) or node.metrics.get("revenue", 0)


def initialize_bio_states(
    graph: CompanyGraph,
    sim_params: SimulationParameters,
    config: BioConfig | None = None,
) -> dict[str, tuple[BioState, BioParams]]:
    """Initialize BioState + BioParams per node from graph metrics and SimulationParameters.

    Maps each node type to appropriate K derivation and initializes
    all ODE state variables.
    """
    config = config or BioConfig()
    multiplier = sim_params.carrying_capacity_multiplier
    result: dict[str, tuple[BioState, BioParams]] = {}

    for node in graph.nodes:
        K = _derive_carrying_capacity(node, sim_params, multiplier)
        r = _derive_growth_rate(node, sim_params)
        population = _derive_population(node)
        cash = _derive_cash(node)
        capital = _derive_capital(node)

        # Compute initial Cobb-Douglas revenue
        if config.cobb_douglas and population > 0 and capital > 0:
            revenue_rate = (
                sim_params.tfp
                * (capital ** sim_params.capital_elasticity)
                * (population ** sim_params.labor_elasticity)
            )
        else:
            revenue_rate = node.metrics.get("revenue", 0.0)

        # Cost rate
        node_fixed = sim_params.fixed_costs
        if len(graph.nodes) > 1:
            node_fixed = sim_params.fixed_costs / len(graph.nodes)
        cost_rate = node_fixed + sim_params.variable_cost_per_unit * population

        # Health score
        losses = max(0.0, cost_rate - revenue_rate)
        denom = cash + losses
        health_score = max(0.0, min(1.0, cash / denom)) if denom > 0 else (1.0 if cash >= 0 else 0.0)

        bio_state = BioState(
            population=population,
            carrying_capacity=K,
            growth_rate=r,
            cash=cash,
            capital=capital,
            revenue_rate=revenue_rate,
            cost_rate=cost_rate,
            health_score=health_score,
        )

        # Phase 2: initialize apoptosis state for dept/team/cost_center nodes
        if config.apoptosis and node.type in ("department", "team", "cost_center"):
            bio_state.apoptosis = ApoptosisState(bcl2=1.0, bax=0.0)

        # Phase 4: initialize cell cycle for department nodes
        if config.cell_cycle and node.type == "department":
            resource_rate = revenue_rate / max(cost_rate, 1.0)
            bio_state.cell_cycle = CellCycleState(
                cyclin=min(0.1, resource_rate * 0.1),
                cdk_active=0.0,
                phase="G1",
            )

        bio_params = BioParams(
            r=r,
            K=K,
            tfp=sim_params.tfp,
            alpha=sim_params.capital_elasticity,
            beta=sim_params.labor_elasticity,
            fixed_costs=node_fixed,
            variable_cost_rate=sim_params.variable_cost_per_unit,
        )

        result[node.id] = (bio_state, bio_params)

    return result
