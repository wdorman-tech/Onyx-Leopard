from __future__ import annotations

import logging
import math

import numpy as np
from scipy.integrate import solve_ivp

from src.schemas import CompanyGraph, SimulationParameters
from src.simulation.biomath.adoption import compute_r0, step_sir
from src.simulation.biomath.apoptosis import apply_wind_down, check_apoptosis, step_apoptosis
from src.simulation.biomath.cell_cycle import check_division_ready, step_cell_cycle
from src.simulation.biomath.competition import (
    compute_competition_matrix,
    step_competition,
)
from src.simulation.biomath.fba import run_fba
from src.simulation.biomath.initializer import initialize_bio_states
from src.simulation.biomath.mapk import compute_cascade_response
from src.simulation.biomath.models import (
    ActionConstraints,
    BioConfig,
    BioParams,
    BioState,
    FluxSolution,
)
from src.simulation.biomath.replicator import (
    build_strategy_payoff_matrix,
    step_replicator,
)
from src.simulation.biomath.signals import hill_coefficient_for_structure, propagate_signal
from src.simulation.state import AgentAction

logger = logging.getLogger(__name__)


def _node_ode(
    t: float,
    y: list[float],
    params: BioParams,
) -> list[float]:
    """Per-node ODE system: logistic growth + cash dynamics.

    y = [population, cash]
    dN/dt = r * N * (1 - N/K)
    dCash/dt = revenue_rate - cost_rate
    where revenue_rate = Cobb-Douglas, cost_rate = fixed + variable * N
    """
    N, cash = y[0], y[1]
    N = max(N, 0.0)

    dN = params.r * N * (1.0 - N / max(params.K, 1e-6))

    # Cobb-Douglas revenue
    capital = max(cash, 0.0) + 1.0  # use cash as capital proxy, minimum 1
    revenue = params.tfp * (capital ** params.alpha) * (max(N, 0.1) ** params.beta)
    costs = params.fixed_costs + params.variable_cost_rate * N

    dCash = revenue - costs

    return [dN, dCash]


class BioMathEngine:
    """Orchestrates all biological mathematics computations per simulation tick.

    Runs in sequence each tick:
    1. Replicator dynamics (every N ticks) — strategy emphasis
    2. FBA — resource allocation LP
    3. Lotka-Volterra — market competition
    4. Per-node logistic growth + Cobb-Douglas
    5. Signal propagation (MAPK or Hill) for events
    6. SIR adoption for active launches
    7. Apoptosis check + wind-down
    8. Cell cycle checkpoints
    """

    def __init__(
        self,
        graph: CompanyGraph,
        sim_params: SimulationParameters,
        config: BioConfig | None = None,
        structure_type: str | None = None,
    ):
        self.sim_params = sim_params
        self.config = config or BioConfig()
        self.structure_type = structure_type
        self._tick = 0

        # Initialize per-node bio states and params
        states = initialize_bio_states(graph, sim_params, self.config)
        self.bio_states: dict[str, BioState] = {nid: s for nid, (s, _) in states.items()}
        self.bio_params: dict[str, BioParams] = {nid: p for nid, (_, p) in states.items()}

        # FBA solution (updated each tick)
        self.flux_solution: FluxSolution = FluxSolution()

        # Replicator state
        self._strategy_names: list[str] = ["growth", "profitability", "efficiency"]
        self._strategy_freq: np.ndarray = np.array([0.4, 0.3, 0.3])

        # MAPK cascade states per node (6-variable state)
        self._mapk_states: dict[str, np.ndarray | None] = {}

        # Track which nodes are in apoptosis wind-down
        self._winding_down: set[str] = set()

    def step(self, graph: CompanyGraph, events: list[dict] | None = None, dt: float = 1.0) -> dict[str, BioState]:
        """Run all bio-math computations for one tick. Returns updated bio states."""
        self._tick += 1
        events = events or []

        # 1. Replicator dynamics (every N ticks)
        if self.config.replicator and self._tick % self.config.replicator_interval == 0:
            self._step_replicator(graph)

        # 2. FBA — resource allocation
        if self.config.fba:
            self._step_fba(graph)

        # 3. Lotka-Volterra competition
        if self.config.lotka_volterra:
            self._step_competition(graph, dt)

        # 4. Per-node logistic growth + Cobb-Douglas
        if self.config.logistic_growth or self.config.cobb_douglas:
            self._step_node_odes(dt)

        # 5. Signal propagation for events
        if events and (self.config.hill_signals or self.config.mapk_cascade):
            self._step_signals(graph, events)

        # 6. SIR adoption
        if self.config.sir_adoption:
            self._step_sir(dt)

        # 7. Apoptosis
        if self.config.apoptosis:
            self._step_apoptosis(graph, dt)

        # 8. Cell cycle
        if self.config.cell_cycle:
            self._step_cell_cycle()

        # Update health scores
        self._update_health_scores()

        return dict(self.bio_states)

    def compute_constraints(self) -> dict[str, ActionConstraints]:
        """Derive action constraints from current bio-math state."""
        constraints: dict[str, ActionConstraints] = {}

        for nid, state in self.bio_states.items():
            params = self.bio_params.get(nid)
            if not params:
                continue

            # Max hire: can't exceed carrying capacity
            headroom = max(0, math.floor(state.carrying_capacity - state.population))
            max_fire = max(0, math.floor(state.population))

            # Budget constraints
            max_increase = max(0.0, state.cash * 0.2)  # can increase by 20% of cash
            max_decrease = max(0.0, state.cash * 0.5)  # can decrease by 50%

            # Health status
            if state.health_score >= 0.6:
                health_status = "healthy"
            elif state.health_score >= 0.3:
                health_status = "stressed"
            else:
                health_status = "critical"

            # Capacity utilization
            cap_util = state.population / max(state.carrying_capacity, 1.0)

            # Cell cycle gating for expand actions
            can_expand = True
            if self.config.cell_cycle and state.cell_cycle:
                can_expand = check_division_ready(state.cell_cycle)

            # Override for apoptosis
            if state.apoptosis and state.apoptosis.triggered:
                headroom = 0
                max_increase = 0.0
                can_expand = False
                health_status = "critical"

            constraints[nid] = ActionConstraints(
                max_hire=headroom,
                max_fire=max_fire,
                max_budget_increase=max_increase,
                max_budget_decrease=max_decrease,
                health_status=health_status,
                can_expand=can_expand,
                capacity_utilization=min(1.0, cap_util),
            )

        return constraints

    def validate_actions(self, actions: list[AgentAction]) -> list[AgentAction]:
        """Clamp/reject actions violating conservation laws."""
        if not self.config.conservation_laws:
            return actions

        validated: list[AgentAction] = []
        constraints = self.compute_constraints()

        for action in actions:
            c = constraints.get(action.agent_id)
            if not c:
                validated.append(action)
                continue

            match action.action_type:
                case "hire":
                    count = action.params.get("count", 1)
                    clamped = min(int(count), c.max_hire)
                    if clamped <= 0:
                        action = AgentAction(
                            agent_id=action.agent_id,
                            action_type="report",
                            params={"original_action": "hire", "reason": "at_capacity"},
                            reasoning=f"Hire rejected: at capacity ({c.capacity_utilization:.0%})",
                        )
                    else:
                        action.params["count"] = clamped

                case "fire":
                    count = action.params.get("count", 1)
                    clamped = min(int(count), c.max_fire)
                    action.params["count"] = max(clamped, 0)

                case "reallocate_budget":
                    amount = action.params.get("amount", 0)
                    if amount > 0:
                        amount = min(amount, c.max_budget_increase)
                    else:
                        amount = max(amount, -c.max_budget_decrease)
                    action.params["amount"] = amount

                case "invest":
                    amount = action.params.get("amount", 0)
                    state = self.bio_states.get(action.agent_id)
                    if state and amount > state.cash:
                        action.params["amount"] = max(0, state.cash * 0.5)

                case "expand":
                    if not c.can_expand:
                        action = AgentAction(
                            agent_id=action.agent_id,
                            action_type="report",
                            params={"original_action": "expand", "reason": "cell_cycle_blocked"},
                            reasoning="Expansion blocked: insufficient resource accumulation",
                        )

                case "cut_costs":
                    amount = action.params.get("amount", 0)
                    state = self.bio_states.get(action.agent_id)
                    if state:
                        action.params["amount"] = min(amount, state.cash)

            # Block most actions during apoptosis wind-down
            if action.agent_id in self._winding_down and action.action_type not in ("report", "fire"):
                action = AgentAction(
                    agent_id=action.agent_id,
                    action_type="report",
                    params={"original_action": action.action_type, "reason": "shutting_down"},
                    reasoning="Node is shutting down — only reporting allowed",
                )

            validated.append(action)

        return validated

    def sync_to_graph(self, graph: CompanyGraph) -> None:
        """Write bio-math state back into node.metrics and global_metrics."""
        total_health = 0.0
        total_cap_util = 0.0
        departments_at_risk = 0
        node_count = 0

        for node in graph.nodes:
            state = self.bio_states.get(node.id)
            if not state:
                continue

            node_count += 1

            # Sync core metrics
            if node.type in ("department", "team", "role", "cost_center"):
                node.metrics["headcount"] = max(0, round(state.population))
            elif node.type == "revenue_stream":
                node.metrics["revenue"] = max(0, round(state.population, 2))

            node.metrics["health_score"] = round(state.health_score, 3)
            cap_util = state.population / max(state.carrying_capacity, 1.0)
            node.metrics["capacity_utilization"] = round(min(1.0, cap_util), 3)

            # Phase 2: signal activation
            node.metrics["signal_activation"] = round(state.signal_activation, 3)

            # Phase 2: apoptosis
            if state.apoptosis and state.apoptosis.triggered:
                node.metrics["apoptosis_triggered"] = 1.0
                node.metrics["wind_down_remaining"] = float(state.wind_down_ticks)
            else:
                node.metrics["apoptosis_triggered"] = 0.0

            # Phase 3: SIR adoption
            if state.sir_state and state.sir_state.active:
                r0 = compute_r0(state.sir_state.beta, state.sir_state.gamma)
                node.metrics["adoption_r0"] = round(r0, 2)
                adoption_pct = state.sir_state.infected / max(state.sir_state.total_market, 1.0)
                node.metrics["adoption_pct"] = round(adoption_pct, 3)

            # Phase 4: cell cycle
            if state.cell_cycle:
                phase_map = {"G1": 1.0, "S": 2.0, "G2": 3.0, "M": 4.0}
                node.metrics["cell_cycle_phase"] = phase_map.get(state.cell_cycle.phase, 1.0)

            # Accumulate for globals
            total_health += state.health_score
            total_cap_util += min(1.0, cap_util)
            if state.health_score < 0.3:
                departments_at_risk += 1

        # Update global metrics
        if node_count > 0:
            graph.global_metrics["avg_health_score"] = round(total_health / node_count, 3)
            graph.global_metrics["avg_capacity_utilization"] = round(total_cap_util / node_count, 3)
        graph.global_metrics["departments_at_risk"] = float(departments_at_risk)

        # FBA efficiency
        if self.flux_solution.feasible:
            graph.global_metrics["resource_efficiency"] = round(self.flux_solution.objective_value, 2)

        # Recompute totals
        total_headcount = sum(
            n.metrics.get("headcount", 0) for n in graph.nodes if n.type != "external"
        )
        total_budget = sum(
            n.metrics.get("budget", 0) for n in graph.nodes if n.type != "external"
        )
        total_revenue = sum(
            n.metrics.get("revenue", 0) for n in graph.nodes if n.type == "revenue_stream"
        )
        graph.global_metrics["total_headcount"] = total_headcount
        graph.global_metrics["total_budget"] = total_budget
        graph.global_metrics["revenue"] = total_revenue

    def get_bio_summary(self) -> dict[str, dict]:
        """Serialized bio-state summary for SSE events."""
        summary: dict[str, dict] = {}
        for nid, state in self.bio_states.items():
            entry: dict = {
                "health_score": round(state.health_score, 3),
                "capacity_utilization": round(
                    state.population / max(state.carrying_capacity, 1.0), 3
                ),
                "signal_activation": round(state.signal_activation, 3),
                "apoptosis_triggered": bool(state.apoptosis and state.apoptosis.triggered),
            }
            if state.cell_cycle:
                entry["cell_cycle_phase"] = state.cell_cycle.phase
            if state.sir_state and state.sir_state.active:
                entry["adoption_r0"] = round(
                    compute_r0(state.sir_state.beta, state.sir_state.gamma), 2
                )
            summary[nid] = entry
        return summary

    # --- Internal step methods ---

    def _step_node_odes(self, dt: float) -> None:
        """Run logistic growth + Cobb-Douglas ODE per node."""
        for nid, state in self.bio_states.items():
            params = self.bio_params.get(nid)
            if not params or state.population <= 0:
                continue

            if state.apoptosis and state.apoptosis.triggered:
                continue  # skip growth for dying nodes

            y0 = [state.population, state.cash]

            try:
                sol = solve_ivp(
                    _node_ode,
                    [0, dt],
                    y0,
                    args=(params,),
                    method="RK45",
                    max_step=0.5,
                )

                if sol.success and sol.y.shape[1] > 0:
                    state.population = max(0.0, float(sol.y[0, -1]))
                    state.cash = float(sol.y[1, -1])
                else:
                    # Euler fallback
                    dy = _node_ode(0, y0, params)
                    state.population = max(0.0, state.population + dy[0] * dt)
                    state.cash = state.cash + dy[1] * dt
            except Exception:
                logger.warning("ODE solve failed for node %s, using Euler", nid)
                dy = _node_ode(0, y0, params)
                state.population = max(0.0, state.population + dy[0] * dt)
                state.cash = state.cash + dy[1] * dt

            # Update derived values
            capital = max(state.cash, 0.0) + 1.0
            state.revenue_rate = params.tfp * (capital ** params.alpha) * (max(state.population, 0.1) ** params.beta)
            state.cost_rate = params.fixed_costs + params.variable_cost_rate * state.population

    def _step_competition(self, graph: CompanyGraph, dt: float) -> None:
        """Run Lotka-Volterra competition for revenue_stream + external nodes."""
        revenue_nodes = [n for n in graph.nodes if n.type == "revenue_stream"]
        competitor_nodes = [n for n in graph.nodes if n.type == "external"]

        if not revenue_nodes:
            return

        all_ids = [n.id for n in revenue_nodes] + [n.id for n in competitor_nodes]
        n_company = len(revenue_nodes)
        n_competitors = len(competitor_nodes)

        populations = np.array([
            self.bio_states[nid].population if nid in self.bio_states else 0.0
            for nid in all_ids
        ])
        growth_rates = np.array([
            self.bio_params[nid].r if nid in self.bio_params else 0.05
            for nid in all_ids
        ])
        capacities = np.array([
            self.bio_params[nid].K if nid in self.bio_params else 1.0
            for nid in all_ids
        ])

        comp_shares = [n.metrics.get("market_share", 0.1) for n in competitor_nodes]
        comp_costs = [n.metrics.get("relative_cost", 1.0) for n in competitor_nodes]

        alpha_default = self.sim_params.competition_alpha_default
        alpha = compute_competition_matrix(
            n_company, n_competitors, comp_shares, comp_costs, alpha_default
        )

        new_pops = step_competition(populations, growth_rates, capacities, alpha, dt)

        for i, nid in enumerate(all_ids):
            if nid in self.bio_states:
                self.bio_states[nid].population = max(0.0, float(new_pops[i]))

    def _step_signals(self, graph: CompanyGraph, events: list[dict]) -> None:
        """Propagate events through org hierarchy using MAPK cascade or Hill functions."""
        for event in events:
            strength = event.get("strength", 0.5)
            source = event.get("source_node_id")

            if self.config.mapk_cascade:
                # Use MAPK cascade for ultrasensitive response
                activations = propagate_signal(
                    graph, strength, source,
                    hill_n=2.0, hill_K=0.5,
                )
                for nid, raw_activation in activations.items():
                    cascade_output = compute_cascade_response(
                        raw_activation,
                        org_depth=3,
                        structure_type=self.structure_type,
                    )
                    if nid in self.bio_states:
                        self.bio_states[nid].signal_activation = max(
                            self.bio_states[nid].signal_activation,
                            cascade_output,
                        )
            elif self.config.hill_signals:
                hill_n = hill_coefficient_for_structure(self.structure_type)
                activations = propagate_signal(
                    graph, strength, source,
                    hill_n=hill_n, hill_K=0.5,
                )
                for nid, activation in activations.items():
                    if nid in self.bio_states:
                        self.bio_states[nid].signal_activation = max(
                            self.bio_states[nid].signal_activation,
                            activation,
                        )

    def _step_apoptosis(self, graph: CompanyGraph, dt: float) -> None:
        """Check and advance apoptosis for eligible nodes."""
        threshold = self.sim_params.apoptosis_threshold

        for nid, state in self.bio_states.items():
            if state.apoptosis is None:
                continue

            if state.apoptosis.triggered:
                apply_wind_down(state)
                self._winding_down.add(nid)
                if state.wind_down_ticks <= 0 and state.population < 1.0:
                    self._winding_down.discard(nid)
                continue

            # Run the bistable switch ODE
            state.apoptosis = step_apoptosis(state, dt)

            # Also check health-based trigger
            if check_apoptosis(state, threshold):
                state.apoptosis.triggered = True
                state.apoptosis.caspase = 0.1
                state.wind_down_ticks = 4
                self._winding_down.add(nid)
                logger.info("Node %s triggered apoptosis (health=%.2f)", nid, state.health_score)

    def _step_sir(self, dt: float) -> None:
        """Advance SIR adoption models for active product launches."""
        for nid, state in self.bio_states.items():
            if state.sir_state and state.sir_state.active:
                state.sir_state = step_sir(state.sir_state, dt)
                # SIR infected count drives revenue growth
                adoption_fraction = state.sir_state.infected / max(state.sir_state.total_market, 1.0)
                state.revenue_rate *= (1.0 + adoption_fraction * 0.5)

    def _step_cell_cycle(self) -> None:
        """Advance cell cycle checkpoints for department nodes."""
        for nid, state in self.bio_states.items():
            if state.cell_cycle is None:
                continue
            resource_rate = state.revenue_rate / max(state.cost_rate, 1.0)
            state.cell_cycle = step_cell_cycle(state.cell_cycle, resource_rate)

    def _step_fba(self, graph: CompanyGraph) -> None:
        """Run flux balance analysis for resource allocation."""
        objective = "growth"
        # Use replicator-derived strategy emphasis
        if self._strategy_freq is not None and len(self._strategy_freq) >= 2:
            if self._strategy_freq[1] > self._strategy_freq[0]:
                objective = "profitability"

        self.flux_solution, _, _ = run_fba(graph, objective)

    def _step_replicator(self, graph: CompanyGraph) -> None:
        """Run replicator dynamics to evolve strategy emphasis."""
        # Measure performance of each strategy
        performance: dict[str, float] = {}

        # Growth: measured by headcount change
        total_pop = sum(s.population for s in self.bio_states.values())
        total_K = sum(s.carrying_capacity for s in self.bio_states.values())
        performance["growth"] = total_pop / max(total_K, 1.0)

        # Profitability: measured by average health score
        health_scores = [s.health_score for s in self.bio_states.values()]
        performance["profitability"] = sum(health_scores) / max(len(health_scores), 1)

        # Efficiency: measured by capacity utilization
        utils = [s.population / max(s.carrying_capacity, 1.0) for s in self.bio_states.values()]
        performance["efficiency"] = sum(utils) / max(len(utils), 1)

        payoff_matrix, names = build_strategy_payoff_matrix(performance)
        if len(names) > 0:
            self._strategy_freq = step_replicator(self._strategy_freq, payoff_matrix)
            self._strategy_names = names

    def _update_health_scores(self) -> None:
        """Recompute health scores from cash and cost/revenue rates."""
        for state in self.bio_states.values():
            losses = max(0.0, state.cost_rate - state.revenue_rate)
            denom = state.cash + losses
            if denom > 0:
                state.health_score = max(0.0, min(1.0, state.cash / denom))
            elif state.cash >= 0:
                state.health_score = 1.0
            else:
                state.health_score = 0.0

    def activate_sir_launch(self, node_id: str, market_size: float, growth_rate: float, churn: float) -> None:
        """Activate SIR adoption model for a product launch on a revenue_stream node."""
        from src.simulation.biomath.adoption import create_sir_for_launch

        state = self.bio_states.get(node_id)
        if state:
            state.sir_state = create_sir_for_launch(growth_rate, churn, market_size)
