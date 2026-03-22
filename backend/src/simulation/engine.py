from __future__ import annotations

import asyncio
import json
import logging

from camel.agents import ChatAgent

from src.agents.factory import create_haiku
from src.schemas import CompanyGraph, NodeData, SimulationParameters
from src.simulation.biomath.engine import BioMathEngine
from src.simulation.biomath.models import ActionConstraints, BioConfig
from src.simulation.prompts import (
    AGENT_SYSTEM_PROMPTS,
    BIOMATH_CONTEXT,
    OUTLOOK_CONTEXT,
    TICK_PROMPT,
)
from src.simulation.state import AgentAction, SimulationState, StepResult

logger = logging.getLogger(__name__)

MAX_NODES = 20


class SimulationEngine:
    def __init__(
        self,
        graph: CompanyGraph,
        max_ticks: int = 50,
        outlook: str = "normal",
        sim_params: SimulationParameters | None = None,
    ):
        if len(graph.nodes) > MAX_NODES:
            raise ValueError(f"Max {MAX_NODES} nodes allowed, got {len(graph.nodes)}")
        self.state = SimulationState(graph=graph, max_ticks=max_ticks, outlook=outlook)
        self._agents: dict[str, ChatAgent] = {}
        self._build_agents()

        # Bio-math layer
        self.sim_params = sim_params
        self.biomath: BioMathEngine | None = None
        if sim_params:
            self.biomath = BioMathEngine(
                graph,
                sim_params,
                config=BioConfig(),
                structure_type=None,
            )

    def _build_agents(self) -> None:
        for node in self.state.graph.nodes:
            system_prompt = self._build_system_prompt(node)
            model = create_haiku()
            agent = ChatAgent(system_message=system_prompt, model=model)
            self._agents[node.id] = agent

    def _build_system_prompt(self, node: NodeData) -> str:
        template = AGENT_SYSTEM_PROMPTS.get(node.type, AGENT_SYSTEM_PROMPTS["role"])
        return template.format(
            label=node.label,
            headcount=node.metrics.get("headcount", 0),
            budget=node.metrics.get("budget", 0),
            revenue=node.metrics.get("revenue", 0),
            custom_prompt=node.agent_prompt,
        )

    def _build_biomath_context(self, node: NodeData, constraints: ActionConstraints | None) -> str:
        """Build the bio-math context block for the LLM prompt."""
        if not self.biomath or not constraints:
            return ""

        state = self.biomath.bio_states.get(node.id)
        if not state:
            return ""

        net = state.revenue_rate - state.cost_rate
        utilization = state.population / max(state.carrying_capacity, 1.0)

        extra_lines: list[str] = []

        # Signal context
        if state.signal_activation > 0.05:
            extra_lines.append(f"- Signal activation: {state.signal_activation:.0%}")

        # Competition context
        if node.type == "revenue_stream":
            extra_lines.append(f"- Market position: capacity utilization {utilization:.0%}")

        # Apoptosis warning
        if state.apoptosis and state.apoptosis.triggered:
            extra_lines.append("- WARNING: Department is shutting down. Only fire/report actions allowed.")

        # Cell cycle
        if state.cell_cycle:
            if constraints.can_expand:
                extra_lines.append("- Ready to Expand: resource accumulation threshold reached")
            else:
                extra_lines.append(f"- Expansion blocked: cell cycle phase {state.cell_cycle.phase}")

        # SIR adoption
        if state.sir_state and state.sir_state.active:
            from src.simulation.biomath.adoption import compute_r0
            r0 = compute_r0(state.sir_state.beta, state.sir_state.gamma)
            extra_lines.append(f"- Product adoption R0: {r0:.1f} ({'viral' if r0 > 1 else 'declining'})")

        # Shadow prices from FBA
        if self.biomath.flux_solution.shadow_prices:
            max_price_key = max(
                self.biomath.flux_solution.shadow_prices,
                key=lambda k: abs(self.biomath.flux_solution.shadow_prices[k]),
            )
            max_price = self.biomath.flux_solution.shadow_prices[max_price_key]
            if abs(max_price) > 0.01:
                extra_lines.append(f"- Most constrained resource: node {max_price_key}, marginal value: ${max_price:,.0f}")

        extra_context = "\n".join(extra_lines) + "\n" if extra_lines else ""

        return BIOMATH_CONTEXT.format(
            growth_rate=state.growth_rate,
            population=state.population,
            capacity=state.carrying_capacity,
            utilization=utilization,
            revenue_rate=state.revenue_rate,
            cost_rate=state.cost_rate,
            net=net,
            health_score=state.health_score,
            health_status=constraints.health_status,
            max_hire=constraints.max_hire,
            max_fire=constraints.max_fire,
            max_budget=constraints.max_budget_increase,
            extra_context=extra_context,
        )

    def _build_tick_context(self, node: NodeData, constraints: ActionConstraints | None = None) -> str:
        events_text = ""
        if self.state.event_queue:
            events_text = "Events this week:\n" + "\n".join(
                f"- {e.get('description', str(e))}" for e in self.state.event_queue
            )

        company_context = (
            f"Company: {self.state.graph.name}\n"
            f"Global metrics: {json.dumps(self.state.graph.global_metrics)}\n"
            f"Total nodes: {len(self.state.graph.nodes)}"
        )

        biomath_context = self._build_biomath_context(node, constraints)

        return TICK_PROMPT.format(
            tick=self.state.tick,
            company_context=company_context,
            node_metrics=json.dumps(node.metrics),
            biomath_context=biomath_context,
            outlook_context=OUTLOOK_CONTEXT.get(self.state.outlook, OUTLOOK_CONTEXT["normal"]),
            events=events_text,
        )

    async def _run_agent(self, node: NodeData, constraints: ActionConstraints | None = None) -> AgentAction:
        agent = self._agents[node.id]
        prompt = self._build_tick_context(node, constraints)
        try:
            response = await asyncio.to_thread(agent.step, prompt)
            raw = response.msgs[0].content
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            return AgentAction(
                agent_id=node.id,
                action_type=data.get("action_type", "report"),
                params=data.get("params", {}),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning("Agent %s failed: %s", node.id, e)
            return AgentAction(
                agent_id=node.id,
                action_type="report",
                params={},
                reasoning=f"Agent error: {e}",
            )

    def _apply_actions(self, actions: list[AgentAction]) -> None:
        """Apply agent actions to the graph state with simple conflict resolution."""
        for action in actions:
            node = next((n for n in self.state.graph.nodes if n.id == action.agent_id), None)
            if node is None:
                continue

            match action.action_type:
                case "hire":
                    count = action.params.get("count", 1)
                    node.metrics["headcount"] = node.metrics.get("headcount", 0) + count
                    self.state.graph.global_metrics["total_headcount"] = (
                        self.state.graph.global_metrics.get("total_headcount", 0) + count
                    )
                case "fire":
                    count = min(
                        action.params.get("count", 1),
                        node.metrics.get("headcount", 0),
                    )
                    node.metrics["headcount"] = node.metrics.get("headcount", 0) - count
                    self.state.graph.global_metrics["total_headcount"] = (
                        self.state.graph.global_metrics.get("total_headcount", 0) - count
                    )
                case "reallocate_budget":
                    amount = action.params.get("amount", 0)
                    node.metrics["budget"] = node.metrics.get("budget", 0) + amount
                case "cut_costs":
                    amount = action.params.get("amount", 0)
                    node.metrics["budget"] = max(
                        0, node.metrics.get("budget", 0) - amount
                    )
                case "invest":
                    amount = action.params.get("amount", 0)
                    node.metrics["budget"] = node.metrics.get("budget", 0) - amount
                case "expand":
                    node.metrics["headcount"] = node.metrics.get("headcount", 0) + action.params.get("count", 2)
                case "launch_product":
                    # Activate SIR adoption if bio-math is enabled
                    if self.biomath and node.type == "revenue_stream":
                        market_size = node.metrics.get("revenue", 100000)
                        growth_rate = node.metrics.get("growth_rate", 0.1)
                        self.biomath.activate_sir_launch(
                            node.id, market_size, growth_rate, churn=0.05
                        )
                case _:
                    pass  # report, collaborate, etc. — no metric changes

    async def tick(self) -> StepResult:
        self.state.tick += 1
        bio_summary: dict[str, dict] = {}

        if self.biomath:
            # 1. Run bio-math ODEs (compute mathematical ground truth)
            self.biomath.step(
                self.state.graph,
                events=self.state.event_queue,
                dt=1.0,
            )
            constraints = self.biomath.compute_constraints()

            # 2. Run LLM agents with math context + constraints injected
            tasks = [
                self._run_agent(node, constraints.get(node.id))
                for node in self.state.graph.nodes
            ]
            actions = await asyncio.gather(*tasks)

            # 3. Validate actions against conservation laws
            validated = self.biomath.validate_actions(list(actions))

            # 4. Apply validated actions
            self._apply_actions(validated)

            # 5. Sync bio-state to graph (overwrites metrics with math-computed values)
            self.biomath.sync_to_graph(self.state.graph)

            bio_summary = self.biomath.get_bio_summary()
            final_actions = validated
        else:
            # Legacy path: no bio-math, pure LLM
            tasks = [self._run_agent(node) for node in self.state.graph.nodes]
            actions = await asyncio.gather(*tasks)
            self._apply_actions(list(actions))
            final_actions = list(actions)

        self.state.event_queue.clear()

        result = StepResult(
            tick=self.state.tick,
            graph=self.state.graph.model_copy(deep=True),
            actions=final_actions,
            global_metrics=dict(self.state.graph.global_metrics),
            bio_summary=bio_summary,
        )
        self.state.history.append(result)
        return result

    def inject_event(self, event: dict) -> None:
        self.state.event_queue.append(event)

    @property
    def is_complete(self) -> bool:
        return self.state.tick >= self.state.max_ticks
