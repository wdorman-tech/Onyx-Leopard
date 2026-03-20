from __future__ import annotations

import asyncio
import json
import logging

from camel.agents import ChatAgent

from src.agents.factory import create_haiku
from src.schemas import CompanyGraph, NodeData
from src.simulation.prompts import AGENT_SYSTEM_PROMPTS, OUTLOOK_CONTEXT, TICK_PROMPT
from src.simulation.state import AgentAction, SimulationState, StepResult

logger = logging.getLogger(__name__)

MAX_NODES = 20


class SimulationEngine:
    def __init__(self, graph: CompanyGraph, max_ticks: int = 50, outlook: str = "normal"):
        if len(graph.nodes) > MAX_NODES:
            raise ValueError(f"Max {MAX_NODES} nodes allowed, got {len(graph.nodes)}")
        self.state = SimulationState(graph=graph, max_ticks=max_ticks, outlook=outlook)
        self._agents: dict[str, ChatAgent] = {}
        self._build_agents()

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

    def _build_tick_context(self, node: NodeData) -> str:
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

        return TICK_PROMPT.format(
            tick=self.state.tick,
            company_context=company_context,
            node_metrics=json.dumps(node.metrics),
            outlook_context=OUTLOOK_CONTEXT.get(self.state.outlook, OUTLOOK_CONTEXT["normal"]),
            events=events_text,
        )

    async def _run_agent(self, node: NodeData) -> AgentAction:
        agent = self._agents[node.id]
        prompt = self._build_tick_context(node)
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
                case _:
                    pass  # report, collaborate, etc. — no metric changes

    async def tick(self) -> StepResult:
        self.state.tick += 1
        tasks = [self._run_agent(node) for node in self.state.graph.nodes]
        actions = await asyncio.gather(*tasks)
        self._apply_actions(list(actions))
        self.state.event_queue.clear()

        result = StepResult(
            tick=self.state.tick,
            graph=self.state.graph.model_copy(deep=True),
            actions=list(actions),
            global_metrics=dict(self.state.graph.global_metrics),
        )
        self.state.history.append(result)
        return result

    def inject_event(self, event: dict) -> None:
        self.state.event_queue.append(event)

    @property
    def is_complete(self) -> bool:
        return self.state.tick >= self.state.max_ticks
