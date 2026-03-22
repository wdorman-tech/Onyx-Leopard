from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import CompanyGraph


@dataclass
class AgentAction:
    agent_id: str
    action_type: str  # hire, fire, reallocate_budget, launch_product, cut_costs, etc.
    params: dict
    reasoning: str


@dataclass
class StepResult:
    tick: int
    graph: CompanyGraph
    actions: list[AgentAction]
    global_metrics: dict[str, float]
    bio_summary: dict[str, dict] = field(default_factory=dict)


@dataclass
class SimulationState:
    graph: CompanyGraph
    tick: int = 0
    max_ticks: int = 50
    outlook: str = "normal"
    history: list[StepResult] = field(default_factory=list)
    event_queue: list[dict] = field(default_factory=list)
