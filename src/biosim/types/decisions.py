from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class AgentDecision:
    """Immutable record of a single agent decision."""

    tick: int
    company_index: int
    dept_index: int
    tier: int  # 0=ODE, 1=Heuristic, 2=Haiku, 3=Sonnet
    action: str
    rationale: str
    confidence: float
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionBatch:
    """Collection of decisions from one tick across all agents."""

    tick: int
    decisions: list[AgentDecision] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def add(self, decision: AgentDecision) -> None:
        self.decisions.append(decision)
        self.total_cost_usd += decision.cost_usd
