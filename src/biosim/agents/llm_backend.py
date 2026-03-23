from __future__ import annotations

from typing import Protocol

from biosim.agents.department_agent import DepartmentAgent
from biosim.agents.prompts import DEPARTMENT_DOMAINS


class LLMBackend(Protocol):
    """Interface matching the DecisionRouter's expected protocol."""

    async def call(self, dept_index: int, company_index: int, context: dict) -> dict: ...


class CamelLLMBackend:
    """Manages a pool of DepartmentAgent instances per (company, dept) pair."""

    def __init__(self) -> None:
        self._agents: dict[tuple[int, int], DepartmentAgent] = {}

    def get_or_create_agent(
        self,
        company_index: int,
        dept_index: int,
        company_name: str,
        model_type: str = "haiku",
    ) -> DepartmentAgent:
        """Lazy-create a DepartmentAgent for this company/dept pair."""
        key = (company_index, dept_index)
        if key not in self._agents:
            dept_name = DEPARTMENT_DOMAINS[dept_index]["name"]
            self._agents[key] = DepartmentAgent(
                company_name=company_name,
                company_index=company_index,
                dept_index=dept_index,
                dept_name=dept_name,
                model_type=model_type,
            )
        return self._agents[key]

    async def call(self, dept_index: int, company_index: int, context: dict) -> dict:
        """Protocol method -- called by DecisionRouter for Tier 2/3."""
        company_name = context.get("company_name", f"Company_{company_index}")
        model_type = context.get("model_type", "haiku")
        agent = self.get_or_create_agent(company_index, dept_index, company_name, model_type)
        return await agent.decide(context)

    @property
    def total_cost_usd(self) -> float:
        """Sum cost across all agents."""
        return sum(agent.total_cost_usd for agent in self._agents.values())

    def remove_company(self, company_index: int) -> None:
        """Remove all agents for a dead company."""
        keys_to_remove = [k for k in self._agents if k[0] == company_index]
        for key in keys_to_remove:
            del self._agents[key]
