from __future__ import annotations

import json
import logging
from typing import Any

from biosim.agents.prompts import build_system_prompt
from biosim.agents.tools import build_simulation_tools

logger = logging.getLogger(__name__)

try:
    from camel.agents import ChatAgent
    from camel.configs import AnthropicConfig
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "haiku": {
        "model_id": "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "temperature": 0.2,
        "cost_per_m_tokens": 0.30,
    },
    "sonnet": {
        "model_id": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "temperature": 0.3,
        "cost_per_m_tokens": 4.0,
    },
}

_DEFAULT_RESPONSE = {
    "action": "hold",
    "parameters": {},
    "rationale": "Unable to parse LLM response; defaulting to hold.",
    "confidence": 0.0,
}


class DepartmentAgent:
    """A single department's AI agent, wrapping a CAMEL-AI ChatAgent.

    Lazy-initialized: the ChatAgent is only created on first decide() call.
    This avoids API key requirements during construction and testing.
    """

    def __init__(
        self,
        company_name: str,
        company_index: int,
        dept_index: int,
        dept_name: str,
        model_type: str = "haiku",
    ) -> None:
        self.company_name = company_name
        self.company_index = company_index
        self.dept_index = dept_index
        self.dept_name = dept_name
        self.model_type = model_type
        self._chat_agent: Any | None = None
        self._total_tokens: int = 0

    def _ensure_agent(self) -> Any:
        """Lazy-init the CAMEL ChatAgent on first use."""
        if self._chat_agent is not None:
            return self._chat_agent

        if not CAMEL_AVAILABLE:
            raise RuntimeError(
                "camel-ai is not installed. Install with: pip install 'camel-ai[anthropic]'"
            )

        system_prompt = build_system_prompt(self.company_name, self.dept_name, self.dept_index)
        cfg = _MODEL_CONFIGS[self.model_type]

        model = ModelFactory.create(
            model_platform=ModelPlatformType.ANTHROPIC,
            model_type=cfg["model_id"],
            model_config_dict=AnthropicConfig(
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            ).as_dict(),
        )

        tools = build_simulation_tools()
        self._chat_agent = ChatAgent(
            system_message=system_prompt,
            model=model,
            tools=tools if tools else None,
        )
        return self._chat_agent

    async def decide(self, context: dict) -> dict:
        """Make a decision given simulation context.

        Args:
            context: Dict with keys: financial_state, competitive_context,
                     recent_events, available_actions, constraints.

        Returns:
            Dict with keys: action, parameters, rationale, confidence.
        """
        agent = self._ensure_agent()
        message = self._format_context(context)

        try:
            response = agent.step(message)
            self._total_tokens += getattr(
                getattr(response, "info", None), "total_tokens", 0
            ) or 0
            return self._parse_response(response)
        except Exception:
            logger.exception(
                "Agent %s/%s decision failed", self.company_name, self.dept_name
            )
            return dict(_DEFAULT_RESPONSE)

    def _format_context(self, context: dict) -> str:
        """Format simulation context into a structured message for the agent."""
        parts = ["## Current Situation\n"]

        if "financial_state" in context:
            parts.append("### Financial State")
            parts.append(json.dumps(context["financial_state"], indent=2))

        if "competitive_context" in context:
            parts.append("\n### Competitive Context")
            parts.append(json.dumps(context["competitive_context"], indent=2))

        if "recent_events" in context:
            parts.append("\n### Recent Events")
            for event in context["recent_events"]:
                parts.append(f"- {event}")

        if "available_actions" in context:
            parts.append("\n### Available Actions")
            for action in context["available_actions"]:
                parts.append(f"- {action}")

        if "constraints" in context:
            parts.append("\n### Constraints")
            for constraint in context["constraints"]:
                parts.append(f"- {constraint}")

        parts.append(
            "\n\nRespond with a JSON object containing:"
            " action, parameters, rationale, confidence."
        )
        return "\n".join(parts)

    def _parse_response(self, response: Any) -> dict:
        """Extract structured decision from CAMEL ChatAgent response.

        Parse JSON from response content. Handle malformed responses gracefully.
        """
        try:
            content = response.msgs[-1].content if response.msgs else ""
        except (AttributeError, IndexError):
            content = str(response)

        try:
            result = json.loads(content)
            return _validate_decision(result)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: extract first top-level JSON object from mixed content
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                result = json.loads(content[start:end])
                return _validate_decision(result)
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("Could not parse JSON from agent response: %.200s", content)
        return dict(_DEFAULT_RESPONSE)

    @property
    def total_cost_usd(self) -> float:
        """Estimate cost based on token usage and model pricing."""
        cfg = _MODEL_CONFIGS[self.model_type]
        return (self._total_tokens / 1_000_000) * cfg["cost_per_m_tokens"]


def _validate_decision(result: dict) -> dict:
    """Ensure the decision dict has required keys with correct types."""
    return {
        "action": str(result.get("action", "hold")),
        "parameters": dict(result.get("parameters", {})),
        "rationale": str(result.get("rationale", "")),
        "confidence": float(result.get("confidence", 0.0)),
    }
