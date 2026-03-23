from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from biosim.agents.department_agent import DepartmentAgent, _validate_decision


class TestDepartmentAgentInit:
    def test_init_does_not_require_camel(self):
        """Agent construction works without camel-ai."""
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        assert agent.company_name == "TestCorp"
        assert agent.company_index == 0
        assert agent.dept_index == 0
        assert agent.dept_name == "Finance"
        assert agent.model_type == "haiku"
        assert agent._chat_agent is None

    def test_init_with_sonnet(self):
        agent = DepartmentAgent("TestCorp", 0, 1, "R&D", model_type="sonnet")
        assert agent.model_type == "sonnet"

    def test_ensure_agent_raises_without_camel(self):
        """If camel-ai not installed, _ensure_agent raises RuntimeError."""
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        with patch("biosim.agents.department_agent.CAMEL_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="camel-ai is not installed"):
                agent._ensure_agent()


class TestCostTracking:
    def test_haiku_cost(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance", model_type="haiku")
        agent._total_tokens = 1_000_000
        assert abs(agent.total_cost_usd - 0.30) < 0.01

    def test_sonnet_cost(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance", model_type="sonnet")
        agent._total_tokens = 1_000_000
        assert abs(agent.total_cost_usd - 4.0) < 0.01

    def test_zero_tokens(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        assert agent.total_cost_usd == 0.0


class TestFormatContext:
    def test_format_all_fields(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        ctx = {
            "financial_state": {"cash": 100000, "revenue": 50000},
            "competitive_context": {"market_share": 0.2},
            "recent_events": ["Competitor launched new product"],
            "available_actions": ["adjust_budget", "hold"],
            "constraints": ["Budget cap at $200k"],
        }
        result = agent._format_context(ctx)
        assert "Financial State" in result
        assert "Competitive Context" in result
        assert "Recent Events" in result
        assert "Competitor launched new product" in result
        assert "adjust_budget" in result
        assert "Budget cap at $200k" in result

    def test_format_empty_context(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        result = agent._format_context({})
        assert "Current Situation" in result
        assert "JSON object" in result


class TestParseResponse:
    def test_parse_valid_json(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "action": "adjust_budget",
            "parameters": {"delta_pct": 5.0},
            "rationale": "Revenue is growing.",
            "confidence": 0.85,
        })
        mock_response.msgs = [mock_msg]

        result = agent._parse_response(mock_response)
        assert result["action"] == "adjust_budget"
        assert result["parameters"] == {"delta_pct": 5.0}
        assert result["confidence"] == 0.85

    def test_parse_json_in_markdown_fence(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = (
            'Here is my decision:\n```json\n'
            '{"action": "hold", "parameters": {}, "rationale": "Wait and see.", "confidence": 0.5}'
            "\n```"
        )
        mock_response.msgs = [mock_msg]

        result = agent._parse_response(mock_response)
        assert result["action"] == "hold"
        assert result["confidence"] == 0.5

    def test_parse_malformed_response(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "I think we should do something but I'm not sure what."
        mock_response.msgs = [mock_msg]

        result = agent._parse_response(mock_response)
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0

    def test_parse_empty_msgs(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_response = MagicMock()
        mock_response.msgs = []

        result = agent._parse_response(mock_response)
        assert result["action"] == "hold"


class TestValidateDecision:
    def test_missing_keys_filled(self):
        result = _validate_decision({})
        assert result["action"] == "hold"
        assert result["parameters"] == {}
        assert result["rationale"] == ""
        assert result["confidence"] == 0.0

    def test_correct_types(self):
        result = _validate_decision({
            "action": 123,
            "parameters": {},
            "rationale": 456,
            "confidence": "0.7",
        })
        assert isinstance(result["action"], str)
        assert isinstance(result["rationale"], str)
        assert isinstance(result["confidence"], float)


class TestDecideWithMock:
    def test_decide_returns_parsed_decision(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_chat_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "action": "adjust_budget",
            "parameters": {"delta_pct": 10},
            "rationale": "Invest in growth.",
            "confidence": 0.9,
        })
        mock_response = MagicMock()
        mock_response.msgs = [mock_msg]
        mock_response.info = MagicMock(total_tokens=500)
        mock_chat_agent.step.return_value = mock_response
        agent._chat_agent = mock_chat_agent

        result = asyncio.run(agent.decide({"financial_state": {"cash": 100000}}))
        assert result["action"] == "adjust_budget"
        assert result["confidence"] == 0.9
        mock_chat_agent.step.assert_called_once()

    def test_decide_handles_exception(self):
        agent = DepartmentAgent("TestCorp", 0, 0, "Finance")
        mock_chat_agent = MagicMock()
        mock_chat_agent.step.side_effect = RuntimeError("API failure")
        agent._chat_agent = mock_chat_agent

        result = asyncio.run(agent.decide({"financial_state": {}}))
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0
