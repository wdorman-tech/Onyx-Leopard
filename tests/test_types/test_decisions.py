from __future__ import annotations

import pytest

from biosim.types.decisions import AgentDecision, DecisionBatch


class TestAgentDecision:
    def test_immutable(self) -> None:
        d = AgentDecision(
            tick=1, company_index=0, dept_index=0,
            tier=0, action="hold", rationale="stable", confidence=0.9,
        )
        with pytest.raises(AttributeError):
            d.tick = 2  # type: ignore[misc]

    def test_defaults(self) -> None:
        d = AgentDecision(
            tick=1, company_index=0, dept_index=3,
            tier=1, action="expand", rationale="growth", confidence=0.85,
        )
        assert d.cost_usd == 0.0
        assert d.input_tokens == 0
        assert d.output_tokens == 0
        assert d.metadata == {}

    def test_llm_fields(self) -> None:
        d = AgentDecision(
            tick=5, company_index=2, dept_index=7,
            tier=3, action="acquire", rationale="strategic fit",
            confidence=0.72, cost_usd=0.003,
            input_tokens=512, output_tokens=128,
            metadata={"model": "sonnet"},
        )
        assert d.tier == 3
        assert d.cost_usd == 0.003
        assert d.input_tokens == 512
        assert d.output_tokens == 128
        assert d.metadata["model"] == "sonnet"


class TestDecisionBatch:
    def test_empty_batch(self) -> None:
        batch = DecisionBatch(tick=1)
        assert batch.decisions == []
        assert batch.total_cost_usd == 0.0

    def test_add_accumulates_cost(self) -> None:
        batch = DecisionBatch(tick=10)
        d1 = AgentDecision(
            tick=10, company_index=0, dept_index=0,
            tier=0, action="hold", rationale="ode", confidence=1.0,
        )
        d2 = AgentDecision(
            tick=10, company_index=0, dept_index=7,
            tier=3, action="pivot", rationale="market shift",
            confidence=0.65, cost_usd=0.005,
        )
        batch.add(d1)
        batch.add(d2)
        assert len(batch.decisions) == 2
        assert batch.total_cost_usd == pytest.approx(0.005)

    def test_add_preserves_order(self) -> None:
        batch = DecisionBatch(tick=1)
        for i in range(5):
            batch.add(AgentDecision(
                tick=1, company_index=0, dept_index=i,
                tier=0, action=f"act_{i}", rationale="test", confidence=0.5,
            ))
        assert [d.dept_index for d in batch.decisions] == [0, 1, 2, 3, 4]
