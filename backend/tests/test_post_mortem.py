"""Tests for the end-of-run CEO post-mortem agent."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.orchestrator import CeoDecision
from src.simulation.post_mortem import (
    POST_MORTEM_MODEL_ID,
    CEOReport,
    PostMortemAgent,
)
from src.simulation.replay import CostTracker
from src.simulation.seed import CompanySeed
from src.simulation.stance import CeoStance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_seed() -> CompanySeed:
    return CompanySeed(
        name="Test Co",
        niche="B2B SaaS for QA testing",
        archetype="small_team",
        industry_keywords=["saas", "qa"],
        location_label="Product",
        economics_model="subscription",  # type: ignore[arg-type]
        starting_price=99.0,
        base_unit_cost=20.0,
        daily_fixed_costs=500.0,
        starting_cash=1_000_000.0,
        starting_employees=5,
        base_capacity_per_location=1000,
        margin_target=0.50,
        revenue_per_employee_target=200_000.0,
        tam=1e8,
        competitor_density=3,
        market_growth_rate=0.10,
        customer_unit_label="subscribers",
        seasonality_amplitude=0.10,
        initial_supplier_types=["cloud_provider"],
        initial_revenue_streams=["subscription_revenue"],
        initial_cost_centers=["engineering_payroll"],
        initial_locations=1,
        initial_marketing_intensity=0.30,
        initial_quality_target=0.75,
        initial_price_position="mid",
        initial_capital_runway_months=12.0,
        initial_hiring_pace="steady",
        initial_geographic_scope="national",
        initial_revenue_concentration=0.40,
        initial_customer_acquisition_channel="content",
        cobb_douglas_alpha=0.3,
        cobb_douglas_beta=0.7,
    )


def _make_stance() -> CeoStance:
    return CeoStance(
        archetype="founder_operator",
        risk_tolerance=0.55,
        growth_obsession=0.50,
        quality_floor=0.70,
        hiring_bias="balanced",
        time_horizon="annual",
        cash_comfort=6.0,
        signature_moves=[
            "stay close to the customer",
            "hire only when it hurts",
        ],
        voice="I run a tight ship and I take the calls myself.",
    )


def _make_decision(tick: int, reasoning: str = "Default reasoning citing stance.") -> CeoDecision:
    return CeoDecision(
        spawn_nodes=[],
        retire_nodes=[],
        adjust_params={},
        open_locations=0,
        reasoning=reasoning,
        references_stance=["risk_tolerance"],
        tier="strategic",
        tick=tick,
    )


def _make_company(
    *,
    alive: bool = True,
    cash: float = 250_000.0,
    monthly_revenue: float = 80_000.0,
    tick: int = 365,
    decisions: list[tuple[int, CeoDecision]] | None = None,
) -> Any:
    """Lightweight stand-in for `CompanyAgentV2` exposing only the
    attributes `PostMortemAgent` actually reads. Avoids the heavy
    library / orchestrator / shock-scheduler construction needed for a
    real agent."""
    return SimpleNamespace(
        seed=_make_seed(),
        stance=_make_stance(),
        company_id="co-test-1",
        alive=alive,
        tick=tick,
        cash=cash,
        monthly_revenue=monthly_revenue,
        daily_revenue=monthly_revenue / 30.0,
        daily_costs=monthly_revenue / 30.0 * 0.8,
        market_share=0.12,
        satisfaction=0.78,
        capacity_utilization=0.65,
        spawned_nodes={"founder_engineer": 1, "bd_rep": 2},
        consecutive_insolvent=0 if alive else 14,
        full_decision_log=list(decisions) if decisions else [
            (1, _make_decision(1, "Bootstrap the team.")),
            (30, _make_decision(30, "Defend price after demand wobble.")),
            (90, _make_decision(90, "Quarterly review — hold the line on hiring.")),
            (180, _make_decision(180, "Mid-year — push BD a bit harder.")),
            (300, _make_decision(300, "Year-end — protect cash, no exec hires.")),
        ],
    )


def _content_block(*, tool_input: dict[str, Any] | None = None, text: str | None = None) -> Any:
    block = MagicMock()
    if tool_input is not None:
        block.type = "tool_use"
        block.input = tool_input
        block.text = None
    else:
        block.type = "text"
        block.text = text or ""
        block.input = None
    return block


def _mock_response(
    *,
    tool_input: dict[str, Any] | None = None,
    text: str | None = None,
    input_tokens: int = 1500,
    output_tokens: int = 400,
) -> Any:
    response = MagicMock()
    response.content = [_content_block(tool_input=tool_input, text=text)]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    return response


def _make_mock_client(response: Any) -> Any:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# CEOReport schema
# ─────────────────────────────────────────────────────────────────────────────


def test_ceo_report_accepts_valid_payload() -> None:
    report = CEOReport(
        company_id="co-test-1",
        company_name="Test Co",
        strategy="founder_operator",
        performance_summary="Survived the year.",
        what_went_well="Hired conservatively.",
        what_went_wrong="Slow to react to demand wobble.",
        key_decisions=["Day 1: bootstrap", "Day 90: hold hiring"],
        final_assessment="Solid year; defensible position.",
    )
    assert report.company_id == "co-test-1"
    assert report.strategy == "founder_operator"
    assert len(report.key_decisions) == 2


def test_ceo_report_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        CEOReport(
            company_id="co-1",
            company_name="Test Co",
            strategy="founder_operator",
            performance_summary="Summary.",
            what_went_well="Well.",
            what_went_wrong="Wrong.",
            key_decisions=["one"],
            final_assessment="Final.",
            extra_field="not allowed",  # type: ignore[call-arg]
        )


def test_ceo_report_rejects_empty_required_strings() -> None:
    with pytest.raises(Exception):
        CEOReport(
            company_id="co-1",
            company_name="",  # min_length=1 violation
            strategy="founder_operator",
            performance_summary="x",
            what_went_well="x",
            what_went_wrong="x",
            key_decisions=[],
            final_assessment="x",
        )


def test_normalize_splits_item_tag_string() -> None:
    """Live failure mode: Haiku returns key_decisions as a string with
    <item>...</item> blocks instead of a JSON array."""
    raw = (
        "<item>Day 7: Retired the founder engineer</item>"
        "<item>Day 90: Spawned sales team</item>"
    )
    normalized = PostMortemAgent._normalize_llm_payload(
        {
            "performance_summary": "x",
            "what_went_well": "x",
            "what_went_wrong": "x",
            "key_decisions": raw,
            "final_assessment": "y",
        }
    )
    assert normalized["key_decisions"] == [
        "Day 7: Retired the founder engineer",
        "Day 90: Spawned sales team",
    ]


def test_normalize_synthesizes_missing_final_assessment() -> None:
    """Live failure mode: Haiku omits final_assessment entirely."""
    normalized = PostMortemAgent._normalize_llm_payload(
        {
            "performance_summary": "We started lean. We ended with $168M cash.",
            "what_went_well": "x",
            "what_went_wrong": "x",
            "key_decisions": ["one"],
        }
    )
    assert normalized["final_assessment"]
    # Last sentence of performance_summary is what we pulled.
    assert "168M" in normalized["final_assessment"]


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heuristic_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = PostMortemAgent()
    report = await agent.generate(_make_company(alive=True))
    assert isinstance(report, CEOReport)
    assert report.company_name == "Test Co"
    assert report.strategy == "founder_operator"
    # Survivor wording shows up in final_assessment.
    assert "Survived" in report.final_assessment


@pytest.mark.asyncio
async def test_heuristic_for_casualty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = PostMortemAgent()
    report = await agent.generate(
        _make_company(alive=False, cash=-5000.0, monthly_revenue=10_000.0)
    )
    assert "bankrupt" in report.performance_summary.lower()
    assert "Did not survive" in report.final_assessment


@pytest.mark.asyncio
async def test_heuristic_when_cost_ceiling_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Predicted post-mortem cost (Haiku, ~4000 in + 800 out tokens) is
    # ~$0.008 — a ceiling well below that triggers `would_exceed`.
    tracker = CostTracker(ceiling_usd=0.0001)
    client = _make_mock_client(_mock_response(text="should never be called"))
    agent = PostMortemAgent(cost_tracker=tracker, client=client)
    report = await agent.generate(_make_company(alive=True))
    assert isinstance(report, CEOReport)
    # LLM client must not have been called when preflight skips.
    assert client.messages.create.await_count == 0


@pytest.mark.asyncio
async def test_heuristic_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("transport boom"))
    agent = PostMortemAgent(client=client)
    report = await agent.generate(_make_company(alive=True))
    assert isinstance(report, CEOReport)
    # We do hit the SDK once before falling back.
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_heuristic_when_no_tool_use_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = _make_mock_client(_mock_response(text="plain text, no tool call"))
    agent = PostMortemAgent(client=client)
    report = await agent.generate(_make_company(alive=True))
    # Falls back to heuristic — still valid report, no exception.
    assert isinstance(report, CEOReport)


# ─────────────────────────────────────────────────────────────────────────────
# LLM happy path (forced tool-use)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tool_input = {
        "performance_summary": "Survived the year despite a Q3 wobble.",
        "what_went_well": "Defended price discipline; BD hired late but well.",
        "what_went_wrong": "Slow to react to satisfaction dip in Q2.",
        "key_decisions": [
            "Day 1: bootstrapped solo before adding BD",
            "Day 180: held price under competitor pressure",
            "Day 300: skipped exec hires to preserve cash",
        ],
        "final_assessment": "Defensible position; the founder_operator bet held.",
    }
    client = _make_mock_client(_mock_response(tool_input=tool_input))
    agent = PostMortemAgent(client=client)
    report = await agent.generate(_make_company(alive=True))

    assert report.company_id == "co-test-1"
    assert report.company_name == "Test Co"
    assert report.strategy == "founder_operator"
    assert report.performance_summary == tool_input["performance_summary"]
    assert report.key_decisions == tool_input["key_decisions"]
    # Tool was called exactly once.
    assert client.messages.create.await_count == 1
    # The tool-use forced choice must be wired correctly.
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == POST_MORTEM_MODEL_ID
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_ceo_report"}
    assert kwargs["tools"][0]["name"] == "submit_ceo_report"


@pytest.mark.asyncio
async def test_llm_records_cost_against_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tool_input = {
        "performance_summary": "x",
        "what_went_well": "x",
        "what_went_wrong": "x",
        "key_decisions": ["x"],
        "final_assessment": "x",
    }
    client = _make_mock_client(
        _mock_response(tool_input=tool_input, input_tokens=2000, output_tokens=500)
    )
    tracker = CostTracker(ceiling_usd=10.0)
    agent = PostMortemAgent(cost_tracker=tracker, client=client)

    await agent.generate(_make_company(alive=True))
    # Some non-zero cost should have been recorded.
    assert tracker.total_cost() > 0


@pytest.mark.asyncio
async def test_llm_prompt_includes_decision_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tool_input = {
        "performance_summary": "x",
        "what_went_well": "x",
        "what_went_wrong": "x",
        "key_decisions": ["x"],
        "final_assessment": "x",
    }
    client = _make_mock_client(_mock_response(tool_input=tool_input))
    agent = PostMortemAgent(client=client)

    company = _make_company(
        alive=True,
        decisions=[
            (1, _make_decision(1, "First move — bootstrap team.")),
            (90, _make_decision(90, "Q1 review — hold the line.")),
            (365, _make_decision(365, "Year-end — defensible exit.")),
        ],
    )
    await agent.generate(company)
    kwargs = client.messages.create.await_args.kwargs
    user_prompt = kwargs["messages"][0]["content"]
    payload = json.loads(user_prompt)
    assert payload["company"]["name"] == "Test Co"
    assert payload["final_state"]["alive"] is True
    assert payload["total_decisions"] == 3
    assert len(payload["decision_log_sample"]) >= 3
