"""Tests for `seed_builder.py`.

The Anthropic client is mocked end-to-end — no real API calls. Each test
constructs a `SeedInterview` with an injected mock client whose
`messages.create` coroutine returns scripted assistant responses.
"""

from __future__ import annotations

import json
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.simulation.seed import ARCHETYPES as SEED_ARCHETYPES
from src.simulation.seed import CompanySeed
from src.simulation.seed_builder import (
    DEFAULT_MODEL,
    MAX_VALIDATION_RETRIES,
    InterviewResponse,
    InterviewSession,
    SeedInterview,
    seed_from_archetype,
)
from src.simulation.stance import ARCHETYPES as STANCE_ARCHETYPES
from src.simulation.stance import CeoStance

# ─────────────────────────────────────────────────────────────────────────────
# Mock plumbing
# ─────────────────────────────────────────────────────────────────────────────


def _content_block(text: str) -> Any:
    """Minimal stand-in for an Anthropic content block (only `.text` is read)."""
    block = MagicMock()
    block.text = text
    return block


def _mock_response(text: str) -> Any:
    """Minimal stand-in for `client.messages.create()`'s return value."""
    response = MagicMock()
    response.content = [_content_block(text)]
    return response


def _make_mock_client(scripted_replies: list[str]) -> Any:
    """Build a mock `anthropic.AsyncAnthropic` that yields each reply in order.

    Each call to `client.messages.create(...)` advances the script by one.
    Raises if the script is exhausted — that means the test under-budgeted
    its scripted replies and needs to add more.
    """
    client = MagicMock()
    iterator = iter(scripted_replies)

    async def _create(**_: Any) -> Any:
        try:
            return _mock_response(next(iterator))
        except StopIteration as exc:
            raise AssertionError(
                "mock client ran out of scripted replies — add more"
            ) from exc

    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=_create)
    return client


# Canonical valid FINAL_JSON payload used by tests that just need a successful
# completion. Mirrors the minimal-valid kwargs in `tests/test_seed.py`.
_VALID_PAYLOAD = {
    "seed": {
        "name": "Test Co.",
        "niche": "Test niche for unit tests",
        "archetype": "small_team",
        "industry_keywords": ["test", "fixture"],
        "location_label": "Location",
        "economics_model": "physical",
        "base_price": 14.0,
        "base_unit_cost": 4.0,
        "daily_fixed_costs": 300.0,
        "starting_cash": 50_000.0,
        "starting_employees": 5,
        "base_capacity_per_location": 80,
        "margin_target": 0.55,
        "revenue_per_employee_target": 200_000.0,
        "tam": 1e8,
        "competitor_density": 4,
        "market_growth_rate": 0.05,
        "customer_unit_label": "diners",
        "seasonality_amplitude": 0.15,
        "initial_supplier_types": ["primary_goods_supplier"],
        "initial_revenue_streams": ["storefront_sales"],
        "initial_cost_centers": ["cogs", "labor"],
        "initial_locations": 1,
        "initial_marketing_intensity": 0.3,
        "initial_quality_target": 0.7,
        "initial_price_position": "mid",
        "initial_capital_runway_months": 12.0,
        "initial_hiring_pace": "steady",
        "initial_geographic_scope": "local",
        "initial_revenue_concentration": 0.4,
        "initial_customer_acquisition_channel": "word_of_mouth",
    },
    "stance": {
        "archetype": "founder_operator",
        "risk_tolerance": 0.55,
        "growth_obsession": 0.5,
        "quality_floor": 0.7,
        "hiring_bias": "balanced",
        "time_horizon": "annual",
        "cash_comfort": 9.0,
        "signature_moves": [
            "stay close to the customer",
            "hire only when it hurts",
        ],
        "voice": (
            "I started this thing in my garage and I am still the one taking the "
            "calls when something breaks."
        ),
    },
}


def _final_json_reply(payload: dict | None = None) -> str:
    payload = payload or _VALID_PAYLOAD
    return f"FINAL_JSON: {json.dumps(payload)}"


# ─────────────────────────────────────────────────────────────────────────────
# start_interview
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_interview_returns_first_question() -> None:
    client = _make_mock_client(["What does your company do?"])
    interview = SeedInterview(client=client)

    response = await interview.start_interview("A coffee shop in Brooklyn.")

    assert isinstance(response, InterviewResponse)
    assert response.session_id
    assert response.question == "What does your company do?"
    assert response.complete is False
    assert response.seed is None
    assert response.stance is None

    # Session was created and stored.
    session = interview.get_session(response.session_id)
    assert session is not None
    assert session.user_description == "A coffee shop in Brooklyn."
    assert session.complete is False
    assert session.turn_count == 1
    # Conversation history: opener (user) + first question (assistant).
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_start_interview_calls_default_model() -> None:
    client = _make_mock_client(["Hello, what's the business?"])
    interview = SeedInterview(client=client)

    await interview.start_interview("describe me")

    create = client.messages.create
    assert create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert "system" in kwargs
    assert isinstance(kwargs["messages"], list) and kwargs["messages"]


# ─────────────────────────────────────────────────────────────────────────────
# Full interview flow — 8-10 turns produces valid models
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_interview_eight_turns_produces_valid_models() -> None:
    questions = [f"Question {i}?" for i in range(1, 9)]
    scripted = [*questions, _final_json_reply()]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    first = await interview.start_interview("A SaaS startup.")
    assert first.question == questions[0]

    sid = first.session_id

    # Eight Q&A turns: 7 follow-up questions, then the FINAL_JSON.
    for i in range(1, 8):
        resp = await interview.submit_answer(sid, f"answer {i}")
        assert resp.complete is False
        assert resp.question == questions[i]

    final = await interview.submit_answer(sid, "answer 8")

    assert final.complete is True
    assert final.question is None
    assert isinstance(final.seed, CompanySeed)
    assert isinstance(final.stance, CeoStance)
    assert final.seed.name == "Test Co."
    assert final.stance.archetype == "founder_operator"

    session = interview.get_session(sid)
    assert session is not None
    assert session.complete is True


@pytest.mark.asyncio
async def test_full_interview_ten_turns_produces_valid_models() -> None:
    """Same as 8-turn case but at the upper end of the spec'd range."""
    questions = [f"Question {i}?" for i in range(1, 11)]
    scripted = [*questions, _final_json_reply()]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    first = await interview.start_interview("A consultancy.")
    sid = first.session_id

    for i in range(1, 10):
        await interview.submit_answer(sid, f"answer {i}")

    final = await interview.submit_answer(sid, "answer 10")
    assert final.complete is True
    assert isinstance(final.seed, CompanySeed)
    assert isinstance(final.stance, CeoStance)


@pytest.mark.asyncio
async def test_final_json_with_markdown_fences_still_parses() -> None:
    """Claude sometimes wraps payloads in ```json``` fences despite instructions."""
    fenced_reply = f"```json\nFINAL_JSON: {json.dumps(_VALID_PAYLOAD)}\n```"
    client = _make_mock_client([fenced_reply])
    interview = SeedInterview(client=client)

    first = await interview.start_interview("A test business.")
    assert first.complete is True
    assert isinstance(first.seed, CompanySeed)


# ─────────────────────────────────────────────────────────────────────────────
# Validation retries
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bad_payload_then_good_payload_succeeds() -> None:
    """First FINAL_JSON has bad data; second is correct → interview succeeds."""
    bad_payload = json.loads(json.dumps(_VALID_PAYLOAD))
    bad_payload["seed"]["base_price"] = -5.0  # invalid: must be > 0

    scripted = [
        _final_json_reply(bad_payload),
        _final_json_reply(),  # corrected
    ]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    response = await interview.start_interview("A test business.")
    assert response.complete is True
    assert isinstance(response.seed, CompanySeed)

    session = interview.get_session(response.session_id)
    assert session is not None
    assert session.retries_used == 1
    assert len(session.errors) == 1
    assert "base_price" in session.errors[0]


@pytest.mark.asyncio
async def test_bad_json_triggers_retry_and_eventual_raise() -> None:
    """Three consecutive bad payloads exhaust retries and raise."""
    bad_payload = json.loads(json.dumps(_VALID_PAYLOAD))
    bad_payload["seed"]["base_price"] = -5.0

    # start + 2 retries before final raise = 3 total bad replies needed
    scripted = [_final_json_reply(bad_payload)] * (MAX_VALIDATION_RETRIES + 1)
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    with pytest.raises(ValueError, match=r"failed after .* validation retries"):
        await interview.start_interview("A test business.")


@pytest.mark.asyncio
async def test_unparseable_json_triggers_retry_and_raise() -> None:
    """`FINAL_JSON: {garbage` should be treated as a validation failure."""
    scripted = ["FINAL_JSON: {not valid json at all"] * (MAX_VALIDATION_RETRIES + 1)
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    with pytest.raises(ValueError, match=r"failed after .* validation retries"):
        await interview.start_interview("Test business.")


@pytest.mark.asyncio
async def test_missing_stance_key_triggers_retry() -> None:
    """Payload missing one of the required top-level keys is invalid."""
    bad = {"seed": _VALID_PAYLOAD["seed"]}  # no `stance` key
    good = _final_json_reply()

    scripted = [_final_json_reply(bad), good]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    response = await interview.start_interview("Test")
    assert response.complete is True
    session = interview.get_session(response.session_id)
    assert session is not None
    assert session.retries_used == 1


@pytest.mark.asyncio
async def test_retry_budget_is_per_session_not_global() -> None:
    """Two separate sessions each get their own retry budget."""
    bad_payload = json.loads(json.dumps(_VALID_PAYLOAD))
    bad_payload["seed"]["base_price"] = -5.0

    # Session A: bad → good (uses 1 retry, succeeds)
    # Session B: bad → good (also uses 1 retry, succeeds — proves budget isn't shared)
    scripted = [
        _final_json_reply(bad_payload),
        _final_json_reply(),
        _final_json_reply(bad_payload),
        _final_json_reply(),
    ]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    a = await interview.start_interview("Business A")
    b = await interview.start_interview("Business B")

    assert a.complete is True and b.complete is True
    assert a.session_id != b.session_id


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_answer_unknown_session_raises() -> None:
    client = _make_mock_client([])
    interview = SeedInterview(client=client)

    with pytest.raises(KeyError, match="unknown session_id"):
        await interview.submit_answer("does-not-exist", "any answer")


@pytest.mark.asyncio
async def test_submit_answer_after_completion_raises() -> None:
    client = _make_mock_client([_final_json_reply()])
    interview = SeedInterview(client=client)

    response = await interview.start_interview("Test")
    assert response.complete is True

    with pytest.raises(RuntimeError, match="already complete"):
        await interview.submit_answer(response.session_id, "more input")


@pytest.mark.asyncio
async def test_concurrent_sessions_are_isolated() -> None:
    """Multiple sessions on the same `SeedInterview` instance must not bleed state."""
    payload_a = json.loads(json.dumps(_VALID_PAYLOAD))
    payload_a["seed"]["name"] = "Company Alpha"
    payload_a["stance"]["archetype"] = "venture_growth"
    # venture_growth requires risk_tolerance in [0.75, 0.98]
    payload_a["stance"]["risk_tolerance"] = 0.85

    payload_b = json.loads(json.dumps(_VALID_PAYLOAD))
    payload_b["seed"]["name"] = "Company Beta"
    payload_b["stance"]["archetype"] = "bootstrap"
    payload_b["stance"]["risk_tolerance"] = 0.15

    scripted = [
        "First question for A?",
        "First question for B?",
        _final_json_reply(payload_a),
        _final_json_reply(payload_b),
    ]
    client = _make_mock_client(scripted)
    interview = SeedInterview(client=client)

    a = await interview.start_interview("Aggressive growth co.")
    b = await interview.start_interview("Lean co.")
    assert a.session_id != b.session_id

    a_done = await interview.submit_answer(a.session_id, "answer A")
    b_done = await interview.submit_answer(b.session_id, "answer B")

    assert a_done.complete is True and b_done.complete is True
    assert a_done.seed is not None and b_done.seed is not None
    assert a_done.seed.name == "Company Alpha"
    assert b_done.seed.name == "Company Beta"
    assert a_done.stance is not None and b_done.stance is not None
    assert a_done.stance.archetype == "venture_growth"
    assert b_done.stance.archetype == "bootstrap"

    # Sessions still distinct in the registry.
    sess_a = interview.get_session(a.session_id)
    sess_b = interview.get_session(b.session_id)
    assert sess_a is not None and sess_b is not None
    assert sess_a is not sess_b
    assert sess_a.user_description == "Aggressive growth co."
    assert sess_b.user_description == "Lean co."


# ─────────────────────────────────────────────────────────────────────────────
# No-LLM-at-construction guarantee
# ─────────────────────────────────────────────────────────────────────────────


def test_construction_does_not_call_llm() -> None:
    """Instantiating `SeedInterview` must not touch the API."""
    client = _make_mock_client([])
    interview = SeedInterview(client=client)

    # Mock client should have zero calls.
    assert client.messages.create.await_count == 0
    assert interview._sessions == {}


# ─────────────────────────────────────────────────────────────────────────────
# seed_from_archetype helper
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("archetype", SEED_ARCHETYPES)
def test_seed_from_archetype_returns_valid_pair(archetype: str) -> None:
    rng = random.Random(42)
    seed, stance = seed_from_archetype(
        archetype=archetype,
        niche="Test niche",
        name="Test Co",
        rng=rng,
    )

    assert isinstance(seed, CompanySeed)
    assert isinstance(stance, CeoStance)
    assert seed.archetype == archetype
    assert seed.name == "Test Co"
    assert seed.niche == "Test niche"
    assert stance.archetype in STANCE_ARCHETYPES


def test_seed_from_archetype_uses_archetype_default_stance() -> None:
    """Default stance archetype is the spec'd mapping per company archetype."""
    expected = {
        "solo_founder": "founder_operator",
        "small_team": "bootstrap",
        "venture_funded": "venture_growth",
        "enterprise": "consolidator",
    }
    for company_arch, stance_arch in expected.items():
        _seed, stance = seed_from_archetype(
            archetype=company_arch,
            niche="x",
            name="y",
            rng=random.Random(0),
        )
        assert stance.archetype == stance_arch


def test_seed_from_archetype_respects_stance_override() -> None:
    seed, stance = seed_from_archetype(
        archetype="venture_funded",
        niche="x",
        name="y",
        stance_archetype="turnaround",  # cross-product Monte Carlo
        rng=random.Random(0),
    )
    assert seed.archetype == "venture_funded"
    assert stance.archetype == "turnaround"


def test_seed_from_archetype_rejects_bad_company_archetype() -> None:
    with pytest.raises(ValueError, match="unknown company archetype"):
        seed_from_archetype(
            archetype="hyperscaler",
            niche="x",
            name="y",
            rng=random.Random(0),
        )


def test_seed_from_archetype_rejects_bad_stance_archetype() -> None:
    with pytest.raises(ValueError, match="unknown stance archetype"):
        seed_from_archetype(
            archetype="small_team",
            niche="x",
            name="y",
            stance_archetype="hypergrowth",
            rng=random.Random(0),
        )


def test_seed_from_archetype_is_deterministic_given_rng() -> None:
    a_seed, a_stance = seed_from_archetype(
        archetype="venture_funded",
        niche="x",
        name="y",
        rng=random.Random(123),
    )
    b_seed, b_stance = seed_from_archetype(
        archetype="venture_funded",
        niche="x",
        name="y",
        rng=random.Random(123),
    )
    assert a_seed == b_seed
    assert a_stance == b_stance


def test_seed_from_archetype_does_not_call_llm() -> None:
    """No mock needed — the helper must never touch the API."""
    # If this ever instantiates an Anthropic client, the call would fail
    # without ANTHROPIC_API_KEY. The fact that the test passes proves it's pure-Python.
    seed, stance = seed_from_archetype(
        archetype="solo_founder",
        niche="solo consultant",
        name="Solo Inc",
        rng=random.Random(0),
    )
    assert isinstance(seed, CompanySeed)
    assert isinstance(stance, CeoStance)


# ─────────────────────────────────────────────────────────────────────────────
# InterviewSession dataclass
# ─────────────────────────────────────────────────────────────────────────────


def test_interview_session_defaults() -> None:
    session = InterviewSession(id="s1", user_description="desc")
    assert session.id == "s1"
    assert session.user_description == "desc"
    assert session.messages == []
    assert session.seed_partial == {}
    assert session.stance_partial == {}
    assert session.complete is False
    assert session.errors == []
    assert session.retries_used == 0
    assert session.turn_count == 0


def test_interview_response_only_question_or_models() -> None:
    """Response is either a question (incomplete) or a (seed, stance) pair (complete)."""
    asking = InterviewResponse(session_id="s1", question="why?")
    assert asking.complete is False
    assert asking.seed is None and asking.stance is None

    # Construct dummy models for the complete case.
    seed = CompanySeed(**_VALID_PAYLOAD["seed"])
    stance = CeoStance(**_VALID_PAYLOAD["stance"])
    done = InterviewResponse(
        session_id="s1", complete=True, seed=seed, stance=stance
    )
    assert done.question is None
    assert done.complete is True
