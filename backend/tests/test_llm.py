from __future__ import annotations

import asyncio

import httpx
import pytest

from src.simulation import llm
from src.simulation.llm import LlmError, ModelInfo, load_catalog, parse_json_object
from tests.conftest import GLM_FLASH, FakeRouter, completion, make_llm

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def call(client: llm.OpenRouterClient) -> dict:
    return asyncio.run(
        client.complete_json(system="sys", user="usr", schema_name="t", schema=SCHEMA)
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", 0.0)


def test_request_uses_strict_schema_low_reasoning_and_auth(router: FakeRouter) -> None:
    router.push(body=completion({"answer": "hi"}, cost=0.0002))
    client = make_llm(router)

    assert call(client) == {"answer": "hi"}

    req = router.requests[0]
    assert req.url == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-or-test"
    body = router.bodies()[0]
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["provider"] == {"require_parameters": True}
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert client.budget.spent_usd == pytest.approx(0.0002)
    assert client.budget.calls == 1


def test_reasoning_picks_cheapest_supported_effort_or_none(router: FakeRouter) -> None:
    router.respond = lambda _body: completion({"answer": "x"})
    high_only = ModelInfo(
        id="m/high",
        name="m",
        supported_parameters=frozenset({"reasoning"}),
        reasoning_efforts=("max", "high"),
    )
    call(make_llm(router, model=high_only))
    no_reasoning = ModelInfo(id="m/plain", name="m")
    call(make_llm(router, model=no_reasoning))

    first, second = router.bodies()
    assert first["reasoning"]["effort"] == "high"
    assert "reasoning" not in second


def test_falls_back_to_prompt_json_when_no_structured_provider(router: FakeRouter) -> None:
    router.push(
        404,
        {
            "error": {
                "code": 404,
                "message": "No endpoints found that support the requested parameters",
            }
        },
    )
    router.push(body=completion('Sure! ```json\n{"answer": "ok"}\n```'))
    router.push(body=completion({"answer": "again"}))
    client = make_llm(router)

    assert call(client) == {"answer": "ok"}
    assert call(client) == {"answer": "again"}

    _, fallback, later = router.bodies()
    for body in (fallback, later):
        assert "response_format" not in body
        assert "provider" not in body
        assert '"required":["answer"]' in body["messages"][0]["content"]


def test_bad_key_disables_client_for_the_run(router: FakeRouter) -> None:
    router.push(401, {"error": {"code": 401, "message": "No auth credentials found"}})
    client = make_llm(router)

    with pytest.raises(LlmError) as err:
        call(client)
    assert err.value.fatal
    assert "401" in str(client.disabled_reason)
    assert not client.available

    with pytest.raises(LlmError):
        call(client)
    assert len(router.requests) == 1  # no second HTTP call


def test_transient_errors_retry(router: FakeRouter) -> None:
    router.push(429, {"error": {"message": "rate limited"}})
    router.push(502, {"error": {"message": "upstream"}})
    router.push(body=completion({"answer": "third time"}))

    assert call(make_llm(router)) == {"answer": "third time"}
    assert len(router.requests) == 3


def test_budget_stops_calls(router: FakeRouter) -> None:
    router.respond = lambda _body: completion({"answer": "x"}, cost=0.6)
    client = make_llm(router, budget=1.0)
    call(client)
    call(client)
    assert client.budget.exhausted

    with pytest.raises(LlmError, match="budget"):
        call(client)
    assert len(router.requests) == 2


def test_parse_json_object_tolerates_preamble() -> None:
    assert parse_json_object('Here you go:\n{"a": 1}\nThanks') == {"a": 1}
    with pytest.raises(LlmError):
        parse_json_object("no json here")


def test_catalog_enriches_tiers_and_lists_live_anthropic_models() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        data = [
            {
                "id": GLM_FLASH.id,
                "name": GLM_FLASH.name,
                "pricing": {"prompt": "0.00000003", "completion": "0.000000083"},
                "context_length": 1048576,
                "benchmarks": {"artificial_analysis": {"intelligence_index": 41.8}},
            },
            {"id": "anthropic/claude-haiku-5.5", "name": "Claude Haiku 5.5", "pricing": {}},
            {"id": "anthropic/claude-haiku-5.5:batch", "name": "batch", "pricing": {}},
            {"id": "openai/gpt-9", "name": "nope", "pricing": {}},
        ]
        return httpx.Response(200, json={"data": data})

    llm._catalog_cache = None
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    catalog = asyncio.run(load_catalog(client)).public()
    llm._catalog_cache = None

    assert catalog["live"] is True
    assert catalog["default"] == "z-ai/glm-5.3-flash"
    assert [t["tier"] for t in catalog["tiers"]] == ["bad", "mediocre", "good", "super"]
    good = next(t for t in catalog["tiers"] if t["tier"] == "good")
    assert good["prompt_per_million"] == pytest.approx(0.03)
    assert good["intelligence_index"] == 41.8
    assert [m["id"] for m in catalog["anthropic"]] == ["anthropic/claude-haiku-5.5"]


def test_catalog_offline_falls_back_to_static_list() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    llm._catalog_cache = None
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    catalog = asyncio.run(load_catalog(client)).public()

    assert catalog["live"] is False
    assert len(catalog["tiers"]) == 4
    assert any(m["id"].startswith("anthropic/") for m in catalog["anthropic"])


def test_api_key_is_read_live_from_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.api_key() is None
    llm.ENV_FILE.write_text("OTHER=1\nOPENROUTER_API_KEY=sk-or-pasted-later\n")
    assert llm.api_key() == "sk-or-pasted-later"


def test_reasoning_that_eats_the_budget_is_retried_once(router: FakeRouter) -> None:
    exhausted = completion("", cost=0.0003)
    exhausted["choices"][0]["finish_reason"] = "length"
    router.push(body=exhausted)
    router.push(body=completion({"answer": "second try"}, cost=0.0001))
    client = make_llm(router)

    assert call(client) == {"answer": "second try"}
    assert client.budget.calls == 2
    assert client.budget.spent_usd == pytest.approx(0.0004)

    router.push(body=exhausted)
    router.push(body=exhausted)
    with pytest.raises(LlmError, match="finish_reason=length"):
        call(client)
    assert len(router.requests) == 4


def test_budget_holds_back_a_reserve_for_ceo_reports(router: FakeRouter) -> None:
    from src.simulation.llm import MAX_REPORT_RESERVE_SHARE, REPORT_COST_MULTIPLE, Budget

    budget = Budget(ceiling_usd=1.0, report_calls=2)
    assert budget.reserve_usd == 0.0  # nothing to average yet
    budget.record(0.05)
    assert budget.reserve_usd == pytest.approx(2 * 0.05 * REPORT_COST_MULTIPLE)
    assert Budget(1.0, report_calls=50, spent_usd=0.1, calls=1).reserve_usd == pytest.approx(
        MAX_REPORT_RESERVE_SHARE
    )

    router.respond = lambda _body: completion({"answer": "x"}, cost=0.2)
    client = make_llm(router, budget=1.0)
    client.budget.report_calls = 1
    while client.available:
        call(client)
    # Decisions stopped short of the ceiling, leaving room for the report.
    assert client.budget.spent_usd < 1.0 and client.can_report
    with pytest.raises(LlmError, match="budget"):
        call(client)
    report = asyncio.run(
        client.complete_json(system="s", user="u", schema_name="t", schema=SCHEMA, use_reserve=True)
    )
    assert report == {"answer": "x"}
