from __future__ import annotations

import json
from functools import partial
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import main
from src.simulation.llm import Catalog, OpenRouterClient
from tests.conftest import GLM_FLASH, FakeRouter, completion

client = TestClient(main.app)


def events(session_id: str) -> list[dict[str, Any]]:
    client.post(f"/api/simulate/control/{session_id}", json={"action": "set_speed", "speed": 1e6})
    with client.stream("GET", f"/api/simulate/stream/{session_id}") as resp:
        return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


def test_config_reports_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert client.get("/api/config").json()["ai_available"] is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    assert client.get("/api/config").json()["ai_available"] is True


def test_archetypes_and_companies() -> None:
    arch = client.get("/api/archetypes").json()
    assert {a["id"] for a in arch["ceo"]} == {
        "founder_operator",
        "venture_growth",
        "bootstrap",
        "consolidator",
        "turnaround",
    }
    companies = client.get("/api/companies").json()
    assert {c["name"] for c in companies} >= {"Apple Inc."}
    assert all(c["seed"]["tam"] > 0 for c in companies)


def test_heuristic_run_streams_ticks_reports_and_complete() -> None:
    start = client.post(
        "/api/simulate/start",
        json={
            "ceo_archetypes": ["bootstrap", "venture_growth", "turnaround"],
            "duration_days": 90,
            "ai_enabled": False,
            "rng_seed": 5,
        },
    ).json()
    assert start["model"] is None
    assert len(start["companies"]) == 3

    stream = events(start["session_id"])
    ticks = [e for e in stream if e["type"] == "tick"]
    assert ticks[-1]["tick"] == 90
    assert len(ticks[-1]["companies"]) == 3
    assert [e["type"] for e in stream[-3:]] == ["reports", "saved", "complete"]
    assert {r["source"] for r in stream[-3]["reports"]} == {"heuristic"}

    saved = stream[-2]
    pdf = client.get(saved["files"]["report.pdf"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    csv_text = client.get(saved["files"]["timeseries.csv"]).text
    assert csv_text.splitlines()[0].startswith("day,company_id,company,ceo,alive,cash")
    assert len(csv_text.splitlines()) == 1 + 3 * 90  # daily samples for every company
    run = client.get(saved["files"]["run.json"]).json()
    assert run["run_id"] == saved["run_id"]
    assert {c["report"]["source"] for c in run["companies"]} == {"heuristic"}
    gone = client.post(f"/api/simulate/control/{start['session_id']}", json={"action": "play"})
    assert gone.status_code == 404


def test_imported_profile_run() -> None:
    apple = next(c for c in client.get("/api/companies").json() if c["name"] == "Apple Inc.")
    start = client.post(
        "/api/simulate/start",
        json={
            "seed": apple["seed"],
            "ceo_archetypes": ["consolidator", "bootstrap"],
            "duration_days": 60,
            "ai_enabled": False,
        },
    ).json()
    ticks = [e for e in events(start["session_id"]) if e["type"] == "tick"]
    assert ticks[-1]["companies"][0]["name"] == "Apple Inc."
    assert ticks[-1]["companies"][0]["daily_revenue"] > 1e8


def test_ai_run_without_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resp = client.post(
        "/api/simulate/start", json={"ceo_archetypes": ["bootstrap"], "ai_enabled": True}
    )
    assert resp.status_code == 400
    assert "OPENROUTER_API_KEY" in resp.json()["detail"]


def test_ai_run_end_to_end_with_mocked_openrouter(
    monkeypatch: pytest.MonkeyPatch, router: FakeRouter
) -> None:
    async def fake_catalog() -> Catalog:
        return Catalog(
            default_model=GLM_FLASH.id,
            reasoning_effort="low",
            tiers=[],
            anthropic=[],
            models={GLM_FLASH.id: GLM_FLASH},
            live=True,
        )

    def respond(body: dict[str, Any]) -> dict[str, Any]:
        if body["response_format"]["json_schema"]["name"] == "ceo_report":
            return completion(
                {
                    "performance_summary": "We held share.",
                    "what_went_well": "Pricing.",
                    "what_went_wrong": "Slow hiring.",
                    "key_decisions": ["Day 30: nudged marketing"],
                    "final_assessment": "Defensible.",
                },
                cost=0.001,
            )
        return completion(
            {
                "reasoning": "Hold steady, nudge marketing.",
                "references_stance": ["risk_tolerance"],
                "spawn": [],
                "retire": [],
                "open_locations": 0,
                "price": None,
                "marketing_intensity": 1.2,
                "raise_amount": None,
            },
            cost=0.001,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(main, "load_catalog", fake_catalog)
    monkeypatch.setattr(main, "OpenRouterClient", partial(OpenRouterClient, http=router.client()))
    router.respond = respond

    start = client.post(
        "/api/simulate/start",
        json={
            "ceo_archetypes": ["bootstrap", "venture_growth"],
            "duration_days": 60,
            "ai_enabled": True,
            "shocks_enabled": False,
            "budget_usd": 1.0,
            "rng_seed": 3,
        },
    ).json()
    assert start["model"] == GLM_FLASH.id

    stream = events(start["session_id"])
    assert sum(e["type"] == "thinking" for e in stream) == 2  # days 30 and 60
    ai_decisions = [
        d
        for e in stream
        if e["type"] == "tick"
        for c in e["companies"]
        for d in c["decisions"]
        if d["source"] == "ai"
    ]
    assert len(ai_decisions) == 4  # 2 companies × 2 monthly rounds
    reports = next(e for e in stream if e["type"] == "reports")["reports"]
    assert {r["source"] for r in reports} == {"ai"}
    assert stream[-1]["ai"]["calls"] == 6
    assert stream[-1]["ai"]["spent_usd"] == pytest.approx(0.006)


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/..%2F..%2Fsecrets/run.json",
        "/api/runs/abc/../../.env",
        "/api/runs/abc/notes.txt",
        "/api/runs/does-not-exist/report.pdf",
    ],
)
def test_run_files_are_whitelisted(path: str) -> None:
    assert client.get(path).status_code == 404


def test_config_reports_jev_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert client.get("/api/config").json()["jev_available"] is False
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    assert client.get("/api/config").json()["jev_available"] is True


def _fake_verify(status: str, credit: float | None = None):  # type: ignore[no-untyped-def]
    from src.simulation.llm import KeyCheck

    async def verify(_key: str) -> KeyCheck:
        return KeyCheck(status, credit)

    return verify


GOOD_KEY = "sk-or-v1-" + "a1b2c3d4" * 8


def test_onboarding_saves_key_once_and_keeps_other_env_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.simulation import llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main, "verify_key", _fake_verify("ok", 2.5))
    llm.ENV_FILE.write_text("TYPESAFE_API_KEY=ts-keep-me\n", encoding="utf-8")

    resp = client.post("/api/config/openrouter-key", json={"key": f"  {GOOD_KEY}\n"})
    assert resp.status_code == 200
    assert resp.json() == {"ai_available": True, "verified": True, "credit_remaining": 2.5}
    assert GOOD_KEY not in resp.text
    env = llm.ENV_FILE.read_text(encoding="utf-8").splitlines()
    assert env == ["TYPESAFE_API_KEY=ts-keep-me", f"OPENROUTER_API_KEY={GOOD_KEY}"]
    assert client.get("/api/config").json()["ai_available"] is True

    again = client.post("/api/config/openrouter-key", json={"key": GOOD_KEY.replace("a", "b")})
    assert again.status_code == 409
    assert llm.ENV_FILE.read_text(encoding="utf-8").count("OPENROUTER_API_KEY=") == 1


@pytest.mark.parametrize(
    "key", ["short", "sk-or-v1-abc\nEVIL=1" + "x" * 20, "has spaces in it at all ok"]
)
def test_onboarding_rejects_malformed_keys(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    from src.simulation import llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main, "verify_key", _fake_verify("ok"))
    assert client.post("/api/config/openrouter-key", json={"key": key}).status_code == 400
    assert not llm.ENV_FILE.exists()


def test_onboarding_refuses_key_openrouter_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.simulation import llm

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main, "verify_key", _fake_verify("rejected"))
    resp = client.post("/api/config/openrouter-key", json={"key": GOOD_KEY})
    assert resp.status_code == 400
    assert "rejected" in resp.json()["detail"]
    assert not llm.ENV_FILE.exists()


def test_verify_key_against_openrouter_shapes() -> None:
    import asyncio

    import httpx

    from src.simulation.llm import verify_key

    def reply(status: int, body: dict):  # type: ignore[no-untyped-def]
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(status, json=body))
        )

    ok = asyncio.run(verify_key(GOOD_KEY, reply(200, {"data": {"limit_remaining": 3.5}})))
    assert (ok.status, ok.credit_remaining) == ("ok", 3.5)
    bad = asyncio.run(verify_key(GOOD_KEY, reply(401, {"error": {"message": "User not found."}})))
    assert bad.status == "rejected"

    def offline(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    down = asyncio.run(
        verify_key(GOOD_KEY, httpx.AsyncClient(transport=httpx.MockTransport(offline)))
    )
    assert down.status == "unverified"


def test_post_mortems_stay_ai_written_when_decisions_hit_the_budget(
    library, router: FakeRouter
) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    from src.simulation.report import write_report
    from tests.conftest import make_company, make_llm

    company = make_company(library)
    llm = make_llm(router, budget=0.01)
    llm.budget.report_calls = 1
    llm.budget.record(0.009)  # decisions have used everything but the reserve
    assert not llm.available and llm.can_report
    router.respond = lambda _body: completion(
        {
            "performance_summary": "Held on.",
            "what_went_well": "Margins.",
            "what_went_wrong": "Late hires.",
            "key_decisions": ["Day 30: held"],
            "final_assessment": "Survived.",
        }
    )
    report = asyncio.run(write_report(company, llm))
    assert report.source == "ai"
