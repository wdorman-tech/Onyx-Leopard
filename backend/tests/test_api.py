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
