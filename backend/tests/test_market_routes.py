"""API-level tests for market simulation routes."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestMarketPresets:
    def test_get_presets_returns_four(self, client: TestClient):
        resp = client.get("/api/simulate/market/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4
        slugs = {p["slug"] for p in data}
        assert slugs == {"price-war", "innovation-race", "monopoly", "commodity"}

    def test_presets_have_required_fields(self, client: TestClient):
        resp = client.get("/api/simulate/market/presets")
        for preset in resp.json():
            assert "slug" in preset
            assert "name" in preset
            assert "description" in preset
            assert "alpha" in preset
            assert "beta" in preset


class TestMarketStart:
    def test_start_with_valid_preset(self, client: TestClient):
        resp = client.post("/api/simulate/start", json={
            "mode": "market",
            "preset": "price-war",
            "max_ticks": 10,
        })
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_start_with_invalid_preset(self, client: TestClient):
        resp = client.post("/api/simulate/start", json={
            "mode": "market",
            "preset": "nonexistent",
            "max_ticks": 10,
        })
        assert resp.status_code == 400

    def test_start_without_preset_in_market_mode(self, client: TestClient):
        resp = client.post("/api/simulate/start", json={
            "mode": "market",
            "max_ticks": 10,
        })
        assert resp.status_code == 400

    def test_growth_mode_still_works(self, client: TestClient):
        resp = client.post("/api/simulate/start", json={
            "mode": "growth",
            "industry": "restaurant",
            "max_ticks": 10,
        })
        assert resp.status_code == 200
        assert "session_id" in resp.json()


class TestMarketStream:
    def test_stream_emits_market_ticks(self, client: TestClient):
        start_resp = client.post("/api/simulate/start", json={
            "mode": "market",
            "preset": "monopoly",
            "max_ticks": 5,
        })
        session_id = start_resp.json()["session_id"]

        with client.stream("GET", f"/api/simulate/stream/{session_id}") as resp:
            ticks = []
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["type"] == "tick":
                    ticks.append(data)
                elif data["type"] == "complete":
                    break

        assert len(ticks) == 5
        first = ticks[0]
        assert first["mode"] == "market"
        assert "tam" in first
        assert "hhi" in first
        assert "agents" in first
        assert len(first["agents"]) >= 3  # monopoly starts with 3


class TestControlStillWorks:
    def test_pause_play_market_session(self, client: TestClient):
        start_resp = client.post("/api/simulate/start", json={
            "mode": "market",
            "preset": "commodity",
            "max_ticks": 100,
        })
        session_id = start_resp.json()["session_id"]

        resp = client.post(f"/api/simulate/control/{session_id}", json={
            "action": "pause",
        })
        assert resp.status_code == 200

        resp = client.post(f"/api/simulate/control/{session_id}", json={
            "action": "set_speed",
            "speed": 5,
        })
        assert resp.status_code == 200
