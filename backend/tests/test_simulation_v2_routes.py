"""Tests for `simulation_v2.py` FastAPI routes."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.simulation.library_loader import _reset_library_cache, get_library
from src.simulation.seed import sample_seed_for_archetype
from src.simulation.stance import sample_stance


@pytest.fixture(autouse=True)
def reset_library():
    _reset_library_cache()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def library():
    return get_library()


def _valid_refs_for(library, economics_model):
    s, r, c = [], [], []
    for key, node in sorted(library.nodes.items()):
        if economics_model not in node.applicable_economics:
            continue
        if node.category == "supplier" and not s:
            s.append(key)
        elif node.category == "revenue" and not r:
            r.append(key)
        elif node.category == "ops" and not c:
            c.append(key)
    return s, r, c


@pytest.fixture
def valid_seed_dict(library):
    rng = random.Random(7)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    s, r, c = _valid_refs_for(library, seed.economics_model)
    seed = seed.model_copy(update={
        "initial_supplier_types": s,
        "initial_revenue_streams": r,
        "initial_cost_centers": c,
    })
    return seed.model_dump()


@pytest.fixture
def valid_stance_dict():
    rng = random.Random(7)
    return sample_stance("bootstrap", rng=rng).model_dump()


# ─── Health + metadata ──────────────────────────────────────────────────


def test_health_endpoint(client):
    r = client.get("/api/v2/simulate/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "v2"}


def test_archetypes_endpoint(client):
    r = client.get("/api/v2/simulate/archetypes")
    assert r.status_code == 200
    body = r.json()
    assert "seed_archetypes" in body
    assert "stance_archetypes" in body
    assert "small_team" in body["seed_archetypes"]
    assert "bootstrap" in body["stance_archetypes"]


def test_library_endpoint(client):
    r = client.get("/api/v2/simulate/library")
    assert r.status_code == 200
    body = r.json()
    assert body["node_count"] > 0
    assert "nodes" in body
    # Spot-check one node entry
    sample_key = next(iter(body["nodes"]))
    sample = body["nodes"][sample_key]
    assert "label" in sample
    assert "category" in sample
    assert "hire_cost" in sample
    assert "modifier_keys" in sample


# ─── Sampling helpers ──────────────────────────────────────────────────


def test_sample_seed_endpoint(client):
    r = client.post("/api/v2/simulate/seed/sample/small_team?rng_seed=42")
    assert r.status_code == 200
    seed = r.json()
    assert seed["archetype"] == "small_team"
    assert "starting_cash" in seed


def test_sample_seed_unknown_archetype(client):
    r = client.post("/api/v2/simulate/seed/sample/not_an_archetype")
    assert r.status_code == 400


def test_sample_stance_endpoint(client):
    r = client.post("/api/v2/simulate/stance/sample/bootstrap?rng_seed=42")
    assert r.status_code == 200
    stance = r.json()
    assert stance["archetype"] == "bootstrap"


def test_sample_stance_unknown_archetype(client):
    r = client.post("/api/v2/simulate/stance/sample/cowboy")
    assert r.status_code == 400


# ─── Start ─────────────────────────────────────────────────────────────


def test_start_creates_session(client, valid_seed_dict, valid_stance_dict):
    body = {
        "seed": valid_seed_dict,
        "stance": valid_stance_dict,
        "num_companies": 1,
        "duration_ticks": 30,
    }
    r = client.post("/api/v2/simulate/start", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "session_id" in payload
    assert payload["num_companies"] == 1
    assert payload["max_ticks"] == 30


def test_start_rejects_invalid_seed_refs(client, valid_stance_dict, library):
    rng = random.Random(7)
    seed = sample_seed_for_archetype("small_team", rng=rng)
    seed = seed.model_copy(update={
        "initial_supplier_types": ["totally_made_up_node"],
    })
    r = client.post("/api/v2/simulate/start", json={
        "seed": seed.model_dump(),
        "stance": valid_stance_dict,
        "duration_ticks": 30,
    })
    assert r.status_code == 400
    assert "node key" in r.json()["detail"].lower()


def test_start_validates_required_fields(client):
    r = client.post("/api/v2/simulate/start", json={"num_companies": 1})
    assert r.status_code == 422  # missing required seed/stance


# ─── Control ───────────────────────────────────────────────────────────


def test_control_pause_play(client, valid_seed_dict, valid_stance_dict):
    body = {
        "seed": valid_seed_dict,
        "stance": valid_stance_dict,
        "duration_ticks": 30,
    }
    start = client.post("/api/v2/simulate/start", json=body).json()
    sid = start["session_id"]

    r1 = client.post(f"/api/v2/simulate/control/{sid}", json={"action": "pause"})
    assert r1.status_code == 200
    assert r1.json()["paused"] is True

    r2 = client.post(f"/api/v2/simulate/control/{sid}", json={"action": "play"})
    assert r2.status_code == 200
    assert r2.json()["paused"] is False


def test_control_set_speed(client, valid_seed_dict, valid_stance_dict):
    body = {
        "seed": valid_seed_dict,
        "stance": valid_stance_dict,
        "duration_ticks": 30,
    }
    start = client.post("/api/v2/simulate/start", json=body).json()
    sid = start["session_id"]
    r = client.post(
        f"/api/v2/simulate/control/{sid}",
        json={"action": "set_speed", "speed": 0.01},
    )
    assert r.status_code == 200
    assert r.json()["speed"] == 0.01


def test_control_unknown_session(client):
    r = client.post("/api/v2/simulate/control/nonexistent", json={"action": "pause"})
    assert r.status_code == 404


def test_control_set_speed_requires_speed(client, valid_seed_dict, valid_stance_dict):
    body = {
        "seed": valid_seed_dict,
        "stance": valid_stance_dict,
        "duration_ticks": 30,
    }
    sid = client.post("/api/v2/simulate/start", json=body).json()["session_id"]
    r = client.post(
        f"/api/v2/simulate/control/{sid}",
        json={"action": "set_speed"},
    )
    assert r.status_code == 400


# ─── Stream ─────────────────────────────────────────────────────────────


def test_stream_emits_tick_events(client, valid_seed_dict, valid_stance_dict):
    """Run a short sim end-to-end via SSE; verify tick events arrive."""
    body = {
        "seed": valid_seed_dict,
        "stance": valid_stance_dict,
        "num_companies": 1,
        "duration_ticks": 10,
    }
    sid = client.post("/api/v2/simulate/start", json=body).json()["session_id"]

    # Set speed very high to drain the sim quickly
    client.post(f"/api/v2/simulate/control/{sid}", json={"action": "set_speed", "speed": 0.001})

    with client.stream("GET", f"/api/v2/simulate/stream/{sid}") as resp:
        assert resp.status_code == 200
        events_seen = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events_seen.append(line[6:])
            if len(events_seen) >= 6:  # 5 ticks + complete
                break

    assert len(events_seen) >= 1
    # Should see at least one tick event
    assert any('"type": "tick"' in e for e in events_seen)


def test_stream_unknown_session(client):
    r = client.get("/api/v2/simulate/stream/nonexistent")
    assert r.status_code == 404
