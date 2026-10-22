from __future__ import annotations

import asyncio
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.main import COMPANIES_DIR, StartRequest, build_sim
from src.simulation import dynamics
from src.simulation.engine import INSOLVENT_DAYS_TO_DEATH, MarketSim
from src.simulation.library import LibraryValidationError, NodeLibrary, load_library
from src.simulation.seed import ARCHETYPES, sample_seed, seed_from_oleo
from src.simulation.shocks import make_shock
from src.simulation.stance import ARCHETYPES as CEO_ARCHETYPES

ALL_CEOS = list(CEO_ARCHETYPES)


def sim_for(days: int = 365, rng_seed: int = 42, **kw) -> MarketSim:
    req = StartRequest(
        ceo_archetypes=kw.pop("ceos", ALL_CEOS),
        duration_days=days,
        ai_enabled=False,
        rng_seed=rng_seed,
        **kw,
    )
    return asyncio.run(build_sim(req))


def run(sim: MarketSim) -> list[dict]:
    async def go() -> list[dict]:
        out = []
        while not sim.is_complete:
            await sim.step()
            out.append(sim.payload())
        return out

    return asyncio.run(go())


def test_library_loads_and_every_modifier_feeds_a_bucket(library: NodeLibrary) -> None:
    assert len(library.nodes) > 100
    for econ in ("physical", "subscription", "service"):
        assert library.primary_location(econ) is not None
    assert library.founder() is not None


def test_library_rejects_prerequisite_cycles(tmp_path: Path) -> None:
    node = {
        "category": "ops",
        "label": "X",
        "hire_cost": 1,
        "daily_fixed_costs": 1,
        "employees_count": 1,
        "capacity_contribution": 0,
        "category_caps": {"soft_cap": 1, "hard_cap": 1},
        "applicable_economics": ["service"],
    }
    path = tmp_path / "lib.yaml"
    path.write_text(
        yaml.safe_dump(
            {"a": {**node, "prerequisites": ["b"]}, "b": {**node, "prerequisites": ["a"]}}
        )
    )
    with pytest.raises(LibraryValidationError, match="cycle"):
        load_library(path)


@pytest.mark.parametrize("archetype", ARCHETYPES)
@pytest.mark.parametrize("economics", ["physical", "subscription", "service"])
def test_sampled_seeds_are_valid_and_start_solvent(
    library: NodeLibrary, archetype: str, economics: str
) -> None:
    seed = sample_seed(archetype, random.Random(3), library, economics)  # type: ignore[arg-type]
    library.validate_seed(seed)
    sim = sim_for(days=60, ceos=["bootstrap"], seed=seed)
    run(sim)
    company = sim.companies[0]
    assert company.alive
    assert company.daily_revenue > 0


def test_oleo_profiles_convert_to_valid_seeds(library: NodeLibrary) -> None:
    files = sorted(COMPANIES_DIR.glob("*.oleo"))
    assert len(files) == 4
    for path in files:
        seed = seed_from_oleo(json.loads(path.read_text(encoding="utf-8")))
        library.validate_seed(seed)


def test_runs_are_deterministic_for_a_seed() -> None:
    def fingerprint(payloads: list[dict]) -> list:
        return [(c["cash"], c["market_share"]) for c in payloads[-1]["companies"]]

    assert fingerprint(run(sim_for(rng_seed=9))) == fingerprint(run(sim_for(rng_seed=9)))
    assert fingerprint(run(sim_for(rng_seed=9))) != fingerprint(run(sim_for(rng_seed=10)))


@pytest.mark.slow
def test_five_year_run_invariants() -> None:
    payloads = run(sim_for(days=5 * 365))
    assert len(payloads) == 5 * 365
    for p in payloads[::30]:
        alive = [c for c in p["companies"] if c["alive"]]
        if alive:
            assert sum(c["market_share"] for c in alive) == pytest.approx(1.0)
        for c in p["companies"]:
            assert math.isfinite(c["cash"])
            assert 0.0 <= c["satisfaction"] <= 1.0
            assert c["graph"]["nodes"], "every company keeps at least its founder"


def test_higher_price_loses_share() -> None:
    sim = sim_for(days=30, ceos=["bootstrap", "bootstrap"], shocks_enabled=False)
    a, b = sim.companies
    b.market_strength = a.market_strength
    b.price = a.price * 2
    asyncio.run(sim.step())
    assert a.market_share > b.market_share
    expected = 1 / (1 + 2**-1.5)  # logit with elasticity 1.5, all else equal
    assert a.market_share == pytest.approx(expected, rel=0.05)


def test_negative_cash_for_a_month_is_bankruptcy() -> None:
    sim = sim_for(days=200, ceos=["bootstrap", "venture_growth"], shocks_enabled=False)
    doomed = sim.companies[0]

    async def go() -> None:
        for day in range(INSOLVENT_DAYS_TO_DEATH + 5):
            doomed.cash = -1e12
            await sim.step()
            if not doomed.alive:
                assert day + 1 == INSOLVENT_DAYS_TO_DEATH
                return
        raise AssertionError("company never died")

    asyncio.run(go())
    events = sim.payload()["events"]
    assert any("bankrupt" in e for e in events)


def test_market_shock_hits_every_company_and_satisfaction_recovers() -> None:
    sim = sim_for(days=400, ceos=["bootstrap", "turnaround"], shocks_enabled=False)
    run_days = 20
    for _ in range(run_days):
        asyncio.run(sim.step())
    baseline = [c.daily_revenue for c in sim.companies]

    crash = make_shock("market_crash", "severe", sim.tick + 1, random.Random(1))
    sim.active_market_shocks.append(crash)
    departure = make_shock("key_employee_departure", "severe", sim.tick + 1, random.Random(1))
    sim.companies[0].shocks.append(departure)
    for _ in range(10):
        asyncio.run(sim.step())

    assert all(c.env.get("market_demand_mult", 1.0) < 1.0 for c in sim.companies), (
        "market shocks reach everyone"
    )
    assert sim.companies[0].satisfaction > 0.25, "shock moves the target, not a daily compounding"
    assert any(c.daily_revenue <= b for c, b in zip(sim.companies, baseline, strict=True))


def test_lotka_volterra_coexistence_for_similar_firms() -> None:
    alpha = dynamics.competition_matrix(3, 0.3, 0.0, np.random.default_rng(0))
    pops, k = np.full(3, 10.0), np.full(3, 100.0)
    for _ in range(2000):
        pops = dynamics.lotka_volterra_step(pops, np.full(3, 0.05), k, alpha)
    assert pops == pytest.approx(100 / (1 + 0.3 * 2), rel=1e-3)
