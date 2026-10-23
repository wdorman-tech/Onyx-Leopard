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
    b.price = a.price * 2
    asyncio.run(sim.step())
    assert a.market_share > b.market_share
    # Customer bases start at the logit share of day-1 offers, all else equal.
    expected = 1 / (1 + 2**-sim.share_price_elasticity)
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


def test_lv_rest_point_is_the_target_share_so_small_firms_survive() -> None:
    alpha = dynamics.competition_matrix(4, 0.3, 0.1, np.random.default_rng(1))
    target = np.array([0.62, 0.2, 0.13, 0.05])
    k = dynamics.lv_capacities_for_equilibrium(alpha, target)
    pops = np.full(4, 0.25)
    for _ in range(3000):
        pops = dynamics.lotka_volterra_step(pops, np.full(4, 0.05), k, alpha)
    assert pops == pytest.approx(target, rel=1e-3), "no competitive exclusion at rest"


def test_market_share_does_not_follow_headcount(library: NodeLibrary) -> None:
    sim = sim_for(days=120, ceos=["bootstrap", "bootstrap"], shocks_enabled=False)
    a, b = sim.companies
    # Staff with no effect on satisfaction, reach or price: pure headcount.
    plain = next(
        k
        for k, n in library.nodes.items()
        if n.category == "ops"
        and n.employees_count > 0
        and not library.effects(k)
        and a.seed.economics_model in n.applicable_economics
        and library.prerequisites_met(k, b.spawned)
    )
    b.spawned[plain] = b.spawned.get(plain, 0) + library.nodes[plain].category_caps.hard_cap
    assert b.employees > a.employees
    for _ in range(90):
        asyncio.run(sim.step())
    assert a.market_share == pytest.approx(b.market_share, abs=0.01)


def test_sales_headcount_buys_share_with_diminishing_returns(library: NodeLibrary) -> None:
    sim = sim_for(days=30, ceos=["bootstrap", "bootstrap"], shocks_enabled=False)
    plain, seller = sim.companies
    # Stack every sales/marketing node that is open to this business, to its cap.
    for key, node in library.nodes.items():
        if (
            library.effects(key).get("marketing", 0) > 0
            and node.category in ("sales", "marketing")
            and seller.seed.economics_model in node.applicable_economics
        ):
            seller.spawned[key] = node.category_caps.hard_cap
    reach = [c.attraction_inputs()[1] for c in (plain, seller)]
    assert reach[1] / reach[0] > 10
    asyncio.run(sim.step())
    pull_ratio = seller.target_share / plain.target_share
    assert pull_ratio == pytest.approx((reach[1] / reach[0]) ** 0.3, rel=0.01)
    assert seller.target_share < 0.8, "a sales army wins share, it does not wipe out rivals"


def test_customers_move_toward_a_better_offer_over_weeks() -> None:
    sim = sim_for(days=365, ceos=["bootstrap"] * 4, shocks_enabled=False)
    asyncio.run(sim.step())
    cutter = sim.companies[0]
    cutter.price *= 0.8
    asyncio.run(sim.step())
    start, target = cutter.market_share, cutter.target_share
    assert start == pytest.approx(0.25, abs=1e-3), "no overnight jump"
    assert target > start + 0.01
    for _ in range(60):
        asyncio.run(sim.step())
    closed = (cutter.market_share - start) / (target - start)
    assert 0.4 < closed < 0.9, (
        f"a mid-share base closes most of the gap in ~2 months ({closed:.2f})"
    )


def test_matched_price_cut_shrinks_an_inelastic_market() -> None:
    def spend_at(mult: float) -> tuple[float, float]:
        sim = sim_for(days=30, ceos=["bootstrap"] * 3, shocks_enabled=False, rng_seed=4)
        for c in sim.companies:
            c.price = c.seed.starting_price * mult
        asyncio.run(sim.step())
        return sim.daily_spend, sim.market_price_elasticity

    (base, eps), (cut, _) = spend_at(1.0), spend_at(0.5)
    assert eps < 1.0
    assert cut / base == pytest.approx(0.5 ** (1 - eps), rel=1e-6)


def test_starting_price_is_the_nash_price() -> None:
    sim = sim_for(days=30, ceos=["bootstrap"] * 4, shocks_enabled=False)
    asyncio.run(sim.step())
    company = sim.companies[0]
    # Lerner condition: gross margin = 1 / own-price elasticity at the start.
    assert company.gross_margin() * company.own_price_elasticity == pytest.approx(1.0, rel=1e-3)
    margin, share, eps_m = 0.4, 0.2, 0.7
    eps_s = dynamics.nash_share_elasticity(margin, share, eps_m)
    assert dynamics.own_price_elasticity(eps_s, share, eps_m, share) == pytest.approx(1 / margin)


def test_shock_type_cannot_stack_on_itself() -> None:
    from src.simulation.shocks import ShockScheduler

    # λ ≈ 45 arrivals per type per tick: without dead time every type would stack.
    scheduler = ShockScheduler("market", rng_seed=0, rate_multiplier=50_000.0)
    first = scheduler.arrivals(1)
    assert len({s.name for s in first}) == len(first), "one arrival per type per tick"
    running = [s for s in first if s.name == "market_crash"]
    assert running and not [s for s in scheduler.arrivals(2, running) if s.name == "market_crash"]
    assert [s for s in scheduler.arrivals(2) if s.name == "market_crash"], "free once it ends"


def test_severity_compounds_in_log_space_and_impacts_scale_with_the_company() -> None:
    from src.simulation.shocks import MULT_SUFFIX, ONE_SHOT_KEYS, SEVERITY_INTENSITY, SHOCK_TYPES

    rng = random.Random(0)
    worst = min(
        make_shock("market_crash", "severe", 1, rng).impact["market_demand_mult"]
        for _ in range(200)
    )
    assert worst >= (1 - 0.20) ** SEVERITY_INTENSITY["severe"] - 1e-9 > 0.5
    for spec in SHOCK_TYPES.values():
        for key in spec["impact"]:
            assert key.endswith(MULT_SUFFIX) or key in ONE_SHOT_KEYS, f"{key} is a fixed amount"


def test_core_sales_never_exceed_market_spend() -> None:
    for p in run(sim_for(days=365, shocks_enabled=False))[::7]:
        core = sum(c["core_revenue"] for c in p["companies"])
        assert core <= p["tam"] / 365 * (1 + 1e-9)
