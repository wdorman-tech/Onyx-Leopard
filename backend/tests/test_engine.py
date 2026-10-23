from __future__ import annotations

import asyncio
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.main import COMPANIES_DIR, StartRequest, build_sim
from src.simulation import dynamics
from src.simulation.engine import INSOLVENT_DAYS_TO_DEATH, PRICE_BOUNDS_FACTOR, MarketSim
from src.simulation.library import LibraryValidationError, NodeLibrary, load_library
from src.simulation.seed import ARCHETYPES, sample_seed, seed_from_oleo
from src.simulation.shocks import environment, make_shock
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


def steps(sim: MarketSim, days: int) -> None:
    """Advance `days` days on one event loop (a loop per day exhausts Windows sockets)."""

    async def go() -> None:
        for _ in range(days):
            await sim.step()

    asyncio.run(go())


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


@pytest.mark.parametrize("archetype", ARCHETYPES)
@pytest.mark.parametrize("economics", ["physical", "subscription", "service"])
def test_sampled_seeds_start_in_the_designed_operating_margin_band(
    library: NodeLibrary, archetype: str, economics: str
) -> None:
    from src.simulation.seed import CAPACITY_HEADROOM, STARTING_OPERATING_MARGIN_BAND
    from tests.conftest import make_company

    for value in range(5):
        company = make_company(
            library, company_archetype=archetype, economics=economics, seed_value=value
        )
        seed = company.seed
        # At target revenue, with the uplift and cost bucket the engine applies.
        units = seed.base_capacity_per_location * seed.initial_locations / CAPACITY_HEADROOM
        revenue = units * seed.starting_price * company.revenue_mult
        costs = units * seed.base_unit_cost + company._base_fixed_costs() * company.fixed_cost_mult
        lo, hi = STARTING_OPERATING_MARGIN_BAND
        assert lo - 1e-3 <= 1 - costs / revenue <= hi + 1e-3


def test_oleo_import_keeps_the_profiles_gross_margin(library: NodeLibrary) -> None:
    from src.simulation.engine import Company
    from src.simulation.shocks import ShockScheduler
    from src.simulation.stance import sample_stance

    for path in sorted(COMPANIES_DIR.glob("*.oleo")):
        data = json.loads(path.read_text(encoding="utf-8"))
        margin = data["profile"].get("financials", {}).get("gross_margin")
        seed = seed_from_oleo(data, library)
        company = Company(
            company_id="co-0",
            name=seed.name,
            seed=seed,
            stance=sample_stance("bootstrap", random.Random(0)),
            library=library,
            rng=random.Random(0),
            shock_scheduler=ShockScheduler("company", 0, 0.0),
        )
        if isinstance(margin, float):
            assert company.gross_margin() == pytest.approx(margin, abs=1e-3), path.name


def test_seed_gross_margin_is_checked_against_revenue_per_customer(library: NodeLibrary) -> None:
    seed = sample_seed("small_team", random.Random(1), library, "service")
    uplift = library.starting_modifiers(
        [*seed.initial_supplier_types, *seed.initial_revenue_streams, *seed.initial_cost_centers],
        seed.economics_model,
        seed.initial_locations,
    ).revenue_multiplier()
    assert uplift > 1
    # Above list price but below revenue per customer: a valid business.
    library.validate_seed(seed.model_copy(update={"base_unit_cost": seed.starting_price * 1.1}))
    with pytest.raises(LibraryValidationError, match="negative gross margin"):
        library.validate_seed(
            seed.model_copy(update={"base_unit_cost": seed.starting_price * uplift * 1.01})
        )


@pytest.mark.parametrize("margin", [0.10, 0.5, 0.85])
@pytest.mark.parametrize("profile", ["amazon", "microsoft"])
def test_oleo_imports_start_within_caps_and_profitable(
    library: NodeLibrary, profile: str, margin: float
) -> None:
    data = json.loads((COMPANIES_DIR / f"{profile}.oleo").read_text(encoding="utf-8"))
    data["profile"]["financials"]["gross_margin"] = margin
    seed = seed_from_oleo(data, library)
    library.validate_seed(seed)  # day-0 node counts within hard caps
    sim = frozen(sim_for(days=60, ceos=["bootstrap"] * 4, shocks_enabled=False, seed=seed))
    steps(sim, 30)
    assert all(c.daily_revenue > c.daily_costs for c in sim.companies)


def test_thin_margin_oleo_import_starts_profitable(library: NodeLibrary) -> None:
    data = json.loads((COMPANIES_DIR / "microsoft.oleo").read_text(encoding="utf-8"))
    data["profile"]["financials"]["gross_margin"] = 0.02  # floored to the import minimum
    seed = seed_from_oleo(data, library)
    sim = sim_for(days=30, ceos=["bootstrap"] * 3, shocks_enabled=False, seed=seed)
    asyncio.run(sim.step())
    company = sim.companies[0]
    assert company.gross_margin() == pytest.approx(0.10, abs=1e-3)
    assert company.contribution_margin() > 0
    assert company.daily_revenue > company.daily_costs


def test_seed_refs_cannot_repeat_past_a_nodes_hard_cap(library: NodeLibrary) -> None:
    seed = sample_seed("small_team", random.Random(1), library, "service")
    stream = next(
        k for k in seed.initial_revenue_streams if library.nodes[k].category_caps.hard_cap == 1
    )
    padded = seed.model_copy(update={"initial_revenue_streams": [stream] * 3})
    with pytest.raises(LibraryValidationError, match="hard cap"):
        library.validate_seed(padded)


def test_oleo_import_with_a_missing_library_node_is_a_validation_error(
    library: NodeLibrary,
) -> None:
    from src.simulation.library import NodeLibrary as Library

    data = json.loads((COMPANIES_DIR / "microsoft.oleo").read_text(encoding="utf-8"))
    seed = seed_from_oleo(data, library)
    missing = seed.initial_revenue_streams[0]
    partial = Library({k: n for k, n in library.nodes.items() if k != missing})
    with pytest.raises(LibraryValidationError):
        seed_from_oleo(data, partial)


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
    steps(sim, run_days)
    baseline = [c.daily_revenue for c in sim.companies]

    crash = make_shock("market_crash", "severe", sim.tick + 1, random.Random(1))
    sim.active_market_shocks.append(crash)
    departure = make_shock("key_employee_departure", "severe", sim.tick + 1, random.Random(1))
    sim.companies[0].shocks.append(departure)
    steps(sim, 10)

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
    steps(sim, 90)
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
    steps(sim, 60)
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
    # Linear demand: spend = S0 · (P/P0) · (1 + ε (1 - P/P0)).
    assert cut / base == pytest.approx(0.5 * (1 + eps * 0.5), rel=1e-6)
    assert cut < base


def demand_bound_sim(library: NodeLibrary, economics: str, rivals: int, rng_seed: int = 7):  # type: ignore[no-untyped-def]
    """Rule CEOs, no shocks, flat TAM, capacity far above demand, deep cash."""
    seed = sample_seed("small_team", random.Random(rng_seed), library, economics)  # type: ignore[arg-type]
    seed = seed.model_copy(
        update={
            "base_capacity_per_location": seed.base_capacity_per_location * 1_000,
            "market_growth_rate": 0.0,
            "starting_cash": seed.starting_cash * 1_000,
        }
    )
    return sim_for(days=900, ceos=["bootstrap"] * rivals, shocks_enabled=False, seed=seed)


@pytest.mark.parametrize("rivals", [4, 10])
@pytest.mark.parametrize("economics", ["physical", "subscription", "service"])
def test_starting_price_is_the_nash_price_while_demand_binds(
    library: NodeLibrary, economics: str, rivals: int
) -> None:
    from src.simulation.engine import MARKETING_SPEND_FRACTION

    sim = demand_bound_sim(library, economics, rivals)
    asyncio.run(sim.step())
    company = sim.companies[0]
    assert company.binding == "demand"
    # Lerner on the contribution margin, at the starting price.
    margin = company.contribution_margin()
    assert margin * company.own_price_elasticity == pytest.approx(1.0, rel=1e-3)

    # Independent check of the first-order condition. Net revenue per unit of
    # list price is the revenue-stream uplift (from the library) times what is
    # left after revenue-linked marketing; long-run profit is flat at p0.
    p0, eps_m = company.seed.starting_price, company.seed.market_price_elasticity
    k = library.modifiers(company.spawned).revenue_multiplier() * (1 - MARKETING_SPEND_FRACTION)
    cost = company.seed.base_unit_cost

    def profit(p: float) -> float:
        prices = [p] + [p0] * (rivals - 1)
        shares = dynamics.logit_shares(
            [1.0] * rivals,
            [1.0] * rivals,
            prices,
            [1.0] * rivals,
            quality_exponent=1.0,
            marketing_exponent=1.0,
            price_elasticity=sim.share_price_elasticity,
        )
        index = 1 / sum(s / q for s, q in zip(shares, prices, strict=True))
        units = dynamics.linear_demand_spend(1.0, index, p0, eps_m) * shares[0] / p
        return units * (k * p - cost)

    h = p0 * 1e-4
    slope = (profit(p0 + h) - profit(p0 - h)) / (2 * h) * p0 / profit(p0)
    assert abs(slope) < 1e-3, f"d ln profit / d ln p at p0 = {slope:.4f}"


@pytest.mark.slow
@pytest.mark.parametrize("economics", ["subscription", "service"])
def test_no_unilateral_price_move_pays_once_customers_settle(
    library: NodeLibrary, economics: str
) -> None:
    """Engine-level: 10 rivals, demand binding, only company 0 moves its price."""
    from src.simulation.engine import MARKETING_SPEND_FRACTION

    def contribution(mult: float) -> float:
        sim = demand_bound_sim(library, economics, rivals=10)
        mover = sim.companies[0]
        mover.price = mover.seed.starting_price * mult

        async def go() -> float:
            total = 0.0
            for day in range(1, 801):
                await sim.step()
                if day > 700:  # customer bases have settled
                    total += mover.daily_revenue - mover.served * mover.unit_cost()
                    total -= (
                        MARKETING_SPEND_FRACTION * mover.marketing_intensity * mover.daily_revenue
                    )
            return total

        total = asyncio.run(go())
        assert mover.binding == "demand"
        return total

    base = contribution(1.0)
    for mult in (0.95, 1.05):
        # Deterministic sims: a correct calibration loses ~2-3e-4 either way, so
        # any tolerance here would let a miscalibrated margin through.
        assert contribution(mult) < base, f"price ×{mult} beat the starting price"


def frozen(sim: MarketSim) -> MarketSim:
    """Rule CEOs that never act, so only the test moves prices."""
    for ceo in sim.heuristic.values():
        ceo.propose = lambda company, day: None  # type: ignore[method-assign]
    return sim


@pytest.mark.parametrize("rivals", [4, 10])
def test_best_response_is_the_starting_price_in_a_demand_bound_market(
    library: NodeLibrary, rivals: int
) -> None:
    sim = demand_bound_sim(library, "subscription", rivals)
    asyncio.run(sim.step())
    company = sim.companies[0]
    best = sim.best_response_price(company)
    assert best is not None
    price, gain = best
    assert price == pytest.approx(company.seed.starting_price, rel=5e-3)
    assert gain == pytest.approx(0.0, abs=1e-6 * company.daily_revenue)


def test_best_response_raises_a_capacity_bound_price_until_demand_meets_capacity() -> None:
    sim = frozen(sim_for(days=365, ceos=["bootstrap"] * 4, shocks_enabled=False, rng_seed=21))
    asyncio.run(sim.step())
    company = sim.companies[0]
    assert company.binding == "capacity" and company.settled_demand_vs_capacity() > 1
    price, gain = sim.best_response_price(company)  # type: ignore[misc]
    assert price > company.price and gain > 0


def test_a_monopoly_on_inelastic_demand_has_a_finite_best_price() -> None:
    """Round 6: constant elasticity sent a lone firm to the 10x bound (profit x40-75)."""
    sim = frozen(sim_for(days=60, ceos=["bootstrap"], shocks_enabled=False))
    asyncio.run(sim.step())
    company = sim.companies[0]
    eps = company.seed.market_price_elasticity
    assert eps < 1
    choke = company.seed.starting_price * (1 + 1 / eps)
    price, _ = sim.best_response_price(company)  # type: ignore[misc]
    assert price < choke < company.seed.starting_price * PRICE_BOUNDS_FACTOR[1]
    # Demand-bound monopoly on linear demand: p* = (choke + per-unit cost) / 2.
    company.base_capacity_per_location = 0  # unused; capacity comes from the seed
    k = company.revenue_mult * (1 - company._marketing_share())
    interior = (choke + company.unit_cost() / k) / 2
    assert price >= interior * (1 - 1e-3), "capacity can only push the price up"


def test_an_offer_at_the_choke_sells_nothing_whatever_its_customer_base() -> None:
    """Round 8: a choked firm kept selling to its old customer base for months."""
    sim = frozen(sim_for(days=60, ceos=["bootstrap"] * 3, shocks_enabled=False, rng_seed=1))
    asyncio.run(sim.step())
    choked = sim.companies[0]
    choked.price = sim.choke_price * 1.5
    steps(sim, 7)
    assert choked.customer_base and choked.customer_base > 0.01, "customers linger"
    assert choked.served == 0.0 and choked.core_revenue == 0.0, "but nobody pays that price"


def test_a_copied_best_response_never_rounds_onto_the_choke() -> None:
    """Round 8 repro: 2109.398748 was shown as 2109.4, past a choke of 2109.39875."""
    from src.simulation.ceo import _price

    sim = sim_for(
        days=720,
        ceos=["consolidator", "turnaround"],
        shocks_enabled=False,
        rng_seed=3,
        company_archetype="solo_founder",
    )
    steps(sim, 30)
    for company in sim.alive:
        price, _ = sim.best_response_price(company)  # type: ignore[misc]
        assert _price(price) < sim.choke_price


def test_linear_demand_point_elasticity_and_choke() -> None:
    eps = 0.6
    assert dynamics.linear_demand_elasticity(1.0, 1.0, eps) == pytest.approx(eps)
    h = 1e-6
    numeric = -(
        math.log(dynamics.linear_demand_spend(1, 1.2 + h, 1, eps) / (1.2 + h))
        - math.log(dynamics.linear_demand_spend(1, 1.2 - h, 1, eps) / (1.2 - h))
    ) / (math.log(1.2 + h) - math.log(1.2 - h))
    assert dynamics.linear_demand_elasticity(1.2, 1.0, eps) == pytest.approx(numeric, rel=1e-5)
    assert dynamics.linear_demand_spend(1.0, 1 + 1 / eps, 1.0, eps) == 0.0
    assert math.isinf(dynamics.linear_demand_elasticity(1 + 1 / eps, 1.0, eps))


def test_best_response_is_finite_for_extreme_seeds(library: NodeLibrary) -> None:
    seed = sample_seed("small_team", random.Random(3), library, "physical")
    seed = seed.model_copy(
        update={
            "market_price_elasticity": 0.001,
            "base_capacity_per_location": seed.base_capacity_per_location / 10,
        }
    )
    sim = sim_for(days=60, ceos=["bootstrap"], shocks_enabled=False, seed=seed)
    asyncio.run(sim.step())
    price, gain = sim.best_response_price(sim.companies[0])  # type: ignore[misc]
    lo, hi = (seed.starting_price * f for f in PRICE_BOUNDS_FACTOR)
    assert lo <= price <= hi and math.isfinite(gain)


def test_under_a_price_cap_the_best_response_lists_at_the_uncapped_optimum() -> None:
    """Rounds 5-6: cutting list to the cap, or freezing it, loses money when the cap lifts."""
    sim = frozen(sim_for(days=60, ceos=["bootstrap"] * 3, shocks_enabled=False, rng_seed=21))
    asyncio.run(sim.step())
    company = sim.companies[0]
    # The best price for once the capping shock is over: the market without it.
    uncapped, _ = sim.best_response_price(company)  # type: ignore[misc]
    entry = make_shock("new_competitor_entry", "severe", sim.tick, random.Random(0))
    company.shocks = [entry]
    company.env = environment(company.shocks)
    cap = company.seed.starting_price * company.env["price_ceiling_mult"]
    assert cap < uncapped, "the cap binds at the free optimum"
    price, gain = sim.best_response_price(company)  # type: ignore[misc]
    assert price == pytest.approx(uncapped, rel=1e-3)
    assert gain == pytest.approx(0.0, abs=1e-9 * company.daily_revenue), "same pay under the cap"


def test_best_response_counts_the_customer_base_floor(library: NodeLibrary) -> None:
    """Round 5: with a steep elasticity the floor, not the logit, sets a high-price sliver."""
    seed = sample_seed("small_team", random.Random(11), library, "physical")
    seed = seed.model_copy(update={"base_unit_cost": seed.starting_price * 0.9})
    sim = frozen(sim_for(days=60, ceos=["bootstrap"] * 3, shocks_enabled=False, seed=seed))
    assert sim.share_price_elasticity == 10.0
    for rival, mult in zip(sim.companies[1:], (0.35, 2.49), strict=True):
        rival.price = seed.starting_price * mult
    steps(sim, 2)
    price, _ = sim.best_response_price(sim.companies[0])  # type: ignore[misc]
    # Harvesting the floor's sliver pays all the way up to what anyone will pay.
    assert sim.choke_price * 0.95 < price < sim.choke_price


def test_best_response_uses_capacity_added_since_the_morning() -> None:
    sim = frozen(sim_for(days=60, ceos=["bootstrap"] * 4, shocks_enabled=False, rng_seed=21))
    asyncio.run(sim.step())
    company = sim.companies[0]
    before, _ = sim.best_response_price(company)  # type: ignore[misc]
    loc = company.primary_location
    assert loc is not None and company.binding == "capacity"
    company.spawned[loc] += 3  # opened by a decision after today's operate()
    assert company.capacity_and_production(company.env)[0] > company.capacity
    after, _ = sim.best_response_price(company)  # type: ignore[misc]
    assert after < before, "more capacity to fill means a lower clearing price"


def test_no_best_response_work_once_the_ai_is_off(library: NodeLibrary, router) -> None:  # type: ignore[no-untyped-def]
    from src.simulation.ceo import AiCeo
    from tests.conftest import make_llm

    sim = sim_for(days=90, ceos=["bootstrap"] * 2, shocks_enabled=False)
    sim.llm = make_llm(router)
    sim.llm.disabled_reason = "test: AI off"
    sim.ai = {c.company_id: AiCeo(sim.llm, c) for c in sim.companies}

    def boom(company):  # type: ignore[no-untyped-def]
        raise AssertionError("best response computed with the AI off")

    sim.best_response_price = boom  # type: ignore[method-assign]
    steps(sim, 90)


@pytest.mark.parametrize("rng_seed", [21, 49, 89])
def test_following_the_best_response_converges_and_pays(rng_seed: int) -> None:
    """Round 3-4 review: jumping to local formulas each month cycled or spiralled."""

    def run(follow: bool) -> tuple[float, list[float]]:
        sim = frozen(
            sim_for(days=720, ceos=["bootstrap"] * 4, shocks_enabled=False, rng_seed=rng_seed)
        )
        mover = sim.companies[0]

        async def go() -> tuple[float, list[float]]:
            total, prices = 0.0, []
            while not sim.is_complete:
                await sim.step()
                total += mover.daily_revenue - mover.daily_costs
                if follow and sim.tick % 30 == 0:
                    mover.price = sim.best_response_price(mover)[0]  # type: ignore[index]
                    prices.append(mover.price)
            return total, prices

        return asyncio.run(go())

    followed, prices = run(follow=True)
    held, _ = run(follow=False)
    late = prices[len(prices) // 2 :]
    assert max(abs(b / a - 1) for a, b in itertools.pairwise(late)) < 0.03, late
    assert followed > held


def test_zero_marketing_costs_some_share_not_the_brand() -> None:
    from src.simulation.engine import MARKETING_INTENSITY_ELASTICITY, SHARE_MARKETING_EXPONENT

    sim = sim_for(days=30, ceos=["bootstrap", "bootstrap"], shocks_enabled=False)
    quiet, loud = sim.companies
    quiet.marketing_intensity = 0.0
    asyncio.run(sim.step())
    pull = 0.5 ** (2 * MARKETING_INTENSITY_ELASTICITY * SHARE_MARKETING_EXPONENT)
    assert quiet.target_share / loud.target_share == pytest.approx(pull, rel=1e-6)
    assert 0.85 < pull < 0.9, "stopping all marketing hurts, it does not erase the brand"


def test_own_price_elasticity_matches_a_finite_difference_at_asymmetric_prices() -> None:
    eps_s, eps_m = 1.3, 0.6
    prices = [0.8, 1.0, 1.25]

    def units(p0: float) -> float:
        ps = [p0, *prices[1:]]
        shares = dynamics.logit_shares(
            [1.0] * 3,
            [1.0] * 3,
            ps,
            [1.0] * 3,
            quality_exponent=1.0,
            marketing_exponent=1.0,
            price_elasticity=eps_s,
        )
        index = 1 / sum(s / p for s, p in zip(shares, ps, strict=True))
        return dynamics.linear_demand_spend(1.0, index, 1.0, eps_m) * shares[0] / p0

    h = 1e-6
    numeric = -(math.log(units(prices[0] * (1 + h))) - math.log(units(prices[0] * (1 - h)))) / (
        math.log(1 + h) - math.log(1 - h)
    )
    shares = dynamics.logit_shares(
        [1.0] * 3,
        [1.0] * 3,
        prices,
        [1.0] * 3,
        quality_exponent=1.0,
        marketing_exponent=1.0,
        price_elasticity=eps_s,
    )
    index = 1 / sum(s / p for s, p in zip(shares, prices, strict=True))
    unit_share = shares[0] / prices[0] * index
    category = dynamics.linear_demand_elasticity(index, 1.0, eps_m)  # at today's price level
    formula = dynamics.own_price_elasticity(eps_s, shares[0], category, unit_share)
    assert formula == pytest.approx(numeric, rel=1e-5)


@pytest.mark.parametrize(
    ("price", "cost", "ceos"),
    [(12.39, 12.39, 4), (0.5, 0.5, 2), (40.0, 39.7, 2), (13_346.0, 13_212.54, 10)],
)
def test_thin_or_zero_margin_seeds_run(library: NodeLibrary, price, cost, ceos) -> None:  # type: ignore[no-untyped-def]
    seed = sample_seed("small_team", random.Random(3), library, "physical").model_copy(
        update={"starting_price": price, "base_unit_cost": cost}
    )
    sim = sim_for(days=30, ceos=["bootstrap"] * ceos, shocks_enabled=False, seed=seed)
    assert sim.share_price_elasticity <= 10.0
    sim.companies[0].price = price * 1.01  # any gap is now a steep but finite pull
    steps(sim, 5)
    shares = [c.market_share for c in sim.companies]
    assert all(math.isfinite(x) for x in shares) and sum(shares) == pytest.approx(1.0)


def test_a_bankrupt_firms_customers_choose_by_todays_offers() -> None:
    sim = sim_for(days=60, ceos=["bootstrap"] * 3, shocks_enabled=False)
    asyncio.run(sim.step())
    big, bad, good = sim.companies
    big.customer_base, bad.customer_base, good.customer_base = 0.8, 0.1, 0.1
    bad.price *= 3  # a much worse offer than `good`
    big.alive, big.died_on = False, sim.tick
    asyncio.run(sim.step())
    # The freed 0.8 splits by today's targets; the stale bases were 50/50.
    expected = 0.1 + 0.8 * good.target_share
    assert good.target_share > 0.55
    assert good.market_share == pytest.approx(expected, rel=0.02)
    assert good.market_share > 0.55, "not in proportion to the stale bases"
    assert big.customer_base == 0.0


def test_one_shot_costs_use_todays_nodes_and_shocks(library: NodeLibrary) -> None:
    from src.simulation.engine import fixed_cost_multiplier

    sim = sim_for(days=60, ceos=["bootstrap", "bootstrap"], shocks_enabled=False)
    asyncio.run(sim.step())
    company = sim.companies[0]
    # Change the cost bucket after the last operate(): spawn a cost-cutting node.
    cutter = next(
        k
        for k, n in library.nodes.items()
        if library.effects(k).get("cost", 0) < 0
        and company.seed.economics_model in n.applicable_economics
    )
    company.spawned[cutter] = company.spawned.get(cutter, 0) + 1
    regulation = make_shock("regulatory_change", "severe", sim.tick + 1, random.Random(0))
    departure = make_shock("key_employee_departure", "minor", sim.tick + 1, random.Random(0))
    company.shock_scheduler.arrivals = lambda tick, active=(): [departure]  # type: ignore[method-assign]

    env = environment([regulation, departure])
    expected = (
        departure.impact["replacement_cost_employee_days"]
        * company._base_fixed_costs()
        * fixed_cost_multiplier(library.modifiers(company.spawned), env)
        / company.employees
    )
    assert fixed_cost_multiplier(library.modifiers(company.spawned), env) != company.fixed_cost_mult
    before = company.cash
    company.receive_shocks(sim.tick + 1, [regulation])
    assert before - company.cash == pytest.approx(expected)


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
