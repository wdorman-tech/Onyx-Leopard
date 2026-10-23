from __future__ import annotations

import asyncio
import json
import random

import pytest

from src.simulation.ceo import AiCeo, HeuristicCeo, sanitize
from src.simulation.engine import HEURISTIC_CADENCE_DAYS
from src.simulation.library import NodeLibrary
from src.simulation.shocks import environment, make_shock
from tests.conftest import FakeRouter, completion, make_company, make_llm


def test_sanitize_enforces_library_and_engine_bounds(library: NodeLibrary) -> None:
    company = make_company(library, archetype="bootstrap")
    founder = library.founder()
    locked = next(
        k
        for k, n in library.nodes.items()
        if n.prerequisites
        and not library.prerequisites_met(k, company.spawned)
        and "physical" in n.applicable_economics
    )
    saas_only = next(
        k for k, n in library.nodes.items() if n.applicable_economics == ["subscription"]
    )

    d = sanitize(
        company,
        day=30,
        source="ai",
        reasoning="test",
        spawn=["not_a_node", locked, saas_only],
        retire=[founder, "not_spawned"],
        price=company.seed.starting_price * 50,
        marketing_intensity=9.0,
        raise_amount=1e9,
        references_stance=["cash_comfort", "made_up_trait"],
    )

    assert d.spawn == []
    assert d.retire == []
    assert d.price is None, "out of range is rejected, never clamped into a 90% cut"
    assert d.marketing_intensity == 2.0
    assert d.raise_amount is None  # bootstrap cannot raise
    assert d.references_stance == ["cash_comfort"]
    assert "[engine:" in d.reasoning


def test_sanitize_nets_opening_and_closing_the_same_node(library: NodeLibrary) -> None:
    """Seen live: a model sent open_locations=5 every month while retiring locations."""
    company = make_company(library, archetype="bootstrap")
    loc = company.primary_location
    assert loc is not None
    company.spawned[loc] = 10
    d = sanitize(
        company, day=30, source="ai", reasoning="shrink", retire=[loc] * 3, open_locations=5
    )
    assert (d.open_locations, d.retire.count(loc)) == (2, 0), "+5 -3 = +2, no paid churn"
    assert "netted" in d.reasoning

    hire = next(k for k in AiCeo._spawnable(company) if k != loc and k in company.spawned)
    d = sanitize(company, day=30, source="ai", reasoning="x", spawn=[hire], retire=[hire])
    assert d.spawn == [] and d.retire == []


def test_schema_says_levers_are_changes_not_targets(library: NodeLibrary) -> None:
    from src.simulation.ceo import decision_schema

    props = decision_schema(["a"], ["b"], locations=7)["properties"]
    assert "NEW" in props["open_locations"]["description"]
    assert "7 you run" in props["open_locations"]["description"]
    assert props["retire"]["items"]["enum"] == ["b"] and "description" in props["retire"]


def test_sanitize_caps_raises_for_allowed_archetypes(library: NodeLibrary) -> None:
    company = make_company(library, archetype="venture_growth")
    d = sanitize(company, day=1, source="ai", reasoning="raise", raise_amount=1e15)
    assert d.raise_amount == pytest.approx(company.seed.starting_cash)


def test_heuristic_opens_location_when_capacity_binds(library: NodeLibrary) -> None:
    company = make_company(library, archetype="venture_growth")
    company.cash = company.seed.starting_cash * 100
    company.binding, company.utilization = "capacity", 1.0
    ceo = HeuristicCeo()
    decisions = [ceo.decide(company, HEURISTIC_CADENCE_DAYS * i) for i in range(1, 4)]
    assert decisions[:2] == [None, None]
    assert decisions[2] is not None and decisions[2].open_locations == 1

    before = company.locations
    company.apply(decisions[2])
    assert company.locations == before + 1


def test_heuristic_cuts_costs_in_a_cash_crisis(library: NodeLibrary) -> None:
    company = make_company(library)
    company.cash, company.daily_revenue, company.daily_costs = 1_000.0, 100.0, 5_000.0
    d = HeuristicCeo().decide(company, HEURISTIC_CADENCE_DAYS)
    assert d is not None and len(d.retire) == 1
    assert library.nodes[d.retire[0]].category not in ("exec", "location")


def test_ai_ceo_prompt_schema_and_applied_decision(
    library: NodeLibrary, router: FakeRouter
) -> None:
    company = make_company(library, archetype="founder_operator")
    rival = make_company(library, company_id="co-1", seed_value=2)
    spawnable = AiCeo._spawnable(company)
    pick = spawnable[0]
    new_price = round(company.seed.starting_price * 0.9, 2)
    router.push(
        body=completion(
            {
                "reasoning": "Undercut the rival and add capacity.",
                "references_stance": ["growth_obsession"],
                "spawn": [pick],
                "retire": [],
                "open_locations": 1,
                "price": new_price,
                "marketing_intensity": 1.4,
                "raise_amount": None,
            }
        )
    )
    ceo = AiCeo(make_llm(router), company)

    decision = asyncio.run(ceo.decide(company, 30, [company, rival], 1e7))

    assert decision is not None and decision.source == "ai"
    assert decision.spawn == [pick] and decision.open_locations == 1
    body = router.bodies()[0]
    system, user = body["messages"][0]["content"], json.loads(body["messages"][1]["content"])
    assert company.stance.voice in system
    assert user["day"] == 30
    assert [row[0] for row in user["can_spawn"]] == spawnable
    assert user["market"]["rivals"][0]["name"] == rival.name
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["spawn"]["items"]["enum"] == spawnable
    assert set(schema["required"]) == set(schema["properties"])

    company.apply(decision)
    assert company.price == new_price
    assert company.marketing_intensity == 1.4
    assert company.spawned[pick] >= 1


def test_ai_prompt_shows_margins_elasticity_and_market_shocks(
    library: NodeLibrary, router: FakeRouter
) -> None:
    company = make_company(library, archetype="venture_growth")
    company.target_share, company.own_price_elasticity = 0.3, 1.8
    crash = make_shock("market_crash", "severe", 25, random.Random(0))
    router.push(body=completion({"reasoning": "Hold.", "spawn": [], "retire": []}))
    ceo = AiCeo(make_llm(router), company)

    asyncio.run(ceo.decide(company, 30, [company], 1e7, [crash]))

    body = router.bodies()[0]
    system, user = body["messages"][0]["content"], json.loads(body["messages"][1]["content"])
    you = user["you"]
    assert you["gross_margin"] == pytest.approx(
        1 - company.seed.base_unit_cost / company.seed.starting_price, abs=1e-3
    )
    assert you["price_elasticity"] == 1.8 and you["target_share"] == 0.3
    assert "monthly_gross_profit" in you
    assert user["market"]["category_price_elasticity"] == company.seed.market_price_elasticity
    assert [s["name"] for s in user["shocks"]] == ["market_crash"]
    assert user["shocks"][0]["scope"] == "market"
    assert "undercutting rivals wins share" not in system
    assert "best_response_price" in system
    assert {"settled_demand_vs_capacity", "best_response_price", "price_bounds"} <= set(you)


def test_ai_prompt_separates_list_price_from_what_customers_pay(
    library: NodeLibrary, router: FakeRouter
) -> None:
    company = make_company(library, archetype="venture_growth")
    rival = make_company(library, company_id="co-1", seed_value=2)
    entry = make_shock("new_competitor_entry", "severe", 25, random.Random(0))
    company.shocks.append(entry)
    company.env = environment(company.shocks)
    cap = company.seed.starting_price * company.env["price_ceiling_mult"]
    company.price = company.seed.starting_price * 1.2  # above the cap
    router.push(body=completion({"reasoning": "Hold.", "spawn": [], "retire": []}))

    asyncio.run(AiCeo(make_llm(router), company).decide(company, 30, [company, rival], 1e7))

    user = json.loads(router.bodies()[0]["messages"][1]["content"])
    you = user["you"]
    assert you["price"] == pytest.approx(company.price, rel=1e-3)
    assert (
        you["effective_price"] == pytest.approx(cap, rel=1e-3)
        and you["price_cap"] == you["effective_price"]
    )
    revenue = cap * company.revenue_mult
    assert you["revenue_per_customer"] == pytest.approx(revenue, rel=1e-3)
    assert you["gross_margin"] == pytest.approx(1 - company.unit_cost() / revenue, abs=1e-3)
    assert you["contribution_margin"] < you["gross_margin"]
    rival_row = user["market"]["rivals"][0]
    assert rival_row["price"] == pytest.approx(rival.price, rel=1e-3), "list price, like you.price"
    assert rival_row["effective_price"] == pytest.approx(rival.effective_price(rival.env), rel=1e-3)


def test_ai_budget_is_paced_across_the_whole_run(library: NodeLibrary, router: FakeRouter) -> None:
    from src.main import StartRequest, build_sim

    sim = asyncio.run(
        build_sim(
            StartRequest(
                ceo_archetypes=["bootstrap", "venture_growth"],
                duration_days=360,
                ai_enabled=False,
                jev_enabled=False,
                shocks_enabled=False,
                rng_seed=2,
            )
        )
    )
    cost = 0.001
    sim.llm = make_llm(router, budget=10 * cost)  # 5 of the 12 monthly rounds
    sim.ai = {c.company_id: AiCeo(sim.llm, c) for c in sim.companies}
    router.respond = lambda _body: completion({"reasoning": "Hold.", "spawn": []}, cost=cost)

    async def go() -> list[str]:
        events = []
        while not sim.is_complete:
            await sim.step()
            events += sim.payload()["events"]
        return events

    events = asyncio.run(go())
    days = sorted({d.day for c in sim.companies for d in c.decisions if d.source == "ai"})
    # Unpaced, the five affordable rounds were days 30-150 and the AI went quiet.
    assert days == [30, 90, 150, 240, 300]
    assert sim.ai_status()["paced_rounds"] == 5
    assert sum("ahead of pace" in e for e in events) == 1


def test_ai_ceo_failure_is_reported_not_raised(library: NodeLibrary, router: FakeRouter) -> None:
    company = make_company(library)
    router.push(400, {"error": {"code": 400, "message": "context too long"}})
    ceo = AiCeo(make_llm(router), company)

    assert asyncio.run(ceo.decide(company, 30, [company], 1e7)) is None
    assert any("Day 30" in e and "AI CEO call failed" in e for e in company.events)
    assert ceo.available


def test_bad_key_disables_ai_without_per_company_noise(
    library: NodeLibrary, router: FakeRouter
) -> None:
    company = make_company(library)
    router.push(401, {"error": {"code": 401, "message": "User not found."}})
    ceo = AiCeo(make_llm(router), company)

    assert asyncio.run(ceo.decide(company, 30, [company], 1e7)) is None
    assert company.events == []
    assert not ceo.available


def test_crisis_cuts_only_nodes_that_cost_more_than_they_earn(library: NodeLibrary) -> None:
    """Rounds 5-7: cut by net saving, not by category or by cost alone."""
    from src.simulation.library import COST_BUCKET_GAIN, REVENUE_BUCKET_GAIN

    company = make_company(library, economics="service")
    company.cash, company.daily_revenue, company.daily_costs = 1_000.0, 100.0, 5_000.0
    company.core_revenue = 50_000.0  # revenue streams here earn far more than they cost

    def saving(key: str) -> float:
        effects = library.effects(key)
        return (
            library.nodes[key].daily_fixed_costs * company.fixed_cost_mult
            - REVENUE_BUCKET_GAIN * effects.get("revenue", 0.0) * company.core_revenue
            + COST_BUCKET_GAIN * effects.get("cost", 0.0) * company._base_fixed_costs()
        )

    earners = [k for k in company.spawned if library.effects(k).get("revenue", 0) > 0]
    assert earners and all(saving(k) < 0 for k in earners)
    ceo = HeuristicCeo()
    for i in range(1, 30):
        d = ceo.decide(company, HEURISTIC_CADENCE_DAYS * i)
        if d is None:
            break
        assert d.retire and all(saving(k) > 0 for k in d.retire)
        assert not set(d.retire) & set(earners)
        company.apply(d)
