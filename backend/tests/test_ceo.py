from __future__ import annotations

import asyncio
import json

import pytest

from src.simulation.ceo import AiCeo, HeuristicCeo, sanitize
from src.simulation.engine import HEURISTIC_CADENCE_DAYS
from src.simulation.library import NodeLibrary
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
    assert d.price is None
    assert d.marketing_intensity == 2.0
    assert d.raise_amount is None  # bootstrap cannot raise
    assert d.references_stance == ["cash_comfort"]
    assert "[engine:" in d.reasoning


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
