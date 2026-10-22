from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.simulation.ceo import HeuristicCeo
from src.simulation.engine import HEURISTIC_CADENCE_DAYS
from src.simulation.jev import BASELINE_ALPHA, JevJudge, build_state
from src.simulation.library import NodeLibrary
from tests.conftest import make_company


class FakeTypeSafe:
    """Async stand-in for typesafe_sdk.AsyncTypeSafeClient: replays Noul answers."""

    def __init__(self, answers: list[float] | Exception) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, *, state: dict[str, Any], questions: dict[str, Any]) -> Any:
        self.calls.append({"state": state, "questions": questions})
        if isinstance(self.answers, Exception):
            raise self.answers
        p = self.answers.pop(0)
        return SimpleNamespace(
            answers={"stronger": SimpleNamespace(noul=p)},
            usage=SimpleNamespace(input_tokens=400),
        )


def capacity_bound(library: NodeLibrary):  # type: ignore[no-untyped-def]
    company = make_company(library, archetype="venture_growth")
    company.cash = company.seed.starting_cash * 100
    company.binding, company.utilization = "capacity", 1.0
    return company


def propose(ceo: HeuristicCeo, company) -> Any:  # type: ignore[no-untyped-def]
    proposal = None
    for i in range(1, 4):
        proposal = ceo.propose(company, HEURISTIC_CADENCE_DAYS * i)
    assert proposal is not None and proposal.open_locations == 1
    return proposal


def test_state_is_precomputed_and_question_is_literal(library: NodeLibrary) -> None:
    company = capacity_bound(library)
    proposal = propose(HeuristicCeo(), company)
    fake = FakeTypeSafe([0.6])
    asyncio.run(JevJudge(client=fake).review(company, proposal, 21))

    state = fake.calls[0]["state"]
    assert state["proposal"]["action"].startswith("open 1 more ")
    assert state["proposal"]["one_time_cost"] > 0
    assert state["proposal"]["cash_after_one_time_cost"] < company.cash
    assert state["ceo"]["archetype"] == "venture growth"
    json.dumps(state)  # plain JSON, nothing for Jev to compute
    noul = fake.calls[0]["questions"]["stronger"]
    assert "`proposal.action`" in str(noul)


def test_approval_commits_and_veto_leaves_no_side_effects(library: NodeLibrary) -> None:
    company = capacity_bound(library)
    judge = JevJudge(client=FakeTypeSafe([0.5, 0.2]))

    # First answer sets the baseline and is approved (deviation 0).
    ceo = HeuristicCeo()
    proposal = propose(ceo, company)
    first = asyncio.run(judge.review(company, proposal, 21))
    assert first is not None and first.approved
    assert first.decision.source == "jev" and first.decision.open_locations == 1
    assert "approved" in first.decision.reasoning

    # A clearly lower answer is vetoed into a hold.
    ceo = HeuristicCeo()
    proposal = propose(ceo, company)
    second = asyncio.run(judge.review(company, proposal, 42))
    assert second is not None and not second.approved
    held = second.decision
    assert (held.open_locations, held.spawn, held.retire) == (0, [], [])
    assert held.reasoning.startswith("Held: rule proposed to open 1 more")
    assert second.baseline == pytest.approx((1 - BASELINE_ALPHA) * 0.5 + BASELINE_ALPHA * 0.2)

    before = company.locations
    ceo.defer()
    company.apply(held)
    assert company.locations == before
    assert ceo.propose(company, HEURISTIC_CADENCE_DAYS * 4) is None  # window restarts
    assert judge.status()["approved"] == 1 and judge.status()["vetoed"] == 1


def test_bad_typesafe_key_disables_jev_and_rules_act_alone(library: NodeLibrary) -> None:
    from typesafe_sdk import TypeSafeAuthenticationError

    company = capacity_bound(library)
    error = TypeSafeAuthenticationError(401, {"message": "bad key"}, {}, "bad key")
    judge = JevJudge(client=FakeTypeSafe(error))
    assert asyncio.run(judge.review(company, propose(HeuristicCeo(), company), 21)) is None
    assert not judge.available
    assert any("Jev rejected the TypeSafe key" in e for e in company.events)


def test_engine_routes_weekly_rules_through_jev(library: NodeLibrary) -> None:
    from src.main import StartRequest, build_sim

    sim = asyncio.run(
        build_sim(
            StartRequest(
                ceo_archetypes=["venture_growth", "bootstrap"],
                duration_days=365,
                ai_enabled=False,
                jev_enabled=False,
                shocks_enabled=False,
                rng_seed=8,
            )
        )
    )
    sim.jev = JevJudge(client=FakeTypeSafe([0.9, 0.1] * 500))

    async def go() -> None:
        while not sim.is_complete:
            await sim.step()

    asyncio.run(go())
    jev_calls = [d for c in sim.companies for d in c.decisions if d.source == "jev"]
    assert jev_calls, "rules fired and Jev judged them"
    assert all(d.jev_p is not None for d in jev_calls)
    assert not [d for c in sim.companies for d in c.decisions if d.source == "heuristic"]
    status = sim.jev.status()
    assert status["calls"] == len(jev_calls)
    assert status["approved"] + status["vetoed"] == status["calls"]
    assert build_state(sim.companies[0], jev_calls[0])["company"]["employees"] > 0
