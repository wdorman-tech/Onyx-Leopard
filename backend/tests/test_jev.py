from __future__ import annotations

import asyncio
import json
import math
import random
from types import SimpleNamespace
from typing import Any

import pytest

from src.simulation.ceo import HeuristicCeo
from src.simulation.engine import HEURISTIC_CADENCE_DAYS
from src.simulation.jev import (
    BASELINE_ALPHA,
    HORIZON_DAYS,
    PRIOR_VARIANCE,
    VETO_Z,
    JevJudge,
    build_state,
)
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


def cash_crisis(library: NodeLibrary, runway_days: float):  # type: ignore[no-untyped-def]
    company = make_company(library)
    company.daily_revenue, company.daily_costs = 100.0, 5_100.0
    company.cash = runway_days * 5_000.0
    return company


def test_state_is_precomputed_and_question_is_literal(library: NodeLibrary) -> None:
    company = capacity_bound(library)
    proposal = propose(HeuristicCeo(), company)
    fake = FakeTypeSafe([0.6])
    asyncio.run(JevJudge(client=fake).review(company, proposal, 21))

    state = fake.calls[0]["state"]
    assert state["proposal"]["action"].startswith("open 1 more ")
    assert state["proposal"]["one_time_cost"] > 0
    acts, holds = state["if_it_acts"], state["if_it_holds"]
    assert acts["cash_now"] == holds["cash_now"] - state["proposal"]["one_time_cost"]
    assert acts[f"cash_in_{HORIZON_DAYS}_days"] < holds[f"cash_in_{HORIZON_DAYS}_days"]
    assert "ceo" not in state, "Jev judges the company's outcome, not a persona's tastes"
    json.dumps(state)  # plain JSON, nothing for Jev to compute
    noul = str(fake.calls[0]["questions"]["stronger"])
    assert "`proposal.action`" in noul and "`if_it_acts`" in noul and "`if_it_holds`" in noul


def test_approval_commits_and_veto_leaves_no_side_effects(library: NodeLibrary) -> None:
    company = capacity_bound(library)
    judge = JevJudge(client=FakeTypeSafe([0.6, 0.05]))

    # First answer of a kind is its own pivot and goes ahead.
    ceo = HeuristicCeo()
    proposal = propose(ceo, company)
    first = asyncio.run(judge.review(company, proposal, 21))
    assert first is not None and first.approved
    assert first.decision.source == "jev" and first.decision.open_locations == 1
    assert "approved" in first.decision.reasoning

    # An answer far below the baseline (z < -VETO_Z under the prior spread) is vetoed.
    ceo = HeuristicCeo()
    proposal = propose(ceo, company)
    second = asyncio.run(judge.review(company, proposal, 42))
    assert second is not None and not second.approved
    assert (0.05 - 0.6) / math.sqrt(PRIOR_VARIANCE * (1 - BASELINE_ALPHA)) < -VETO_Z
    held = second.decision
    assert (held.open_locations, held.spawn, held.retire) == (0, [], [])
    assert held.reasoning.startswith("Held: rule proposed to open 1 more")
    assert second.baseline == pytest.approx(0.6)  # gated on earlier answers only

    before = company.locations
    ceo.defer()
    company.apply(held)
    assert company.locations == before
    assert ceo.propose(company, HEURISTIC_CADENCE_DAYS * 4) is None  # window restarts
    assert judge.status()["approved"] == 1 and judge.status()["vetoed"] == 1


def test_repeated_proposals_are_not_a_coin_flip(library: NodeLibrary) -> None:
    """The old gate (beat your own running mean) vetoed about half of identical repeats."""
    company = capacity_bound(library)
    rng = random.Random(0)
    answers = [min(1.0, max(0.0, rng.gauss(0.6, 0.03))) for _ in range(60)]
    judge = JevJudge(client=FakeTypeSafe(list(answers)))

    async def go() -> list[bool]:  # one event loop: a loop per call exhausts Windows sockets
        verdicts = []
        for i in range(len(answers)):
            verdict = await judge.review(company, propose(HeuristicCeo(), company), 21 * i)
            assert verdict is not None
            verdicts.append(verdict.approved)
        return verdicts

    verdicts = asyncio.run(go())
    assert sum(not v for v in verdicts) <= len(verdicts) * 0.15


def test_cut_that_holding_off_cannot_survive_skips_jev(library: NodeLibrary) -> None:
    company = cash_crisis(library, runway_days=HORIZON_DAYS / 2)
    proposal = HeuristicCeo().propose(company, HEURISTIC_CADENCE_DAYS)
    assert proposal is not None and proposal.retire
    fake = FakeTypeSafe([0.0])
    judge = JevJudge(client=fake)

    assert asyncio.run(judge.review(company, proposal, HEURISTIC_CADENCE_DAYS)) is None
    assert fake.calls == [] and judge.status()["forced"] == 1

    # With runway past the horizon the cut is discretionary and Jev rules on it.
    company = cash_crisis(library, runway_days=HORIZON_DAYS * 2)
    proposal = HeuristicCeo().propose(company, HEURISTIC_CADENCE_DAYS)
    assert proposal is not None and proposal.retire
    assert asyncio.run(judge.review(company, proposal, HEURISTIC_CADENCE_DAYS)) is not None
    assert len(fake.calls) == 1


def test_cuts_and_growth_moves_keep_separate_baselines(library: NodeLibrary) -> None:
    grower = capacity_bound(library)
    judge = JevJudge(client=FakeTypeSafe([0.9] * 10 + [0.3]))
    for i in range(10):
        assert asyncio.run(judge.review(grower, propose(HeuristicCeo(), grower), 21 * i))

    cutter = cash_crisis(library, runway_days=HORIZON_DAYS * 2)
    cutter.company_id = grower.company_id
    proposal = HeuristicCeo().propose(cutter, HEURISTIC_CADENCE_DAYS)
    assert proposal is not None and proposal.retire
    verdict = asyncio.run(judge.review(cutter, proposal, 300))
    assert verdict is not None and verdict.approved, "a cut is not measured against growth moves"


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
    rule_only = [d for c in sim.companies for d in c.decisions if d.source == "heuristic"]
    status = sim.jev.status()
    assert len(rule_only) == status["forced"], "only forced survival cuts bypass Jev"
    assert status["calls"] == len(jev_calls)
    assert status["approved"] + status["vetoed"] == status["calls"]
    assert build_state(sim.companies[0], jev_calls[0])["company"]["employees"] > 0
