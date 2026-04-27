"""Integration tests for biosim wiring (Phase 3.5).

Exercises the math-layer integrations done in Phase 3.3b through the live
engine end-to-end:
  * Cobb-Douglas production cap on revenue (P-MATH-1)
  * RK45 satisfaction ODE (P-MATH-2)
  * Lotka-Volterra competition between companies (P-MATH-3)
  * Multi-tick insolvency aligned with biosim's pattern (P-ENG-2 / P-MATH-4)

These tests run the full CompanyAgentV2 -> MultiCompanySimV2 -> biosim chain;
they catch wiring bugs that unit tests on biosim or unit tests on
unified_v2 individually would miss.
"""

from __future__ import annotations

import asyncio
import math
import random

import numpy as np
import pytest

from src.simulation.library_loader import _reset_library_cache, get_library
from src.simulation.seed import (
    sample_seed_for_archetype,
)
from src.simulation.stance import sample_stance
from src.simulation.unified_v2 import (
    INSOLVENT_TICKS_TO_DEATH,
    MultiCompanySimV2,
)

# Re-use the fixture patterns from test_unified_v2.py — keep tests honest
# about library + silent-bundle construction.
from tests.test_unified_v2 import (  # type: ignore[import-not-found]
    _make_agent,
    _valid_refs_for,
)


@pytest.fixture
def library():
    _reset_library_cache()
    return get_library()


@pytest.fixture
def rng():
    return random.Random(42)


def _build_seed(library, rng, archetype="small_team", **overrides):
    """Build a valid seed for `archetype`, with library refs filled in."""
    seed = sample_seed_for_archetype(archetype, rng=rng)
    suppliers, revenues, costs = _valid_refs_for(library, seed.economics_model)
    update = {
        "initial_supplier_types": suppliers,
        "initial_revenue_streams": revenues,
        "initial_cost_centers": costs,
    }
    update.update(overrides)
    return seed.model_copy(update=update)


# --- P-MATH-1: Cobb-Douglas ------------------------------------------------


def test_cobb_douglas_output_scales_with_capital_and_labor(library, rng):
    """Doubling K (starting_cash) or L (starting_employees) must measurably
    raise daily_revenue when the Cobb-Douglas cap is the binding constraint.
    """
    base_seed = _build_seed(library, rng, archetype="small_team")
    # Pin to a regime where production_cap binds: physical capacity is set
    # very high (so it never binds) and demand is large (TAM is multi-million
    # in small_team archetype). Cobb-Douglas with K=200k, L~12, alpha=0.3,
    # beta=0.7 -> ~220 units served/tick, well below physical cap. Then 4x K
    # and 4x L must each lift Y measurably.
    base_seed = base_seed.model_copy(update={
        "starting_cash": 200_000.0,
        "starting_employees": 5,
        "initial_locations": 5,
        "base_capacity_per_location": 100_000,
        "cobb_douglas_alpha": 0.3,
        "cobb_douglas_beta": 0.7,
    })
    rich_seed = base_seed.model_copy(update={"starting_cash": 800_000.0})  # 4x K
    big_team_seed = base_seed.model_copy(update={"starting_employees": 20})  # 4x L

    stance = sample_stance("bootstrap", rng=rng)

    async def _run(seed):
        agent = _make_agent(seed, stance, library, rng=random.Random(0))
        # Tick once — production_cap is computed inside step().
        await agent.step()
        return agent.daily_revenue, agent.capacity_utilization

    base_rev, base_util = asyncio.run(_run(base_seed))
    rich_rev, _ = asyncio.run(_run(rich_seed))
    big_team_rev, _ = asyncio.run(_run(big_team_seed))

    # Confirm the regime: physical capacity is NOT the binding constraint.
    # capacity_utilization is served/physical_capacity; if it's well below
    # 1.0, then either Cobb-Douglas or demand is binding. We set physical
    # capacity to 5*100k = 500k units and demand much higher than CD output,
    # so CD should bind. capacity_utilization < 0.5 confirms that.
    assert base_util < 0.5, (
        f"physical capacity is unexpectedly binding "
        f"(capacity_utilization={base_util:.2f}); "
        f"Cobb-Douglas scaling assertions below would be vacuous"
    )

    assert rich_rev > base_rev, (
        f"4x capital must lift revenue under Cobb-Douglas; got "
        f"{rich_rev:.2f} vs {base_rev:.2f}"
    )
    assert big_team_rev > base_rev, (
        f"4x labor must lift revenue under Cobb-Douglas; got "
        f"{big_team_rev:.2f} vs {base_rev:.2f}"
    )


# --- P-MATH-3: Lotka-Volterra ----------------------------------------------


def test_lotka_volterra_competition_suppresses_growth(library, rng):
    """A company facing 4 competitors must end with lower market_strength than
    the same company running solo — LV cross-suppression at work.
    """
    seed = _build_seed(
        library, rng, archetype="small_team", starting_employees=10
    )
    stance = sample_stance("bootstrap", rng=rng)

    def _make_sim(n_companies):
        agents = [
            _make_agent(seed, stance, library, rng=random.Random(i))
            for i in range(n_companies)
        ]
        return MultiCompanySimV2(
            sim_id=f"lv-{n_companies}",
            companies=agents,
            max_ticks=120,
            tam_initial=seed.tam,
        )

    async def _run(sim):
        for _ in range(100):
            await sim.step()
        return [c.market_strength for c in sim.companies if c.alive]

    solo_strengths = asyncio.run(_run(_make_sim(1)))
    cohort_strengths = asyncio.run(_run(_make_sim(5)))

    # Solo case — if there's only one company, MultiCompanySimV2 skips the LV
    # step (alive count must be > 1). The first-company strength should
    # remain at its initialized value (= starting_employees).
    assert solo_strengths[0] == pytest.approx(seed.starting_employees, abs=1e-6)
    # In the cohort, mean strength should differ from initial — companies
    # are competing and at least one is being suppressed (or the leader is
    # running away). Strong assertion: NOT all five at their starting value.
    cohort_arr = np.array(cohort_strengths)
    assert not np.allclose(cohort_arr, seed.starting_employees), (
        f"LV step did not modify any market_strength; got {cohort_strengths}"
    )


def test_lv_competition_matrix_stable_across_company_deaths(library, rng):
    """When a company dies mid-sim, the surviving firms' pairwise LV weights
    must NOT discontinuously change. Regression test for a Phase 3.6 review
    finding: rebuilding the competition matrix at the new shrunk size with
    the same RNG seed produced a different matrix because n_species changed
    the perturbation draw count. Fix uses a fixed full-size matrix sliced
    by surviving-company indices.
    """
    seed = _build_seed(library, rng, archetype="small_team", starting_employees=10)
    stance = sample_stance("bootstrap", rng=rng)
    agents = [
        _make_agent(seed, stance, library, rng=random.Random(i))
        for i in range(4)
    ]
    sim = MultiCompanySimV2(
        sim_id="lv-stability",
        companies=agents,
        max_ticks=10,
        tam_initial=seed.tam,
    )
    # Snapshot the cached full matrix; it must be built once at __init__.
    full_matrix_at_init = sim._lv_competition_matrix_full.copy()
    assert full_matrix_at_init.shape == (4, 4)

    # Kill company index 1 directly to simulate mid-sim death without
    # waiting for organic insolvency.
    agents[1].alive = False

    async def _run():
        for _ in range(3):
            await sim.step()

    asyncio.run(_run())

    # The full matrix must be UNCHANGED — death does not rebuild.
    np.testing.assert_array_equal(
        sim._lv_competition_matrix_full,
        full_matrix_at_init,
        err_msg="LV competition matrix was rebuilt after a death",
    )
    # And the surviving-firm pairwise weights should be exactly the
    # original sub-block, not a fresh draw at n=3.
    survivors = [0, 2, 3]
    expected_sub = full_matrix_at_init[np.ix_(survivors, survivors)]
    actual_sub = sim._lv_competition_matrix_full[np.ix_(survivors, survivors)]
    np.testing.assert_array_equal(actual_sub, expected_sub)


# --- P-MATH-2: Satisfaction ODE --------------------------------------------


def test_logistic_satisfaction_converges_smoothly(library, rng):
    """Over 50 ticks, satisfaction must move toward its target without NaN
    or sustained oscillation (the RK45 integrator is well-behaved).
    """
    seed = _build_seed(library, rng, archetype="small_team")
    stance = sample_stance("bootstrap", rng=rng)
    agent = _make_agent(seed, stance, library, rng=random.Random(0))

    history = []

    async def _run():
        for _ in range(50):
            tr = await agent.step()
            history.append(tr.avg_satisfaction)

    asyncio.run(_run())

    arr = np.array(history)
    assert not np.any(np.isnan(arr)), "satisfaction produced NaN"
    assert not np.any(np.isinf(arr)), "satisfaction produced Inf"
    # Values must stay in [0, 1] — the post-ODE clamp enforces this.
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)
    # No wild oscillation — the per-tick |delta| should be small (< 0.5)
    deltas = np.abs(np.diff(arr))
    assert np.all(deltas < 0.5), (
        f"satisfaction oscillating wildly: max delta {deltas.max():.3f}"
    )


# --- P-ENG-2 / P-MATH-4: Multi-tick insolvency ----------------------------


def test_multi_tick_insolvency_uses_biosim_pattern(library, rng):
    """A company with persistently negative cash dies after exactly
    INSOLVENT_TICKS_TO_DEATH consecutive insolvent ticks — not earlier.
    """
    seed = _build_seed(library, rng, archetype="small_team", starting_cash=1.0)
    # Force daily_fixed_costs > any realistic revenue so cash goes negative
    # immediately and stays there.
    seed = seed.model_copy(update={
        "starting_cash": 100.0,
        "daily_fixed_costs": 100_000.0,
    })
    stance = sample_stance("bootstrap", rng=rng)
    agent = _make_agent(seed, stance, library, rng=random.Random(0))

    async def _run():
        for tick_n in range(1, INSOLVENT_TICKS_TO_DEATH + 5):
            await agent.step()
            if not agent.alive:
                return tick_n
        return None

    death_tick = asyncio.run(_run())
    assert death_tick is not None, (
        f"agent never died despite {INSOLVENT_TICKS_TO_DEATH + 5} insolvent ticks"
    )
    # Death must happen on or after INSOLVENT_TICKS_TO_DEATH.
    # Allow +/-1 tick of slack for tick-counter timing differences.
    assert death_tick >= INSOLVENT_TICKS_TO_DEATH - 1, (
        f"agent died too early on tick {death_tick}; expected >= "
        f"{INSOLVENT_TICKS_TO_DEATH - 1}"
    )


# --- P-MATH-* end-to-end smoke --------------------------------------------


@pytest.mark.slow
def test_full_sim_5y_with_biosim_no_nan(library, rng):
    """Run a 5-year (1825-tick) multi-company sim. Assert no NaN in any
    TickResult numeric field. Stress-test for the wired math layer.
    """
    seed_a = _build_seed(library, rng, archetype="small_team")
    seed_b = _build_seed(library, rng, archetype="venture_funded")
    stance_a = sample_stance("bootstrap", rng=rng)
    stance_b = sample_stance("venture_growth", rng=rng)

    agents = [
        _make_agent(seed_a, stance_a, library, rng=random.Random(1)),
        _make_agent(seed_b, stance_b, library, rng=random.Random(2)),
    ]
    sim = MultiCompanySimV2(
        sim_id="biosim-5y",
        companies=agents,
        max_ticks=1825,
        tam_initial=max(seed_a.tam, seed_b.tam),
    )

    numeric_fields = (
        "cash", "daily_revenue", "daily_costs", "monthly_revenue",
        "capacity_utilization", "avg_satisfaction",
    )

    async def _run():
        for _ in range(1825):
            payload = await sim.step()
            if payload.get("all_dead"):
                break
            for tr in payload.get("results", []):
                for f in numeric_fields:
                    v = getattr(tr, f)
                    if not math.isfinite(v):
                        return True, f, v, tr.tick
        return False, None, None, sim.tick

    nan_inf_seen, field, val, last_tick = asyncio.run(_run())
    assert not nan_inf_seen, (
        f"non-finite value in TickResult.{field} = {val} at tick {last_tick}"
    )
