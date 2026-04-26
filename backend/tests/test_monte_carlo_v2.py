"""Tests for `monte_carlo_v2.py` — heuristic-only sweeps over CompanySeed fields."""

from __future__ import annotations

import asyncio

import pytest

from src.simulation.library_loader import _reset_library_cache, get_library
from src.simulation.monte_carlo_v2 import (
    CompanyResultV2,
    MonteCarloConfigV2,
    MonteCarloReportV2,
    MonteCarloRunnerV2,
    RunResultV2,
    SeedVariation,
    run_single_v2,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def library():
    _reset_library_cache()
    return get_library()


def _valid_refs_for(library, economics_model):
    suppliers, revenues, costs = [], [], []
    for key, node in sorted(library.nodes.items()):
        if economics_model not in node.applicable_economics:
            continue
        if node.category == "supplier" and not suppliers:
            suppliers.append(key)
        elif node.category == "revenue" and not revenues:
            revenues.append(key)
        elif node.category == "ops" and not costs:
            costs.append(key)
    return suppliers, revenues, costs


@pytest.fixture
def physical_refs(library):
    """Library refs guaranteed to validate against physical economics."""
    suppliers, revenues, costs = _valid_refs_for(library, "physical")
    return suppliers, revenues, costs


@pytest.fixture
def base_config(physical_refs):
    s, r, c = physical_refs
    return MonteCarloConfigV2(
        seed_archetype="small_team",
        stance_archetype="bootstrap",
        num_runs=3,
        num_companies_per_run=1,
        ticks_per_run=20,
        sample_interval=5,
        base_seed=99,
        initial_supplier_types=s,
        initial_revenue_streams=r,
        initial_cost_centers=c,
        pin_economics_model="physical",
    )


# ─── Config validation ────────────────────────────────────────────────────


def test_config_rejects_extra_fields():
    with pytest.raises(Exception):
        MonteCarloConfigV2(unknown_field=42)  # type: ignore[call-arg]


def test_config_defaults_sensible():
    c = MonteCarloConfigV2()
    assert c.num_runs >= 1
    assert c.ticks_per_run >= 10
    assert c.transcript_dir is None  # heuristic-only by default


# ─── Single-run execution ─────────────────────────────────────────────────


def test_run_single_executes(base_config, library):
    result = asyncio.run(
        run_single_v2(
            run_index=0,
            config=base_config,
            library=library,
            rng_seed=12345,
            lhs_row=__import__("numpy").zeros(1),
        )
    )
    assert isinstance(result, RunResultV2)
    assert result.run_index == 0
    assert result.rng_seed == 12345
    assert result.final_tick > 0
    assert len(result.companies) == 1


def test_run_single_company_archetype_matches_stance(base_config, library):
    result = asyncio.run(
        run_single_v2(
            run_index=0, config=base_config, library=library, rng_seed=1,
            lhs_row=__import__("numpy").zeros(1),
        )
    )
    assert result.companies[0].archetype == "bootstrap"


# ─── Runner orchestration ─────────────────────────────────────────────────


def test_runner_completes_all_runs(base_config, library):
    runner = MonteCarloRunnerV2(config=base_config, library=library)
    report = asyncio.run(runner.run_sequential())
    assert isinstance(report, MonteCarloReportV2)
    assert report.num_runs == base_config.num_runs
    assert len(report.results) == base_config.num_runs
    assert runner.is_complete


def test_runner_aggregates_archetype(base_config, library):
    runner = MonteCarloRunnerV2(config=base_config, library=library)
    report = asyncio.run(runner.run_sequential())
    assert report.stance_archetype == "bootstrap"
    assert report.seed_archetype == "small_team"


def test_runner_deterministic_with_same_base_seed(base_config, library):
    """Two runners with the same base_seed produce identical results."""
    r1 = MonteCarloRunnerV2(config=base_config, library=library)
    rep1 = asyncio.run(r1.run_sequential())
    r2 = MonteCarloRunnerV2(config=base_config, library=library)
    rep2 = asyncio.run(r2.run_sequential())

    cash1 = [c.final_cash for r in rep1.results for c in r.companies]
    cash2 = [c.final_cash for r in rep2.results for c in r.companies]
    assert cash1 == cash2


def test_runner_different_seeds_diverge(base_config, library):
    """Different base_seeds produce different aggregate stats."""
    cfg2 = base_config.model_copy(update={"base_seed": 7777})
    r1 = MonteCarloRunnerV2(config=base_config, library=library)
    r2 = MonteCarloRunnerV2(config=cfg2, library=library)
    rep1 = asyncio.run(r1.run_sequential())
    rep2 = asyncio.run(r2.run_sequential())
    # rng_seed lists should differ
    seeds1 = [r.rng_seed for r in rep1.results]
    seeds2 = [r.rng_seed for r in rep2.results]
    assert seeds1 != seeds2


# ─── LHS variation ───────────────────────────────────────────────────────


def test_lhs_variation_produces_distinct_seeds(physical_refs, library):
    s, r, c = physical_refs
    cfg = MonteCarloConfigV2(
        seed_archetype="small_team",
        stance_archetype="bootstrap",
        num_runs=4,
        ticks_per_run=10,
        sample_interval=5,
        base_seed=42,
        parameter_variations=[
            SeedVariation(field="starting_price", low=10.0, high=100.0),
            SeedVariation(field="competitor_density", low=1, high=8),
        ],
        initial_supplier_types=s,
        initial_revenue_streams=r,
        initial_cost_centers=c,
        pin_economics_model="physical",
    )
    runner = MonteCarloRunnerV2(config=cfg, library=library)
    report = asyncio.run(runner.run_sequential())

    prices = sorted({r.varied_params["starting_price"] for r in report.results})
    assert len(prices) == 4, f"LHS should produce 4 distinct prices, got: {prices}"


def test_lhs_variation_unknown_field_raises(physical_refs, library):
    s, r, c = physical_refs
    cfg = MonteCarloConfigV2(
        seed_archetype="small_team",
        stance_archetype="bootstrap",
        num_runs=2,
        ticks_per_run=10,
        sample_interval=5,
        parameter_variations=[
            SeedVariation(field="not_a_real_field", low=0, high=1),
        ],
        initial_supplier_types=s,
        initial_revenue_streams=r,
        initial_cost_centers=c,
        pin_economics_model="physical",
    )
    runner = MonteCarloRunnerV2(config=cfg, library=library)
    with pytest.raises(ValueError, match="not a CompanySeed attribute"):
        asyncio.run(runner.run_sequential())


# ─── Shocks integration ───────────────────────────────────────────────────


def test_shock_lambda_overrides_increase_arrivals(physical_refs, library):
    s, r, c = physical_refs
    cfg_no_shocks = MonteCarloConfigV2(
        seed_archetype="small_team",
        stance_archetype="bootstrap",
        num_runs=2, ticks_per_run=50, sample_interval=10,
        shock_lambdas={},
        initial_supplier_types=s, initial_revenue_streams=r, initial_cost_centers=c,
        pin_economics_model="physical",
    )
    cfg_high_shocks = MonteCarloConfigV2(
        seed_archetype="small_team",
        stance_archetype="bootstrap",
        num_runs=2, ticks_per_run=50, sample_interval=10,
        shock_lambdas={"market_crash": 0.5},  # high arrival rate
        initial_supplier_types=s, initial_revenue_streams=r, initial_cost_centers=c,
        pin_economics_model="physical",
    )

    rep_low = asyncio.run(MonteCarloRunnerV2(config=cfg_no_shocks, library=library).run_sequential())
    rep_high = asyncio.run(MonteCarloRunnerV2(config=cfg_high_shocks, library=library).run_sequential())

    assert rep_high.mean_shocks_per_run > rep_low.mean_shocks_per_run


# ─── Report ────────────────────────────────────────────────────────────────


def test_report_includes_aggregate_stats(base_config, library):
    runner = MonteCarloRunnerV2(config=base_config, library=library)
    report = asyncio.run(runner.run_sequential())
    assert 0.0 <= report.survival_rate <= 1.0
    assert report.std_final_cash >= 0.0
    assert report.varied_fields == []
