"""Tests for Monte Carlo simulation runner."""

from __future__ import annotations

import pytest

from src.simulation.monte_carlo import MonteCarloRunner, _set_nested_param
from src.simulation.monte_carlo_models import MonteCarloConfig, ParameterVariation
from src.simulation.unified_models import UnifiedParams


class TestMonteCarloBasic:
    def test_identical_seed_produces_identical_results(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=3, ticks_per_run=200,
            num_companies=2, seed=42,
        )
        report1 = MonteCarloRunner(config).run_sequential()
        report2 = MonteCarloRunner(config).run_sequential()

        for r1, r2 in zip(report1.results, report2.results):
            for c1, c2 in zip(r1.companies, r2.companies):
                assert c1.final_cash == c2.final_cash, f"Determinism failed: {c1.name}"

    def test_runs_complete_without_error(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=5, ticks_per_run=300,
            num_companies=3,
        )
        report = MonteCarloRunner(config).run_sequential()
        assert report.num_runs == 5
        assert len(report.results) == 5
        for r in report.results:
            assert r.final_tick == 300
            assert len(r.companies) >= 3

    def test_report_has_aggregate_stats(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=5, ticks_per_run=200,
            num_companies=2,
        )
        report = MonteCarloRunner(config).run_sequential()
        assert len(report.survival_rates) > 0
        assert len(report.mean_final_cash) > 0
        assert len(report.std_final_cash) > 0
        assert len(report.mean_final_share) > 0
        for rate in report.survival_rates.values():
            assert 0.0 <= rate <= 1.0


class TestParameterVariation:
    def test_varied_tam_produces_different_results(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=5, ticks_per_run=300,
            num_companies=2,
            parameter_variations=[
                ParameterVariation(param="tam_0", low=10000, high=50000),
            ],
        )
        report = MonteCarloRunner(config).run_sequential()

        # Different TAM values should produce different cash outcomes
        cash_values = [r.companies[0].final_cash for r in report.results]
        assert len(set(round(c, 0) for c in cash_values)) > 1, "All runs produced same cash"

        # Verify varied params are recorded
        for r in report.results:
            assert "tam_0" in r.varied_params
            assert 10000 <= r.varied_params["tam_0"] <= 50000

    def test_multiple_variations(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=4, ticks_per_run=200,
            num_companies=2,
            parameter_variations=[
                ParameterVariation(param="tam_0", low=15000, high=40000),
                ParameterVariation(param="alpha", low=0.5, high=1.0),
            ],
        )
        report = MonteCarloRunner(config).run_sequential()
        assert report.num_runs == 4
        assert len(report.varied_parameters) == 2
        for r in report.results:
            assert "tam_0" in r.varied_params
            assert "alpha" in r.varied_params


class TestParameterPathResolution:
    def test_flat_name_sets_attr(self):
        params = UnifiedParams()
        _set_nested_param(params, "tam_0", 12345.0)
        assert params.tam_0 == 12345.0

    def test_params_prefix_is_stripped(self):
        params = UnifiedParams()
        _set_nested_param(params, "params.alpha", 0.42)
        assert params.alpha == 0.42

    def test_unknown_attr_raises(self):
        params = UnifiedParams()
        with pytest.raises(ValueError, match="no attribute 'nonexistent'"):
            _set_nested_param(params, "nonexistent", 1.0)

    def test_unknown_nested_attr_raises(self):
        params = UnifiedParams()
        with pytest.raises(ValueError, match="no attribute 'foo'"):
            _set_nested_param(params, "foo.bar", 1.0)

    def test_invalid_path_in_runner_raises(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=2, ticks_per_run=100,
            num_companies=2,
            parameter_variations=[
                ParameterVariation(param="not_a_real_param", low=0.0, high=1.0),
            ],
        )
        with pytest.raises(ValueError, match="not_a_real_param"):
            MonteCarloRunner(config)


class TestMonteCarloThreaded:
    def test_threaded_matches_sequential(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=4, ticks_per_run=200,
            num_companies=2, seed=99,
        )
        seq_report = MonteCarloRunner(config).run_sequential()
        par_report = MonteCarloRunner(config).run_all(max_workers=2)

        # Same seed + same configs should produce same results regardless of execution order
        for sr, pr in zip(seq_report.results, par_report.results):
            for sc, pc in zip(sr.companies, pr.companies):
                assert sc.final_cash == pc.final_cash


class TestSampledTimeSeries:
    def test_sampled_data_at_intervals(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=2, ticks_per_run=300,
            num_companies=2, sample_interval=50,
        )
        report = MonteCarloRunner(config).run_sequential()
        for r in report.results:
            assert len(r.sampled_ticks) > 0
            assert len(r.sampled_tam) == len(r.sampled_ticks)
            assert len(r.sampled_hhi) == len(r.sampled_ticks)
            # Should sample at tick 50, 100, 150, 200, 250, 300
            assert r.sampled_ticks[0] == 50

    def test_hhi_in_valid_range(self):
        config = MonteCarloConfig(
            industry="restaurant", num_runs=3, ticks_per_run=200,
            num_companies=4,
        )
        report = MonteCarloRunner(config).run_sequential()
        assert 0.0 <= report.mean_hhi <= 1.0
