"""Tests for market preset configurations."""

from __future__ import annotations

import pytest

from src.simulation.market.engine import MarketEngine
from src.simulation.market.presets import MARKET_PRESETS


class TestPresetValidity:
    def test_all_presets_exist(self):
        assert len(MARKET_PRESETS) == 4
        assert set(MARKET_PRESETS.keys()) == {"price-war", "innovation-race", "monopoly", "commodity"}

    def test_slugs_are_unique(self):
        slugs = [p.slug for p in MARKET_PRESETS.values()]
        assert len(slugs) == len(set(slugs))

    def test_all_have_valid_params(self):
        for slug, preset in MARKET_PRESETS.items():
            assert preset.params.tam_0 > 0, f"{slug}: TAM_0 must be positive"
            assert preset.params.n_0 > 0, f"{slug}: n_0 must be positive"
            assert preset.params.alpha >= 0, f"{slug}: alpha must be non-negative"
            assert preset.params.beta >= 0, f"{slug}: beta must be non-negative"
            assert 0 <= preset.params.delta <= 1, f"{slug}: delta must be in [0, 1]"
            assert preset.params.t_q > 0, f"{slug}: t_q must be positive"
            assert preset.params.t_death > 0, f"{slug}: t_death must be positive"
            assert preset.params.tau_decay > 0, f"{slug}: tau_decay must be positive"

    def test_all_have_name_and_description(self):
        for slug, preset in MARKET_PRESETS.items():
            assert len(preset.name) > 0, f"{slug}: missing name"
            assert len(preset.description) > 10, f"{slug}: description too short"


class TestPresetCharacter:
    def test_price_war_marketing_dominant(self):
        p = MARKET_PRESETS["price-war"].params
        assert p.alpha > p.beta

    def test_innovation_race_quality_dominant(self):
        p = MARKET_PRESETS["innovation-race"].params
        assert p.beta > p.alpha

    def test_monopoly_low_entry(self):
        p = MARKET_PRESETS["monopoly"].params
        assert p.lambda_entry <= 0.02

    def test_commodity_high_churn(self):
        p = MARKET_PRESETS["commodity"].params
        assert p.delta >= 0.10

    def test_commodity_most_agents(self):
        counts = {slug: p.params.n_0 for slug, p in MARKET_PRESETS.items()}
        assert counts["commodity"] == max(counts.values())

    def test_all_presets_r_exceeds_delta(self):
        """r_min must exceed delta for every preset so revenue can grow."""
        for slug, preset in MARKET_PRESETS.items():
            p = preset.params
            assert p.r_range[0] > p.delta, (
                f"{slug}: r_min={p.r_range[0]} <= delta={p.delta} — revenue will always decay"
            )


class TestPresetSmoke:
    """Each preset should run 200 ticks without crashing."""

    @pytest.mark.parametrize("preset_slug", list(MARKET_PRESETS.keys()))
    def test_runs_200_ticks(self, preset_slug: str):
        params = MARKET_PRESETS[preset_slug].params
        engine = MarketEngine(params, max_ticks=200, seed=42)
        last_result = None
        for _ in range(200):
            last_result = engine.tick()

        assert last_result is not None
        assert last_result["tick"] == 200
        assert engine.is_complete

    @pytest.mark.parametrize("preset_slug", list(MARKET_PRESETS.keys()))
    def test_has_active_agents_at_tick_50(self, preset_slug: str):
        """At least some agents should survive the first 50 ticks."""
        params = MARKET_PRESETS[preset_slug].params
        engine = MarketEngine(params, max_ticks=500, seed=42)
        for _ in range(50):
            result = engine.tick()
        assert result["agent_count"] > 0, f"{preset_slug}: all agents dead by tick 50"
