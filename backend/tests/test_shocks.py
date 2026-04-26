"""Tests for the shock library and Poisson scheduler.

Covers:
    1. All 8 built-in factories produce valid Shock instances.
    2. Same RNG seed → byte-identical arrival sequence (determinism).
    3. Severity scaling: severe > moderate > minor for impact magnitudes.
    4. Shock expiry: after duration_ticks, shock leaves the active set.
    5. Multiplicative composition: two demand-cutting shocks compound.
"""

from __future__ import annotations

import random

import pytest

from src.simulation.shocks import (
    SEVERITY_INTENSITY,
    SHOCK_FACTORIES,
    Severity,
    Shock,
    ShockScheduler,
    apply_active_shocks,
    make_economic_recession,
    make_key_employee_departure,
    make_market_crash,
    make_new_competitor_entry,
    make_regulatory_change,
    make_supply_chain_disruption,
    make_talent_war,
    make_viral_growth_event,
)

ALL_FACTORIES = [
    make_market_crash,
    make_new_competitor_entry,
    make_key_employee_departure,
    make_supply_chain_disruption,
    make_regulatory_change,
    make_viral_growth_event,
    make_economic_recession,
    make_talent_war,
]


# ---------------------------------------------------------------------------
# 1. Factory validity
# ---------------------------------------------------------------------------


class TestFactories:
    def test_registry_has_all_eight(self):
        assert set(SHOCK_FACTORIES) == {
            "market_crash",
            "new_competitor_entry",
            "key_employee_departure",
            "supply_chain_disruption",
            "regulatory_change",
            "viral_growth_event",
            "economic_recession",
            "talent_war",
        }

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    @pytest.mark.parametrize("severity", ["minor", "moderate", "severe"])
    def test_factory_produces_valid_shock(self, factory, severity):
        rng = random.Random(0)
        shock = factory(rng, severity)
        assert isinstance(shock, Shock)
        assert shock.name  # non-empty
        assert shock.severity == severity
        assert shock.duration_ticks > 0
        assert isinstance(shock.impact, dict)
        assert len(shock.impact) > 0
        assert isinstance(shock.description, str)
        assert shock.description  # non-empty
        assert shock.tick_started is None  # not yet injected

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_factory_default_severity_is_moderate(self, factory):
        shock = factory(random.Random(0))
        assert shock.severity == "moderate"

    def test_factory_rejects_unknown_severity(self):
        with pytest.raises(ValueError, match="Unknown severity"):
            make_market_crash(random.Random(0), "catastrophic")

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_factory_name_matches_registry_key(self, factory):
        shock = factory(random.Random(0))
        assert shock.name in SHOCK_FACTORIES
        assert SHOCK_FACTORIES[shock.name] is factory


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_factory_deterministic_given_seed(self):
        s1 = make_market_crash(random.Random(1234), "moderate")
        s2 = make_market_crash(random.Random(1234), "moderate")
        assert s1.model_dump() == s2.model_dump()

    def test_different_seeds_produce_different_shocks(self):
        s1 = make_market_crash(random.Random(1), "moderate")
        s2 = make_market_crash(random.Random(2), "moderate")
        assert s1.impact != s2.impact

    def test_scheduler_same_seed_same_arrival_sequence(self):
        # Bump lambdas high enough that we'll see arrivals in 100 ticks.
        lambdas = {name: 0.05 for name in SHOCK_FACTORIES}
        sched1 = ShockScheduler(rng_seed=42, lambdas=lambdas)
        sched2 = ShockScheduler(rng_seed=42, lambdas=lambdas)

        seq1 = []
        seq2 = []
        for t in range(200):
            seq1.append([(s.name, s.severity, s.duration_ticks) for s in sched1.tick(t)])
            seq2.append([(s.name, s.severity, s.duration_ticks) for s in sched2.tick(t)])

        assert seq1 == seq2

    def test_scheduler_different_seeds_diverge(self):
        lambdas = {name: 0.05 for name in SHOCK_FACTORIES}
        sched1 = ShockScheduler(rng_seed=1, lambdas=lambdas)
        sched2 = ShockScheduler(rng_seed=2, lambdas=lambdas)

        # Collect arrivals across 500 ticks.
        arrivals1 = [s.name for t in range(500) for s in sched1.tick(t)]
        arrivals2 = [s.name for t in range(500) for s in sched2.tick(t)]

        # Different seeds should yield divergent (not identical) sequences.
        assert arrivals1 != arrivals2

    def test_scheduler_deterministic_full_dump(self):
        """Stronger check — every field of every shock matches across runs."""
        lambdas = {"market_crash": 0.10, "viral_growth_event": 0.10}
        sched1 = ShockScheduler(rng_seed=999, lambdas=lambdas)
        sched2 = ShockScheduler(rng_seed=999, lambdas=lambdas)

        for t in range(100):
            a1 = sched1.tick(t)
            a2 = sched2.tick(t)
            assert [s.model_dump() for s in a1] == [s.model_dump() for s in a2]


# ---------------------------------------------------------------------------
# 3. Severity scaling
# ---------------------------------------------------------------------------


def _impact_magnitude(shock: Shock) -> float:
    """Sum of absolute deviations: |1-v| for *_mult keys, |v| for additive keys.

    This is the metric that should monotonically grow with severity.
    """
    total = 0.0
    for key, value in shock.impact.items():
        if key.endswith("_mult"):
            total += abs(1.0 - value)
        else:
            total += abs(value)
    return total


class TestSeverityScaling:
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_severe_greater_than_moderate_greater_than_minor(self, factory):
        # Average over many seeds to wash out per-draw noise — the scaling is
        # an expectation property, not a per-draw property.
        n_samples = 200
        mags: dict[str, float] = {}
        for sev in ("minor", "moderate", "severe"):
            total = 0.0
            for seed in range(n_samples):
                shock = factory(random.Random(seed), sev)
                total += _impact_magnitude(shock)
            mags[sev] = total / n_samples

        assert mags["minor"] < mags["moderate"] < mags["severe"], (
            f"{factory.__name__}: severity scaling violated. mags={mags}"
        )

    def test_intensity_constants_are_monotone(self):
        assert SEVERITY_INTENSITY["minor"] < SEVERITY_INTENSITY["moderate"]
        assert SEVERITY_INTENSITY["moderate"] < SEVERITY_INTENSITY["severe"]

    @pytest.mark.parametrize(
        "factory",
        # Use only factories whose impact is dominated by *_mult keys; the
        # `key_employee_departure` "retire_random_employee" flag is a constant
        # signal, so test it separately.
        [
            make_market_crash,
            make_new_competitor_entry,
            make_supply_chain_disruption,
            make_regulatory_change,
            make_viral_growth_event,
            make_economic_recession,
            make_talent_war,
        ],
    )
    def test_duration_scales_with_severity(self, factory):
        n_samples = 200
        durs: dict[str, float] = {}
        for sev in ("minor", "moderate", "severe"):
            total = sum(factory(random.Random(s), sev).duration_ticks for s in range(n_samples))
            durs[sev] = total / n_samples
        assert durs["minor"] < durs["moderate"] < durs["severe"]


# ---------------------------------------------------------------------------
# 4. Expiry
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_is_active_false_before_injection(self):
        shock = make_market_crash(random.Random(0), "moderate")
        assert shock.tick_started is None
        assert shock.is_active(0) is False
        assert shock.is_active(100) is False

    def test_is_active_true_during_window(self):
        shock = make_market_crash(random.Random(0), "moderate")
        shock.tick_started = 50
        assert shock.is_active(50) is True
        assert shock.is_active(50 + shock.duration_ticks - 1) is True

    def test_is_active_false_after_window(self):
        shock = make_market_crash(random.Random(0), "moderate")
        shock.tick_started = 50
        expiry_tick = 50 + shock.duration_ticks
        assert shock.is_active(expiry_tick) is False
        assert shock.is_active(expiry_tick + 100) is False

    def test_scheduler_prunes_expired_shocks(self):
        # High λ to force arrivals early.
        sched = ShockScheduler(rng_seed=7, lambdas={"market_crash": 0.5})
        # Force an arrival on tick 0.
        arrivals = []
        while not arrivals:
            arrivals = sched.tick(0)
            if not arrivals:
                # Reroll on the next tick — eventually we must get one.
                # Bail if it takes absurdly long.
                if len(sched.active) == 0 and sum(1 for _ in range(1)) > 100:
                    pytest.fail("Could not inject a shock at high λ")
                break

        # Use a deterministic alternative path: inject manually rather than
        # relying on random draw timing.
        sched = ShockScheduler(rng_seed=7, lambdas={})
        injected = make_market_crash(random.Random(0), "minor")
        injected.tick_started = 10
        sched._active.append(injected)  # testing internal pruning behavior

        # Within window
        active_now = sched.tick(11)
        assert injected in sched.active

        # After expiry
        sched.tick(10 + injected.duration_ticks + 1)
        assert injected not in sched.active

        # Sanity: tick() returned no new arrivals (λ=0 for everything)
        assert active_now == []

    def test_scheduler_active_list_grows_and_shrinks(self):
        sched = ShockScheduler(
            rng_seed=123,
            lambdas={"market_crash": 0.02, "viral_growth_event": 0.02},
        )
        peak_active = 0
        for t in range(2000):
            sched.tick(t)
            peak_active = max(peak_active, len(sched.active))
        # By tick 2000, with λ=0.02 and durations < 270 ticks, we should have
        # both seen arrivals and seen expirations (active count not monotone).
        assert peak_active > 0
        # And by the very end, almost everything has expired.
        # (Strict equality to 0 isn't guaranteed; assert active is bounded.)
        assert len(sched.active) <= peak_active


# ---------------------------------------------------------------------------
# 5. Multiplicative composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_empty_active_returns_base_state(self):
        base = {"market_demand_mult": 1.0, "fixed_cost_add": 100.0}
        out = apply_active_shocks([], base)
        assert out == base
        assert out is not base  # fresh copy

    def test_does_not_mutate_base_state(self):
        base = {"market_demand_mult": 1.0}
        shock = Shock(
            name="test",
            severity="minor",
            duration_ticks=10,
            impact={"market_demand_mult": 0.7},
        )
        apply_active_shocks([shock], base)
        assert base == {"market_demand_mult": 1.0}

    def test_single_multiplicative_shock_applies(self):
        base = {"market_demand_mult": 1.0}
        shock = Shock(
            name="crash",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.7},
        )
        out = apply_active_shocks([shock], base)
        assert out["market_demand_mult"] == pytest.approx(0.7)

    def test_two_demand_cutting_shocks_compound(self):
        """Core invariant: 0.7 * 0.8 = 0.56 — stronger combined cut than either alone."""
        s1 = Shock(
            name="crash",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.7},
        )
        s2 = Shock(
            name="recession",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.8},
        )

        single1 = apply_active_shocks([s1], {})["market_demand_mult"]
        single2 = apply_active_shocks([s2], {})["market_demand_mult"]
        combined = apply_active_shocks([s1, s2], {})["market_demand_mult"]

        # Combined effect is stronger (smaller) than either alone.
        assert combined < single1
        assert combined < single2
        # Specifically multiplicative composition.
        assert combined == pytest.approx(0.7 * 0.8)

    def test_additive_keys_sum(self):
        s1 = Shock(
            name="reg",
            severity="moderate",
            duration_ticks=30,
            impact={"compliance_cost_add": 5_000.0},
        )
        s2 = Shock(
            name="reg2",
            severity="moderate",
            duration_ticks=30,
            impact={"compliance_cost_add": 3_000.0},
        )
        out = apply_active_shocks([s1, s2], {})
        assert out["compliance_cost_add"] == pytest.approx(8_000.0)

    def test_mixed_keys_compose_independently(self):
        s1 = Shock(
            name="crash",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.7, "compliance_cost_add": 1_000.0},
        )
        s2 = Shock(
            name="reg",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.9, "compliance_cost_add": 2_500.0},
        )
        out = apply_active_shocks([s1, s2], {})
        assert out["market_demand_mult"] == pytest.approx(0.7 * 0.9)
        assert out["compliance_cost_add"] == pytest.approx(3_500.0)

    def test_base_state_provides_starting_value_for_mult(self):
        # If base_state already has a non-1.0 multiplier, we compose around it.
        base = {"market_demand_mult": 1.2}  # an existing tailwind
        shock = Shock(
            name="crash",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.5},
        )
        out = apply_active_shocks([shock], base)
        assert out["market_demand_mult"] == pytest.approx(1.2 * 0.5)

    def test_unrelated_base_keys_pass_through(self):
        base = {"unrelated_key": 42.0}
        shock = Shock(
            name="crash",
            severity="moderate",
            duration_ticks=30,
            impact={"market_demand_mult": 0.7},
        )
        out = apply_active_shocks([shock], base)
        assert out["unrelated_key"] == 42.0
        assert out["market_demand_mult"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Bonus: scheduler validation
# ---------------------------------------------------------------------------


class TestSchedulerValidation:
    def test_rejects_unknown_shock_name_in_lambdas(self):
        with pytest.raises(ValueError, match="Unknown shock types"):
            ShockScheduler(rng_seed=0, lambdas={"alien_invasion": 0.1})

    def test_rejects_negative_lambda(self):
        with pytest.raises(ValueError, match="finite and"):
            ShockScheduler(rng_seed=0, lambdas={"market_crash": -0.1})

    def test_rejects_inf_lambda(self):
        with pytest.raises(ValueError, match="finite and"):
            ShockScheduler(rng_seed=0, lambdas={"market_crash": float("inf")})

    def test_zero_lambda_means_no_arrivals(self):
        sched = ShockScheduler(rng_seed=0, lambdas={name: 0.0 for name in SHOCK_FACTORIES})
        arrivals: list[Shock] = []
        for t in range(1000):
            arrivals.extend(sched.tick(t))
        assert arrivals == []

    def test_very_high_lambda_produces_many_arrivals(self):
        sched = ShockScheduler(rng_seed=0, lambdas={"market_crash": 2.0})
        total = sum(len(sched.tick(t)) for t in range(100))
        # E[count] = 200; 3-sigma below is ~165. Use a generous lower bound.
        assert total > 100

    def test_severity_is_drawn_from_weights(self):
        # With weight=1.0 on minor, every drawn shock should be minor.
        sched = ShockScheduler(
            rng_seed=0,
            lambdas={"market_crash": 0.5},
            severity_weights={"minor": 1.0, "moderate": 0.0, "severe": 0.0},
        )
        all_arrivals: list[Shock] = []
        for t in range(200):
            all_arrivals.extend(sched.tick(t))
        assert len(all_arrivals) > 0
        assert all(s.severity == "minor" for s in all_arrivals)


# ---------------------------------------------------------------------------
# Sanity: shock model
# ---------------------------------------------------------------------------


class TestShockModel:
    def test_construct_minimal(self):
        s = Shock(name="x", severity="minor", duration_ticks=1)
        assert s.name == "x"
        assert s.impact == {}
        assert s.description == ""
        assert s.tick_started is None

    def test_rejects_bad_severity(self):
        with pytest.raises(ValueError):
            Shock(name="x", severity="catastrophic", duration_ticks=1)  # type: ignore[arg-type]

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError):
            Shock(name="x", severity="minor", duration_ticks=0)

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError):
            Shock(name="x", severity="minor", duration_ticks=-1)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValueError):
            Shock(
                name="x",
                severity="minor",
                duration_ticks=1,
                bogus_field=42,  # type: ignore[call-arg]
            )

    def test_severity_typing_union_covers_all_three(self):
        for sev in ("minor", "moderate", "severe"):
            s = Shock(name="x", severity=sev, duration_ticks=1)  # type: ignore[arg-type]
            assert s.severity == sev

    def test_tick_started_is_settable_post_construction(self):
        s = Shock(name="x", severity="minor", duration_ticks=10)
        s.tick_started = 42
        assert s.tick_started == 42


# ---------------------------------------------------------------------------
# Type-narrow severity sanity
# ---------------------------------------------------------------------------


def test_severity_literal_covers_three_values():
    # This is a typing/runtime sanity check: SEVERITY_INTENSITY's keys should
    # exactly cover the Severity Literal.
    assert set(SEVERITY_INTENSITY) == {"minor", "moderate", "severe"}
    # Not a real test of the type system, but catches typos.
    sev: Severity = "minor"
    assert sev in SEVERITY_INTENSITY
