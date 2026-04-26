"""Tests for `orchestrator.HeuristicOrchestrator` — the operational tier.

These tests deliberately use a small synthetic node library (defined in
`_make_library()`) rather than loading `node_library.yaml`. The orchestrator's
contract is "library is passed in as a dict" — so we exercise it that way and
keep the suite fast and independent of `library_loader`.

What we cover:
  * Capacity > 95% sustained for 3+ ticks → hire (with prereqs satisfied)
  * Capacity > 95% for only 2 ticks → no hire (history matters)
  * Capacity < 40% sustained for 5+ ticks + cash buffer → layoff
  * Cash crisis → cost-cut decision (highest-cost non-exec)
  * Every emitted decision has non-empty `references_stance` (role-lock)
  * Every emitted decision has `tier == "heuristic"`
  * Cadence — only emits on `tick % 7 == 0`
  * Soft cap respected (non-emergency); hard cap respected (always)
"""

from __future__ import annotations

import pytest

from src.simulation.orchestrator import (
    DAYS_PER_MONTH,
    HEURISTIC_CADENCE_TICKS,
    CeoDecision,
    CompanyState,
    HeuristicOrchestrator,
)
from src.simulation.seed import CompanySeed
from src.simulation.stance import CeoStance

# ---------------------------------------------------------------------------
# Fixtures — minimal but realistic seed/stance/library
# ---------------------------------------------------------------------------


def _make_seed(economics_model: str = "subscription") -> CompanySeed:
    """A small but valid CompanySeed for tests. Subscription by default
    because most of our synthetic library nodes are subscription-applicable."""
    return CompanySeed(
        name="Test Co",
        niche="B2B SaaS for QA testing",
        archetype="small_team",
        industry_keywords=["saas", "qa"],
        location_label="Product",
        economics_model=economics_model,  # type: ignore[arg-type]
        base_price=99.0,
        base_unit_cost=20.0,
        daily_fixed_costs=500.0,
        starting_cash=1_000_000.0,
        starting_employees=5,
        base_capacity_per_location=1000,
        margin_target=0.50,
        revenue_per_employee_target=200_000.0,
        tam=1e8,
        competitor_density=3,
        market_growth_rate=0.10,
        customer_unit_label="subscribers",
        seasonality_amplitude=0.10,
        initial_supplier_types=["cloud_provider"],
        initial_revenue_streams=["subscription_revenue"],
        initial_cost_centers=["engineering_payroll"],
        initial_locations=1,
        initial_marketing_intensity=0.30,
        initial_quality_target=0.75,
        initial_price_position="mid",
        initial_capital_runway_months=12.0,
        initial_hiring_pace="steady",
        initial_geographic_scope="national",
        initial_revenue_concentration=0.40,
        initial_customer_acquisition_channel="content",
    )


def _make_stance(
    *,
    cash_comfort: float = 6.0,
    risk_tolerance: float = 0.5,
    hiring_bias: str = "balanced",
) -> CeoStance:
    return CeoStance(
        archetype="founder_operator",
        risk_tolerance=risk_tolerance,
        growth_obsession=0.55,
        quality_floor=0.7,
        hiring_bias=hiring_bias,  # type: ignore[arg-type]
        time_horizon="annual",
        cash_comfort=cash_comfort,
        signature_moves=[
            "stay close to the customer",
            "hire only when it hurts",
        ],
        voice="I run a tight ship and I take the calls myself.",
    )


def _make_library() -> dict[str, dict]:
    """Synthetic library covering every category the heuristic touches.

    Includes:
      * 1 founder ops node (no prereqs, free)
      * 1 first-engineer ops node (requires founder)
      * 1 mid-tier engineer ops node (requires first-engineer; this is
        what the capacity rule is expected to spawn after prereqs are met)
      * 1 cheap-but-prereq-blocked ops node (proves we honor prereqs)
      * 1 supplier (so replenish can fire)
      * 1 expensive non-exec node (cost-cut target)
      * 1 exec node (proves we don't cut execs in cash crisis)
      * 1 sales-category node (proves capacity hire only picks ops)
    """
    return {
        # ── ops ──
        "founder_engineer": {
            "category": "ops",
            "label": "Founder Engineer",
            "hire_cost": 0.0,
            "daily_fixed_costs": 0.0,  # zero TCO sentinel
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": [],
            "category_caps": {"soft_cap": 1, "hard_cap": 1},
            "applicable_economics": ["subscription", "service"],
        },
        "first_engineer": {
            "category": "ops",
            "label": "First Engineer",
            "hire_cost": 30000.0,
            "daily_fixed_costs": 300.0,
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": ["founder_engineer"],
            "category_caps": {"soft_cap": 1, "hard_cap": 1},
            "applicable_economics": ["subscription", "service"],
        },
        "junior_engineer": {
            "category": "ops",
            "label": "Junior Engineer",
            "hire_cost": 20000.0,
            "daily_fixed_costs": 200.0,  # cheapest non-zero TCO with prereq met
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": ["first_engineer"],
            "category_caps": {"soft_cap": 12, "hard_cap": 60},
            "applicable_economics": ["subscription", "service"],
        },
        "senior_engineer": {
            "category": "ops",
            "label": "Senior Engineer",
            "hire_cost": 32000.0,
            "daily_fixed_costs": 350.0,
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": ["first_engineer"],
            "category_caps": {"soft_cap": 8, "hard_cap": 50},
            "applicable_economics": ["subscription", "service"],
        },
        "data_team": {
            "category": "ops",
            "label": "Data Team",
            "hire_cost": 70000.0,
            "daily_fixed_costs": 700.0,  # expensive — cost-cut target
            "employees_count": 3,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": [],
            "category_caps": {"soft_cap": 1, "hard_cap": 4},
            "applicable_economics": ["subscription", "service"],
        },
        "blocked_node": {
            # Has an unsatisfied prereq → must never be picked by the
            # capacity rule even though it would otherwise be cheapest.
            "category": "ops",
            "label": "Blocked Node",
            "hire_cost": 1.0,
            "daily_fixed_costs": 1.0,
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": ["does_not_exist"],
            "category_caps": {"soft_cap": 5, "hard_cap": 5},
            "applicable_economics": ["subscription", "service"],
        },
        # ── supplier ──
        "cloud_provider": {
            "category": "supplier",
            "label": "Cloud Provider",
            "hire_cost": 0.0,
            "daily_fixed_costs": 50.0,
            "employees_count": 0,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": [],
            "category_caps": {"soft_cap": 1, "hard_cap": 3},
            "applicable_economics": ["subscription", "service"],
        },
        # ── exec — protected from cost cuts ──
        "founder_ceo": {
            "category": "exec",
            "label": "Founder CEO",
            "hire_cost": 0.0,
            "daily_fixed_costs": 0.0,
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": [],
            "category_caps": {"soft_cap": 1, "hard_cap": 1},
            "applicable_economics": ["physical", "subscription", "service"],
        },
        # ── sales — proves capacity hire only picks ops ──
        "bd_rep": {
            "category": "sales",
            "label": "BD Rep",
            "hire_cost": 8000.0,
            "daily_fixed_costs": 100.0,  # cheaper than every ops candidate
            "employees_count": 1,
            "capacity_contribution": 0,
            "modifier_keys": {},
            "prerequisites": [],
            "category_caps": {"soft_cap": 6, "hard_cap": 12},
            "applicable_economics": ["subscription", "service"],
        },
    }


def _make_state(
    *,
    tick: int,
    cash: float = 1_000_000.0,
    daily_burn: float = 1_000.0,
    monthly_revenue: float = 50_000.0,
    spawned: dict[str, int] | None = None,
    capacity_utilization: float = 0.5,
    avg_satisfaction: float = 0.7,
    employee_count: int = 5,
    recent_decisions: list[CeoDecision] | None = None,
) -> CompanyState:
    return CompanyState(
        tick=tick,
        cash=cash,
        daily_burn=daily_burn,
        monthly_revenue=monthly_revenue,
        spawned_nodes=dict(spawned) if spawned else {},
        capacity_utilization=capacity_utilization,
        avg_satisfaction=avg_satisfaction,
        employee_count=employee_count,
        active_shocks=[],
        recent_decisions=list(recent_decisions) if recent_decisions else [],
    )


def _drive_history(
    orch: HeuristicOrchestrator,
    *,
    starting_tick: int,
    n_ticks: int,
    utilization: float,
    spawned: dict[str, int],
    cash: float = 1_000_000.0,
    daily_burn: float = 1_000.0,
) -> list[CeoDecision | None]:
    """Tick the orchestrator `n_ticks` times in a row with the given inputs.

    Returns the decision list (one per tick, possibly mostly None). Lets a
    test build up "for X ticks" history before asserting on the trigger tick.
    """
    out: list[CeoDecision | None] = []
    for offset in range(n_ticks):
        state = _make_state(
            tick=starting_tick + offset,
            cash=cash,
            daily_burn=daily_burn,
            spawned=spawned,
            capacity_utilization=utilization,
        )
        out.append(orch.tick(state))
    return out


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_constructor_rejects_empty_library() -> None:
    with pytest.raises(ValueError):
        HeuristicOrchestrator(_make_seed(), _make_stance(), {})


def test_off_cadence_returns_none() -> None:
    """The heuristic only speaks on tick % 7 == 0; everything else is silent."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    # Ticks 1..6 are all off-cadence (1, 2, 3, 4, 5, 6 mod 7 != 0).
    for tick in range(1, HEURISTIC_CADENCE_TICKS):
        result = orch.tick(_make_state(tick=tick, capacity_utilization=0.5))
        assert result is None, f"expected silence on tick {tick}, got {result}"


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def test_cadence_only_emits_on_multiples_of_seven() -> None:
    """Every decision (whichever rule fires) must land on tick % 7 == 0."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    spawned = {"founder_engineer": 1, "first_engineer": 1, "cloud_provider": 1}
    decisions: list[CeoDecision] = []
    # Drive 50 ticks at high utilization so we get plenty of would-be hires.
    for tick in range(50):
        state = _make_state(
            tick=tick,
            spawned=spawned,
            capacity_utilization=0.99,
        )
        d = orch.tick(state)
        if d is not None:
            decisions.append(d)
    assert decisions, "expected at least one decision over 50 ticks of high util"
    for d in decisions:
        assert d.tick % HEURISTIC_CADENCE_TICKS == 0, (
            f"decision emitted off-cadence at tick {d.tick}"
        )


# ---------------------------------------------------------------------------
# Capacity-driven hire
# ---------------------------------------------------------------------------


def test_capacity_hire_fires_after_three_sustained_ticks() -> None:
    """Capacity > 95% for 3+ ticks → hire of an applicable ops node."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    spawned = {"founder_engineer": 1, "first_engineer": 1}

    # Walk forward to a cadence tick with three consecutive high-util samples.
    # We pick tick=14 as the trigger: ticks 12, 13, 14 all > 95%, and 14 is
    # divisible by 7 (cadence-aligned). The orchestrator records util on
    # every tick, so we tick all three.
    decisions = _drive_history(
        orch,
        starting_tick=12,
        n_ticks=3,
        utilization=0.99,
        spawned=spawned,
    )
    # Tick 12 and 13 are off-cadence → None. Tick 14 is on-cadence → decision.
    assert decisions[0] is None
    assert decisions[1] is None
    decision = decisions[2]
    assert decision is not None, "expected a hire decision on tick 14"
    assert decision.spawn_nodes, "expected non-empty spawn_nodes for capacity hire"

    spawned_key = decision.spawn_nodes[0]
    node_def = _make_library()[spawned_key]
    # Must be an ops node, applicable, with prereqs satisfied.
    assert node_def["category"] == "ops"
    assert "subscription" in node_def["applicable_economics"]
    for prereq in node_def["prerequisites"]:
        assert spawned.get(prereq, 0) > 0, (
            f"prereq {prereq} of chosen node {spawned_key} not satisfied"
        )
    # Specifically, with our library it should be junior_engineer (cheapest
    # non-zero-TCO ops node with prereqs met), NOT bd_rep (sales) or
    # blocked_node (unmet prereq).
    assert spawned_key == "junior_engineer", (
        f"expected cheapest applicable ops node 'junior_engineer'; got {spawned_key}"
    )


def test_capacity_hire_does_not_fire_at_two_ticks_history() -> None:
    """Capacity > 95% for only 2 ticks must NOT trigger — history matters."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    spawned = {"founder_engineer": 1, "first_engineer": 1}

    # Tick 13 (off-cadence): util 0.99 (sample 1)
    orch.tick(_make_state(tick=13, spawned=spawned, capacity_utilization=0.99))
    # Tick 14 (on-cadence): util 0.99 (sample 2). Only 2 high samples — must
    # NOT fire the capacity rule.
    decision = orch.tick(
        _make_state(tick=14, spawned=spawned, capacity_utilization=0.99)
    )
    if decision is not None:
        # Acceptable only if some other rule (replenish on tick 0 cadence)
        # fired and emitted a non-spawn decision. The capacity rule itself
        # must not be the source.
        assert not decision.spawn_nodes, (
            "capacity hire fired with only 2 ticks of high-util history"
        )


def test_capacity_hire_skips_non_ops_categories() -> None:
    """The capacity rule must pick only ops nodes — not sales/marketing/etc.

    `bd_rep` has the lowest TCO in the library but is `sales`. The hire rule
    must skip it.
    """
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    spawned = {"founder_engineer": 1, "first_engineer": 1}
    # Build 3+ ticks of high util, fire on cadence tick 21.
    decisions = _drive_history(
        orch,
        starting_tick=14,
        n_ticks=8,  # ticks 14..21
        utilization=0.99,
        spawned=spawned,
    )
    # The decision on tick 21 (or 14, depending on history alignment) must be
    # a spawn that picks ops, never bd_rep.
    spawn_decisions = [
        d for d in decisions if d is not None and d.spawn_nodes
    ]
    assert spawn_decisions, "no spawn decisions emitted across 8 high-util ticks"
    for d in spawn_decisions:
        for k in d.spawn_nodes:
            assert k != "bd_rep", "capacity rule incorrectly picked sales node"


# ---------------------------------------------------------------------------
# Layoff
# ---------------------------------------------------------------------------


def test_layoff_fires_after_five_low_ticks_with_runway() -> None:
    """Capacity < 40% for 5+ ticks AND 6+ months of runway → LIFO ops layoff."""
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), library)
    # Pre-seed the orchestrator's internal LIFO with two ops hires so it
    # knows what to retire. Mimics what would have happened in earlier ticks
    # if the capacity rule had previously fired.
    orch._spawn_history.extend(["junior_engineer", "senior_engineer"])
    spawned = {
        "founder_engineer": 1,
        "first_engineer": 1,
        "junior_engineer": 1,
        "senior_engineer": 1,
    }
    # Big cash buffer: 10M cash / (1k * 30) ≈ 333 months of runway, well above
    # the 6-month layoff floor.
    _drive_history(
        orch,
        starting_tick=21,
        n_ticks=5,  # ticks 21..25; tick 21 is on-cadence (21 % 7 == 0)
        utilization=0.10,
        spawned=spawned,
        cash=10_000_000.0,
        daily_burn=1_000.0,
    )
    # Tick 21 is on-cadence, but only one low-util sample yet → no layoff.
    # Tick 28 would be next on-cadence; but we only ran to tick 25. Run more.
    more = _drive_history(
        orch,
        starting_tick=26,
        n_ticks=3,  # ticks 26..28
        utilization=0.10,
        spawned=spawned,
        cash=10_000_000.0,
        daily_burn=1_000.0,
    )
    # Tick 28 is on-cadence; by then we have 8 consecutive low-util samples
    # (ticks 21..28 — but UTILIZATION_HISTORY_LEN=5 caps to last 5).
    decision = more[-1]
    assert decision is not None, "expected a layoff decision on tick 28"
    assert decision.retire_nodes, "expected non-empty retire_nodes for layoff"
    # LIFO → senior_engineer was the most recent push.
    assert decision.retire_nodes[0] == "senior_engineer", (
        f"expected LIFO retire of 'senior_engineer'; got {decision.retire_nodes}"
    )


def test_layoff_blocked_when_runway_below_floor() -> None:
    """Low utilization + thin cash → layoff rule must defer to cash-crisis rule."""
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(cash_comfort=6.0), library)
    orch._spawn_history.append("junior_engineer")
    spawned = {
        "founder_engineer": 1,
        "first_engineer": 1,
        "junior_engineer": 1,
        "data_team": 1,
    }
    # Cash = 30k, daily_burn = 1k → 1 month of runway. Below the 6-month
    # layoff floor and below the 6-month cash_comfort → cash-crisis fires
    # first, not the layoff rule.
    decisions = _drive_history(
        orch,
        starting_tick=21,
        n_ticks=8,
        utilization=0.10,
        spawned=spawned,
        cash=30_000.0,
        daily_burn=1_000.0,
    )
    crisis_decisions = [d for d in decisions if d is not None]
    assert crisis_decisions, "expected at least one cash-crisis decision"
    # Cash-crisis cites cash_comfort + risk_tolerance; layoff cites only cash_comfort.
    chosen = crisis_decisions[-1]
    assert "risk_tolerance" in chosen.references_stance, (
        f"expected cash-crisis (cites risk_tolerance); got {chosen.references_stance}"
    )


# ---------------------------------------------------------------------------
# Cash crisis
# ---------------------------------------------------------------------------


def test_cash_crisis_retires_highest_cost_non_exec() -> None:
    """Cash < cash_comfort * burn * 30 → retire highest-cost non-exec node.

    With our library, `data_team` has daily_fixed_costs=700 (highest). The
    rule must pick it, not `founder_ceo` (exec, protected) or
    `cloud_provider` (lower cost).
    """
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(cash_comfort=6.0), library)
    spawned = {
        "founder_engineer": 1,
        "founder_ceo": 1,
        "data_team": 1,
        "cloud_provider": 1,
    }
    # 30k cash / 1k burn = 1 month runway, 6-month comfort → crisis trigger.
    state = _make_state(
        tick=HEURISTIC_CADENCE_TICKS,  # 7 — on cadence
        cash=30_000.0,
        daily_burn=1_000.0,
        spawned=spawned,
    )
    decision = orch.tick(state)
    assert decision is not None, "expected cash-crisis decision"
    assert decision.retire_nodes == ["data_team"], (
        f"expected highest-cost non-exec 'data_team'; got {decision.retire_nodes}"
    )
    assert "cash_comfort" in decision.references_stance
    assert "risk_tolerance" in decision.references_stance


def test_cash_crisis_skips_exec_nodes() -> None:
    """When the only spawned non-exec node has zero daily cost, the crisis
    rule must not retire the founder_ceo (exec, protected). It returns None
    and lets a higher tier intervene."""
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(cash_comfort=6.0), library)
    spawned = {"founder_ceo": 1, "founder_engineer": 1}  # both zero daily cost
    state = _make_state(
        tick=HEURISTIC_CADENCE_TICKS,
        cash=10.0,
        daily_burn=1_000.0,
        spawned=spawned,
    )
    decision = orch.tick(state)
    # Either no decision (crisis can't act → defer) or a non-retire decision.
    if decision is not None:
        assert "founder_ceo" not in decision.retire_nodes


def test_no_cash_crisis_when_cash_above_threshold() -> None:
    """Cash above comfort threshold → no crisis decision."""
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(cash_comfort=6.0), library)
    spawned = {"founder_ceo": 1, "data_team": 1}
    state = _make_state(
        tick=HEURISTIC_CADENCE_TICKS,
        # 6 months runway = comfort floor. 7+ months → safe.
        cash=7.0 * 1_000.0 * DAYS_PER_MONTH,
        daily_burn=1_000.0,
        spawned=spawned,
        capacity_utilization=0.5,  # neutral util — no other rule should fire
    )
    decision = orch.tick(state)
    assert decision is None or not decision.retire_nodes


# ---------------------------------------------------------------------------
# Role-lock — references_stance always non-empty
# ---------------------------------------------------------------------------


def test_every_decision_has_non_empty_references_stance() -> None:
    """Drive a wide range of states; every emitted decision must cite stance."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    decisions: list[CeoDecision] = []

    # Phase 1: high util sustained → capacity hire
    for _ in range(5):
        d = orch.tick(
            _make_state(
                tick=HEURISTIC_CADENCE_TICKS,
                spawned={"founder_engineer": 1, "first_engineer": 1},
                capacity_utilization=0.99,
            )
        )
        if d:
            decisions.append(d)

    # Phase 2: low util sustained + cash → layoff
    orch._spawn_history.append("junior_engineer")
    for tick in range(2 * HEURISTIC_CADENCE_TICKS, 6 * HEURISTIC_CADENCE_TICKS):
        d = orch.tick(
            _make_state(
                tick=tick,
                spawned={
                    "founder_engineer": 1,
                    "first_engineer": 1,
                    "junior_engineer": 1,
                },
                capacity_utilization=0.10,
                cash=10_000_000.0,
            )
        )
        if d:
            decisions.append(d)

    # Phase 3: cash crisis
    d = orch.tick(
        _make_state(
            tick=10 * HEURISTIC_CADENCE_TICKS,
            spawned={"founder_ceo": 1, "data_team": 1},
            cash=10.0,
            daily_burn=1_000.0,
        )
    )
    if d:
        decisions.append(d)

    assert decisions, "test setup produced zero decisions"
    for d in decisions:
        assert d.references_stance, (
            f"role-lock violation: empty references_stance on tick={d.tick}, "
            f"reasoning={d.reasoning!r}"
        )


# ---------------------------------------------------------------------------
# Tier label
# ---------------------------------------------------------------------------


def test_every_decision_is_tier_heuristic() -> None:
    """`tier == 'heuristic'` is the structural marker for this layer."""
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), _make_library())
    spawned = {"founder_engineer": 1, "first_engineer": 1}
    decisions = []
    for tick in range(40):
        d = orch.tick(
            _make_state(
                tick=tick,
                spawned=spawned,
                capacity_utilization=0.99,
            )
        )
        if d:
            decisions.append(d)
    assert decisions
    for d in decisions:
        assert d.tier == "heuristic", f"got tier={d.tier!r} on tick {d.tick}"


# ---------------------------------------------------------------------------
# Cap policy
# ---------------------------------------------------------------------------


def test_hard_cap_never_breached() -> None:
    """Even under sustained capacity emergency, hard_cap is absolute.

    Set state so junior_engineer is already at hard_cap=60. The capacity rule
    must skip it and fall back to the next-cheapest applicable ops node
    (senior_engineer, hard_cap=50).
    """
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), library)
    spawned = {
        "founder_engineer": 1,
        "first_engineer": 1,
        "junior_engineer": 60,  # at hard_cap
    }
    decisions = _drive_history(
        orch,
        starting_tick=12,
        n_ticks=3,  # tick 14 on-cadence with 3 high-util samples
        utilization=0.99,
        spawned=spawned,
    )
    decision = decisions[-1]
    assert decision is not None
    # Must not pick junior_engineer (capped) or blocked_node (unmet prereq).
    assert "junior_engineer" not in decision.spawn_nodes
    assert "blocked_node" not in decision.spawn_nodes
    # Should fall back to senior_engineer.
    assert decision.spawn_nodes == ["senior_engineer"], (
        f"expected fallback to 'senior_engineer'; got {decision.spawn_nodes}"
    )


def test_soft_cap_breached_only_under_emergency() -> None:
    """Capacity hire treats sustained > 95% as emergency → soft_cap is breachable.

    Library node `senior_engineer` has soft_cap=8. With 8 already spawned
    AND junior_engineer at hard_cap, the next emergency hire should still
    pick senior_engineer (breach soft_cap, since hard_cap=50 isn't reached).
    """
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), library)
    spawned = {
        "founder_engineer": 1,
        "first_engineer": 1,
        "junior_engineer": 60,  # at hard_cap, blocked
        "senior_engineer": 8,  # at soft_cap, but emergency-eligible
    }
    decisions = _drive_history(
        orch,
        starting_tick=12,
        n_ticks=3,
        utilization=0.99,
        spawned=spawned,
    )
    decision = decisions[-1]
    assert decision is not None
    assert decision.spawn_nodes == ["senior_engineer"], (
        f"expected emergency-soft-cap-breach pick; got {decision.spawn_nodes}"
    )


def test_no_spawn_when_all_caps_exhausted() -> None:
    """If every applicable ops node is at hard_cap, capacity rule returns None.

    The orchestrator must not panic-spawn an inapplicable node or breach hard_cap.
    """
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), library)
    spawned = {
        "founder_engineer": 1,  # hard_cap=1 → blocked
        "first_engineer": 1,    # hard_cap=1 → blocked
        "junior_engineer": 60,  # hard_cap=60 → blocked
        "senior_engineer": 50,  # hard_cap=50 → blocked
        "data_team": 4,         # hard_cap=4 → blocked
    }
    decisions = _drive_history(
        orch,
        starting_tick=12,
        n_ticks=3,
        utilization=0.99,
        spawned=spawned,
    )
    decision = decisions[-1]
    # Either None (no rule fires) or some non-spawn rule (replenish, etc).
    if decision is not None:
        assert not decision.spawn_nodes, (
            f"orchestrator spawned despite all caps exhausted: {decision.spawn_nodes}"
        )


# ---------------------------------------------------------------------------
# Replenish
# ---------------------------------------------------------------------------


def test_replenish_fires_only_when_supplier_present() -> None:
    """Replenish rule needs at least one supplier node spawned."""
    library = _make_library()
    orch = HeuristicOrchestrator(_make_seed(), _make_stance(), library)
    # No supplier → no replenish even at long ticks.
    for tick in (HEURISTIC_CADENCE_TICKS, 30 * HEURISTIC_CADENCE_TICKS):
        d = orch.tick(_make_state(tick=tick, spawned={"founder_engineer": 1}))
        if d is not None:
            assert "replenish_supplier" not in d.adjust_params

    # With supplier → eventually replenish fires.
    saw_replenish = False
    spawned = {"founder_engineer": 1, "cloud_provider": 1}
    for tick in range(0, 100):
        d = orch.tick(_make_state(tick=tick, spawned=spawned, capacity_utilization=0.5))
        if d and "replenish_supplier" in d.adjust_params:
            saw_replenish = True
            assert d.references_stance, "replenish must cite stance attrs"
            break
    assert saw_replenish, "expected at least one replenish over 100 ticks with supplier"
