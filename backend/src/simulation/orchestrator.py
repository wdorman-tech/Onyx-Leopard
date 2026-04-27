"""CEO orchestrator — all three tiers (Onyx Leopard v2).

Per `V2_REDESIGN_PLAN.md` §4, the CEO orchestrator runs at three cadences:

    | Cadence    | Tick interval               | Driver                |
    |------------|-----------------------------|-----------------------|
    | Operational| every 7 ticks               | Heuristic Python rules|
    | Tactical   | every 30 ticks              | Haiku 4.5             |
    | Strategic  | every 90 ticks / on shock   | Sonnet 4.6            |

This module hosts all three:

  * `HeuristicOrchestrator` — pure Python, no LLM calls.
  * `TacticalOrchestrator` — Haiku 4.5, mid-cadence parameter / mid-level-hire
    decisions. Async LLM calls wired through `replay_or_call` for determinism.
  * `StrategicOrchestrator` — Sonnet 4.6, slow-cadence executive decisions
    (exec hires, expansion, raises, layoffs > 5%, vertical entry). Wakes
    immediately on any active severe shock via `force_wake=True`.
  * `OrchestratorBundle` — owns one of each tier and routes a single tick
    through the correct cadences in priority order.

All three tiers emit the same `CeoDecision` shape (subclass of
`replay.CeoDecision` plus `tier`/`tick` metadata), so downstream code can
consume any decision uniformly.

Heuristic responsibilities (preserved from the operational-only version):
  1. Capacity-driven hire — sustained utilization > 95% triggers a mechanical
     ops hire (cheapest applicable node honoring prerequisites + caps).
  2. Layoff — sustained utilization < 40% with healthy runway triggers a LIFO
     retirement of the most-recently-spawned ops node.
  3. Replenish — supplier nodes need periodic replenishment; emitted on a
     30-tick cadence (MVP — no per-supplier signal yet).
  4. Cost cut on cash crisis — when cash drops below the stance's comfort
     threshold, retire the highest-cost non-exec node.

LLM tier design constraints (deliberate, do not relax):
  * Every LLM call goes through `replay_or_call(transcript, prompt, ...)` so
    runs with `transcript.mode == "replay"` reproduce exactly without
    hitting the API.
  * Cost is preflighted via `cost_tracker.would_exceed(...)` before every
    call. If projected cost would push past the per-sim ceiling, the
    orchestrator skips, logs, and returns None — never partially commits.
  * Prompts are built as JSON strings. The replay layer's
    `canonicalize_prompt()` re-serializes JSON deterministically, so
    dict-key ordering and pretty-print whitespace can drift across runs
    without invalidating the recorded SHA.
  * `references_stance` is validated non-empty post-parse. Empty list =
    role-lock violation = retry once with explicit feedback. Persistent
    empty triggers a None return so the caller can fall back.
  * Spawn list is filtered post-parse: hard_cap breaches and unmet
    prerequisites are removed (with a note appended to `reasoning`) before
    the decision is returned to the engine.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.simulation.library_loader import NodeDef, NodeLibrary


def _get_client() -> AsyncAnthropic:
    """Build an AsyncAnthropic client from env credentials."""
    return AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env


from src.simulation.replay import (
    CeoDecision as _BaseCeoDecision,
)
from src.simulation.replay import (
    CostCeilingExceededError,
    CostTracker,
    Tier,
    Transcript,
    replay_or_call,
)
from src.simulation.seed import CompanySeed
from src.simulation.shocks import Shock
from src.simulation.stance import CeoStance, to_system_prompt

# Critic is imported eagerly at module bottom to keep the cycle one-directional:
# `critic.py` only references orchestrator types behind `TYPE_CHECKING`, so this
# direction is safe.
from src.simulation.critic import (
    CRITIC_VIOLATION_THRESHOLD,
    CriticAgent,
    CriticScore,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — every numeric threshold this module uses, named and documented.
# ---------------------------------------------------------------------------

#: Operational cadence per v2 spec §4. Heuristic orchestrator only emits
#: decisions on ticks where `tick % HEURISTIC_CADENCE_TICKS == 0`.
HEURISTIC_CADENCE_TICKS: int = 7

#: How many ticks of utilization history we keep. Long enough to satisfy the
#: "sustained for 5 ticks" layoff rule; short enough to react quickly.
UTILIZATION_HISTORY_LEN: int = 5

#: Capacity-utilization thresholds (0..1). Above HIGH for HIGH_PERSIST_TICKS in
#: a row → hire trigger. Below LOW for LOW_PERSIST_TICKS in a row → layoff.
CAPACITY_HIGH_THRESHOLD: float = 0.95
CAPACITY_LOW_THRESHOLD: float = 0.40
HIGH_PERSIST_TICKS: int = 3
LOW_PERSIST_TICKS: int = 5

#: Layoffs are only safe if the company has a healthy runway. Measured as
#: months of cash given current daily burn. CEOs with high `cash_comfort`
#: need an even bigger buffer; this is the floor.
LAYOFF_MIN_RUNWAY_MONTHS: float = 6.0

#: Approximate days per month, used to convert daily burn to monthly runway.
#: 30.0 is the same convention used elsewhere in the codebase (location.py,
#: monte_carlo, market presets).
DAYS_PER_MONTH: float = 30.0

#: Replenish cadence in ticks. MVP — fires unconditionally on multiples of
#: this from spawn time; the LLM tiers can override with smarter logic.
REPLENISH_CADENCE_TICKS: int = 30

#: Sentinel for "no cap" when a node definition omits soft_cap or hard_cap.
#: Conceptually unbounded; practically a value far above any realistic
#: spawn count (10**9). Replaces the previous bare `10**9` literal.
NO_CAP_SENTINEL: int = 10**9

#: Cost-cut policy: never retire an exec under cash crisis (it would just
#: trigger a leadership-vacuum cascade and is strategic-tier territory).
COST_CUT_PROTECTED_CATEGORIES: tuple[str, ...] = ("exec",)

#: Emergency soft-cap policy. The heuristic tier will breach `soft_cap` for
#: ops nodes when capacity utilization is critically high — but never breach
#: `hard_cap`. Strategic decisions are the LLM tier's job; this rule only
#: applies inside the operational loop.
SOFT_CAP_EMERGENCY_OK: bool = True


# ---------------------------------------------------------------------------
# adjust_params bounds (P-AI-2 — Phase 2.1) — orchestrator-side validation.
#
# These mirror the engine-side last-line clamps in `unified_v2.py`
# (PRICE_ADJUST_*, MARKETING_INTENSITY_*, RAISE_ALLOWED_ARCHETYPES). They
# are duplicated rather than imported because `unified_v2.py` already
# imports from this module — pulling them the other way would create a
# cycle. A follow-up phase that consolidates the bounds will move the
# canonical definition here and have `unified_v2.py` import from this
# module instead.
# ---------------------------------------------------------------------------

#: Lower bound multiplier on `seed.starting_price` for CEO-proposed prices.
#: A `price` value below `seed.starting_price * PRICE_LOWER_BOUND_MULT` is
#: dropped (not clamped) — extreme repricing should be a strategic decision,
#: not silently floored.
PRICE_LOWER_BOUND_MULT: float = 0.1

#: Upper bound multiplier on `seed.starting_price` for CEO-proposed prices.
#: A `price` above `seed.starting_price * PRICE_UPPER_BOUND_MULT` is dropped.
PRICE_UPPER_BOUND_MULT: float = 10.0

#: Bounds on `marketing_intensity` from the CEO. Out-of-range values are
#: CLAMPED (not dropped) — marketing intensity is a continuous knob, so a
#: clamp gives the engine a usable signal while the warning surfaces drift.
MARKETING_INTENSITY_LOWER: float = 0.0
MARKETING_INTENSITY_UPPER: float = 2.0

#: Bounds on `replenish_supplier` signal. The engine treats this key as a
#: trigger flag: any positive value invokes a one-shot supplier-burn debit
#: of fixed magnitude (see ``unified_v2._apply_param_adjustments``); the
#: numeric value carries no further information. We still clamp into
#: ``[0, 1]`` to make the semantics legible to operators reading
#: transcripts (0 = "do nothing", >0 = "trigger replenish") and to keep
#: the LLM from emitting absurd magnitudes that downstream tooling might
#: misread.
REPLENISH_SUPPLIER_LOWER: float = 0.0
REPLENISH_SUPPLIER_UPPER: float = 1.0

#: Stance archetypes that may submit a nonzero `raise_amount`. Bootstrap /
#: founder_operator / turnaround stances do not access external capital
#: markets — a `raise_amount` from one of those archetypes is dropped
#: outright with a warning. Matches `unified_v2.RAISE_ALLOWED_ARCHETYPES`.
RAISE_ALLOWED_STANCE_ARCHETYPES: frozenset[str] = frozenset(
    {"venture_growth", "consolidator"}
)

#: The set of `adjust_params` keys this module knows how to validate. Any
#: key NOT in this set is dropped with a warning — the orchestrator refuses
#: to forward inputs it hasn't bounded.
KNOWN_ADJUST_PARAMS_KEYS: frozenset[str] = frozenset(
    {"price", "marketing_intensity", "raise_amount", "replenish_supplier"}
)


# ---------------------------------------------------------------------------
# LLM tier constants — cadences, model IDs, prompt sizing, retry policy.
# ---------------------------------------------------------------------------

#: Tactical tier cadence — Haiku 4.5 fires on `tick % TACTICAL_CADENCE_TICKS == 0`.
TACTICAL_CADENCE_TICKS: int = 30

#: Strategic tier cadence — Sonnet 4.6 fires on `tick % STRATEGIC_CADENCE_TICKS == 0`
#: OR on any active severe shock (force_wake).
STRATEGIC_CADENCE_TICKS: int = 90

#: How many state-window snapshots each LLM tier keeps in its own rolling
#: history. Spec §4 calls for "last 10 ticks of revenue/cash/satisfaction/
#: capacity/market_share/employee_count" — we honor that here.
STATE_WINDOW_LEN: int = 10

#: How many of THIS tier's recent decisions to include in the prompt for
#: continuity. Spec §4: "last 3 decisions at this cadence with reasoning".
DECISION_HISTORY_LEN: int = 3

#: How many ticks of shock history to surface to the LLM in the user prompt.
#: Spec §4: "any events in last 30 ticks". Independent of cadence intervals.
SHOCK_LOOKBACK_TICKS: int = 30

#: Pinned model IDs per CLAUDE.md and `replay.MODEL_PRICING` keys. The
#: cost-tracker pricing map keys these exact strings — keep aligned.
TACTICAL_MODEL_ID: str = "claude-haiku-4-5"
STRATEGIC_MODEL_ID: str = "claude-sonnet-4-6"

#: Token budget envelopes (per call) used for the cost-preflight `would_exceed`
#: check. These are intentionally conservative upper bounds — Haiku and Sonnet
#: rarely hit them but the preflight is a guard against runaway prompt growth
#: blowing the per-sim ceiling. Tune with empirical data once monitoring is up.
TACTICAL_PREDICTED_INPUT_TOKENS: int = 4000
TACTICAL_PREDICTED_OUTPUT_TOKENS: int = 600
STRATEGIC_PREDICTED_INPUT_TOKENS: int = 6000
STRATEGIC_PREDICTED_OUTPUT_TOKENS: int = 800

#: Hard cap on `max_tokens` for the LLM call itself. Larger than the predicted
#: output so a verbose reasoning paragraph never gets truncated, but small
#: enough that one runaway response can't blow the whole budget.
TACTICAL_MAX_TOKENS: int = 1024
STRATEGIC_MAX_TOKENS: int = 1500

#: How many extra retry attempts a malformed JSON / empty references_stance
#: response gets before we log + return None. The first call is attempt 0;
#: with `LLM_MAX_RETRIES = 1` the LLM gets exactly one second chance.
LLM_MAX_RETRIES: int = 1

#: Frozen set of valid `CeoStance` field names. Used post-parse to filter the
#: LLM's `references_stance` list — anything not in this set is hallucinated
#: and dropped before the decision reaches the engine. Built from
#: `CeoStance.model_fields` so it stays in sync if stance gains/loses fields
#: without touching this module.
_STANCE_FIELDS: frozenset[str] = frozenset(CeoStance.model_fields.keys())


# ---------------------------------------------------------------------------
# Pydantic models — public contract of this module.
# ---------------------------------------------------------------------------


class CeoDecision(_BaseCeoDecision):
    """A single orchestrator decision, emitted once per cadence (or None).

    Subclass of `replay.CeoDecision` (the canonical, foundational schema —
    v2 spec §4 verbatim) that adds two orchestrator-only metadata fields:

      * `tier` — which orchestrator layer produced the call (heuristic /
        tactical / strategic). Lets transcripts and the critic agent audit
        each layer separately.
      * `tick` — the simulation tick the decision was emitted on. Distinct
        from `TranscriptEntry.tick` so a decision can be reasoned about
        outside the recording layer (tests, in-process logging).

    These fields are intentionally NOT on `replay.CeoDecision` because the
    transcript layer already records `tier` and `tick` per entry — duplicating
    them on the decision would make the on-disk format redundant. They live
    here on the orchestrator's view of the decision because that's where the
    orchestrator (and its tests) need them.
    """

    model_config = ConfigDict(extra="forbid")

    tier: Literal["heuristic", "tactical", "strategic"]
    tick: int = Field(ge=0)


class CompanyState(BaseModel):
    """Snapshot of company state passed to the orchestrator on every cadence.

    Contains everything the heuristic rules need to make a decision without
    reaching back into engine internals. The LLM tiers will accept the same
    object plus an additional state-window of historical metrics.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tick: int = Field(ge=0)
    cash: float
    daily_burn: float = Field(ge=0.0)
    monthly_revenue: float
    spawned_nodes: dict[str, int] = Field(default_factory=dict)
    capacity_utilization: float = Field(ge=0.0, le=2.0)
    avg_satisfaction: float = Field(ge=0.0, le=1.0)
    employee_count: int = Field(ge=0)
    active_shocks: list[Shock] = Field(default_factory=list)
    recent_decisions: list[CeoDecision] = Field(
        default_factory=list,
        description="Most-recent decisions across tiers, oldest-first. Heuristic uses last 3.",
    )


# ---------------------------------------------------------------------------
# HeuristicOrchestrator — operational base class.
# ---------------------------------------------------------------------------


class HeuristicOrchestrator:
    """Operational-tier CEO orchestrator. Pure Python, no LLM.

    This class is also the operational base the LLM tiers compose with. The
    follow-up `orchestrator_llm.py` module will instantiate a
    `HeuristicOrchestrator` per company and call its `tick()` first; tactical
    and strategic decisions layer on top at their own cadences.

    Responsibilities:
        * Maintain rolling utilization history (deque of last N ticks).
        * Track spawn/retire order so LIFO layoffs are deterministic.
        * Track per-node spawn ticks so the replenish cadence can fire from
          actual instantiation time, not just absolute simulation time.
        * Validate the role-lock invariant on every emitted decision
          (`references_stance` must be non-empty).
    """

    def __init__(
        self,
        seed: CompanySeed,
        stance: CeoStance,
        library: dict[str, dict],
    ) -> None:
        """Construct an orchestrator for one simulated company.

        Args:
            seed: Company seed (immutable for the run).
            stance: Locked CEO persona (immutable for the run).
            library: Node library as a dict of `{node_key: node_def_dict}`.
                Pass the loaded contents of `node_library.yaml` — this class
                deliberately does **not** import `library_loader` so it stays
                composable with mocked libraries in tests.

        Raises:
            ValueError: if `library` is empty (a heuristic tier with no nodes
                to spawn is structurally broken).
        """
        if not library:
            raise ValueError(
                "HeuristicOrchestrator requires a non-empty node library; "
                f"got {len(library)} entries."
            )
        self.seed: CompanySeed = seed
        self.stance: CeoStance = stance
        self.library: dict[str, dict] = library

        # Rolling history of capacity_utilization. Bounded so memory stays
        # constant per simulation regardless of run length.
        self._utilization_history: deque[float] = deque(maxlen=UTILIZATION_HISTORY_LEN)

        # LIFO ordering of operational hires this orchestrator made. Used to
        # pick the most-recently-spawned ops node for layoffs. Source-of-truth
        # for spawned counts is `state.spawned_nodes`; this list is purely a
        # *recency* memo. Entries persist after retirement is requested
        # because retirement is fire-and-forget at this layer — the engine
        # applies it asynchronously and reflects in `state.spawned_nodes` on
        # the next tick.
        self._spawn_history: list[str] = []

        # Last tick the replenish signal fired. -1 sentinel means "never".
        self._last_replenish_tick: int = -1

    # ---- Public API -------------------------------------------------------

    def tick(self, state: CompanyState) -> CeoDecision | None:
        """Advance the orchestrator one tick.

        Always updates internal history (so off-cadence ticks still feed the
        rolling window). Only emits a decision on cadence ticks. Returns
        `None` between cadences and when no rule fires.
        """
        # 1. Always record utilization history. The "for N ticks" rules below
        #    require a real timeline, not a single-snapshot guess.
        self._utilization_history.append(state.capacity_utilization)

        # 2. Cadence gate. Off-cadence ticks return None but still bookkeep.
        if state.tick % HEURISTIC_CADENCE_TICKS != 0:
            return None

        # 3. Rule priority: cash crisis first (preservation), then capacity
        #    pressure, then layoff, then replenish. Order matters — if the
        #    company is bleeding cash we don't want to hire on the same call.
        decision = (
            self._cash_crisis_rule(state)
            or self._capacity_hire_rule(state)
            or self._layoff_rule(state)
            or self._replenish_rule(state)
        )

        if decision is None:
            return None

        # 4. Role-lock invariant: every heuristic must cite which stance
        #    attributes drove it. Empty list = bug.
        assert decision.references_stance, (
            "Role-lock violation: heuristic emitted a decision with no "
            "stance references. This is a bug in HeuristicOrchestrator."
        )

        # 5. Track the spawn/retire effects for our internal LIFO bookkeeping.
        for node_key in decision.spawn_nodes:
            self._spawn_history.append(node_key)
        for node_key in decision.retire_nodes:
            # Remove the most recent entry of that key, if present. Mirrors
            # LIFO retirement semantics.
            for i in range(len(self._spawn_history) - 1, -1, -1):
                if self._spawn_history[i] == node_key:
                    self._spawn_history.pop(i)
                    break

        return decision

    # ---- Rule 1: cash crisis ---------------------------------------------

    def _cash_crisis_rule(self, state: CompanyState) -> CeoDecision | None:
        """Retire highest-cost non-exec node when cash runway is below comfort.

        Trigger: cash < stance.cash_comfort * daily_burn * 30. (i.e. fewer
        months of runway than this CEO is comfortable with.)
        """
        if state.daily_burn <= 0.0:
            # No burn → no runway concern. Heuristic can't help here.
            return None

        crisis_floor = self.stance.cash_comfort * state.daily_burn * DAYS_PER_MONTH
        if state.cash >= crisis_floor:
            return None

        target = self._highest_cost_unprotected_node(state)
        if target is None:
            # Nothing safe to cut — strategic tier's problem.
            return None

        runway_months = state.cash / (state.daily_burn * DAYS_PER_MONTH)
        node_def = self.library[target]
        reasoning = (
            f"Cash crisis: ${state.cash:,.0f} = {runway_months:.1f} months of "
            f"runway, below my {self.stance.cash_comfort:.1f}-month comfort. "
            f"Cutting {node_def.get('label', target)} (highest-cost "
            f"non-exec node) to extend runway. Risk tolerance "
            f"{self.stance.risk_tolerance:.2f} says preserve optionality."
        )
        return CeoDecision(
            spawn_nodes=[],
            retire_nodes=[target],
            adjust_params={},
            open_locations=0,
            reasoning=reasoning,
            references_stance=["cash_comfort", "risk_tolerance"],
            tier="heuristic",
            tick=state.tick,
        )

    # ---- Rule 2: capacity-driven hire ------------------------------------

    def _capacity_hire_rule(self, state: CompanyState) -> CeoDecision | None:
        """Spawn cheapest applicable ops node when utilization is sustained-high."""
        if not self._utilization_persistent_above(CAPACITY_HIGH_THRESHOLD, HIGH_PERSIST_TICKS):
            return None

        candidate = self._cheapest_ops_candidate(state)
        if candidate is None:
            # Nothing legal to spawn — wait for strategic tier or different state.
            return None

        node_def = self.library[candidate]
        cost_estimate = float(node_def.get("hire_cost", 0.0)) + DAYS_PER_MONTH * float(
            node_def.get("daily_fixed_costs", 0.0)
        )
        reasoning = (
            f"Capacity utilization {state.capacity_utilization:.0%} sustained "
            f"above {CAPACITY_HIGH_THRESHOLD:.0%} for {HIGH_PERSIST_TICKS}+ "
            f"ticks. Hiring {node_def.get('label', candidate)} (cheapest "
            f"applicable ops node, ~${cost_estimate:,.0f} 30-day TCO). "
            f"Hiring bias '{self.stance.hiring_bias}' supports a mechanical "
            f"capacity hire here."
        )
        return CeoDecision(
            spawn_nodes=[candidate],
            retire_nodes=[],
            adjust_params={},
            open_locations=0,
            reasoning=reasoning,
            references_stance=["hiring_bias"],
            tier="heuristic",
            tick=state.tick,
        )

    # ---- Rule 3: layoff ---------------------------------------------------

    def _layoff_rule(self, state: CompanyState) -> CeoDecision | None:
        """Retire most-recently-spawned ops node on sustained underutilization.

        Only fires if the company has at least LAYOFF_MIN_RUNWAY_MONTHS of
        cash buffer — otherwise the cash-crisis rule handles it.
        """
        if not self._utilization_persistent_below(CAPACITY_LOW_THRESHOLD, LOW_PERSIST_TICKS):
            return None
        if state.daily_burn <= 0.0:
            return None
        runway_months = state.cash / (state.daily_burn * DAYS_PER_MONTH)
        if runway_months < LAYOFF_MIN_RUNWAY_MONTHS:
            return None

        target = self._most_recent_ops_node(state)
        if target is None:
            return None

        node_def = self.library[target]
        reasoning = (
            f"Capacity utilization {state.capacity_utilization:.0%} sustained "
            f"below {CAPACITY_LOW_THRESHOLD:.0%} for {LOW_PERSIST_TICKS}+ "
            f"ticks with {runway_months:.1f} months of runway. Retiring "
            f"{node_def.get('label', target)} (most recently hired ops node) "
            f"to right-size capacity. Cash comfort {self.stance.cash_comfort:.1f} "
            f"months is preserved by the cut."
        )
        return CeoDecision(
            spawn_nodes=[],
            retire_nodes=[target],
            adjust_params={},
            open_locations=0,
            reasoning=reasoning,
            references_stance=["cash_comfort"],
            tier="heuristic",
            tick=state.tick,
        )

    # ---- Rule 4: replenish ------------------------------------------------

    def _replenish_rule(self, state: CompanyState) -> CeoDecision | None:
        """Emit a replenish signal on a fixed cadence as MVP.

        v2 spec §6 says supplier interaction will eventually flow through
        auto-derived bridge modifiers; until then, this rule emits a generic
        `replenish_supplier=1.0` parameter on a 30-tick cadence whenever the
        company has at least one supplier node spawned. The LLM tiers can
        override or replace this with smarter logic.
        """
        # Only meaningful if there's a supplier on the books.
        suppliers = self._spawned_nodes_in_category(state, "supplier")
        if not suppliers:
            return None

        if (
            self._last_replenish_tick >= 0
            and state.tick - self._last_replenish_tick < REPLENISH_CADENCE_TICKS
        ):
            return None

        self._last_replenish_tick = state.tick

        reasoning = (
            f"Routine supplier replenishment cycle ({REPLENISH_CADENCE_TICKS}-"
            f"tick cadence). Time horizon '{self.stance.time_horizon}' makes "
            f"this an automatic operational call."
        )
        return CeoDecision(
            spawn_nodes=[],
            retire_nodes=[],
            adjust_params={"replenish_supplier": 1.0},
            open_locations=0,
            reasoning=reasoning,
            references_stance=["time_horizon"],
            tier="heuristic",
            tick=state.tick,
        )

    # ---- Helpers ---------------------------------------------------------

    def _utilization_persistent_above(self, threshold: float, ticks: int) -> bool:
        """True iff the *last* `ticks` recorded utilizations are all > threshold.

        Returns False until we have at least `ticks` samples — we never make
        a "sustained" claim from partial history.
        """
        if len(self._utilization_history) < ticks:
            return False
        # Slice the rightmost `ticks` entries.
        recent = list(self._utilization_history)[-ticks:]
        return all(u > threshold for u in recent)

    def _utilization_persistent_below(self, threshold: float, ticks: int) -> bool:
        """True iff the *last* `ticks` recorded utilizations are all < threshold."""
        if len(self._utilization_history) < ticks:
            return False
        recent = list(self._utilization_history)[-ticks:]
        return all(u < threshold for u in recent)

    def _spawned_nodes_in_category(self, state: CompanyState, category: str) -> list[str]:
        """Return node_keys of currently-spawned nodes in `category` (count > 0)."""
        out: list[str] = []
        for node_key, count in state.spawned_nodes.items():
            if count <= 0:
                continue
            node_def = self.library.get(node_key)
            if node_def is None:
                # Unknown node in state — silently skip; library_loader is
                # responsible for cross-validation. We don't crash here.
                continue
            if node_def.get("category") == category:
                out.append(node_key)
        return out

    def _prerequisites_satisfied(self, node_key: str, state: CompanyState) -> bool:
        """True iff every prerequisite of `node_key` has count > 0 in state."""
        node_def = self.library.get(node_key)
        if node_def is None:
            return False
        for prereq in node_def.get("prerequisites", []) or []:
            if state.spawned_nodes.get(prereq, 0) <= 0:
                return False
        return True

    def _is_applicable(self, node_key: str) -> bool:
        """True iff this node fits the seed's economics model."""
        node_def = self.library.get(node_key)
        if node_def is None:
            return False
        applicable = node_def.get("applicable_economics", []) or []
        return self.seed.economics_model in applicable

    def _within_caps(
        self,
        node_key: str,
        state: CompanyState,
        *,
        emergency: bool = False,
    ) -> bool:
        """Cap policy: hard_cap is absolute; soft_cap may be breached in emergency.

        We treat `_capacity_hire_rule` invocations as `emergency=True` because
        sustained > 95% utilization is the operational definition of an
        emergency. Other heuristics (none currently) would pass `False`.
        """
        node_def = self.library.get(node_key)
        if node_def is None:
            return False
        caps = node_def.get("category_caps", {}) or {}
        soft = int(caps.get("soft_cap", NO_CAP_SENTINEL))
        hard = int(caps.get("hard_cap", NO_CAP_SENTINEL))
        current = int(state.spawned_nodes.get(node_key, 0))
        if current >= hard:
            return False
        return not (not emergency and current >= soft)

    def _cheapest_ops_candidate(self, state: CompanyState) -> str | None:
        """Pick the cheapest applicable ops node honoring prereqs + caps.

        Cost is `hire_cost + 30 * daily_fixed_costs` (one-month TCO). Sorting
        on TCO instead of just `hire_cost` avoids hiring "free" founder roles
        that have hire_cost=0 but still burn salary, and avoids preferring a
        cheap-to-hire-but-expensive-to-keep candidate.

        Only returns a node whose category is `ops`. The library has many
        categories (sales/marketing/finance/exec/etc.) but capacity hires are
        operational by definition — sales hires are demand-side and belong in
        the tactical (Haiku) tier.
        """
        candidates: list[tuple[float, str]] = []
        for node_key, node_def in self.library.items():
            if node_def.get("category") != "ops":
                continue
            if not self._is_applicable(node_key):
                continue
            if not self._prerequisites_satisfied(node_key, state):
                continue
            if not self._within_caps(node_key, state, emergency=SOFT_CAP_EMERGENCY_OK):
                continue
            tco = float(node_def.get("hire_cost", 0.0)) + DAYS_PER_MONTH * float(
                node_def.get("daily_fixed_costs", 0.0)
            )
            candidates.append((tco, node_key))
        if not candidates:
            return None
        # Stable sort: cheapest TCO first; ties broken by lexicographic key
        # so this is deterministic across runs.
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][1]

    def _most_recent_ops_node(self, state: CompanyState) -> str | None:
        """LIFO over `_spawn_history` filtered to ops nodes still present."""
        # Walk newest→oldest. Skip anything that isn't ops or that the state
        # no longer reports as present (count > 0).
        for node_key in reversed(self._spawn_history):
            node_def = self.library.get(node_key)
            if node_def is None:
                continue
            if node_def.get("category") != "ops":
                continue
            if state.spawned_nodes.get(node_key, 0) <= 0:
                continue
            return node_key
        # Fallback: nothing in history → look at any current ops node and
        # pick the one with the most instances (likely the most-cuttable).
        # This handles the "orchestrator restored from snapshot, history
        # empty" path gracefully.
        ops_nodes = self._spawned_nodes_in_category(state, "ops")
        if not ops_nodes:
            return None
        ops_nodes.sort(key=lambda k: (-state.spawned_nodes[k], k))
        return ops_nodes[0]

    def _highest_cost_unprotected_node(self, state: CompanyState) -> str | None:
        """Pick the highest daily-cost spawned node not in a protected category.

        Used by the cash-crisis rule. Skips exec roles (cutting the CEO under
        a cash crisis is exactly the kind of decision that requires the
        strategic tier, not a heuristic).
        """
        candidates: list[tuple[float, str]] = []
        for node_key, count in state.spawned_nodes.items():
            if count <= 0:
                continue
            node_def = self.library.get(node_key)
            if node_def is None:
                continue
            if node_def.get("category") in COST_CUT_PROTECTED_CATEGORIES:
                continue
            daily = float(node_def.get("daily_fixed_costs", 0.0))
            if daily <= 0.0:
                continue
            candidates.append((daily, node_key))
        if not candidates:
            return None
        # Highest daily cost first; tie-break lexicographically for determinism.
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][1]


# ---------------------------------------------------------------------------
# LLM tiers — Haiku tactical + Sonnet strategic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StateSnapshot:
    """One row of the rolling state window the LLM tiers maintain.

    Spec §4 lists revenue, cash, satisfaction, capacity, market share, and
    employee count as the "last 10 ticks" payload. Market share isn't on the
    `CompanyState` shape today (it's derived in the engine), so we record what
    we have and let the engine populate it later via a state extension —
    keeping `market_share` as `None` until then.
    """

    tick: int
    cash: float
    monthly_revenue: float
    capacity_utilization: float
    avg_satisfaction: float
    employee_count: int


class _LlmOrchestratorBase:
    """Shared machinery for the Haiku-tactical and Sonnet-strategic tiers.

    Subclasses fix four class-level constants:
      * `TIER` — `"tactical"` | `"strategic"` (becomes the `CeoDecision.tier`
        and the `Tier` literal handed to `replay_or_call`).
      * `MODEL_ID` — Anthropic model id; must be a key of `MODEL_PRICING`.
      * `CADENCE_TICKS` — emit on `tick % CADENCE_TICKS == 0`.
      * `DOMAIN_INSTRUCTION` — short paragraph appended to the system prompt
        that tells the CEO what kinds of decisions this tier owns.

    Plus three sizing constants used by the cost-tracker preflight:
      * `PREDICTED_INPUT_TOKENS`, `PREDICTED_OUTPUT_TOKENS`, `MAX_TOKENS`.

    Construction is sync; `tick()` is async because the underlying Anthropic
    SDK call is async. Tests mock the `AsyncAnthropic` client by injecting
    one through the constructor; production paths let `_get_client()` build it
    lazily on first call.
    """

    TIER: ClassVar[Literal["tactical", "strategic"]]
    MODEL_ID: ClassVar[str]
    CADENCE_TICKS: ClassVar[int]
    DOMAIN_INSTRUCTION: ClassVar[str]
    PREDICTED_INPUT_TOKENS: ClassVar[int]
    PREDICTED_OUTPUT_TOKENS: ClassVar[int]
    MAX_TOKENS: ClassVar[int]

    def __init__(
        self,
        seed: CompanySeed,
        stance: CeoStance,
        library: NodeLibrary,
        transcript: Transcript,
        cost_tracker: CostTracker,
        client: AsyncAnthropic | None = None,
        *,
        sim_id: str = "sim-default",
        company_id: str | None = None,
    ) -> None:
        """Construct an LLM orchestrator for one company.

        `seed`, `stance`, and `library` are immutable for the run.
        `transcript` and `cost_tracker` are shared with the bundle and the
        rest of the sim — they belong to the caller, not the orchestrator.
        `client` is optional; lazy-built via `ceo_agent._get_client()` on
        first LLM call when None. Tests inject a mock here.

        `sim_id` and `company_id` are passed through to `replay_or_call` so
        recorded entries are correctly keyed. `company_id` defaults to the
        seed's name (collisions only matter inside one transcript file, which
        belongs to one sim, so seed.name is unique enough for the common case).
        """
        if not isinstance(library, NodeLibrary):
            raise TypeError(f"library must be a NodeLibrary; got {type(library).__name__}")
        self.seed: CompanySeed = seed
        self.stance: CeoStance = stance
        self.library: NodeLibrary = library
        self.transcript: Transcript = transcript
        self.cost_tracker: CostTracker = cost_tracker
        self._client: AsyncAnthropic | None = client
        self._sim_id: str = sim_id
        self._company_id: str = company_id or seed.name

        # Rolling state window — bounded so memory is constant per sim length.
        self._state_window: deque[_StateSnapshot] = deque(maxlen=STATE_WINDOW_LEN)

    # -- Public API ---------------------------------------------------------

    async def tick(self, state: CompanyState, force_wake: bool = False) -> CeoDecision | None:
        """Advance the orchestrator one tick.

        Always pushes a snapshot onto the rolling state window. Only invokes
        the LLM on cadence ticks (`tick % CADENCE_TICKS == 0`) or on
        `force_wake=True` (the strategic-tier severe-shock interrupt).

        Returns `None` between cadences, when the cost ceiling would be
        breached, or when the LLM persistently produces invalid output. Never
        raises on an LLM-side failure — the sim must keep running.
        """
        # 1. Always record the state snapshot. Off-cadence ticks still feed
        #    the rolling window so on-cadence calls have a real history.
        self._record_snapshot(state)

        # 2. Cadence gate — but `force_wake` overrides for severe-shock interrupts.
        if not force_wake and state.tick % self.CADENCE_TICKS != 0:
            return None

        # 3. Cost preflight. If the conservative budget projection would push
        #    us past the ceiling, skip the call rather than attempt-and-fail.
        if self.cost_tracker.would_exceed(
            self.PREDICTED_INPUT_TOKENS,
            self.PREDICTED_OUTPUT_TOKENS,
            self.MODEL_ID,
        ):
            log.warning(
                "%s tier skipping tick=%d (company=%s): "
                "projected cost would exceed ceiling $%.4f (current $%.4f)",
                self.TIER,
                state.tick,
                self._company_id,
                self.cost_tracker.ceiling_usd,
                self.cost_tracker.total_cost(),
            )
            return None

        # 4. Build the prompt + system message ONCE per tick. The retry path
        #    appends a corrective user-side hint but reuses the system prompt.
        system_prompt = self._build_system_prompt()
        feedback: str | None = None

        for attempt in range(LLM_MAX_RETRIES + 1):
            user_prompt = self._build_user_prompt(state, feedback=feedback)
            try:
                base_decision = await self._invoke_llm(state.tick, system_prompt, user_prompt)
            except CostCeilingExceededError as exc:
                # Mid-call ceiling breach (a richer-than-predicted response).
                # Tracker is unchanged; we just have to drop this call.
                log.warning(
                    "%s tier ceiling breached mid-call tick=%d: %s",
                    self.TIER,
                    state.tick,
                    exc,
                )
                return None
            except Exception as exc:
                # Any other LLM exception (transport, schema, parse) — treat as
                # a bad attempt and either retry or bail.
                log.warning(
                    "%s tier LLM attempt %d/%d failed at tick=%d: %s",
                    self.TIER,
                    attempt + 1,
                    LLM_MAX_RETRIES + 1,
                    state.tick,
                    exc,
                )
                if attempt >= LLM_MAX_RETRIES:
                    return None
                feedback = (
                    f"Your previous response was malformed: {exc}. "
                    "Reply with VALID JSON only matching the schema. "
                    "No markdown fences, no preamble."
                )
                continue

            # 5. Validate role-lock invariant — `references_stance` must contain
            #    at least one real `CeoStance` field after filtering hallucinations.
            #    Filter FIRST, retry if the survivor list is empty, return None
            #    if the retry also yields nothing real.
            filtered_refs = _validate_references_stance(
                list(base_decision.references_stance)
            )
            if not filtered_refs:
                if attempt >= LLM_MAX_RETRIES:
                    log.warning(
                        "%s tier returned empty references_stance after filter+retry "
                        "at tick=%d (company=%s); raw refs were %r; returning None",
                        self.TIER,
                        state.tick,
                        self._company_id,
                        list(base_decision.references_stance),
                    )
                    return None
                valid_fields = ", ".join(sorted(_STANCE_FIELDS))
                feedback = (
                    "Your previous `references_stance` contained no recognized "
                    "stance attributes (after filtering hallucinated names). "
                    f"You MUST cite at least one of: {valid_fields}. "
                    "Reply with VALID JSON only matching the schema."
                )
                continue

            # 6. Build the tiered decision (adds tier + tick metadata). Use the
            #    FILTERED references list so hallucinated attribute names never
            #    reach the engine, transcript, or critic agent.
            decision = CeoDecision(
                spawn_nodes=list(base_decision.spawn_nodes),
                retire_nodes=list(base_decision.retire_nodes),
                adjust_params=dict(base_decision.adjust_params),
                open_locations=base_decision.open_locations,
                reasoning=base_decision.reasoning,
                references_stance=filtered_refs,
                tier=self.TIER,
                tick=state.tick,
            )

            # 7. Filter post-parse: drop unmet prereqs / hard-cap breaches
            #    in `spawn_nodes`, then bounds-check `adjust_params`. Append
            #    a note to reasoning so the next call can correct course.
            decision = self._filter_invalid_decision(decision, state)
            return decision

        return None

    # -- Internals ----------------------------------------------------------

    def _record_snapshot(self, state: CompanyState) -> None:
        """Append the current tick's state to the rolling window."""
        self._state_window.append(
            _StateSnapshot(
                tick=state.tick,
                cash=state.cash,
                monthly_revenue=state.monthly_revenue,
                capacity_utilization=state.capacity_utilization,
                avg_satisfaction=state.avg_satisfaction,
                employee_count=state.employee_count,
            )
        )

    def _build_system_prompt(self) -> str:
        """Stance role-lock + tier-specific instruction + JSON-only directive."""
        return (
            to_system_prompt(self.stance)
            + "\n\n"
            + self.DOMAIN_INSTRUCTION
            + "\n\n"
            + "Output strictly valid JSON matching CeoDecision schema. "
            + "No markdown fences, no preamble."
        )

    def _build_user_prompt(self, state: CompanyState, feedback: str | None = None) -> str:
        """Build the user prompt as a JSON string for canonical hashing.

        Sections (all keys deterministic for replay-hash stability):
          * `tick`
          * `state_window` (last STATE_WINDOW_LEN snapshots)
          * `current_state` (the present tick — what to act on)
          * `recent_shocks` (active shocks within SHOCK_LOOKBACK_TICKS)
          * `recent_decisions` (last DECISION_HISTORY_LEN at THIS tier)
          * `available_nodes` (library, filtered + summarized)
          * `budget` (cost-tracker totals)
          * `feedback` (only present on retry attempts)
        """
        payload: dict[str, Any] = {
            "tick": state.tick,
            "state_window": [snap.__dict__ for snap in self._state_window],
            "current_state": {
                "cash": state.cash,
                "daily_burn": state.daily_burn,
                "monthly_revenue": state.monthly_revenue,
                "capacity_utilization": state.capacity_utilization,
                "avg_satisfaction": state.avg_satisfaction,
                "employee_count": state.employee_count,
                "spawned_nodes": dict(state.spawned_nodes),
            },
            "recent_shocks": self._summarize_shocks(state),
            "recent_decisions": self._summarize_recent_decisions(state),
            "available_nodes": self._summarize_available_nodes(state),
            "budget": {
                "spent_usd": round(self.cost_tracker.total_cost(), 6),
                "ceiling_usd": self.cost_tracker.ceiling_usd,
                "remaining_usd": round(self.cost_tracker.remaining_budget(), 6),
            },
            "schema_hint": {
                "spawn_nodes": "list[str] — node_library keys to add",
                "retire_nodes": "list[str] — node_library keys to remove",
                "adjust_params": "dict[str, float] — param mutations",
                "open_locations": "int >= 0 — explicit count this period",
                "reasoning": "str — required, becomes part of decision history",
                "references_stance": (
                    "list[str] non-empty — stance attribute names that drove this decision"
                ),
            },
        }
        if feedback is not None:
            payload["feedback"] = feedback
        # Sorted keys + compact separators so the canonical hash is stable
        # regardless of dict-insertion order.
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _summarize_shocks(self, state: CompanyState) -> list[dict[str, Any]]:
        """Active shocks whose start tick is within `SHOCK_LOOKBACK_TICKS`."""
        out: list[dict[str, Any]] = []
        for shock in state.active_shocks:
            started = shock.tick_started if shock.tick_started is not None else state.tick
            if state.tick - started > SHOCK_LOOKBACK_TICKS:
                continue
            out.append(
                {
                    "name": shock.name,
                    "severity": shock.severity,
                    "tick_started": started,
                    "duration_ticks": shock.duration_ticks,
                    "impact": dict(shock.impact),
                    "description": shock.description,
                }
            )
        return out

    def _summarize_recent_decisions(self, state: CompanyState) -> list[dict[str, Any]]:
        """Last `DECISION_HISTORY_LEN` decisions at THIS tier, oldest-first.

        `state.recent_decisions` mixes tiers; we filter to our own so the LLM
        sees its own continuity, not the heuristic chatter.
        """
        same_tier = [d for d in state.recent_decisions if d.tier == self.TIER]
        return [
            {
                "tick": d.tick,
                "reasoning": d.reasoning,
                "spawn_nodes": list(d.spawn_nodes),
                "retire_nodes": list(d.retire_nodes),
                "adjust_params": dict(d.adjust_params),
                "open_locations": d.open_locations,
                "references_stance": list(d.references_stance),
            }
            for d in same_tier[-DECISION_HISTORY_LEN:]
        ]

    def _summarize_available_nodes(self, state: CompanyState) -> list[dict[str, Any]]:
        """Library nodes filtered to seed.economics_model, summarized for prompt.

        Includes current spawn count + hard_cap so the LLM can see what's
        already at the ceiling, plus prerequisites so it can sequence its
        spawns. Sorted by `(category, key)` for stable hashing.
        """
        out: list[dict[str, Any]] = []
        for key in sorted(self.library.nodes.keys()):
            node: NodeDef = self.library.nodes[key]
            if self.seed.economics_model not in node.applicable_economics:
                continue
            current = int(state.spawned_nodes.get(key, 0))
            out.append(
                {
                    "key": key,
                    "category": node.category,
                    "label": node.label,
                    "hire_cost": node.hire_cost,
                    "daily_fixed_costs": node.daily_fixed_costs,
                    "current_spawn_count": current,
                    "soft_cap": node.category_caps.soft_cap,
                    "hard_cap": node.category_caps.hard_cap,
                    "prerequisites": list(node.prerequisites),
                    "modifier_keys": dict(node.modifier_keys),
                }
            )
        return out

    async def _invoke_llm(
        self, tick: int, system_prompt: str, user_prompt: str
    ) -> _BaseCeoDecision:
        """Wrap the AsyncAnthropic call into the `replay_or_call` contract.

        The closure that does the actual SDK round-trip returns the 5-tuple
        `(decision, raw_response, model, input_tokens, output_tokens)`. Replay
        mode never reaches the closure — `replay_or_call` short-circuits to
        the recorded decision.

        `replay_or_call` returns the canonical `replay.CeoDecision`; we lift
        it to our subclass in the caller (`tick()`) by adding tier+tick.
        """
        client = self._client if self._client is not None else _get_client()

        async def _make_call(
            prompt: str,
        ) -> tuple[_BaseCeoDecision, str, str, int, int]:
            response = await client.messages.create(
                model=self.MODEL_ID,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            data = _parse_llm_json(raw)
            try:
                decision = _BaseCeoDecision(**data)
            except ValidationError as exc:
                raise ValueError(f"CeoDecision schema mismatch: {exc.errors()}") from exc
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0)) if usage else 0
            output_tokens = int(getattr(usage, "output_tokens", 0)) if usage else 0
            return decision, raw, self.MODEL_ID, input_tokens, output_tokens

        # `replay_or_call` is sync — it returns the decision directly. The
        # async LLM call is invoked synchronously inside the closure (which
        # is fine because we're already inside an async function and we do
        # NOT need to `await` the closure return — `replay_or_call` will
        # call it as a regular callable). To make this work, we run the
        # async closure via `asyncio.run` is wrong (we're already in a loop)
        # — instead we awaited the call directly above, so we wrap differently.
        #
        # Cleanest path: do the LLM round-trip BEFORE handing to replay_or_call,
        # using a sync stub that returns the already-fetched data. This keeps
        # `replay_or_call`'s sync contract intact while letting us await the
        # actual API call.
        #
        # In replay mode the closure is never invoked — we still need to call
        # replay_or_call to get the recorded decision. So we run a no-op
        # closure that errors loudly if invoked unexpectedly.

        # Disambiguate the replay-key by suffixing the tier. The replay
        # layer indexes entries on `(tick, company_id)` only — without this,
        # a tick that fires both tactical and strategic (e.g. 90 % 30 == 0
        # AND 90 % 90 == 0) would emit two entries with the same key and
        # fail to load in replay mode. Suffixing keeps the layers isolated.
        replay_company_id = f"{self._company_id}:{self.TIER}"

        if self.transcript.mode == "replay":
            # Replay path: never call the API.
            def _replay_stub(_p: str) -> tuple[_BaseCeoDecision, str, str, int, int]:
                raise AssertionError(
                    "LLM closure must not be invoked in replay mode "
                    "(replay_or_call short-circuits to recorded entry)"
                )

            base_decision = replay_or_call(
                self.transcript,
                user_prompt,
                _replay_stub,
                sim_id=self._sim_id,
                tick=tick,
                company_id=replay_company_id,
                tier=self._replay_tier(),
                cost_tracker=None,  # cost was paid on the recording run
            )
            return base_decision

        # record / off path: actually call the API.
        decision, raw, model, in_tok, out_tok = await _make_call(user_prompt)

        def _sync_replay_call(
            _p: str,
        ) -> tuple[_BaseCeoDecision, str, str, int, int]:
            return decision, raw, model, in_tok, out_tok

        base_decision = replay_or_call(
            self.transcript,
            user_prompt,
            _sync_replay_call,
            sim_id=self._sim_id,
            tick=tick,
            company_id=replay_company_id,
            tier=self._replay_tier(),
            cost_tracker=self.cost_tracker,
        )
        return base_decision

    def _replay_tier(self) -> Tier:
        """Map our `TIER` literal to the `replay.Tier` literal.

        `replay.Tier` uses `"operational"` for the heuristic; ours uses
        `"heuristic"`. For the LLM tiers the literal matches one-to-one.
        """
        # tactical → "tactical", strategic → "strategic" (same string)
        return self.TIER  # type: ignore[return-value]

    def _filter_invalid_decision(
        self, decision: CeoDecision, state: CompanyState
    ) -> CeoDecision:
        """Sanitize the decision before it leaves the orchestrator.

        Two passes:
          1. Drop spawns whose prerequisites aren't met or that breach
             ``hard_cap``.
          2. Filter / clamp ``adjust_params`` via ``_validate_adjust_params``
             (P-AI-2): out-of-bounds ``price`` is dropped, out-of-range
             ``marketing_intensity`` and ``replenish_supplier`` are clamped,
             ``raise_amount`` is gated by ``stance.archetype``, unknown keys
             are dropped.

        Returns the same decision instance when nothing is dropped/clamped.
        Otherwise returns a NEW ``CeoDecision`` with the surviving spawns,
        the validated ``adjust_params``, and a note appended to
        ``reasoning`` explaining what changed — so the CEO can re-plan on
        the next call.
        """
        kept: list[str] = []
        dropped: list[tuple[str, str]] = []
        # Track per-key "would-be" spawn counts so two requests for the same
        # key in one decision don't both pass the hard_cap test.
        in_flight: dict[str, int] = {}
        spawned_set: set[str] = {k for k, c in state.spawned_nodes.items() if c > 0}

        for key in decision.spawn_nodes:
            node = self.library.get(key)
            if node is None:
                dropped.append((key, "unknown node key"))
                continue
            if not self.library.prerequisites_satisfied(spawned_set, key):
                missing = [p for p in node.prerequisites if p not in spawned_set]
                dropped.append((key, f"prerequisites not satisfied: {missing}"))
                continue
            current = int(state.spawned_nodes.get(key, 0)) + in_flight.get(key, 0)
            if current >= node.category_caps.hard_cap:
                dropped.append(
                    (
                        key,
                        f"hard_cap reached ({current}/{node.category_caps.hard_cap})",
                    )
                )
                continue
            kept.append(key)
            in_flight[key] = in_flight.get(key, 0) + 1

        # P-AI-2: bounds-check every adjust_params key. Always runs (even if
        # the spawn list is fully accepted) so the CEO can't sneak an
        # unbounded float past the orchestrator.
        validated_params = self._validate_adjust_params(
            dict(decision.adjust_params), self.seed, self.stance
        )
        params_changed = validated_params != decision.adjust_params

        if not dropped and not params_changed:
            return decision

        reasoning = decision.reasoning
        if dropped:
            notes = "; ".join(f"{k} ({why})" for k, why in dropped)
            reasoning = (
                f"{reasoning} "
                f"[orchestrator note: dropped invalid spawns — {notes}. "
                "Adjust on next tick.]"
            )
        if params_changed:
            reasoning = (
                f"{reasoning} "
                "[orchestrator note: adjust_params filtered/clamped to "
                "stay within engine bounds. See logs for per-key detail.]"
            )

        return CeoDecision(
            spawn_nodes=kept,
            retire_nodes=list(decision.retire_nodes),
            adjust_params=validated_params,
            open_locations=decision.open_locations,
            reasoning=reasoning,
            references_stance=list(decision.references_stance),
            tier=decision.tier,
            tick=decision.tick,
        )

    def _validate_adjust_params(
        self,
        params: dict[str, float],
        seed: CompanySeed,
        stance: CeoStance,
    ) -> dict[str, float]:
        """Bounds-check every CEO-proposed ``adjust_params`` key (P-AI-2).

        Returns a NEW dict containing only the keys that survived
        validation. Does NOT mutate ``params``. Per-key behavior:

          * ``price`` — must be in
            ``[seed.starting_price * PRICE_LOWER_BOUND_MULT,
              seed.starting_price * PRICE_UPPER_BOUND_MULT]``.
            Out-of-range or non-finite → dropped (logged).
          * ``marketing_intensity`` — clamped into
            ``[MARKETING_INTENSITY_LOWER, MARKETING_INTENSITY_UPPER]``.
            Logged when clamping changes the value.
          * ``raise_amount`` — must be ``>= 0`` AND
            ``stance.archetype in RAISE_ALLOWED_STANCE_ARCHETYPES``.
            Either condition violated → dropped (logged).
          * ``replenish_supplier`` — clamped into
            ``[REPLENISH_SUPPLIER_LOWER, REPLENISH_SUPPLIER_UPPER]``.
            Logged when clamping changes the value.

        Any key not in ``KNOWN_ADJUST_PARAMS_KEYS`` is dropped with a
        warning — the orchestrator refuses to forward inputs it cannot
        bound.
        """
        result: dict[str, float] = {}

        for key, raw_value in params.items():
            if key not in KNOWN_ADJUST_PARAMS_KEYS:
                log.warning(
                    "%s tier: dropping unknown adjust_params key %r=%r "
                    "(company=%s, archetype=%s)",
                    self.TIER, key, raw_value, self._company_id, stance.archetype,
                )
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                log.warning(
                    "%s tier: dropping adjust_params[%s]=%r (not coercible to float) "
                    "(company=%s)",
                    self.TIER, key, raw_value, self._company_id,
                )
                continue

            # Reject non-finite values up-front for every key — NaN/Inf
            # can't be clamped meaningfully and indicate a model malfunction.
            if value != value or value in (float("inf"), float("-inf")):
                log.warning(
                    "%s tier: dropping adjust_params[%s]=%r (non-finite) "
                    "(company=%s)",
                    self.TIER, key, value, self._company_id,
                )
                continue

            if key == "price":
                lo = seed.starting_price * PRICE_LOWER_BOUND_MULT
                hi = seed.starting_price * PRICE_UPPER_BOUND_MULT
                if value < lo or value > hi:
                    log.warning(
                        "%s tier: dropping price=%.4f — outside "
                        "[%.4f, %.4f] (company=%s, starting_price=%.4f)",
                        self.TIER, value, lo, hi,
                        self._company_id, seed.starting_price,
                    )
                    continue
                result[key] = value

            elif key == "marketing_intensity":
                clamped = max(
                    MARKETING_INTENSITY_LOWER,
                    min(MARKETING_INTENSITY_UPPER, value),
                )
                if clamped != value:
                    log.warning(
                        "%s tier: clamped marketing_intensity %.4f -> %.4f "
                        "(allowed [%.2f, %.2f], company=%s)",
                        self.TIER, value, clamped,
                        MARKETING_INTENSITY_LOWER, MARKETING_INTENSITY_UPPER,
                        self._company_id,
                    )
                result[key] = clamped

            elif key == "raise_amount":
                if value < 0:
                    log.warning(
                        "%s tier: dropping raise_amount=%.2f (must be >= 0) "
                        "(company=%s)",
                        self.TIER, value, self._company_id,
                    )
                    continue
                if stance.archetype not in RAISE_ALLOWED_STANCE_ARCHETYPES:
                    log.warning(
                        "%s tier: dropping raise_amount=%.2f — stance "
                        "archetype %r does not permit external capital raises "
                        "(company=%s, allowed=%s)",
                        self.TIER, value, stance.archetype, self._company_id,
                        sorted(RAISE_ALLOWED_STANCE_ARCHETYPES),
                    )
                    continue
                result[key] = value

            elif key == "replenish_supplier":
                clamped = max(
                    REPLENISH_SUPPLIER_LOWER,
                    min(REPLENISH_SUPPLIER_UPPER, value),
                )
                if clamped != value:
                    log.warning(
                        "%s tier: clamped replenish_supplier %.4f -> %.4f "
                        "(allowed [%.2f, %.2f], company=%s)",
                        self.TIER, value, clamped,
                        REPLENISH_SUPPLIER_LOWER, REPLENISH_SUPPLIER_UPPER,
                        self._company_id,
                    )
                result[key] = clamped

        return result


def _validate_references_stance(refs: list[str]) -> list[str]:
    """Filter `refs` down to keys that exist on `CeoStance`.

    LLMs occasionally hallucinate stance attributes ("aggressiveness",
    "founder_intuition") that have no engine meaning. Anything not in
    `_STANCE_FIELDS` is dropped — order of survivors is preserved so the
    transcript stays stable.
    """
    return [r for r in refs if r in _STANCE_FIELDS]


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a possibly-fenced / preambled LLM response.

    Mirrors `ceo_agent._parse_json_response` — but lives here so the LLM
    tiers don't depend on the v1 ceo_agent module beyond `_get_client`.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences anywhere in the response.
    import re

    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1).strip())
    # Last resort: outermost { ... }.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object in response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Concrete tiers
# ---------------------------------------------------------------------------


class TacticalOrchestrator(_LlmOrchestratorBase):
    """Haiku 4.5 — mid-cadence tactical decisions (every 30 ticks).

    Owns price tweaks, marketing pace, and mid-level hires (BD reps, ops
    leads, support engineers, junior delivery staff). Strategic moves
    (exec hires, expansion, raises) are NOT this tier's job — escalates
    to strategic by abstaining.
    """

    TIER = "tactical"
    MODEL_ID = TACTICAL_MODEL_ID
    CADENCE_TICKS = TACTICAL_CADENCE_TICKS
    PREDICTED_INPUT_TOKENS = TACTICAL_PREDICTED_INPUT_TOKENS
    PREDICTED_OUTPUT_TOKENS = TACTICAL_PREDICTED_OUTPUT_TOKENS
    MAX_TOKENS = TACTICAL_MAX_TOKENS
    DOMAIN_INSTRUCTION = (
        "TACTICAL TIER — you fire every 30 ticks. Your authority covers:\n"
        " - Price tweaks (set `adjust_params['price']` if changing).\n"
        " - Marketing intensity / pace "
        "(set `adjust_params['marketing_intensity']` 0..1).\n"
        " - Mid-level hires: BD reps, operations leads, support, junior "
        "delivery roles (`spawn_nodes`).\n"
        "DO NOT propose: executive hires, location expansion, capital raises, "
        "layoffs > 5%, vertical entry, or partnerships — those are the "
        "strategic tier's domain. If the situation calls for one of those, "
        "return a no-op decision (empty spawn/retire) and explain in your "
        "reasoning that you are deferring to the strategic tier."
    )


class StrategicOrchestrator(_LlmOrchestratorBase):
    """Sonnet 4.6 — slow-cadence strategic decisions (every 90 ticks or on shock).

    Owns exec hires (CFO, COO, VPs), location/territory expansion, capital
    raises, layoffs > 5%, vertical entry, and partnerships. Wakes immediately
    on any active severe shock via `force_wake=True` (the bundle wires this).
    """

    TIER = "strategic"
    MODEL_ID = STRATEGIC_MODEL_ID
    CADENCE_TICKS = STRATEGIC_CADENCE_TICKS
    PREDICTED_INPUT_TOKENS = STRATEGIC_PREDICTED_INPUT_TOKENS
    PREDICTED_OUTPUT_TOKENS = STRATEGIC_PREDICTED_OUTPUT_TOKENS
    MAX_TOKENS = STRATEGIC_MAX_TOKENS
    DOMAIN_INSTRUCTION = (
        "STRATEGIC TIER — you fire every 90 ticks AND on every severe shock. "
        "Your authority covers:\n"
        " - Executive hires (CFO, COO, VPs, vertical leads).\n"
        " - Location / territory expansion (`open_locations` count this period).\n"
        " - Capital raises (set `adjust_params['raise_amount']` if raising).\n"
        " - Layoffs > 5% of headcount (`retire_nodes`).\n"
        " - Vertical entry, partnerships, and other multi-quarter bets.\n"
        "Tactical noise (small price tweaks, marketing pace, mid-level hires) "
        "is the tactical tier's job — focus on multi-month strategy. If a "
        "severe shock is active, your reasoning MUST address how your "
        "decision responds to it."
    )


# ---------------------------------------------------------------------------
# OrchestratorBundle — routes one tick through all three tiers in priority order.
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorBundle:
    """Owns one of each tier and routes ticks to the correct cadences.

    Ordering on a single tick is deterministic and matches the spec:
      1. Heuristic (every 7 ticks) — if it fires, runs first.
      2. Tactical (every 30 ticks) — if it fires, runs second.
      3. Strategic (every 90 ticks OR severe-shock interrupt) — runs last.
      4. Critic (Phase 2.3, Decision 3A) — if a strategic decision fired
         AND `critic` + `stance` are wired, score it and append to the
         per-tick `critic_scores` list. Critic NEVER blocks the decision —
         below-threshold scores are logged but the decision still applies.

    Returns the tuple `(decisions, critic_scores)`. The decisions list has
    0-3 entries; `critic_scores` has 0-1 entries (one per strategic-tier
    decision in the same tick). Order matters for `decisions` — the engine
    should apply heuristic first (fast operational fixes), then tactical,
    then strategic, so each higher tier sees the lower tiers' state
    effects on the next tick.

    Severe-shock interrupt: any active shock with `severity == "severe"`
    forces a strategic-tier wake regardless of cadence. The interrupt does
    NOT also wake tactical or heuristic — those wait for their own cadences.

    Backward-compat: `critic`, `stance`, and `company_id` default to
    `None`/empty so existing test fixtures that construct
    `OrchestratorBundle(heuristic=..., tactical=..., strategic=...)` keep
    working. The critic only fires when `critic` AND `stance` are both
    non-None — this is intentional, so the engine can opt out cleanly by
    leaving `critic=None`.
    """

    heuristic: HeuristicOrchestrator
    tactical: TacticalOrchestrator
    strategic: StrategicOrchestrator
    critic: CriticAgent | None = None
    stance: CeoStance | None = None
    company_id: str = ""

    async def tick(
        self, state: CompanyState
    ) -> tuple[list[CeoDecision], list[CriticScore]]:
        """Run one tick. Returns `(decisions, critic_scores)`."""
        decisions: list[CeoDecision] = []
        critic_scores: list[CriticScore] = []

        # 1. Heuristic — sync, no LLM.
        h = self.heuristic.tick(state)
        if h is not None:
            decisions.append(h)

        # 2. Tactical — async, may skip on cadence/budget/persistent failure.
        t = await self.tactical.tick(state)
        if t is not None:
            decisions.append(t)

        # 3. Strategic — async; force_wake on any severe shock active right now.
        force_wake = any(shock.severity == "severe" for shock in state.active_shocks)
        s = await self.strategic.tick(state, force_wake=force_wake)
        if s is not None:
            decisions.append(s)

        # 4. Critic — Haiku-scored stance alignment for the strategic
        #    decision (if any). Per Decision 3A: never blocks, only logs.
        if s is not None and self.critic is not None and self.stance is not None:
            score = await self.critic.score(
                s, self.stance, state, company_id=self.company_id
            )
            if score is not None:
                critic_scores.append(score)
                if score.score < CRITIC_VIOLATION_THRESHOLD:
                    log.warning(
                        "Stance violation tick=%d company=%s score=%.2f "
                        "violations=%s",
                        state.tick,
                        self.company_id,
                        score.score,
                        score.violations,
                    )

        return decisions, critic_scores


__all__ = [
    "CAPACITY_HIGH_THRESHOLD",
    "CAPACITY_LOW_THRESHOLD",
    "DAYS_PER_MONTH",
    "DECISION_HISTORY_LEN",
    "HEURISTIC_CADENCE_TICKS",
    "HIGH_PERSIST_TICKS",
    "LAYOFF_MIN_RUNWAY_MONTHS",
    "LLM_MAX_RETRIES",
    "LOW_PERSIST_TICKS",
    "NO_CAP_SENTINEL",
    "REPLENISH_CADENCE_TICKS",
    "SHOCK_LOOKBACK_TICKS",
    "STATE_WINDOW_LEN",
    "STRATEGIC_CADENCE_TICKS",
    "STRATEGIC_MAX_TOKENS",
    "STRATEGIC_MODEL_ID",
    "STRATEGIC_PREDICTED_INPUT_TOKENS",
    "STRATEGIC_PREDICTED_OUTPUT_TOKENS",
    "TACTICAL_CADENCE_TICKS",
    "TACTICAL_MAX_TOKENS",
    "TACTICAL_MODEL_ID",
    "TACTICAL_PREDICTED_INPUT_TOKENS",
    "TACTICAL_PREDICTED_OUTPUT_TOKENS",
    "UTILIZATION_HISTORY_LEN",
    "CeoDecision",
    "CompanyState",
    "HeuristicOrchestrator",
    "OrchestratorBundle",
    "StrategicOrchestrator",
    "TacticalOrchestrator",
]
