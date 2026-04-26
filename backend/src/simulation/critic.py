"""Stance-alignment critic — Haiku-scored validation of strategic decisions.

Per V2 PRD Appendix B (Decision 3A — LOCKED) and the V2 remediation
checklist task 2.3, the critic agent inspects every STRATEGIC-tier
`CeoDecision` immediately after it leaves the `StrategicOrchestrator` and
scores it against the locked `CeoStance`. The critic exists to surface
stance drift the LLM tier might silently introduce — it does NOT block
decisions, it only logs them. (Blocking would defeat the "emergence over
scripting" principle in `CLAUDE.md` — a controversial decision that
violates the stance is exactly the kind of behaviour the simulation is
designed to expose, not suppress.)

Design constraints (do not relax without owner sign-off):
  * Haiku-only. Strategic decisions already cost Sonnet — adding another
    Sonnet pass per strategic tick would double the per-call cost without
    meaningful gain. Haiku is sufficient to detect "this decision
    contradicts the stance" because the stance is locked + small.
  * Returns `None` silently when `cost_tracker.would_exceed(...)` is
    true. The critic is best-effort telemetry — never the reason a sim
    aborts. Silence is the correct failure mode here.
  * Replay-mode reads from the transcript via
    `Transcript.lookup_critic(...)`. Replay never hits the API.
  * Tactical and heuristic decisions are NOT scored — `OrchestratorBundle`
    only invokes the critic on the strategic decision (if any) per tick.

Public surface:
  * `CRITIC_MODEL_ID` — pinned Haiku 4.5 model id (matches MODEL_PRICING).
  * `CRITIC_VIOLATION_THRESHOLD` — score below which we log a warning.
  * `CriticScore` — pydantic model: `{score, violations, reasoning}`.
  * `CriticAgent` — async `score(decision, stance, state) -> CriticScore | None`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.simulation.orchestrator import CeoDecision, CompanyState
from src.simulation.replay import (
    MissingTranscriptEntryError,
    Transcript,
    CostTracker,
)
from src.simulation.stance import CeoStance, to_system_prompt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — every numeric / model-id constant this module uses.
# ---------------------------------------------------------------------------

#: Pinned Anthropic model id for the critic. Haiku 4.5 — keep aligned with
#: `replay.MODEL_PRICING` (the cost-tracker requires the model id be a
#: known pricing key).
CRITIC_MODEL_ID: str = "claude-haiku-4-5"

#: Below this score the critic considers the decision a "violation" of the
#: stance and emits a warning log line. The decision STILL applies — this
#: threshold gates logging severity only, never blocking. 0.6 puts the
#: boundary above "ambiguous" (0.5) but below "aligned" (≥0.7).
CRITIC_VIOLATION_THRESHOLD: float = 0.6

#: Cost-tracker preflight envelope. Same shape as the orchestrator's
#: PREDICTED_INPUT/OUTPUT_TOKENS — conservative upper bounds for the
#: would_exceed check. The actual call usually consumes far less.
CRITIC_PREDICTED_INPUT_TOKENS: int = 2000
CRITIC_PREDICTED_OUTPUT_TOKENS: int = 300

#: Hard cap on `max_tokens` for the LLM call itself. Larger than the
#: predicted output so a verbose `reasoning` field never gets truncated.
CRITIC_MAX_TOKENS: int = 512


# ---------------------------------------------------------------------------
# CriticScore — public contract
# ---------------------------------------------------------------------------


class CriticScore(BaseModel):
    """Stance-alignment score for one strategic decision.

    `score`: 1.0 = perfectly stance-aligned; 0.5 = ambiguous; 0.0 = direct
    contradiction. The critic is asked to populate `violations` with short
    strings naming the specific stance attribute(s) the decision pulls
    against (`["risk_tolerance", "cash_comfort"]`) when score < 1.0;
    `reasoning` is a one-paragraph explanation of the score for audit logs
    and the optional frontend display.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------


def _get_client() -> AsyncAnthropic:
    """Build an AsyncAnthropic client from env credentials. Mirrors orchestrator."""
    return AsyncAnthropic()  # reads ANTHROPIC_API_KEY


class CriticAgent:
    """Haiku-scored stance-alignment critic for strategic decisions.

    Construct one per company (or per simulation if `transcript`+`cost_tracker`
    are sim-scoped). Owns no mutable state of its own — `score()` is a pure
    function of `(decision, stance, state)` plus the shared transcript/tracker.
    """

    def __init__(
        self,
        *,
        transcript: Transcript | None = None,
        cost_tracker: CostTracker | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Construct a critic. All three deps optional for test convenience.

        Args:
            transcript: Shared per-sim transcript. If `mode == "replay"` the
                critic reads scores from disk; if `mode == "record"` it
                writes them. `mode == "off"` or `None` skips persistence.
            cost_tracker: Shared per-sim cost ceiling. When `would_exceed`
                returns True the critic returns `None` silently.
            client: Optional injected `AsyncAnthropic` for tests. Lazy in
                production.
        """
        self.transcript: Transcript | None = transcript
        self.cost_tracker: CostTracker | None = cost_tracker
        self._client: AsyncAnthropic | None = client

    # ── Public API ───────────────────────────────────────────────────────

    async def score(
        self,
        decision: CeoDecision,
        stance: CeoStance,
        state: CompanyState,
        *,
        company_id: str = "",
    ) -> CriticScore | None:
        """Score a strategic decision against the stance.

        Returns `None` when:
          * `cost_tracker.would_exceed(...)` is true (silent skip — never
            blocks the decision the critic is judging).

        Returns the parsed `CriticScore` otherwise. In replay mode the
        score comes from the transcript verbatim (raises
        `MissingTranscriptEntryError` if absent — the recording run was
        incomplete, which is a bug we want loud, not silent).

        `company_id` is needed to key the transcript entry. Pass the same
        value the bundle / engine uses for this company; defaults to empty
        string for tests that aren't exercising the persistence path.
        """
        # 1. Cost preflight — silent skip if budget exhausted.
        if self.cost_tracker is not None and self.cost_tracker.would_exceed(
            CRITIC_PREDICTED_INPUT_TOKENS,
            CRITIC_PREDICTED_OUTPUT_TOKENS,
            CRITIC_MODEL_ID,
        ):
            log.debug(
                "Critic skipping tick=%d (company=%s): would exceed cost ceiling",
                state.tick,
                company_id,
            )
            return None

        # 2. Replay-mode short-circuit.
        if self.transcript is not None and self.transcript.mode == "replay":
            recorded = self.transcript.lookup_critic(state.tick, company_id)
            if recorded is None:
                raise MissingTranscriptEntryError(
                    f"no recorded critic score for tick={state.tick} "
                    f"company_id={company_id!r} in {self.transcript.path} "
                    "— state has diverged from the recording, or the critic "
                    "is being invoked at a tick where the recording run did "
                    "not run it."
                )
            return recorded

        # 3. Real Haiku call.
        client = self._client if self._client is not None else _get_client()
        system_prompt = self._build_system_prompt(stance)
        user_prompt = self._build_user_prompt(decision, state)

        response = await client.messages.create(
            model=CRITIC_MODEL_ID,
            max_tokens=CRITIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        parsed = self._parse_score(raw)

        # 4. Charge the cost tracker (mirrors the orchestrator pattern —
        #    we pay full price even on parse failure since the tokens were
        #    consumed).
        if self.cost_tracker is not None:
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0)) if usage else 0
            output_tokens = int(getattr(usage, "output_tokens", 0)) if usage else 0
            # `record` may raise `CostCeilingExceededError` if the actual
            # usage was much larger than predicted. Same contract as the
            # orchestrator: we don't swallow it — the next critic call
            # will preflight-skip naturally.
            try:
                self.cost_tracker.record(input_tokens, output_tokens, CRITIC_MODEL_ID)
            except Exception as exc:  # noqa: BLE001 — log+continue; critic is best-effort
                log.warning(
                    "Critic cost-record failed at tick=%d (company=%s): %s",
                    state.tick,
                    company_id,
                    exc,
                )

        # 5. Persist to transcript so replay reproduces.
        if self.transcript is not None and self.transcript.mode == "record":
            self.transcript.record_critic(state.tick, company_id, parsed)

        return parsed

    # ── Prompt builders ──────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(stance: CeoStance) -> str:
        """System prompt — stance role-lock + critic rubric + JSON shape."""
        return (
            to_system_prompt(stance)
            + "\n\n"
            + (
                "ROLE OVERRIDE: For THIS call only, you are NOT the CEO. "
                "You are a STANCE-ALIGNMENT CRITIC reviewing a decision the "
                "CEO above just proposed. Your job is to score how well "
                "that decision aligns with the locked stance described.\n\n"
                "Scoring rubric (0.0 – 1.0):\n"
                "  1.0 — perfectly aligned; the decision is exactly what "
                "this stance would produce.\n"
                "  0.7 – 0.9 — broadly aligned; minor friction with one or "
                "two stance attributes.\n"
                "  0.5 — ambiguous; the decision could be argued either way.\n"
                "  0.2 – 0.4 — misaligned; the decision pulls against "
                "multiple stance attributes.\n"
                "  0.0 — direct contradiction of the stance.\n\n"
                "Populate `violations` with short stance-attribute names "
                "(e.g. \"risk_tolerance\", \"cash_comfort\") that the "
                "decision contradicts. Empty list when score == 1.0.\n\n"
                "Return STRICTLY valid JSON matching this schema:\n"
                "  {\"score\": float in [0,1], "
                "\"violations\": list[str], "
                "\"reasoning\": str}\n"
                "No markdown fences. No preamble. JSON only."
            )
        )

    @staticmethod
    def _build_user_prompt(decision: CeoDecision, state: CompanyState) -> str:
        """User prompt — serialize the decision + the state context the CEO had."""
        payload: dict[str, Any] = {
            "tick": state.tick,
            "decision": {
                "tier": decision.tier,
                "spawn_nodes": list(decision.spawn_nodes),
                "retire_nodes": list(decision.retire_nodes),
                "adjust_params": dict(decision.adjust_params),
                "open_locations": decision.open_locations,
                "reasoning": decision.reasoning,
                "references_stance": list(decision.references_stance),
            },
            "context": {
                "cash": state.cash,
                "daily_burn": state.daily_burn,
                "monthly_revenue": state.monthly_revenue,
                "capacity_utilization": state.capacity_utilization,
                "avg_satisfaction": state.avg_satisfaction,
                "employee_count": state.employee_count,
                "spawned_nodes": dict(state.spawned_nodes),
                "active_shocks": [
                    {
                        "name": s.name,
                        "severity": s.severity,
                        "duration_ticks": s.duration_ticks,
                    }
                    for s in state.active_shocks
                ],
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # ── Parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_score(raw: str) -> CriticScore:
        """Extract a `CriticScore` from a possibly-fenced LLM response.

        Mirrors `orchestrator._parse_llm_json` so the critic stays robust
        to the same Haiku-shaped failure modes (markdown fences,
        preambles, trailing prose). Falls back to outermost { ... } as
        last resort.
        """
        text = raw.strip()
        if not text:
            raise ValueError("Empty critic LLM response")
        try:
            return CriticScore.model_validate_json(text)
        except (ValidationError, ValueError):
            pass
        # Strip markdown fences anywhere in the response.
        import re

        fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence:
            return CriticScore.model_validate_json(fence.group(1).strip())
        # Last resort: outermost { ... }.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return CriticScore.model_validate_json(text[start : end + 1])
        raise ValueError(f"No JSON object in critic response: {text[:200]!r}")


__all__ = [
    "CRITIC_MAX_TOKENS",
    "CRITIC_MODEL_ID",
    "CRITIC_PREDICTED_INPUT_TOKENS",
    "CRITIC_PREDICTED_OUTPUT_TOKENS",
    "CRITIC_VIOLATION_THRESHOLD",
    "CriticAgent",
    "CriticScore",
]
