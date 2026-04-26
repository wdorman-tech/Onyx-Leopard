"""Short Q&A interview that produces a `CompanySeed` + `CeoStance` pair.

Replaces the v1 `profile_builder.py` (819 LOC, generated per-industry YAMLs)
with a much shorter conversation: 8-10 questions covering identity, economics,
market, starting position and CEO personality. The output is two pydantic
models (`CompanySeed`, `CeoStance`) — never YAML.

Two entry points:

* `SeedInterview` — stateful Q&A loop. Sessions are kept in-memory on the
  instance (no persistence; this is a local-only platform). Used by the
  adaptive setup UI.
* `seed_from_archetype()` — single-call helper for Monte Carlo bulk runs.
  Skips the LLM entirely; samples from the wave-1 archetype profiles.

Async because the underlying Anthropic SDK call is async (mirrors the
pattern in `profile_builder.py` / `ceo_agent.py`). Tests mock
`client.messages.create` directly.
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

from src.simulation.seed import ARCHETYPES as SEED_ARCHETYPES


def _get_client() -> anthropic.AsyncAnthropic:
    """Build an AsyncAnthropic client from env credentials."""
    return anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env


from src.simulation.seed import CompanySeed, sample_seed_for_archetype
from src.simulation.stance import ARCHETYPES as STANCE_ARCHETYPES
from src.simulation.stance import CeoStance, sample_stance

log = logging.getLogger(__name__)

# Model id pinned per CLAUDE.md guidance ("Use the latest model IDs"). The
# orchestrator/CEO modules already standardise on `claude-sonnet-4-6`.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Max number of additional follow-up questions allowed when Claude returns a
# final payload that fails seed/stance validation. After this many retries
# we give up and raise — the conversation is unrecoverable.
MAX_VALIDATION_RETRIES = 3

# Soft cap on conversational turns before we instruct Claude to wrap up. The
# spec calls for "8-10 questions max"; we let Claude self-terminate but stop
# the user from looping forever if the model keeps asking.
MAX_INTERVIEW_TURNS = 12

SessionId = str


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────


# Inception-prompt-style instructions. Schema fields are listed so Claude
# knows what it's collecting; the final-payload format is anchored with a
# verbatim JSON skeleton.
_SYSTEM_PROMPT = """You are a business analyst conducting a short interview \
to seed a business simulation.

Your job is to ask the founder ENOUGH questions to populate a `CompanySeed`
(~30 fields describing the company at t=0) and a `CeoStance` (~10 fields
describing the CEO's locked persona). You do NOT need to ask about every
field — infer reasonable defaults from the founder's earlier answers.

Topics you must cover, in roughly this order:
1. Company name + niche + archetype (solo_founder | small_team | venture_funded | enterprise)
2. Economics model (physical | subscription | service) + base price + base unit cost + capacity
3. Market: TAM rough order of magnitude, competitor density (0=greenfield .. 10=saturated)
4. Starting position: cash on hand, headcount, runway months, locations
5. CEO personality: risk tolerance (0..1), growth obsession (0..1), time horizon
   (quarterly | annual | decade), what stance archetype best fits
   (founder_operator | venture_growth | bootstrap | consolidator | turnaround)

RULES:
- Ask ONE focused question at a time. Keep questions short (1-2 sentences).
- Be conversational but efficient. Aim for 8-10 questions total. Never exceed 12.
- Adapt follow-ups based on prior answers. If a previous answer implied a value
  for a later field, do not re-ask it.
- For the FIRST message, briefly introduce yourself and ask what the company does.
- When you have enough information, output ONLY a JSON object on its own line
  with the prefix `FINAL_JSON:` followed by the payload. No preamble, no markdown.
- The payload has exactly two keys, `seed` and `stance`, each containing the
  full set of fields for `CompanySeed` and `CeoStance` respectively. Use snake_case
  field names. Lists must be JSON arrays. Numbers must be plain numbers (no $ or %).

FINAL JSON SHAPE (this is the literal format expected — fill every field):

FINAL_JSON: {
  "seed": {
    "name": "...",
    "niche": "...",
    "archetype": "small_team",
    "industry_keywords": ["...", "..."],
    "location_label": "Location",
    "economics_model": "physical",
    "base_price": 14.0,
    "base_unit_cost": 4.0,
    "daily_fixed_costs": 300.0,
    "starting_cash": 50000.0,
    "starting_employees": 5,
    "base_capacity_per_location": 80,
    "margin_target": 0.55,
    "revenue_per_employee_target": 200000.0,
    "tam": 100000000.0,
    "competitor_density": 4,
    "market_growth_rate": 0.05,
    "customer_unit_label": "diners",
    "seasonality_amplitude": 0.15,
    "initial_supplier_types": ["primary_goods_supplier"],
    "initial_revenue_streams": ["storefront_sales"],
    "initial_cost_centers": ["cogs", "labor"],
    "initial_locations": 1,
    "initial_marketing_intensity": 0.3,
    "initial_quality_target": 0.7,
    "initial_price_position": "mid",
    "initial_capital_runway_months": 12.0,
    "initial_hiring_pace": "steady",
    "initial_geographic_scope": "local",
    "initial_revenue_concentration": 0.4,
    "initial_customer_acquisition_channel": "word_of_mouth"
  },
  "stance": {
    "archetype": "founder_operator",
    "risk_tolerance": 0.55,
    "growth_obsession": 0.5,
    "quality_floor": 0.7,
    "hiring_bias": "balanced",
    "time_horizon": "annual",
    "cash_comfort": 9.0,
    "signature_moves": ["stay close to the customer", "hire only when it hurts"],
    "voice": "I started this thing in my garage and I am still the one taking the calls."
  }
}

If you receive a SYSTEM RETRY message describing a validation error, do NOT
emit another question. Re-emit the FINAL_JSON payload with the error fixed."""


# ─────────────────────────────────────────────────────────────────────────────
# Session + response models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InterviewSession:
    """Per-conversation state tracked by `SeedInterview`.

    Lives in memory only. `messages` is the running Anthropic-format chat
    history (alternating user/assistant). `seed_partial` and `stance_partial`
    are populated incrementally as Claude commits to values mid-interview
    (currently used only for diagnostic logging; the authoritative payload
    is the final FINAL_JSON emission).
    """

    id: SessionId
    user_description: str
    messages: list[dict] = field(default_factory=list)
    seed_partial: dict = field(default_factory=dict)
    stance_partial: dict = field(default_factory=dict)
    complete: bool = False
    errors: list[str] = field(default_factory=list)
    retries_used: int = 0
    turn_count: int = 0


class InterviewResponse(BaseModel):
    """Returned by `start_interview` and `submit_answer`.

    Either `question` carries the next prompt for the user, or `complete`
    is `True` and `seed`/`stance` carry the validated final models.
    """

    session_id: SessionId
    question: str | None = None
    complete: bool = False
    seed: CompanySeed | None = None
    stance: CeoStance | None = None
    error: str | None = Field(
        default=None,
        description="Populated when validation fails on the latest LLM payload.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_FINAL_JSON_RE = re.compile(r"FINAL_JSON\s*:\s*(\{.*\})\s*$", re.DOTALL)


def _extract_final_json(text: str) -> dict | None:
    """Pull the JSON object out of a `FINAL_JSON: {...}` reply.

    Returns `None` if the marker isn't present (Claude is still asking
    questions). Raises `ValueError` if the marker is present but the
    payload doesn't parse — that's an LLM bug we want to surface as a retry.
    """
    text = text.strip()
    # Strip markdown code fences if Claude wrapped the payload despite instructions.
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    if "FINAL_JSON" not in text:
        return None

    match = _FINAL_JSON_RE.search(text)
    if match:
        candidate = match.group(1)
    else:
        # Marker present but regex didn't match cleanly — fall back to first
        # `{` after the marker through last `}` in the string.
        marker_idx = text.find("FINAL_JSON")
        start = text.find("{", marker_idx)
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(
                f"FINAL_JSON marker present but no JSON object found: {text[:200]!r}"
            )
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"FINAL_JSON payload did not parse as JSON: {exc}") from exc


def _validate_payload(payload: dict) -> tuple[CompanySeed, CeoStance]:
    """Validate a raw FINAL_JSON dict against `CompanySeed` + `CeoStance`.

    Raises `ValueError` with an LLM-friendly description on failure so the
    retry loop can pass the message back to Claude.
    """
    if not isinstance(payload, dict):
        raise ValueError("FINAL_JSON payload must be a JSON object")
    if "seed" not in payload or "stance" not in payload:
        raise ValueError("FINAL_JSON must contain both 'seed' and 'stance' keys")

    try:
        seed = CompanySeed(**payload["seed"])
    except ValidationError as exc:
        raise ValueError(f"`seed` failed validation: {exc.errors()}") from exc
    try:
        stance = CeoStance(**payload["stance"])
    except ValidationError as exc:
        raise ValueError(f"`stance` failed validation: {exc.errors()}") from exc
    return seed, stance


# ─────────────────────────────────────────────────────────────────────────────
# SeedInterview
# ─────────────────────────────────────────────────────────────────────────────


class SeedInterview:
    """Stateful 8-10 question interview that yields a `CompanySeed` + `CeoStance`.

    No LLM calls happen at construction time — only inside `submit_answer`
    and `start_interview`. This lets tests instantiate the class freely and
    inject a mocked `anthropic.AsyncAnthropic` client.
    """

    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client  # lazy: real client built on first use if None
        self._model = model
        self._max_tokens = max_tokens
        self._sessions: dict[SessionId, InterviewSession] = {}

    # ── Public API ──

    async def start_interview(self, user_description: str) -> InterviewResponse:
        """Open a new session and produce the first reply.

        Normally returns a question, but Claude is allowed to skip straight
        to FINAL_JSON if the opening description was rich enough — the
        response will then carry `complete=True` with seed/stance populated.
        """
        session = InterviewSession(
            id=str(uuid.uuid4()), user_description=user_description.strip()
        )
        self._sessions[session.id] = session

        opener = (
            "Begin the interview. The founder has given you this initial "
            f"description of their business:\n\n{session.user_description}\n\n"
            "Ask your first question now."
        )
        session.messages.append({"role": "user", "content": opener})

        return await self._exchange(session)

    async def submit_answer(self, session_id: str, answer: str) -> InterviewResponse:
        """Submit a user answer; return either the next question or the final payload."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session.complete:
            raise RuntimeError(
                f"session {session_id!r} is already complete; create a new interview"
            )

        session.messages.append({"role": "user", "content": answer.strip()})

        # If the conversation has run too long, nudge Claude to finalize.
        if session.turn_count >= MAX_INTERVIEW_TURNS:
            session.messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM: This conversation has reached the turn limit. "
                        "You MUST respond with the FINAL_JSON payload now. Use "
                        "your best inference for any field you are uncertain about."
                    ),
                }
            )

        return await self._exchange(session)

    def get_session(self, session_id: str) -> InterviewSession | None:
        """Read-only accessor for diagnostics / debugging."""
        return self._sessions.get(session_id)

    # ── Internals ──

    async def _ask_claude(self, session: InterviewSession) -> str:
        """Single LLM round-trip. Returns the raw assistant text."""
        client = self._client if self._client is not None else _get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=session.messages,
        )
        # Anthropic returns content blocks; the interview only ever produces
        # plain text, so the first block's `text` is the full reply.
        return response.content[0].text

    async def _exchange(self, session: InterviewSession) -> InterviewResponse:
        """Send the pending messages, classify the reply, and loop on retries.

        Three outcomes:
        * Reply has no FINAL_JSON marker → next question, return to caller.
        * Reply has FINAL_JSON and validates → mark complete, return models.
        * Reply has FINAL_JSON but is bad → inject a SYSTEM RETRY message
          and call Claude again. After `MAX_VALIDATION_RETRIES` failed
          attempts, raise `ValueError`.
        """
        reply = await self._ask_claude(session)
        session.messages.append({"role": "assistant", "content": reply})
        session.turn_count += 1

        while True:
            try:
                payload = _extract_final_json(reply)
            except ValueError as exc:
                error = str(exc)
                payload = None
            else:
                error = None

            if error is None and payload is None:
                # Claude is still interviewing.
                return InterviewResponse(session_id=session.id, question=reply)

            if error is None:
                assert payload is not None
                try:
                    seed, stance = _validate_payload(payload)
                except ValueError as exc:
                    error = str(exc)
                else:
                    session.complete = True
                    session.seed_partial = payload.get("seed", {})
                    session.stance_partial = payload.get("stance", {})
                    return InterviewResponse(
                        session_id=session.id,
                        complete=True,
                        seed=seed,
                        stance=stance,
                    )

            # Validation or parse error path.
            session.retries_used += 1
            session.errors.append(error)
            log.warning(
                "FINAL_JSON validation failed for session %s (retry %d/%d): %s",
                session.id,
                session.retries_used,
                MAX_VALIDATION_RETRIES,
                error,
            )
            if session.retries_used >= MAX_VALIDATION_RETRIES:
                raise ValueError(
                    f"Seed interview failed after {MAX_VALIDATION_RETRIES} "
                    f"validation retries. Last error: {error}"
                )

            session.messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM RETRY: Your previous FINAL_JSON payload failed "
                        f"validation with this error:\n\n{error}\n\nRe-emit the "
                        "FINAL_JSON payload with the error corrected. Do NOT ask "
                        "another question."
                    ),
                }
            )
            reply = await self._ask_claude(session)
            session.messages.append({"role": "assistant", "content": reply})
            session.turn_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo bulk-seed helper
# ─────────────────────────────────────────────────────────────────────────────


# Mapping from the 4 CompanySeed archetypes to a sensible default CEO stance
# archetype. A solo founder defaults to founder_operator; venture_funded to
# venture_growth; small_team to bootstrap (lean default, callers can pass
# `stance_archetype` to override); enterprise to consolidator. No `turnaround`
# default — turnaround is a deliberate stress-test choice the caller must request.
_SEED_TO_STANCE_DEFAULT: dict[str, str] = {
    "solo_founder": "founder_operator",
    "small_team": "bootstrap",
    "venture_funded": "venture_growth",
    "enterprise": "consolidator",
}


def seed_from_archetype(
    archetype: str,
    niche: str,
    name: str,
    *,
    stance_archetype: str | None = None,
    rng: random.Random | None = None,
    economics_model: Literal["physical", "subscription", "service"] | None = None,
) -> tuple[CompanySeed, CeoStance]:
    """Bulk-Monte-Carlo helper — skip the interview, sample directly.

    Pulls a `CompanySeed` from `sample_seed_for_archetype()` and a `CeoStance`
    from `sample_stance()`. The CEO archetype defaults to a sensible match
    for the company archetype (see `_SEED_TO_STANCE_DEFAULT`); callers can
    override via `stance_archetype` for cross-product Monte Carlo sweeps
    (e.g. "what does a turnaround CEO do at a venture-funded startup").

    No LLM calls. Safe to invoke in tight loops.

    Raises:
        ValueError: if `archetype` or `stance_archetype` is unknown.
    """
    if archetype not in SEED_ARCHETYPES:
        raise ValueError(
            f"unknown company archetype {archetype!r}; expected one of {SEED_ARCHETYPES}"
        )
    chosen_stance = stance_archetype or _SEED_TO_STANCE_DEFAULT[archetype]
    if chosen_stance not in STANCE_ARCHETYPES:
        raise ValueError(
            f"unknown stance archetype {chosen_stance!r}; expected one of "
            f"{STANCE_ARCHETYPES}"
        )

    rng = rng or random.Random()

    seed = sample_seed_for_archetype(
        archetype,  # type: ignore[arg-type]
        rng=rng,
        name=name,
        niche=niche,
        economics_model=economics_model,
    )
    stance = sample_stance(chosen_stance, rng)
    return seed, stance


__all__ = [
    "DEFAULT_MODEL",
    "MAX_INTERVIEW_TURNS",
    "MAX_VALIDATION_RETRIES",
    "InterviewResponse",
    "InterviewSession",
    "SeedInterview",
    "SessionId",
    "seed_from_archetype",
]
