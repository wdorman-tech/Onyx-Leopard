"""Determinism layer for the v2 stance-driven CEO orchestrator.

The orchestrator (see `V2_REDESIGN_PLAN.md` §4) delegates strategic decisions
to LLM calls, which are non-deterministic. To make sims reproducible — required
for unit tests, Monte Carlo replication, and CI — every CEO decision is logged
to a per-sim JSONL transcript. Tests run in `replay` mode: instead of calling
the LLM, they look up the recorded decision for `(tick, company_id)` and
return it verbatim.

This module is foundational for:
    - `orchestrator.py` (uses `replay_or_call` to wrap every LLM call)
    - `monte_carlo.py`  (uses `Transcript` + `CostTracker` per simulation)
    - `test_orchestrator.py`, `test_determinism.py` (drive via `replay` mode)

Public surface:
    - `Transcript`           — JSONL writer/reader (record | replay | off)
    - `TranscriptEntry`      — pydantic model for one recorded decision
    - `CeoDecision`               — pydantic model matching `V2_REDESIGN_PLAN.md` §4
    - `replay_or_call`            — integration helper used by orchestrator
    - `CostTracker`               — running token + USD ledger with hard ceiling
    - `CostCeilingExceededErrorError`  — raised when a `record` would exceed ceiling
    - `PromptHashMismatchErrorError`   — raised when replayed SHA != recorded SHA
    - `MissingTranscriptEntryErrorError` — raised when replay finds no recorded entry
    - `canonicalize_prompt`       — deterministic prompt-canonicalization rule
    - `prompt_sha256`             — SHA-256 of the canonicalized prompt
    - `MODEL_PRICING`             — per-token USD pricing constants (Haiku, Sonnet)

Determinism contract: `replay_or_call` hashes the prompt via
`canonicalize_prompt` before comparing against the recorded `prompt_sha256`.
The canonicalization rule is documented on `canonicalize_prompt` and MUST be
treated as part of the on-disk format — changing it invalidates every
existing transcript on disk.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Constants — model pricing
# ---------------------------------------------------------------------------
#
# Anthropic published per-token pricing (USD), as of 2026-04-15 (the date this
# module was authored). Source: Claude API skill cache + platform.claude.com.
# These WILL go stale. When a new model lands or pricing changes, update both
# the constants AND the version pinned by `MODEL_PRICING_VERSION` so callers
# can detect a stale tracker against a fresh price sheet.
#
# Conversion: published prices are USD per 1M tokens; we store USD per token.
#   $1 / 1M   → 1e-6
#   $3 / 1M   → 3e-6
#   $5 / 1M   → 5e-6
#   $15 / 1M  → 15e-6
# ---------------------------------------------------------------------------

#: Per-token USD pricing for the models the v2 orchestrator uses.
#: Keys are the canonical Anthropic model IDs (no date suffix); values are
#: `(input_per_token, output_per_token)`. Cache-read / cache-write tiers are
#: NOT modelled here — the orchestrator pays full price for every call. If we
#: add prompt caching later, extend this map and `CostTracker.record`.
MODEL_PRICING: Final[dict[str, tuple[float, float]]] = {
    # Claude Haiku 4.5 — $1.00 input / $5.00 output per 1M tokens
    "claude-haiku-4-5": (1.0e-6, 5.0e-6),
    # Claude Sonnet 4.6 — $3.00 input / $15.00 output per 1M tokens
    "claude-sonnet-4-6": (3.0e-6, 15.0e-6),
}

#: Bumped whenever `MODEL_PRICING` changes. Embedded in transcripts so a stale
#: cost-replay can be detected.
MODEL_PRICING_VERSION: Final[str] = "2026-04-15"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CostCeilingExceededError(RuntimeError):
    """Raised when a `CostTracker.record` would push past the per-sim ceiling.

    The tracker is left unchanged when this is raised — the failed record is
    not partially applied. Caller's responsibility to halt the sim (or fall
    back to a cheaper tier).
    """


class PromptHashMismatchError(RuntimeError):
    """Raised in replay mode when the prompt SHA differs from the recorded one.

    State has diverged between the recording run and this run. Continuing
    would silently apply a decision the LLM never actually made for the
    current state — defeats the entire point of replay. Halt and fix the
    divergence (or re-record).
    """


class MissingTranscriptEntryError(LookupError):
    """Raised in replay mode when no entry matches `(tick, company_id)`."""


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------


class CeoDecision(BaseModel):
    """Structured CEO decision per `V2_REDESIGN_PLAN.md` §4.

    Defined here (not in `orchestrator.py`) because `replay.py` is foundational
    — orchestrator.py imports from here. Field set matches the spec exactly.
    """

    model_config = ConfigDict(extra="forbid")

    spawn_nodes: list[str] = Field(default_factory=list)
    retire_nodes: list[str] = Field(default_factory=list)
    adjust_params: dict[str, float] = Field(default_factory=dict)
    open_locations: int = Field(default=0, ge=0)
    reasoning: str = Field(..., min_length=1)
    references_stance: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Transcript entry model
# ---------------------------------------------------------------------------


Tier = Literal["operational", "tactical", "strategic"]


class TranscriptEntry(BaseModel):
    """One recorded LLM call. Keyed by `(sim_id, tick, company_id, decision_id)`.

    `decision_id` is a UUID (not derived from the other fields) so that a
    single tick can carry multiple decisions for the same company without
    collision — e.g. a strategic call and an emergency shock-driven call on
    the same tick.

    `prompt_sha256` is the canonical prompt hash (see `canonicalize_prompt`),
    NOT a hash of the raw prompt string — the canonicalization rule is part
    of the file format.

    `kind` discriminator: defaults to `"decision"` so transcripts written
    before the critic was added still parse cleanly. `iter_entries` and
    `_load_index` use the on-disk `kind` field to dispatch between
    `TranscriptEntry` (CEO decisions) and `CriticEntry` (critic scores).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["decision"] = "decision"
    sim_id: str = Field(..., min_length=1)
    tick: int = Field(..., ge=0)
    company_id: str = Field(..., min_length=1)
    decision_id: str = Field(..., min_length=1)
    tier: Tier
    prompt_sha256: str = Field(..., min_length=64, max_length=64)
    raw_response: str
    parsed_decision: CeoDecision
    model: str = Field(..., min_length=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)


class CriticEntry(BaseModel):
    """One recorded critic score. Keyed by `(tick, company_id)` only.

    Separate model from `TranscriptEntry` — the critic doesn't have a
    prompt-SHA invariant (it's deterministic given decision+stance+state),
    doesn't carry a tier, and isn't billed against `MODEL_PRICING` checks
    in the orchestrator's hot path. Sharing the schema would force a sea
    of nullable fields on either side; keeping them separate keeps both
    shapes auditable.

    `kind` discriminator: literal `"critic"`. `_load_index` dispatches on
    this field so a single JSONL file can interleave decisions and critic
    scores without ambiguity.

    `parsed_score` carries the actual `CriticScore`. The schema is defined
    here as a forward-compatible dict (we import `CriticScore` lazily in
    the agent — `replay.py` is foundational and must not import upward).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["critic"] = "critic"
    tick: int = Field(..., ge=0)
    company_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Prompt canonicalization
# ---------------------------------------------------------------------------


def canonicalize_prompt(prompt: str) -> bytes:
    """Canonicalize a prompt for hashing.

    The rule (treat as part of the file format):
        1. If `prompt` is a JSON-decodable string, re-serialize it with
           `json.dumps(..., sort_keys=True, separators=(",", ":"),
           ensure_ascii=False)` so dict-key order and whitespace can vary
           between runs without breaking replay.
        2. Otherwise, strip trailing whitespace from each line and join with
           "\\n" — handles minor template-rendering whitespace drift.
        3. UTF-8 encode the result.

    The JSON branch is the common case: orchestrator prompts are built from
    structured state (stance + state window + recent shocks + library), and
    pretty-printed JSON whitespace is the most likely silent invalidator.
    The text branch handles future free-form templates without requiring
    every caller to JSON-wrap.

    Changing this rule invalidates every recorded transcript. If you must
    change it, bump a transcript-format version and refuse to replay older
    transcripts.
    """
    try:
        decoded = json.loads(prompt)
    except (ValueError, TypeError):
        # Not JSON — fall back to whitespace-normalized text.
        normalized_lines = [line.rstrip() for line in prompt.splitlines()]
        return "\n".join(normalized_lines).encode("utf-8")
    else:
        return json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")


def prompt_sha256(prompt: str) -> str:
    """Hex SHA-256 of the canonicalized prompt. 64-char lowercase hex string."""
    return hashlib.sha256(canonicalize_prompt(prompt)).hexdigest()


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


TranscriptMode = Literal["record", "replay", "off"]


class Transcript:
    """Per-sim JSONL transcript file (one JSON object per line).

    Three modes:
        - `"record"` — open append-only, write every entry, never read.
        - `"replay"` — open read-only at construction, build an in-memory index,
                       look up by `(tick, company_id)`. No writes allowed.
        - `"off"`    — no-op for legacy tests / production runs that skip the
                       determinism layer entirely. All methods are safe but do
                       nothing observable.

    Concurrency: `record` uses a per-instance `threading.Lock` plus
    `Path.open("a", encoding="utf-8")` with explicit `flush()` so concurrent
    sim runs writing to DIFFERENT files don't interfere. Two sims sharing the
    same path would still serialize through the lock within a process, but
    cross-process locking is out of scope — if you Monte-Carlo across
    processes, give each its own transcript path.
    """

    def __init__(self, path: Path, mode: TranscriptMode) -> None:
        if mode not in ("record", "replay", "off"):
            raise ValueError(
                f"unknown transcript mode {mode!r}; expected record|replay|off"
            )
        self._path = path
        self._mode: TranscriptMode = mode
        self._lock = threading.Lock()
        # Index built on construction in replay mode; lazy in record/off.
        self._index: dict[tuple[int, str], TranscriptEntry] = {}
        # Separate critic index — different value type, different keying
        # semantics (no decision_id since one critic score per strategic
        # decision per tick). `_load_index` populates both in replay mode.
        self._critic_index: dict[tuple[int, str], CriticEntry] = {}

        if mode == "replay":
            self._load_index()
        elif mode == "record":
            # Ensure parent dir exists so the first append doesn't 404.
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mode(self) -> TranscriptMode:
        return self._mode

    def record(self, entry: TranscriptEntry) -> None:
        """Append `entry` to the JSONL file. No-op in `replay` and `off` modes.

        Each call writes exactly one line (with trailing newline) and flushes
        + fsyncs so a crashed process leaves the file in a parseable state.
        Pydantic's `model_dump_json` produces compact, deterministic JSON.
        """
        if self._mode == "off":
            return
        if self._mode == "replay":
            raise RuntimeError("Transcript opened in replay mode; record is forbidden")

        line = entry.model_dump_json()
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
            f.flush()
            # Best-effort fsync — guarantees the line is on disk if the
            # process dies between calls. Skip if the OS doesn't support
            # it (e.g., some virtualized filesystems).
            with contextlib.suppress(OSError, AttributeError):
                os.fsync(f.fileno())

    def lookup(self, tick: int, company_id: str) -> TranscriptEntry | None:
        """Return the recorded entry for `(tick, company_id)`, or `None`.

        Only meaningful in `replay` mode. In `record` and `off` modes always
        returns `None` — there is no in-memory index of writes.
        """
        if self._mode != "replay":
            return None
        return self._index.get((tick, company_id))

    # ── Critic API ────────────────────────────────────────────────────────

    def record_critic(
        self, tick: int, company_id: str, score: Any
    ) -> None:
        """Append a critic score to the JSONL file. Mirrors `record(...)`.

        `score` is typed `Any` to avoid an upward import dependency on the
        `critic` module. In practice it must be a `critic.CriticScore`
        (duck-typed: needs `.score`, `.violations`, `.reasoning`). Passing
        anything else raises ValidationError when the `CriticEntry` is
        constructed.

        Duplicate-detection on `(tick, company_id)` mirrors the decision
        path so a buggy bundle that calls the critic twice for the same
        strategic decision fails loudly rather than silently producing a
        non-reproducible replay.
        """
        if self._mode == "off":
            return
        if self._mode == "replay":
            raise RuntimeError(
                "Transcript opened in replay mode; record_critic is forbidden"
            )

        entry = CriticEntry(
            tick=tick,
            company_id=company_id,
            score=float(score.score),
            violations=list(score.violations),
            reasoning=str(score.reasoning),
        )
        line = entry.model_dump_json()
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
            f.flush()
            with contextlib.suppress(OSError, AttributeError):
                os.fsync(f.fileno())

    def lookup_critic(self, tick: int, company_id: str) -> Any | None:
        """Return the recorded `CriticScore` for `(tick, company_id)`, or `None`.

        Returns `critic.CriticScore` in practice — typed `Any` here to keep
        `replay.py` foundational (no upward import). The critic module
        adapts the on-disk `CriticEntry` back to `CriticScore` itself.
        """
        if self._mode != "replay":
            return None
        entry = self._critic_index.get((tick, company_id))
        if entry is None:
            return None
        # Lazy import — keeps `replay.py` foundational and import-cycle-free.
        from src.simulation.critic import CriticScore

        return CriticScore(
            score=entry.score,
            violations=list(entry.violations),
            reasoning=entry.reasoning,
        )

    def iter_entries(self) -> Iterator[TranscriptEntry | CriticEntry]:
        """Yield every entry in the on-disk file in write order.

        Works in all three modes (re-reads the file from disk every call so
        you see entries written after construction in `record` mode). Used by
        analysis tooling, not by the hot orchestrator path.

        Dispatches on the on-disk `kind` field — decisions become
        `TranscriptEntry`, critic scores become `CriticEntry`. Lines
        missing `kind` default to `"decision"` for backward-compat with
        transcripts written before the critic was added.
        """
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Cheap kind dispatch — decode once, branch on kind.
                obj = json.loads(line)
                kind = obj.get("kind", "decision")
                if kind == "critic":
                    yield CriticEntry.model_validate(obj)
                else:
                    yield TranscriptEntry.model_validate(obj)

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Build the `(tick, company_id) → entry` indices. Strict — duplicates fail.

        A duplicate `(tick, company_id)` is a recording bug for either
        kind. Refusing to load is safer than silently picking one and
        producing a non-reproducible replay.
        """
        if not self._path.exists():
            return  # Empty replay is allowed; lookups return None.
        for entry in self.iter_entries():
            if isinstance(entry, CriticEntry):
                key = (entry.tick, entry.company_id)
                if key in self._critic_index:
                    raise ValueError(
                        f"duplicate critic entry for tick={entry.tick} "
                        f"company_id={entry.company_id!r} in {self._path}; "
                        "this is a recording bug — replay is non-deterministic"
                    )
                self._critic_index[key] = entry
                continue
            # Decision entry path.
            key = (entry.tick, entry.company_id)
            if key in self._index:
                raise ValueError(
                    f"duplicate transcript entry for tick={entry.tick} "
                    f"company_id={entry.company_id!r} in {self._path}; "
                    "this is a recording bug — replay is non-deterministic"
                )
            self._index[key] = entry


# ---------------------------------------------------------------------------
# replay_or_call — the orchestrator integration helper
# ---------------------------------------------------------------------------


def replay_or_call(
    transcript: Transcript,
    prompt: str,
    llm_callable: Callable[[str], tuple[CeoDecision, str, str, int, int]],
    *,
    sim_id: str,
    tick: int,
    company_id: str,
    tier: Tier,
    cost_tracker: CostTracker | None = None,
) -> CeoDecision:
    """Either replay a recorded decision or call the LLM and record it.

    The `llm_callable` contract: takes the prompt string, returns a 5-tuple
        `(decision, raw_response, model, input_tokens, output_tokens)`
    where `decision` is a `CeoDecision`, `model` is the canonical model ID
    (must be a key of `MODEL_PRICING` if `cost_tracker` is passed), and the
    token counts come from the API `usage` block. The orchestrator wraps its
    Anthropic SDK call in a tiny adapter that returns this tuple.

    Mode behaviour:
        - `record`: invoke `llm_callable`, charge `cost_tracker` (if any),
                    record the entry, return the decision.
        - `replay`: look up by `(tick, company_id)`. Verify the prompt SHA
                    matches the recorded one; if not, raise
                    `PromptHashMismatchError`. Return the recorded `parsed_decision`
                    without invoking the LLM. `cost_tracker` is not charged
                    (the cost was paid in the original recording run).
        - `off`:    invoke `llm_callable`, charge `cost_tracker` (if any),
                    return the decision. No recording, no replay.

    `cost_tracker` is optional so this helper is usable from contexts that
    don't enforce a budget (analysis tooling, single-call sanity checks).
    """
    if transcript.mode == "replay":
        entry = transcript.lookup(tick, company_id)
        if entry is None:
            raise MissingTranscriptEntryError(
                f"no recorded decision for tick={tick} company_id={company_id!r} "
                f"in {transcript.path} — state has diverged from the recording, "
                "or the orchestrator is calling at an unrecorded cadence"
            )
        observed_sha = prompt_sha256(prompt)
        if observed_sha != entry.prompt_sha256:
            raise PromptHashMismatchError(
                f"prompt SHA mismatch at tick={tick} company_id={company_id!r}: "
                f"observed={observed_sha} recorded={entry.prompt_sha256} "
                f"(in {transcript.path}). State has diverged between the "
                "recording run and this replay. Re-record the transcript or "
                "fix the divergence."
            )
        return entry.parsed_decision

    # record / off — both call the LLM.
    decision, raw_response, model, input_tokens, output_tokens = llm_callable(prompt)

    cost_usd = 0.0
    if cost_tracker is not None:
        # Compute the pure cost FIRST so `cost_usd` is attributed to this call's
        # tokens before any mutation. `record` may raise CostCeilingExceededError;
        # it leaves the tracker unchanged so the partial mutation is bounded.
        cost_usd = cost_tracker.cost_for(input_tokens, output_tokens, model)
        cost_tracker.record(input_tokens, output_tokens, model)

    if transcript.mode == "record":
        entry = TranscriptEntry(
            sim_id=sim_id,
            tick=tick,
            company_id=company_id,
            decision_id=str(uuid.uuid4()),
            tier=tier,
            prompt_sha256=prompt_sha256(prompt),
            raw_response=raw_response,
            parsed_decision=decision,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        transcript.record(entry)

    return decision


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Per-sim running ledger of token usage + USD spend, with a hard ceiling.

    Thread-safe within a single process. Cross-process Monte Carlo runs each
    get their own tracker; the per-sim ceiling is per-tracker, not global.
    """

    def __init__(self, ceiling_usd: float = 50.0) -> None:
        if ceiling_usd <= 0.0:
            raise ValueError(f"ceiling_usd must be > 0; got {ceiling_usd}")
        self._ceiling_usd = ceiling_usd
        self._total_usd = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._lock = threading.Lock()

    # ── Pricing ───────────────────────────────────────────────────────────

    @staticmethod
    def cost_for(input_tokens: int, output_tokens: int, model: str) -> float:
        """USD cost of a single call. Pure function — does not mutate state."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"token counts must be non-negative; got "
                f"input={input_tokens} output={output_tokens}"
            )
        try:
            in_rate, out_rate = MODEL_PRICING[model]
        except KeyError as exc:
            known = ", ".join(sorted(MODEL_PRICING))
            raise ValueError(
                f"no pricing for model {model!r}; known models: {known}. "
                "Add it to MODEL_PRICING (and bump MODEL_PRICING_VERSION)."
            ) from exc
        return input_tokens * in_rate + output_tokens * out_rate

    # ── State ─────────────────────────────────────────────────────────────

    @property
    def ceiling_usd(self) -> float:
        return self._ceiling_usd

    def total_cost(self) -> float:
        with self._lock:
            return self._total_usd

    def total_tokens(self) -> tuple[int, int]:
        """Return `(input_tokens, output_tokens)` totals."""
        with self._lock:
            return (self._total_input_tokens, self._total_output_tokens)

    def remaining_budget(self) -> float:
        with self._lock:
            return max(0.0, self._ceiling_usd - self._total_usd)

    # ── Recording / pre-flight ────────────────────────────────────────────

    def would_exceed(
        self, predicted_input: int, predicted_output: int, model: str
    ) -> bool:
        """True if recording these counts would push past the ceiling.

        Use this BEFORE calling the LLM to skip the call (or fall back to a
        cheaper tier). Read-only — does not mutate state.
        """
        projected = self.cost_for(predicted_input, predicted_output, model)
        with self._lock:
            return (self._total_usd + projected) > self._ceiling_usd

    def record(self, input_tokens: int, output_tokens: int, model: str) -> None:
        """Add a call's tokens to the ledger.

        Raises `CostCeilingExceededError` if the resulting total would exceed the
        ceiling. The tracker is NOT mutated when this raises — the caller can
        retry with a cheaper model or halt the sim.
        """
        delta = self.cost_for(input_tokens, output_tokens, model)
        with self._lock:
            new_total = self._total_usd + delta
            if new_total > self._ceiling_usd:
                raise CostCeilingExceededError(
                    f"recording {delta:.6f} USD ({input_tokens} in / "
                    f"{output_tokens} out on {model}) would push total to "
                    f"{new_total:.6f} USD past ceiling {self._ceiling_usd:.6f} "
                    f"USD (current {self._total_usd:.6f} USD)"
                )
            self._total_usd = new_total
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "MODEL_PRICING",
    "MODEL_PRICING_VERSION",
    "CeoDecision",
    "CostCeilingExceededError",
    "CostTracker",
    "CriticEntry",
    "MissingTranscriptEntryError",
    "PromptHashMismatchError",
    "Tier",
    "Transcript",
    "TranscriptEntry",
    "TranscriptMode",
    "canonicalize_prompt",
    "prompt_sha256",
    "replay_or_call",
]


# Below this line: forward references resolution. `replay_or_call`'s signature
# uses `CostTracker` defined later in the file, which is fine at runtime
# (PEP 563 / `from __future__ import annotations`) but we list it explicitly
# in `Any` form for any tooling that pre-resolves annotations.
_ = (CostTracker, CeoDecision, Any)
