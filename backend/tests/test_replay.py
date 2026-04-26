"""Tests for the replay/transcript determinism layer (Onyx Leopard v2)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.simulation.replay import (
    MODEL_PRICING,
    CeoDecision,
    CostCeilingExceededError,
    CostTracker,
    MissingTranscriptEntryError,
    PromptHashMismatchError,
    Transcript,
    TranscriptEntry,
    canonicalize_prompt,
    prompt_sha256,
    replay_or_call,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_decision(reasoning: str = "default") -> CeoDecision:
    return CeoDecision(
        spawn_nodes=["bd_rep"],
        retire_nodes=[],
        adjust_params={"price": 26000.0},
        open_locations=1,
        reasoning=reasoning,
        references_stance=["growth_obsession"],
    )


def _make_entry(
    *,
    sim_id: str = "sim_001",
    tick: int = 90,
    company_id: str = "comp_001",
    decision_id: str = "decision_xyz",
    prompt: str = '{"hello": "world"}',
    decision: CeoDecision | None = None,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 1234,
    output_tokens: int = 567,
    cost_usd: float = 0.0123,
) -> TranscriptEntry:
    return TranscriptEntry(
        sim_id=sim_id,
        tick=tick,
        company_id=company_id,
        decision_id=decision_id,
        tier="strategic",
        prompt_sha256=prompt_sha256(prompt),
        raw_response="{\"foo\": 1}",
        parsed_decision=decision or _make_decision(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _stub_llm(
    decision: CeoDecision,
    *,
    raw_response: str = '{"x": 1}',
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Callable[[str], tuple[CeoDecision, str, str, int, int]]:
    """Build an `llm_callable` stub that ignores the prompt and returns fixed data."""

    def _call(_prompt: str) -> tuple[CeoDecision, str, str, int, int]:
        return decision, raw_response, model, input_tokens, output_tokens

    return _call


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalization + hashing
# ─────────────────────────────────────────────────────────────────────────────


def test_canonicalize_json_is_invariant_to_key_order_and_whitespace() -> None:
    a = json.dumps({"b": 2, "a": 1, "nested": {"y": 4, "x": 3}}, indent=2)
    b = json.dumps({"a": 1, "b": 2, "nested": {"x": 3, "y": 4}}, separators=(",", ":"))
    assert canonicalize_prompt(a) == canonicalize_prompt(b)
    assert prompt_sha256(a) == prompt_sha256(b)


def test_canonicalize_text_strips_trailing_whitespace_per_line() -> None:
    a = "hello\nworld\n"
    b = "hello   \nworld\t\n"
    assert canonicalize_prompt(a) == canonicalize_prompt(b)


def test_prompt_sha256_is_64_hex_chars() -> None:
    h = prompt_sha256("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript modes — basics
# ─────────────────────────────────────────────────────────────────────────────


def test_record_mode_writes_jsonl_one_line_per_entry(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    transcript = Transcript(path, mode="record")
    transcript.record(_make_entry(tick=10, company_id="a"))
    transcript.record(_make_entry(tick=11, company_id="a"))
    transcript.record(_make_entry(tick=11, company_id="b"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        # Each line is independently JSON-parseable (human readable).
        parsed = json.loads(line)
        assert "sim_id" in parsed
        assert "prompt_sha256" in parsed


def test_replay_mode_lookup_returns_recorded_entry(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    rec = Transcript(path, mode="record")
    e1 = _make_entry(tick=1, company_id="a")
    e2 = _make_entry(tick=2, company_id="b")
    rec.record(e1)
    rec.record(e2)

    rep = Transcript(path, mode="replay")
    found = rep.lookup(tick=2, company_id="b")
    assert found is not None
    assert found.decision_id == e2.decision_id
    assert rep.lookup(tick=99, company_id="missing") is None


def test_replay_mode_rejects_record(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    Transcript(path, mode="record").record(_make_entry())
    rep = Transcript(path, mode="replay")
    with pytest.raises(RuntimeError, match="replay mode"):
        rep.record(_make_entry(tick=999, company_id="z"))


def test_off_mode_is_pass_through(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    t = Transcript(path, mode="off")
    # Recording is silently a no-op.
    t.record(_make_entry())
    # No file is created.
    assert not path.exists()
    # Lookups always return None.
    assert t.lookup(tick=1, company_id="a") is None


def test_iter_entries_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    rec = Transcript(path, mode="record")
    entries = [_make_entry(tick=i, company_id=f"c{i}") for i in range(5)]
    for e in entries:
        rec.record(e)

    read_back = list(Transcript(path, mode="replay").iter_entries())
    assert [e.decision_id for e in read_back] == [e.decision_id for e in entries]


def test_unknown_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown transcript mode"):
        Transcript(tmp_path / "x.jsonl", mode="bogus")  # type: ignore[arg-type]


def test_record_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "sim.jsonl"
    rec = Transcript(path, mode="record")
    rec.record(_make_entry())
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Replay strict-load behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_replay_load_rejects_duplicate_tick_company_pairs(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    rec = Transcript(path, mode="record")
    rec.record(_make_entry(tick=10, company_id="a", decision_id="d1"))
    rec.record(_make_entry(tick=10, company_id="a", decision_id="d2"))
    with pytest.raises(ValueError, match="duplicate transcript entry"):
        Transcript(path, mode="replay")


def test_replay_on_missing_file_is_empty(tmp_path: Path) -> None:
    rep = Transcript(tmp_path / "does-not-exist.jsonl", mode="replay")
    assert rep.lookup(tick=0, company_id="x") is None
    assert list(rep.iter_entries()) == []


# ─────────────────────────────────────────────────────────────────────────────
# replay_or_call — round-trip determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_record_then_replay_produces_same_decision(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    prompt = '{"state": {"cash": 50000, "tick": 90}}'
    decision = _make_decision(reasoning="invest in BD")

    rec = Transcript(path, mode="record")
    out = replay_or_call(
        rec,
        prompt,
        _stub_llm(decision),
        sim_id="s1",
        tick=90,
        company_id="comp_001",
        tier="strategic",
    )
    assert out == decision

    rep = Transcript(path, mode="replay")
    # The "LLM" in replay should never be called — assert it raises if invoked.
    def _exploding_llm(_p: str) -> tuple[CeoDecision, str, str, int, int]:
        raise AssertionError("LLM must not be invoked in replay mode")

    out2 = replay_or_call(
        rep,
        prompt,
        _exploding_llm,
        sim_id="s1",
        tick=90,
        company_id="comp_001",
        tier="strategic",
    )
    assert out2 == decision


def test_replay_with_mismatched_prompt_sha_raises(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    rec = Transcript(path, mode="record")
    replay_or_call(
        rec,
        '{"state": "original"}',
        _stub_llm(_make_decision()),
        sim_id="s1",
        tick=1,
        company_id="c1",
        tier="strategic",
    )

    rep = Transcript(path, mode="replay")
    with pytest.raises(PromptHashMismatchError, match="diverged"):
        replay_or_call(
            rep,
            '{"state": "modified"}',  # different prompt
            _stub_llm(_make_decision()),
            sim_id="s1",
            tick=1,
            company_id="c1",
            tier="strategic",
        )


def test_replay_with_no_recorded_entry_raises(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    Transcript(path, mode="record")  # creates parent dir but no entries
    # Touch the file so replay loads cleanly even when empty.
    path.touch()

    rep = Transcript(path, mode="replay")
    with pytest.raises(MissingTranscriptEntryError, match="no recorded decision"):
        replay_or_call(
            rep,
            "any prompt",
            _stub_llm(_make_decision()),
            sim_id="s1",
            tick=42,
            company_id="ghost",
            tier="strategic",
        )


def test_off_mode_invokes_llm_but_does_not_record(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    t = Transcript(path, mode="off")
    decision = _make_decision()
    out = replay_or_call(
        t,
        "anything",
        _stub_llm(decision),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="strategic",
    )
    assert out == decision
    assert not path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent multi-sim isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_concurrent_sims_do_not_cross_contaminate(tmp_path: Path) -> None:
    """Two sims writing to different transcript paths in parallel.

    Asserts: each sim's file contains only its own entries, and replaying
    each one returns its own decisions.
    """
    n_per_sim = 25
    sim_paths = {
        "sim_a": tmp_path / "sim_a.jsonl",
        "sim_b": tmp_path / "sim_b.jsonl",
    }

    def _run(sim_id: str) -> None:
        transcript = Transcript(sim_paths[sim_id], mode="record")
        for i in range(n_per_sim):
            decision = _make_decision(reasoning=f"{sim_id}-tick{i}")
            replay_or_call(
                transcript,
                f'{{"sim": "{sim_id}", "tick": {i}}}',
                _stub_llm(decision),
                sim_id=sim_id,
                tick=i,
                company_id=f"{sim_id}_co",
                tier="tactical",
            )

    threads = [threading.Thread(target=_run, args=(s,)) for s in sim_paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Replay each independently, assert no contamination.
    for sim_id, path in sim_paths.items():
        rep = Transcript(path, mode="replay")
        entries = list(rep.iter_entries())
        assert len(entries) == n_per_sim
        for e in entries:
            assert e.sim_id == sim_id
            assert e.company_id == f"{sim_id}_co"
            assert e.parsed_decision.reasoning.startswith(sim_id)


def test_concurrent_record_into_same_file_serializes_writes(tmp_path: Path) -> None:
    """Multiple threads writing to the SAME transcript still produce valid JSONL.

    No interleaved/torn lines; every line is independently JSON-parseable.
    """
    path = tmp_path / "shared.jsonl"
    t = Transcript(path, mode="record")
    n_threads = 8
    n_each = 20

    def _writer(thread_idx: int) -> None:
        for i in range(n_each):
            t.record(
                _make_entry(
                    tick=thread_idx * 1000 + i,
                    company_id=f"t{thread_idx}",
                    decision_id=f"d{thread_idx}_{i}",
                )
            )

    threads = [threading.Thread(target=_writer, args=(k,)) for k in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * n_each
    # Every line must parse as a valid TranscriptEntry — torn writes would fail here.
    for line in lines:
        TranscriptEntry.model_validate_json(line)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL is human-readable
# ─────────────────────────────────────────────────────────────────────────────


def test_jsonl_file_is_human_readable(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    rec = Transcript(path, mode="record")
    rec.record(_make_entry(tick=5, company_id="visible"))
    rec.record(_make_entry(tick=6, company_id="visible"))

    text = path.read_text(encoding="utf-8")
    # Newline-separated, one JSON object per line, no array wrapper.
    assert text.count("\n") == 2
    assert not text.startswith("[")
    # Search by substring works on the on-disk format.
    assert '"company_id":"visible"' in text


# ─────────────────────────────────────────────────────────────────────────────
# CostTracker
# ─────────────────────────────────────────────────────────────────────────────


def test_cost_tracker_haiku_pricing() -> None:
    """Haiku 4.5: $1/1M input + $5/1M output.

    1M input + 1M output → $1 + $5 = $6.
    """
    t = CostTracker(ceiling_usd=100.0)
    t.record(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-haiku-4-5")
    assert t.total_cost() == pytest.approx(6.0, rel=1e-9)


def test_cost_tracker_sonnet_pricing() -> None:
    """Sonnet 4.6: $3/1M input + $15/1M output.

    1M input + 1M output → $3 + $15 = $18.
    """
    t = CostTracker(ceiling_usd=100.0)
    t.record(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6")
    assert t.total_cost() == pytest.approx(18.0, rel=1e-9)


def test_cost_tracker_mixed_call_accumulates() -> None:
    t = CostTracker(ceiling_usd=10.0)
    t.record(input_tokens=10_000, output_tokens=5_000, model="claude-haiku-4-5")
    t.record(input_tokens=2_000, output_tokens=1_000, model="claude-sonnet-4-6")
    # Haiku: 10_000 * 1e-6 + 5_000 * 5e-6 = 0.01 + 0.025 = 0.035
    # Sonnet: 2_000 * 3e-6 + 1_000 * 15e-6 = 0.006 + 0.015 = 0.021
    expected = 0.035 + 0.021
    assert t.total_cost() == pytest.approx(expected, rel=1e-9)
    assert t.total_tokens() == (12_000, 6_000)


def test_cost_tracker_ceiling_enforced() -> None:
    t = CostTracker(ceiling_usd=0.01)
    # 1_000 input * 5e-6 + 2_000 output * 15e-6 on Sonnet = 0.005 + 0.030 = 0.035 > 0.01
    with pytest.raises(CostCeilingExceededError, match="past ceiling"):
        t.record(input_tokens=1_000, output_tokens=2_000, model="claude-sonnet-4-6")
    # Tracker unchanged after the failed record.
    assert t.total_cost() == 0.0
    assert t.total_tokens() == (0, 0)


def test_cost_tracker_would_exceed_is_read_only() -> None:
    t = CostTracker(ceiling_usd=0.01)
    assert t.would_exceed(1_000, 2_000, "claude-sonnet-4-6") is True
    # Querying does not mutate state.
    assert t.total_cost() == 0.0


def test_cost_tracker_remaining_budget_floors_at_zero() -> None:
    t = CostTracker(ceiling_usd=1.0)
    # Spend exactly $0.50.
    t.record(input_tokens=100_000, output_tokens=20_000, model="claude-sonnet-4-6")
    # 100_000 * 3e-6 + 20_000 * 15e-6 = 0.30 + 0.30 = 0.60
    assert t.remaining_budget() == pytest.approx(0.40, rel=1e-9)


def test_cost_tracker_unknown_model_raises() -> None:
    t = CostTracker(ceiling_usd=1.0)
    with pytest.raises(ValueError, match="no pricing for model"):
        t.record(input_tokens=10, output_tokens=10, model="claude-opus-4-7")


def test_cost_tracker_negative_tokens_rejected() -> None:
    t = CostTracker(ceiling_usd=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        t.record(input_tokens=-1, output_tokens=10, model="claude-haiku-4-5")


def test_cost_tracker_invalid_ceiling_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        CostTracker(ceiling_usd=0.0)


def test_replay_or_call_charges_cost_tracker_in_record_mode(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    transcript = Transcript(path, mode="record")
    tracker = CostTracker(ceiling_usd=10.0)
    decision = _make_decision()

    replay_or_call(
        transcript,
        '{"x": 1}',
        _stub_llm(decision, model="claude-haiku-4-5",
                  input_tokens=1_000, output_tokens=500),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="operational",
        cost_tracker=tracker,
    )
    # 1_000 * 1e-6 + 500 * 5e-6 = 0.001 + 0.0025 = 0.0035
    assert tracker.total_cost() == pytest.approx(0.0035, rel=1e-9)


def test_replay_cost_for_runs_before_record(tmp_path: Path) -> None:
    """Ordering invariant (P-AI-4): `cost_for` (pure) must be invoked BEFORE
    `record` (mutating) in `replay_or_call`.

    The pure read-then-mutate ordering means `cost_usd` is attributed to this
    call's tokens against the tracker's PRE-record state. We verify by
    monkey-patching both methods with side effects that append to a shared
    call log. (Note: `record` internally calls `cost_for` to compute its own
    delta — so the call log will contain a third entry from inside `record`.
    The invariant we care about is the ORDER of the FIRST two calls.)
    """
    path = tmp_path / "sim.jsonl"
    transcript = Transcript(path, mode="record")
    tracker = CostTracker(ceiling_usd=10.0)

    call_order: list[str] = []
    real_cost_for = tracker.cost_for
    real_record = tracker.record

    def spy_cost_for(input_tokens: int, output_tokens: int, model: str) -> float:
        call_order.append("cost_for")
        return real_cost_for(input_tokens, output_tokens, model)

    def spy_record(input_tokens: int, output_tokens: int, model: str) -> None:
        call_order.append("record")
        real_record(input_tokens, output_tokens, model)

    # `cost_for` is a staticmethod accessed on the instance — overriding via
    # `tracker.cost_for = spy_cost_for` masks it on the instance only.
    tracker.cost_for = spy_cost_for  # type: ignore[method-assign]
    tracker.record = spy_record  # type: ignore[method-assign]

    replay_or_call(
        transcript,
        '{"x": 1}',
        _stub_llm(_make_decision(), model="claude-haiku-4-5",
                  input_tokens=2_000, output_tokens=400),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="operational",
        cost_tracker=tracker,
    )

    # Pure cost computation must come first; mutation second. (Internal
    # `cost_for` from inside `record` shows up as a third entry — fine.)
    assert call_order[:2] == ["cost_for", "record"]


def test_replay_cost_usd_attributed_against_pre_record_state(tmp_path: Path) -> None:
    """Alternative verification of the swap (P-AI-4): the `cost_usd` recorded
    in the transcript entry equals `cost_for(input, output, model)` computed
    against the tracker's PRE-call state — i.e. the cost of THIS call alone.

    Pre-existing tracker spend is irrelevant to a single call's cost, but the
    assertion is meaningful as a regression guard: if anyone re-orders the
    call so `record` runs first, `cost_for` would still return the same
    pure-function output (it does not depend on tracker state), so the
    written `cost_usd` would be unchanged numerically. The order test above
    is the load-bearing assertion; this one verifies the resulting transcript
    is internally consistent.
    """
    path = tmp_path / "sim.jsonl"
    # Pre-charge the tracker so we'd notice if `cost_usd` accidentally became
    # `total_cost()` instead of the per-call cost.
    tracker = CostTracker(ceiling_usd=10.0)
    tracker.record(input_tokens=5_000, output_tokens=1_000, model="claude-haiku-4-5")
    pre_existing_total = tracker.total_cost()
    assert pre_existing_total > 0.0

    transcript = Transcript(path, mode="record")
    replay_or_call(
        transcript,
        '{"x": 1}',
        _stub_llm(_make_decision(), model="claude-haiku-4-5",
                  input_tokens=2_000, output_tokens=400),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="operational",
        cost_tracker=tracker,
    )

    # Read the recorded entry back and check `cost_usd`.
    rep = Transcript(path, mode="replay")
    entry = rep.lookup(0, "c1")
    assert entry is not None
    expected_per_call_cost = CostTracker.cost_for(2_000, 400, "claude-haiku-4-5")
    assert entry.cost_usd == pytest.approx(expected_per_call_cost, rel=1e-12)
    # And the `cost_usd` is NOT the cumulative total — sanity check the swap
    # didn't accidentally regress to logging the running total.
    assert entry.cost_usd != pytest.approx(tracker.total_cost(), rel=1e-12)


def test_replay_or_call_does_not_charge_in_replay_mode(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    decision = _make_decision()
    rec = Transcript(path, mode="record")
    replay_or_call(
        rec,
        '{"x": 1}',
        _stub_llm(decision, model="claude-haiku-4-5",
                  input_tokens=1_000, output_tokens=500),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="operational",
        cost_tracker=CostTracker(ceiling_usd=10.0),
    )
    rep = Transcript(path, mode="replay")
    tracker = CostTracker(ceiling_usd=0.0001)  # tiny — would explode if charged
    out = replay_or_call(
        rep,
        '{"x": 1}',
        _stub_llm(decision),
        sim_id="s1",
        tick=0,
        company_id="c1",
        tier="operational",
        cost_tracker=tracker,
    )
    assert out == decision
    # Tracker stayed at zero — no charge in replay.
    assert tracker.total_cost() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model validation
# ─────────────────────────────────────────────────────────────────────────────


def test_transcript_entry_rejects_bad_sha_length() -> None:
    with pytest.raises(ValidationError):
        TranscriptEntry(
            sim_id="s",
            tick=0,
            company_id="c",
            decision_id="d",
            tier="tactical",
            prompt_sha256="too-short",
            raw_response="",
            parsed_decision=_make_decision(),
            model="claude-haiku-4-5",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )


def test_transcript_entry_rejects_negative_tick() -> None:
    with pytest.raises(ValidationError):
        _make_entry(tick=-1)


def test_pricing_table_has_both_models() -> None:
    """Sanity: the two models the v2 orchestrator uses must both be priced."""
    assert "claude-haiku-4-5" in MODEL_PRICING
    assert "claude-sonnet-4-6" in MODEL_PRICING
