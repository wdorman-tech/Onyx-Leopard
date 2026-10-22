"""V2 simulation routes — seed + stance + library, no industry slugs.

Mounted at `/api/v2/simulate`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from src.simulation.library_loader import (
    LibraryValidationError,
    NodeLibrary,
    get_library,
)
from src.simulation.post_mortem import PostMortemAgent
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import (
    ARCHETYPES as SEED_ARCHETYPES,
)
from src.simulation.seed import (
    CompanySeed,
    sample_seed_for_archetype,
)
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import (
    ARCHETYPES as STANCE_ARCHETYPES,
)
from src.simulation.stance import (
    CeoStance,
    sample_stance,
)
from src.simulation.unified_v2 import (
    CompanyAgentV2,
    MultiCompanySimV2,
)

log = logging.getLogger(__name__)

_TARGET_FRAME_TIME = 0.016  # ~60fps

# Sandbox base for any transcript_dir submitted by a client. Resolved
# absolute path so traversal attempts can be detected by prefix check.
# Override with `ONYX_TRANSCRIPT_BASE` if a different location is desired.
_DEFAULT_TRANSCRIPT_BASE = (
    Path(__file__).resolve().parent.parent.parent / "transcripts"
)
TRANSCRIPT_BASE = Path(
    os.environ.get("ONYX_TRANSCRIPT_BASE", str(_DEFAULT_TRANSCRIPT_BASE))
).resolve()


router = APIRouter(prefix="/api/v2/simulate", tags=["simulation_v2"])


# ─── Request / response models ────────────────────────────────────────────


class StartRequestV2(BaseModel):
    """Body of POST /api/v2/simulate/start."""

    seed: CompanySeed
    stance: CeoStance
    # Optional per-company overrides — when supplied each must have exactly
    # `num_companies` entries. Replaces the cloned single-seed pattern that
    # made the frontend "per-company strategy" UI a label-only feature.
    seeds: list[CompanySeed] | None = None
    stances: list[CeoStance] | None = None
    num_companies: int = Field(default=1, ge=1, le=20)
    duration_ticks: int = Field(default=365, ge=10, le=10_000)
    tam_initial: float = Field(default=1_000_000.0, ge=1.0)
    shock_lambdas: dict[str, float] = Field(default_factory=dict)
    # `transcript_dir` is a path RELATIVE to TRANSCRIPT_BASE. Absolute paths
    # and `..` traversal are rejected by the validator below to prevent the
    # client from writing anywhere on the filesystem.
    transcript_dir: str | None = None
    cost_ceiling_usd: float = Field(default=10.0, gt=0.0)
    rng_seed: int | None = None
    ai_ceo_enabled: bool = True

    @field_validator("transcript_dir")
    @classmethod
    def _validate_transcript_dir(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return v
        candidate = Path(v)
        if candidate.is_absolute():
            raise ValueError(
                "transcript_dir must be a relative path under the transcript base"
            )
        # `..` parts blow past the base. Reject before resolution so symlinks
        # cannot mask traversal either.
        if any(part == ".." for part in candidate.parts):
            raise ValueError("transcript_dir must not contain '..'")
        return v


class ControlRequestV2(BaseModel):
    """Body of POST /api/v2/simulate/control/{session_id}."""

    action: Literal["play", "pause", "stop", "set_speed"]
    speed: float | None = None


# ─── In-process session manager ──────────────────────────────────────────


@dataclass
class SessionV2:
    id: str
    sim: MultiCompanySimV2
    #: Playback multiplier — ticks per real second (UI ships 1, 2, 5, 10, 50,
    #: 100). The event loop converts this to a per-frame sleep. This used to
    #: be a sleep-delay-in-seconds, which silently inverted the UI's "Nx
    #: speed" semantic (100x meant 100s/tick — a hang, not a sprint).
    speed: float = 1.0
    paused: bool = False
    stopped: bool = False
    pause_event: asyncio.Event | None = None

    def __post_init__(self) -> None:
        if self.pause_event is None:
            self.pause_event = asyncio.Event()
            self.pause_event.set()  # not paused initially

    def play(self) -> None:
        self.paused = False
        if self.pause_event is not None:
            self.pause_event.set()

    def pause(self) -> None:
        self.paused = True
        if self.pause_event is not None:
            self.pause_event.clear()

    def stop(self) -> None:
        self.stopped = True
        # Always release any waiters so the SSE loop wakes up to see `stopped`
        if self.pause_event is not None:
            self.pause_event.set()

    def set_speed(self, speed: float) -> None:
        # `speed` is a multiplier (ticks per real second). Clamp at 0.1× so a
        # stray 0 from the UI doesn't divide-by-zero in the event loop.
        self.speed = max(0.1, speed)

    async def wait_if_paused(self) -> bool:
        """Block while paused. Returns False if the session was stopped."""
        if self.stopped:
            return False
        if self.pause_event is not None and not self.pause_event.is_set():
            await self.pause_event.wait()
        return not self.stopped


class _SessionRegistry:
    """In-memory registry; not thread-safe across worker processes."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionV2] = {}

    def create(self, sim: MultiCompanySimV2) -> SessionV2:
        sid = str(uuid.uuid4())
        s = SessionV2(id=sid, sim=sim)
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> SessionV2 | None:
        return self._sessions.get(sid)

    def remove(self, sid: str) -> None:
        self._sessions.pop(sid, None)


_registry = _SessionRegistry()


# ─── Helpers ────────────────────────────────────────────────────────────


def _resolve_transcript_dir(rel: str) -> Path:
    """Resolve a client-supplied relative path under TRANSCRIPT_BASE.

    Raises HTTPException(400) if resolution escapes the base directory —
    catches symlink and case-folding traversal that the field validator
    cannot see at request-parse time.
    """
    candidate = (TRANSCRIPT_BASE / rel).resolve()
    try:
        candidate.relative_to(TRANSCRIPT_BASE)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="transcript_dir escapes the sandbox",
        ) from exc
    return candidate


def _deterministic_sim_id(rng_seed: int | None) -> str:
    """Derive a stable sim_id from `rng_seed` so reproducible runs share it.

    When `rng_seed` is None we fall back to a random `uuid4` — there is no
    reproducibility contract in that case anyway.
    """
    if rng_seed is None:
        return str(uuid.uuid4())
    digest = hashlib.blake2b(
        f"sim:{rng_seed}".encode(), digest_size=8
    ).hexdigest()
    return f"sim-{digest}"


def _build_sim(req: StartRequestV2, library: NodeLibrary) -> MultiCompanySimV2:
    """Construct a MultiCompanySimV2 from a validated StartRequest."""
    # Determinism contract: identical `rng_seed` ⇒ identical sim_id, identical
    # per-company RNG draws, identical LV competition matrix. The previous
    # `uuid.uuid4()` sim_id plus `hash(sim_id)` LV seeding broke this — the
    # process-random `hash` made reruns non-reproducible even within one
    # process. Both are now derived from `rng_seed`.
    rng_master = random.Random(req.rng_seed if req.rng_seed is not None else None)
    sim_id = _deterministic_sim_id(req.rng_seed)

    transcript = None
    if req.transcript_dir:
        sandboxed = _resolve_transcript_dir(req.transcript_dir)
        sandboxed.mkdir(parents=True, exist_ok=True)
        transcript = Transcript(
            path=sandboxed / f"{sim_id}.jsonl",
            mode="record",
        )

    # Validate per-company seed/stance lengths early so the engine doesn't
    # see a half-built company list. Allows clients to send a single
    # `seed`/`stance` (auto-cloned) OR explicit per-company arrays.
    if req.seeds is not None and len(req.seeds) != req.num_companies:
        raise HTTPException(
            status_code=400,
            detail=(
                f"seeds length ({len(req.seeds)}) must match "
                f"num_companies ({req.num_companies})"
            ),
        )
    if req.stances is not None and len(req.stances) != req.num_companies:
        raise HTTPException(
            status_code=400,
            detail=(
                f"stances length ({len(req.stances)}) must match "
                f"num_companies ({req.num_companies})"
            ),
        )

    companies: list[CompanyAgentV2] = []
    for i in range(req.num_companies):
        company_seed = req.seeds[i] if req.seeds is not None else req.seed
        company_stance = req.stances[i] if req.stances is not None else req.stance
        try:
            library.validate_seed(company_seed)
        except LibraryValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"seed[{i}] failed library validation: {e}",
            ) from e
        scheduler = ShockScheduler(
            rng_seed=rng_master.randint(0, 2**31 - 1),
            lambdas=dict(req.shock_lambdas),
        )
        cost_tracker = CostTracker(ceiling_usd=req.cost_ceiling_usd)
        company = CompanyAgentV2(
            seed=company_seed,
            stance=company_stance,
            library=library,
            sim_id=sim_id,
            company_id=f"co-{i}",
            rng=random.Random(rng_master.randint(0, 2**31 - 1)),
            transcript=transcript,
            cost_tracker=cost_tracker,
            shock_scheduler=scheduler,
            ai_ceo_enabled=req.ai_ceo_enabled,
        )
        companies.append(company)

    # Seed `market_growth_rate` is annualized; convert to per-tick (1 tick = 1 day).
    # The market grows at the FIRST company's rate when no per-company seeds are
    # supplied; this matches prior single-seed behavior. When per-company seeds
    # are present, we pick the median growth rate so one outlier company does
    # not warp the shared market.
    if req.seeds is not None:
        rates = sorted(s.market_growth_rate for s in req.seeds)
        annual_growth = rates[len(rates) // 2]
    else:
        annual_growth = req.seed.market_growth_rate
    daily_growth = annual_growth / 365.0
    # Pull an LV seed from rng_master so the competition matrix is also
    # reproducible from the upstream `rng_seed`.
    lv_seed = rng_master.randint(0, 2**31 - 1)
    return MultiCompanySimV2(
        sim_id=sim_id,
        companies=companies,
        max_ticks=req.duration_ticks,
        tam_initial=req.tam_initial,
        market_growth_rate=daily_growth,
        lv_rng_seed=lv_seed,
    )


# ─── Routes ────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "v2"}


@router.get("/library")
async def list_library() -> dict:
    """Return node library summary for frontend UI hints."""
    try:
        lib = get_library()
    except LibraryValidationError as e:
        raise HTTPException(status_code=500, detail=f"Library failed validation: {e}") from e
    return {
        "node_count": len(lib.nodes),
        "categories": sorted({n.category for n in lib.nodes.values()}),
        "nodes": {
            key: {
                "label": node.label,
                "category": node.category,
                "hire_cost": node.hire_cost,
                "daily_fixed_costs": node.daily_fixed_costs,
                "modifier_keys": dict(node.modifier_keys),
                "prerequisites": list(node.prerequisites),
                "applicable_economics": list(node.applicable_economics),
                "soft_cap": node.category_caps.soft_cap,
                "hard_cap": node.category_caps.hard_cap,
            }
            for key, node in sorted(lib.nodes.items())
        },
    }


@router.get("/archetypes")
async def list_archetypes() -> dict:
    """Available seed and stance archetypes for the interview UI."""
    return {
        "seed_archetypes": list(SEED_ARCHETYPES),
        "stance_archetypes": list(STANCE_ARCHETYPES),
    }


@router.post("/seed/sample/{archetype}")
async def sample_seed_endpoint(archetype: str, rng_seed: int | None = None) -> dict:
    """Generate a sample CompanySeed for the given archetype."""
    if archetype not in SEED_ARCHETYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown seed archetype {archetype!r}. Valid: {SEED_ARCHETYPES}",
        )
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    seed = sample_seed_for_archetype(archetype, rng=rng)
    return seed.model_dump()


@router.post("/stance/sample/{archetype}")
async def sample_stance_endpoint(archetype: str, rng_seed: int | None = None) -> dict:
    """Generate a sample CeoStance for the given archetype."""
    if archetype not in STANCE_ARCHETYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stance archetype {archetype!r}. Valid: {STANCE_ARCHETYPES}",
        )
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    stance = sample_stance(archetype, rng=rng)
    return stance.model_dump()


@router.post("/start")
async def start_v2(request: StartRequestV2) -> dict:
    """Create a v2 simulation session. Returns session_id for the SSE stream."""
    try:
        library = get_library()
    except LibraryValidationError as e:
        raise HTTPException(status_code=500, detail=f"Library failed validation: {e}") from e

    try:
        library.validate_seed(request.seed)
    except LibraryValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sim = _build_sim(request, library)
    session = _registry.create(sim)
    return {
        "session_id": session.id,
        "sim_id": sim.sim_id,
        "num_companies": len(sim.companies),
        "max_ticks": sim.max_ticks,
    }


@router.get("/stream/{session_id}")
async def stream_v2(session_id: str):
    session = _registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        session.play()
        last_tick = 0
        sim = session.sim

        try:
            while not sim.is_complete:
                proceed = await session.wait_if_paused()
                if not proceed:
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    return

                # `session.speed` is a ticks-per-second multiplier from the UI.
                # Above 60×, batch multiple ticks into one frame so we stay
                # under ~60 fps SSE pace; at or below 60×, one tick per frame.
                multiplier = max(0.1, session.speed)
                ticks_per_frame = max(1, int(multiplier * _TARGET_FRAME_TIME))
                step_result = None
                ticks_run = 0
                for _ in range(ticks_per_frame):
                    if sim.is_complete:
                        break
                    step_result = await sim.step()
                    last_tick = step_result["tick"]
                    ticks_run += 1

                if step_result is None:
                    break

                # Build payload from step_result
                payload = {
                    "type": "tick",
                    "tick": step_result["tick"],
                    "tam": step_result["tam"],
                    "alive": step_result["alive"],
                    "shares": step_result.get("shares", []),
                    "companies": [
                        {
                            "company_id": c.company_id,
                            "tick": tr.tick,
                            "cash": tr.cash,
                            "daily_revenue": tr.daily_revenue,
                            "daily_costs": tr.daily_costs,
                            "monthly_revenue": tr.monthly_revenue,
                            "capacity_utilization": tr.capacity_utilization,
                            "avg_satisfaction": tr.avg_satisfaction,
                            "employee_count": tr.employee_count,
                            "spawned_nodes": tr.spawned_nodes,
                            "bankrupt": tr.bankrupt,
                            "active_shocks": [s.model_dump() for s in tr.active_shocks],
                            "arriving_shocks": [s.model_dump() for s in tr.arriving_shocks],
                            "decisions": [d.model_dump() for d in tr.decisions],
                            "graph": c.to_graph_snapshot(),
                        }
                        for c, tr in zip(
                            sim.companies,
                            step_result.get("results", []),
                            strict=False,
                        )
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                # Sleep just long enough to maintain the target ticks-per-second.
                # `ticks_run` accounts for the batch case (above 60×); for the
                # common 1-tick-per-frame case this is simply 1 / multiplier.
                if ticks_run > 0:
                    await asyncio.sleep(ticks_run / multiplier)

            # End-of-run CEO reports — one per company, every company gets
            # one regardless of survival. Best-effort: any failure is logged
            # and the report falls back to a heuristic, so the SSE stream
            # always closes cleanly with a `reports` event before `complete`.
            try:
                agent = PostMortemAgent()
                report_objs = await asyncio.gather(
                    *(agent.generate(c) for c in sim.companies),
                    return_exceptions=True,
                )
                reports = []
                for c, r in zip(sim.companies, report_objs, strict=False):
                    if isinstance(r, BaseException):
                        log.warning(
                            "post-mortem failed for company=%s: %s — "
                            "skipping report",
                            c.company_id,
                            r,
                        )
                        continue
                    reports.append(r.model_dump())
                yield f"data: {json.dumps({'type': 'reports', 'reports': reports})}\n\n"
            except Exception as exc:  # truly unexpected — never block complete
                log.warning("post-mortem batch failed: %s", exc)

            yield f"data: {json.dumps({'type': 'complete', 'tick': last_tick})}\n\n"
        finally:
            _registry.remove(session_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/control/{session_id}")
async def control_v2(session_id: str, request: ControlRequestV2) -> dict:
    session = _registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.action == "play":
        session.play()
    elif request.action == "pause":
        session.pause()
    elif request.action == "stop":
        session.stop()
    elif request.action == "set_speed":
        if request.speed is None:
            raise HTTPException(status_code=400, detail="speed is required for set_speed")
        session.set_speed(request.speed)

    return {
        "status": "ok",
        "action": request.action,
        "session_id": session_id,
        "paused": session.paused,
        "stopped": session.stopped,
        "speed": session.speed,
    }
