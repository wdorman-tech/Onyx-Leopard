"""V2 simulation routes — seed + stance + library, no industry slugs.

Mounted at `/api/v2/simulate`. Lives in parallel with the v1 routes at
`/api/simulate` until wave 5 deletion. Front-end migrates to these endpoints
once it's ready to send seed/stance JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.simulation.library_loader import (
    LibraryValidationError,
    NodeLibrary,
    get_library,
)
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import (
    ARCHETYPES as SEED_ARCHETYPES,
    CompanySeed,
    sample_seed_for_archetype,
)
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import (
    ARCHETYPES as STANCE_ARCHETYPES,
    CeoStance,
    sample_stance,
)
from src.simulation.unified_v2 import (
    CompanyAgentV2,
    MultiCompanySimV2,
)

log = logging.getLogger(__name__)

_TARGET_FRAME_TIME = 0.016  # ~60fps


router = APIRouter(prefix="/api/v2/simulate", tags=["simulation_v2"])


# ─── Request / response models ────────────────────────────────────────────


class StartRequestV2(BaseModel):
    """Body of POST /api/v2/simulate/start."""

    seed: CompanySeed
    stance: CeoStance
    num_companies: int = Field(default=1, ge=1, le=20)
    duration_ticks: int = Field(default=365, ge=10, le=10_000)
    tam_initial: float = Field(default=1_000_000.0, ge=1.0)
    shock_lambdas: dict[str, float] = Field(default_factory=dict)
    transcript_dir: str | None = None
    cost_ceiling_usd: float = Field(default=10.0, ge=0.0)
    rng_seed: int | None = None


class ControlRequestV2(BaseModel):
    """Body of POST /api/v2/simulate/control/{session_id}."""

    action: Literal["play", "pause", "stop", "set_speed"]
    speed: float | None = None


# ─── In-process session manager ──────────────────────────────────────────


@dataclass
class SessionV2:
    id: str
    sim: MultiCompanySimV2
    speed: float = 0.05
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
        self.speed = max(0.001, speed)

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


def _build_sim(req: StartRequestV2, library: NodeLibrary) -> MultiCompanySimV2:
    """Construct a MultiCompanySimV2 from a validated StartRequest."""
    rng_master = random.Random(req.rng_seed if req.rng_seed is not None else None)
    sim_id = str(uuid.uuid4())

    transcript = None
    if req.transcript_dir:
        Path(req.transcript_dir).mkdir(parents=True, exist_ok=True)
        transcript = Transcript(
            path=Path(req.transcript_dir) / f"{sim_id}.jsonl",
            mode="record",
        )

    companies: list[CompanyAgentV2] = []
    for i in range(req.num_companies):
        scheduler = ShockScheduler(
            rng_seed=rng_master.randint(0, 2**31 - 1),
            lambdas=dict(req.shock_lambdas),
        )
        cost_tracker = CostTracker(ceiling_usd=req.cost_ceiling_usd)
        company = CompanyAgentV2(
            seed=req.seed,
            stance=req.stance,
            library=library,
            sim_id=sim_id,
            company_id=f"co-{i}",
            rng=random.Random(rng_master.randint(0, 2**31 - 1)),
            transcript=transcript,
            cost_tracker=cost_tracker,
            shock_scheduler=scheduler,
        )
        companies.append(company)

    return MultiCompanySimV2(
        sim_id=sim_id,
        companies=companies,
        max_ticks=req.duration_ticks,
        tam_initial=req.tam_initial,
    )


def _serialize_tick(step_result: dict) -> dict:
    """Convert MultiCompanySimV2.step() output to JSON-safe SSE payload."""
    return {
        "type": "tick",
        "tick": step_result["tick"],
        "tam": step_result["tam"],
        "alive": step_result["alive"],
        "shares": step_result.get("shares", []),
        "companies": [
            {
                "company_id": tr.tick,  # the step result iter — see below
            }
            for tr in step_result.get("results", [])
        ],
    }


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

                # Batch ticks at high speed to reduce SSE overhead
                ticks_per_frame = max(1, int(_TARGET_FRAME_TIME / max(session.speed, 0.001)))
                step_result = None
                for _ in range(ticks_per_frame):
                    if sim.is_complete:
                        break
                    step_result = await sim.step()
                    last_tick = step_result["tick"]

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
                        for c, tr in zip(sim.companies, step_result.get("results", []), strict=False)
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(session.speed)

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
