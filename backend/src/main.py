"""Onyx Leopard API: start a run, stream it over SSE, control playback."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import FileResponse, StreamingResponse

from src.simulation import export
from src.simulation.ceo import AiCeo, HeuristicCeo
from src.simulation.engine import Company, MarketSim
from src.simulation.jev import JevJudge, jev_key
from src.simulation.library import LibraryValidationError, get_library
from src.simulation.llm import API_KEY_ENV, Budget, OpenRouterClient, api_key, load_catalog
from src.simulation.report import CeoReport, write_report
from src.simulation.seed import (
    ARCHETYPES as COMPANY_ARCHETYPES,
)
from src.simulation.seed import (
    Archetype,
    CompanySeed,
    company_name,
    sample_seed,
    seed_from_oleo,
)
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import ARCHETYPES as CEO_ARCHETYPES
from src.simulation.stance import PROFILES, sample_stance

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

COMPANIES_DIR = BACKEND_DIR / "companies"
MAX_COMPANIES = 10
#: Target SSE frame interval; above 60 ticks/s, ticks are batched per frame.
FRAME_SECONDS = 1 / 60
MIN_SPEED = 0.1

app = FastAPI(title="Onyx Leopard", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    # Local-only app: any localhost port (lets a second checkout run side by side).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Sessions ───────────────────────────────────────────────────────────────


@dataclass
class Session:
    sim: MarketSim
    speed: float = 1.0
    stopped: bool = False
    running: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.running.set()


_sessions: dict[str, Session] = {}


def _session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


# ─── Models ────────────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Explicit company seed (e.g. an imported profile). Sampled when omitted.
    seed: CompanySeed | None = None
    company_archetype: Archetype = "small_team"
    #: One CEO archetype per company; every company shares the seed.
    ceo_archetypes: list[str] = Field(..., min_length=1, max_length=MAX_COMPANIES)
    duration_days: int = Field(default=5 * 365, ge=30, le=100 * 365)
    ai_enabled: bool = True
    model: str | None = None
    budget_usd: float = Field(default=1.0, gt=0.0, le=500.0)
    shocks_enabled: bool = True
    #: Let Jev approve or veto the weekly rule CEO (needs TYPESAFE_API_KEY).
    jev_enabled: bool = True
    rng_seed: int | None = None

    @field_validator("ceo_archetypes")
    @classmethod
    def _known_archetypes(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(CEO_ARCHETYPES))
        if unknown:
            raise ValueError(f"unknown CEO archetypes {unknown}; valid: {list(CEO_ARCHETYPES)}")
        return v


class ControlRequest(BaseModel):
    action: Literal["play", "pause", "stop", "set_speed"]
    speed: float | None = None


# ─── Routes ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "ai_available": api_key() is not None,
        "key_env": API_KEY_ENV,
        "jev_available": jev_key() is not None,
    }


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return (await load_catalog()).public()


@app.get("/api/archetypes")
async def archetypes() -> dict[str, Any]:
    return {
        "ceo": [
            {"id": k, "label": p["label"], "summary": p["summary"]} for k, p in PROFILES.items()
        ],
        "company": list(COMPANY_ARCHETYPES),
    }


@app.get("/api/companies")
async def companies() -> list[dict[str, Any]]:
    """Bundled `.oleo` company profiles, converted to seeds."""
    library = get_library()
    out = []
    for path in sorted(COMPANIES_DIR.glob("*.oleo")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            seed = seed_from_oleo(data)
            library.validate_seed(seed)
        except (ValueError, LibraryValidationError) as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        identity = data.get("profile", {}).get("identity", {})
        out.append(
            {
                "filename": path.name,
                "name": seed.name,
                "industry": identity.get("industry", ""),
                "seed": seed.model_dump(),
            }
        )
    return out


@app.post("/api/seed/sample/{archetype}")
async def sample(archetype: Archetype, rng_seed: int | None = None) -> dict[str, Any]:
    return sample_seed(archetype, random.Random(rng_seed), get_library()).model_dump()


@app.post("/api/simulate/start")
async def start(req: StartRequest) -> dict[str, Any]:
    sim = await build_sim(req)
    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(sim=sim)
    return {
        "session_id": session_id,
        "max_ticks": sim.max_ticks,
        "model": sim.llm.model.id if sim.llm else None,
        "companies": [
            {"company_id": c.company_id, "name": c.name, "archetype": c.stance.archetype}
            for c in sim.companies
        ],
    }


async def build_sim(req: StartRequest) -> MarketSim:
    rng = random.Random(req.rng_seed)
    library = get_library()
    seed = req.seed or sample_seed(req.company_archetype, rng, library)
    try:
        library.validate_seed(seed)
    except LibraryValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    llm: OpenRouterClient | None = None
    if req.ai_enabled:
        key = api_key()
        if key is None:
            raise HTTPException(
                400, f"AI CEOs need {API_KEY_ENV} in backend/.env. Rerun `python start.py`."
            )
        catalog = await load_catalog()
        llm = OpenRouterClient(
            api_key=key,
            model=catalog.info(req.model or catalog.default_model),
            budget=Budget(ceiling_usd=req.budget_usd),
            reasoning_effort=catalog.reasoning_effort,
        )

    jev = JevJudge.from_env() if req.jev_enabled and jev_key() is not None else None
    shock_rate = 1.0 if req.shocks_enabled else 0.0
    names = [seed.name]
    while len(names) < len(req.ceo_archetypes):
        candidate = company_name(seed.economics_model, rng)
        if candidate not in names:
            names.append(candidate)

    companies = [
        Company(
            company_id=f"co-{i}",
            name=names[i],
            seed=seed,
            stance=sample_stance(archetype, rng),
            library=library,
            rng=random.Random(rng.getrandbits(32)),
            shock_scheduler=ShockScheduler("company", rng.getrandbits(32), shock_rate),
        )
        for i, archetype in enumerate(req.ceo_archetypes)
    ]
    return MarketSim(
        sim_id=str(uuid.uuid4()),
        companies=companies,
        max_ticks=req.duration_days,
        tam=seed.tam,
        market_growth_rate=seed.market_growth_rate,
        market_shocks=ShockScheduler("market", rng.getrandbits(32), shock_rate),
        heuristic={c.company_id: HeuristicCeo() for c in companies},
        ai={c.company_id: AiCeo(llm, c) for c in companies} if llm else {},
        llm=llm,
        jev=jev,
        settings={
            "ai_enabled": llm is not None,
            "model": llm.model.id if llm else None,
            "budget_usd": req.budget_usd if llm else None,
            "jev_enabled": jev is not None,
            "shocks_enabled": req.shocks_enabled,
            "duration_days": req.duration_days,
            "company_archetype": None if req.seed else req.company_archetype,
            "rng_seed": req.rng_seed,
        },
        lv_seed=rng.getrandbits(32),
    )


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/api/simulate/stream/{session_id}")
async def stream(session_id: str) -> StreamingResponse:
    session = _session(session_id)
    return StreamingResponse(_run(session_id, session), media_type="text/event-stream")


async def _run(session_id: str, session: Session) -> AsyncIterator[str]:
    sim = session.sim
    try:
        while not sim.is_complete:
            await session.running.wait()
            if session.stopped:
                yield _sse({"type": "stopped"})
                return
            batch = max(1, int(session.speed * FRAME_SECONDS))
            ran = 0
            for _ in range(batch):
                if sim.is_complete:
                    break
                if sim.ai_due(sim.tick + 1):
                    if ran:  # flush progress before the AI round blocks
                        yield _sse(sim.payload())
                    yield _sse({"type": "thinking", "tick": sim.tick + 1})
                await sim.step()
                ran += 1
            yield _sse(sim.payload())
            await asyncio.sleep(ran / max(MIN_SPEED, session.speed))

        reports = await asyncio.gather(*(write_report(c, sim.llm) for c in sim.companies))
        yield _sse({"type": "reports", "reports": [r.model_dump() for r in reports]})
        yield _sse(await _save(sim, list(reports)))
        yield _sse({"type": "complete", "tick": sim.tick, "ai": sim.ai_status()})
    finally:
        if sim.llm is not None:
            await sim.llm.aclose()
        if sim.jev is not None:
            await sim.jev.aclose()
        _sessions.pop(session_id, None)


async def _save(sim: MarketSim, reports: list[CeoReport]) -> dict[str, Any]:
    """Write the run to disk (JSON, CSV, PDF) off the event loop; report where it went."""
    data = export.run_data(sim, reports)
    try:
        run_dir = await asyncio.to_thread(export.save_run, data)
    except Exception as exc:
        log.exception("saving run %s failed", data["run_id"])
        return {"type": "saved", "run_id": data["run_id"], "error": f"Saving the run failed: {exc}"}
    run_id = run_dir.name
    return {
        "type": "saved",
        "run_id": run_id,
        "files": {name: f"/api/runs/{run_id}/{name}" for name in export.RUN_FILES},
    }


@app.get("/api/runs/{run_id}/{filename}")
async def run_file(run_id: str, filename: str) -> FileResponse:
    path = export.run_file(run_id, filename)
    if path is None:
        raise HTTPException(404, "run file not found")
    disposition = "inline" if filename.endswith(".pdf") else "attachment"
    return FileResponse(
        path,
        media_type=export.RUN_FILES[filename],
        headers={"Content-Disposition": f'{disposition}; filename="{run_id}-{filename}"'},
    )


@app.post("/api/simulate/control/{session_id}")
async def control(session_id: str, req: ControlRequest) -> dict[str, Any]:
    session = _session(session_id)
    if req.action == "play":
        session.running.set()
    elif req.action == "pause":
        session.running.clear()
    elif req.action == "stop":
        session.stopped = True
        session.running.set()
    else:
        if req.speed is None:
            raise HTTPException(400, "speed is required for set_speed")
        session.speed = max(MIN_SPEED, req.speed)
    return {"paused": not session.running.is_set(), "speed": session.speed}
