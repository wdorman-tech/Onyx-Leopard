from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.simulation.industries import INDUSTRY_REGISTRY
from src.simulation.manager import session_manager
from src.simulation.market.presets import MARKET_PRESETS
from src.simulation.unified import UnifiedEngine
from src.simulation.unified_models import UnifiedStartConfig

router = APIRouter(prefix="/api/simulate", tags=["simulation"])


class StartRequest(BaseModel):
    max_ticks: int = 0
    industry: str = "restaurant"
    mode: str = "growth"  # "growth" | "market" | "unified"
    preset: str | None = None
    # Unified mode options
    start_mode: str = "identical"  # "identical" | "randomized" | "staggered"
    num_companies: int = 4


class ControlRequest(BaseModel):
    action: str  # play, pause, set_speed, focus_company
    speed: float | None = None
    company_id: str | None = None  # for focus_company action


@router.get("/industries")
async def list_industries() -> list[dict]:
    return [
        {
            "slug": cfg.slug,
            "name": cfg.name,
            "description": cfg.description,
            "icon": cfg.icon,
            "playable": cfg.playable,
            "total_nodes": cfg.total_nodes,
            "growth_stages": cfg.growth_stages,
            "key_metrics": list(cfg.key_metrics),
            "example_nodes": list(cfg.example_nodes),
            "categories": cfg.categories,
        }
        for cfg in INDUSTRY_REGISTRY.values()
    ]


@router.get("/market/presets")
async def list_market_presets() -> list[dict]:
    return [
        {
            "slug": preset.slug,
            "name": preset.name,
            "description": preset.description,
            "alpha": preset.params.alpha,
            "beta": preset.params.beta,
            "delta": preset.params.delta,
            "n_0": preset.params.n_0,
        }
        for preset in MARKET_PRESETS.values()
    ]


@router.post("/start")
async def start_simulation(request: StartRequest) -> dict:
    if request.mode == "market":
        if not request.preset or request.preset not in MARKET_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown market preset: {request.preset}")
        session = session_manager.create_session(
            max_ticks=request.max_ticks,
            mode="market",
            preset=request.preset,
        )
        return {"session_id": session.id}

    if request.mode == "unified":
        config = UnifiedStartConfig(
            start_mode=request.start_mode,
            num_companies=request.num_companies,
            max_ticks=request.max_ticks,
        )
        session = session_manager.create_session(
            mode="unified",
            unified_config=config,
        )
        return {"session_id": session.id}

    if request.industry not in INDUSTRY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown industry: {request.industry}")
    if not INDUSTRY_REGISTRY[request.industry].playable:
        raise HTTPException(status_code=400, detail=f"Industry not yet playable: {request.industry}")
    session = session_manager.create_session(
        max_ticks=request.max_ticks,
        industry=request.industry,
    )
    return {"session_id": session.id}


@router.get("/stream/{session_id}")
async def stream_simulation(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        session.play()
        last_tick = 0
        while not session.engine.is_complete:
            should_continue = await session.wait_if_paused()
            if not should_continue:
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                return

            result = session.engine.tick()
            last_tick = result.get("tick", last_tick)
            event_data = {"type": "tick", "mode": session.mode, **result}
            yield f"data: {json.dumps(event_data)}\n\n"
            await asyncio.sleep(session.speed)

        yield f"data: {json.dumps({'type': 'complete', 'tick': last_tick})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/control/{session_id}")
async def control_simulation(session_id: str, request: ControlRequest) -> dict:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    match request.action:
        case "play":
            session.play()
        case "pause":
            session.pause()
        case "set_speed":
            if request.speed is not None:
                session.set_speed(request.speed)
        case "focus_company":
            if not isinstance(session.engine, UnifiedEngine):
                raise HTTPException(status_code=400, detail="focus_company only works in unified mode")
            if request.company_id is not None:
                session.engine.focused_company_id = request.company_id
                # Return the graph immediately so the frontend doesn't wait for the next tick
                focused = next(
                    (c for c in session.engine.companies if c.state.name == request.company_id),
                    None,
                )
                if focused:
                    return {
                        "status": "ok",
                        "action": request.action,
                        "graph": focused.build_graph_snapshot().model_dump(),
                    }
        case _:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    return {"status": "ok", "action": request.action}
