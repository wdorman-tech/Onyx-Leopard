from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.schemas import CompanyGraph, SimulationParameters
from src.simulation.manager import session_manager

router = APIRouter(prefix="/api/simulate", tags=["simulation"])


class StartRequest(BaseModel):
    graph: CompanyGraph
    max_ticks: int = 50
    outlook: str = "normal"
    sim_params: SimulationParameters | None = None


class ControlRequest(BaseModel):
    action: str  # play, pause, set_speed, set_outlook
    speed: float | None = None
    outlook: str | None = None


class InjectRequest(BaseModel):
    description: str
    params: dict = {}


@router.post("/start")
async def start_simulation(request: StartRequest) -> dict:
    session = session_manager.create_session(
        request.graph,
        max_ticks=request.max_ticks,
        outlook=request.outlook,
        sim_params=request.sim_params,
    )
    return {"session_id": session.id, "outlook": request.outlook}


@router.get("/stream/{session_id}")
async def stream_simulation(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        session.play()
        while not session.engine.is_complete:
            should_continue = await session.wait_if_paused()
            if not should_continue:
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                return

            result = await session.engine.tick()
            event_data = {
                "type": "tick",
                "tick": result.tick,
                "graph": result.graph.model_dump(),
                "actions": [asdict(a) for a in result.actions],
                "global_metrics": result.global_metrics,
            }
            if result.bio_summary:
                event_data["bio_summary"] = result.bio_summary
            yield f"data: {json.dumps(event_data)}\n\n"
            await asyncio.sleep(session.speed)

        yield f"data: {json.dumps({'type': 'complete', 'tick': session.engine.state.tick})}\n\n"

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
        case "set_outlook":
            if request.outlook is not None:
                session.engine.state.outlook = request.outlook
        case _:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    return {"status": "ok", "action": request.action}


@router.post("/inject/{session_id}")
async def inject_event(session_id: str, request: InjectRequest) -> dict:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session.engine.inject_event({"description": request.description, **request.params})
    return {"status": "ok", "event": request.description}
