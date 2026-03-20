from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.simulation.manager import session_manager

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


@router.post("/fork/{session_id}")
async def fork_scenario(session_id: str) -> dict:
    new_session = session_manager.fork_session(session_id)
    if new_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": new_session.id, "forked_from": session_id}


@router.get("/compare")
async def compare_scenarios(session_ids: str) -> dict:
    ids = [s.strip() for s in session_ids.split(",")]
    results = {}
    for sid in ids:
        session = session_manager.get_session(sid)
        if session is None:
            continue
        results[sid] = {
            "tick": session.engine.state.tick,
            "global_metrics": dict(session.engine.state.graph.global_metrics),
            "history": [
                {"tick": r.tick, "global_metrics": r.global_metrics}
                for r in session.engine.state.history
            ],
        }
    return {"scenarios": results}
