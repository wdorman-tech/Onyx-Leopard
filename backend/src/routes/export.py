from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.schemas import CompanyGraph, CompanyProfile, OleoExport, SimulationSnapshot
from src.session_store import session_store
from src.simulation.manager import session_manager

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportProfileRequest(BaseModel):
    session_id: str


class ExportSimulationRequest(BaseModel):
    session_id: str
    simulation_session_id: str | None = None


def _build_export(
    profile: CompanyProfile,
    graph: CompanyGraph,
    snapshot: SimulationSnapshot | None = None,
) -> OleoExport:
    exported_at = datetime.now(timezone.utc).isoformat()

    export = OleoExport(
        format="onyx-leopard-export",
        format_version="1.0.0",
        exported_at=exported_at,
        profile=profile,
        graph=graph,
        simulation_snapshot=snapshot,
        checksum="",
    )

    # Compute checksum over everything except the checksum field itself
    payload = export.model_dump_json(exclude={"checksum"})
    checksum = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
    export.checksum = checksum

    return export


@router.post("/profile")
async def export_profile(request: ExportProfileRequest) -> OleoExport:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.graph is None:
        raise HTTPException(status_code=400, detail="No graph generated yet")

    return _build_export(session.profile, session.graph)


@router.post("/simulation")
async def export_simulation(request: ExportSimulationRequest) -> OleoExport:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.graph is None:
        raise HTTPException(status_code=400, detail="No graph generated yet")

    snapshot: SimulationSnapshot | None = None

    if request.simulation_session_id:
        sim_session = session_manager.get_session(request.simulation_session_id)
        if sim_session:
            state = sim_session.engine.state
            history_summary = [
                {
                    "tick": r.tick,
                    "global_metrics": r.global_metrics,
                    "action_count": len(r.actions),
                }
                for r in state.history[-20:]  # Last 20 ticks condensed
            ]
            snapshot = SimulationSnapshot(
                tick=state.tick,
                outlook=state.outlook,
                global_metrics=dict(state.graph.global_metrics),
                history_summary=history_summary,
            )

    return _build_export(session.profile, session.graph, snapshot)
