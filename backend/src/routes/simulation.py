from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.simulation.industries import INDUSTRY_REGISTRY
from src.simulation.manager import session_manager
from src.simulation.market.presets import MARKET_PRESETS
from src.simulation.config_loader import load_industry
from src.simulation.unified import UnifiedEngine
from src.simulation.unified_models import UnifiedStartConfig

_TARGET_FRAME_TIME = 0.016  # ~60fps, controls how many ticks batch per SSE event


def _normalize_growth_result(result: dict) -> dict:
    """Convert unified engine output to growth-mode format for single-company sims."""
    agent = result["agents"][0] if result.get("agents") else {}
    graphs = result.get("graphs", {})
    graph = next(iter(graphs.values()), {"nodes": [], "edges": []})

    return {
        "tick": result["tick"],
        "stage": agent.get("stage", 1),
        "status": "bankrupt" if not agent.get("alive", True) else result.get("status", "operating"),
        "metrics": {
            "cash": agent.get("cash", 0),
            "daily_revenue": agent.get("daily_revenue", 0),
            "daily_costs": agent.get("daily_costs", 0),
            "daily_profit": round(agent.get("daily_revenue", 0) - agent.get("daily_costs", 0), 2),
            "total_locations": agent.get("location_count", 0),
            "avg_satisfaction": agent.get("avg_satisfaction", 0),
            "total_employees": agent.get("total_employees", 0),
        },
        "events": result.get("events", []),
        "graph": graph,
    }

router = APIRouter(prefix="/api/simulate", tags=["simulation"])


class StartRequest(BaseModel):
    max_ticks: int = 0
    industry: str = "restaurant"
    mode: Literal["growth", "market", "unified", "adaptive"] = "growth"
    preset: str | None = None
    start_mode: Literal["identical", "randomized", "staggered"] = "identical"
    num_companies: int = 4
    ai_ceo_enabled: bool = False
    duration_years: int = 5
    company_strategies: dict[int, str] | None = None
    # Adaptive mode fields
    company_name: str | None = None
    niche_description: str | None = None
    niche_summary: str | None = None
    full_description: str | None = None
    economics_model: str | None = None


class ControlRequest(BaseModel):
    action: str  # play, pause, set_speed, focus_company
    speed: float | None = None
    company_id: str | None = None  # for focus_company action


@router.get("/config/{industry}")
async def get_industry_config(industry: str) -> dict:
    """Return industry-specific frontend configuration."""
    spec = load_industry(industry)
    return {
        "stage_labels": spec.display.stage_labels,
        "speed_presets": spec.display.speed_presets,
        "company_names": spec.display.company_names,
        "ceo_strategies": spec.display.ceo_strategies,
        "max_companies": spec.display.max_companies,
        "min_description_words": spec.display.min_description_words,
        "duration_options": spec.display.duration_options,
        "event_noise_filters": spec.display.event_noise_filters,
    }


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
    if request.mode == "adaptive":
        if not request.company_name or not request.niche_description:
            raise HTTPException(
                status_code=400,
                detail="company_name and niche_description are required for adaptive mode",
            )

        from src.simulation.ceo_agent import validate_api_key
        ok, err = await validate_api_key()
        if not ok:
            raise HTTPException(status_code=400, detail=err)

        from src.simulation.industries import commit_industry
        from src.simulation.profile_builder import (
            generate_competitor_names,
            generate_spec_from_niche,
        )

        # Generate industry spec from niche
        spec_dict = await generate_spec_from_niche(
            niche=request.niche_description,
            description=request.niche_summary or request.niche_description,
            full_description=request.full_description or "",
            economics_model=request.economics_model or "physical",
        )

        # Save and register the custom spec atomically (lock-protected).
        slug = f"adaptive_{spec_dict['meta']['slug']}"
        spec_dict["meta"]["slug"] = slug
        spec_dict["meta"]["playable"] = True
        await commit_industry(slug, spec_dict)

        # Generate competitor names
        num_competitors = request.num_companies - 1  # subtract user's company
        competitor_names: list[str] = []
        if num_competitors > 0:
            competitor_names = await generate_competitor_names(
                request.niche_description, num_competitors,
            )

        # Build custom name map: index 0 = user's company, 1..N = competitors
        custom_names: dict[int, str] = {0: request.company_name}
        for i, name in enumerate(competitor_names):
            custom_names[i + 1] = name

        max_ticks = 0
        if request.ai_ceo_enabled:
            spec = load_industry(slug)
            max_ticks = request.duration_years * spec.constants.ticks_per_year

        config = UnifiedStartConfig(
            industry=slug,
            start_mode=request.start_mode,
            num_companies=request.num_companies,
            max_ticks=max_ticks,
            ai_ceo_enabled=request.ai_ceo_enabled,
            duration_years=request.duration_years,
            company_strategies={
                int(k): v
                for k, v in (request.company_strategies or {}).items()
            },
            custom_company_names=custom_names,
        )
        session = session_manager.create_session(
            mode="unified",
            unified_config=config,
        )
        spec = load_industry(slug)
        return {
            "session_id": session.id,
            "spec_display": spec.display.model_dump(),
            "founder_type": spec.roles.founder_type,
            "competitor_names": competitor_names,
        }

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
        max_ticks = request.max_ticks
        if request.ai_ceo_enabled:
            from src.simulation.ceo_agent import validate_api_key
            ok, err = await validate_api_key()
            if not ok:
                raise HTTPException(status_code=400, detail=err)
            ceo_spec = load_industry(request.industry)
            max_ticks = request.duration_years * ceo_spec.constants.ticks_per_year

        config = UnifiedStartConfig(
            industry=request.industry,
            start_mode=request.start_mode,
            num_companies=request.num_companies,
            max_ticks=max_ticks,
            ai_ceo_enabled=request.ai_ceo_enabled,
            duration_years=request.duration_years,
            company_strategies={
                int(k): v
                for k, v in (request.company_strategies or {}).items()
            },
        )
        session = session_manager.create_session(
            mode="unified",
            unified_config=config,
        )
        spec = load_industry(request.industry)
        return {
            "session_id": session.id,
            "spec_display": spec.display.model_dump(),
            "founder_type": spec.roles.founder_type,
        }

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
        engine = session.engine
        is_unified = isinstance(engine, UnifiedEngine)

        while not engine.is_complete:
            should_continue = await session.wait_if_paused()
            if not should_continue:
                yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                return

            # At very high speeds, batch multiple ticks per SSE event
            # to reduce network overhead and make the sim feel faster.
            ticks_per_frame = max(1, int(_TARGET_FRAME_TIME / max(session.speed, 0.001)))
            for _ in range(ticks_per_frame):
                if engine.is_complete:
                    break

                if is_unified and engine.ai_ceo_enabled:
                    # Use async tick that handles CEO decisions inline
                    result = await engine.tick_with_agents()
                else:
                    result = engine.tick()
                last_tick = result.get("tick", last_tick)

                # Break batch if CEO decisions were made (give frontend time to render)
                if result.get("ceo_decisions"):
                    break

            if session.mode == "growth":
                result = _normalize_growth_result(result)

            # Include CEO decisions in the tick event itself
            ceo_decisions = result.pop("ceo_decisions", None)
            event_data = {"type": "tick", "mode": session.mode, **result}
            yield f"data: {json.dumps(event_data)}\n\n"

            if ceo_decisions:
                yield f"data: {json.dumps({'type': 'ceo_decisions', 'tick': last_tick, 'decisions': ceo_decisions})}\n\n"

            await asyncio.sleep(session.speed)

        # End-of-simulation reports for AI CEO mode (not applicable in growth mode)
        if session.mode != "growth" and is_unified and engine.ai_ceo_enabled:
            yield f"data: {json.dumps({'type': 'generating_reports'})}\n\n"
            reports = await engine.generate_reports()
            yield f"data: {json.dumps({'type': 'reports', 'reports': reports})}\n\n"

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
            if request.speed is None:
                raise HTTPException(status_code=400, detail="speed is required for set_speed action")
            session.set_speed(request.speed)
        case "focus_company":
            if not isinstance(session.engine, UnifiedEngine):
                raise HTTPException(status_code=400, detail="focus_company only works in unified mode")
            if request.company_id is not None:
                session.engine.focused_company_id = request.company_id
        case _:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    return {"status": "ok", "action": request.action}
