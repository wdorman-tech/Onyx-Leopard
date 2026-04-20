"""Monte Carlo simulation API — run N parameter-varied simulations and compare results."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.simulation.monte_carlo import MonteCarloRunner, run_single
from src.simulation.monte_carlo_models import MonteCarloConfig, ParameterVariation

router = APIRouter(prefix="/api/simulate/monte-carlo", tags=["monte-carlo"])

# In-memory batch sessions
_batches: dict[str, MonteCarloRunner] = {}


class MonteCarloStartRequest(BaseModel):
    industry: str = "restaurant"
    num_runs: int = Field(default=10, ge=2, le=100)
    ticks_per_run: int = Field(default=1825, ge=100)
    num_companies: int = Field(default=4, ge=1, le=20)
    start_mode: str = "identical"
    parameter_variations: list[ParameterVariation] = Field(default_factory=list)
    sample_interval: int = Field(default=30, ge=1)
    seed: int = 42


@router.post("/start")
async def start_monte_carlo(request: MonteCarloStartRequest) -> dict:
    mc_config = MonteCarloConfig(**request.model_dump())
    batch_id = str(uuid.uuid4())
    runner = MonteCarloRunner(mc_config)
    _batches[batch_id] = runner
    return {
        "batch_id": batch_id,
        "num_runs": mc_config.num_runs,
        "ticks_per_run": mc_config.ticks_per_run,
    }


@router.get("/stream/{batch_id}")
async def stream_monte_carlo(batch_id: str):
    runner = _batches.get(batch_id)
    if runner is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'detail': 'Batch not found'})}\n\n"]),
            media_type="text/event-stream",
        )

    async def event_generator():
        try:
            configs = runner.configs

            for i, (config, varied, seed) in enumerate(configs):
                started = {"type": "run_started", "run_index": i, "varied_params": varied}
                yield f"data: {json.dumps(started)}\n\n"

                # Offload CPU-bound simulation to a worker thread so the
                # FastAPI event loop stays responsive for other requests.
                result = await asyncio.to_thread(
                    run_single,
                    i, config, varied, seed, runner.mc_config.sample_interval,
                )
                runner.results.append(result)
                runner.completed_count += 1

                percent = round(100 * runner.completed_count / runner.total_runs, 1)
                done = {
                    "type": "run_complete",
                    "run_index": i,
                    "percent": percent,
                    "summary": result.model_dump(),
                }
                yield f"data: {json.dumps(done)}\n\n"

            report = runner.build_report()
            runner.is_complete = True
            batch_done = {"type": "batch_complete", "report": report.model_dump()}
            yield f"data: {json.dumps(batch_done)}\n\n"
        finally:
            _batches.pop(batch_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
