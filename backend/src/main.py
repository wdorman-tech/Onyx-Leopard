import logging
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO)

from src.routes.simulation_v2 import router as simulation_v2_router

app = FastAPI(title="Onyx Leopard", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_v2_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # HTTPException carries an intentionally crafted message and status, pass it
    # through so route-level 400/404 still surface usable client errors.
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # Everything else is logged with full traceback server-side and returned
    # with a generic message so we never leak file paths, ORM internals,
    # or vendor stack details to the client.
    tb = traceback.format_exception(exc)
    logging.error("Unhandled error on %s:\n%s", request.url, "".join(tb))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
