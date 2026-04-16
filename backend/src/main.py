import logging
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)

from src.routes.profile_builder import router as profile_router
from src.routes.simulation import router as simulation_router

app = FastAPI(title="Onyx Leopard", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router)
app.include_router(profile_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exception(exc)
    logging.error(f"Unhandled error on {request.url}:\n{''.join(tb)}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
