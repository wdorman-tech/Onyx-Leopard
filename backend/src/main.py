from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.company import router as company_router
from src.routes.simulation import router as simulation_router
from src.routes.scenarios import router as scenarios_router
from src.routes.settings import router as settings_router
from src.routes.onboarding import router as onboarding_router
from src.routes.documents import router as documents_router
from src.routes.export import router as export_router
from src.routes.import_profile import router as import_router

app = FastAPI(title="Black Jaguar", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(company_router)
app.include_router(simulation_router)
app.include_router(scenarios_router)
app.include_router(settings_router)
app.include_router(onboarding_router)
app.include_router(documents_router)
app.include_router(export_router)
app.include_router(import_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
