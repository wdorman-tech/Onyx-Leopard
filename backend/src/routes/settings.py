from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ApiKeyRequest(BaseModel):
    api_key: str


class StatusResponse(BaseModel):
    api_key_set: bool


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    return StatusResponse(api_key_set=bool(settings.anthropic_api_key))


@router.post("/api-key", response_model=StatusResponse)
async def set_api_key(request: ApiKeyRequest) -> StatusResponse:
    settings.anthropic_api_key = request.api_key
    return StatusResponse(api_key_set=True)
