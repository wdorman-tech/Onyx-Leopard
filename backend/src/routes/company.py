from fastapi import APIRouter, HTTPException

from src.agents.parser import parse_company, refine_company
from src.schemas import CompanyGraph, ParseRequest, RefineRequest

router = APIRouter(prefix="/api", tags=["company"])


@router.post("/parse-company", response_model=CompanyGraph)
async def parse_company_endpoint(request: ParseRequest) -> CompanyGraph:
    try:
        return parse_company(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse company: {e}") from e


@router.post("/refine-company", response_model=CompanyGraph)
async def refine_company_endpoint(request: RefineRequest) -> CompanyGraph:
    try:
        return refine_company(request.message, request.current_graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refine company: {e}") from e
