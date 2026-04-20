"""API routes for the AI-powered business profile builder."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from src.simulation.industries import commit_industry
from src.simulation.profile_builder import (
    analyze_niche as _analyze_niche,
    generate_spec,
    get_first_question,
    get_session,
    process_answer,
    process_upload,
    start_session,
)

router = APIRouter(prefix="/api/profile", tags=["profile"])


class AnswerRequest(BaseModel):
    answer: str


class ConfirmRequest(BaseModel):
    slug: str


class NicheRequest(BaseModel):
    company_name: str
    description: str


@router.post("/analyze-niche")
async def analyze_niche_route(request: NicheRequest) -> dict:
    """Analyze a company description and identify the business niche."""
    from src.simulation.ceo_agent import validate_api_key

    ok, err = await validate_api_key()
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    try:
        result = await _analyze_niche(request.company_name, request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_interview() -> dict:
    """Start a new business profile interview."""
    session = start_session()
    first_question = await get_first_question(session)
    return {
        "session_id": session.id,
        "first_question": first_question,
    }


@router.post("/{session_id}/answer")
async def submit_answer(session_id: str, request: AnswerRequest) -> dict:
    """Submit an answer and get the next question or generated spec."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("interviewing",):
        raise HTTPException(
            status_code=400, detail=f"Session is {session.status}, not interviewing"
        )

    result = await process_answer(session, request.answer)

    # If interview is complete, auto-generate the spec
    if result["is_complete"]:
        try:
            spec_dict = await generate_spec(session)
            result["industry_spec"] = spec_dict
        except ValueError as e:
            result["error"] = str(e)

    return result


@router.get("/{session_id}")
async def get_session_state(session_id: str) -> dict:
    """Get the current state of an interview session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.id,
        "status": session.status,
        "transcript": session.transcript,
        "industry_spec": session.generated_spec,
        "error": session.error,
    }


@router.post("/{session_id}/confirm")
async def confirm_industry(session_id: str, request: ConfirmRequest) -> dict:
    """Confirm and save the generated industry spec."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "complete" or session.generated_spec is None:
        raise HTTPException(
            status_code=400, detail="No generated spec to confirm"
        )

    slug = request.slug.lower().replace(" ", "_").replace("-", "_")
    if not slug.isidentifier():
        raise HTTPException(
            status_code=400, detail="Invalid slug — use only letters, numbers, underscores"
        )

    # Ensure the slug in the spec matches; profile-builder slugs are prefixed
    # with `custom_` to keep them visually separate from curated/adaptive ones.
    namespaced = f"custom_{slug}"
    session.generated_spec["meta"]["slug"] = namespaced
    session.generated_spec["meta"]["playable"] = True

    path = await commit_industry(namespaced, session.generated_spec)

    return {
        "slug": namespaced,
        "path": str(path),
        "message": f"Industry '{namespaced}' saved and ready for simulation",
    }


@router.post("/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile) -> dict:
    """Upload a business document for analysis during the interview."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("interviewing", "generating"):
        raise HTTPException(
            status_code=400, detail=f"Session is {session.status}, cannot upload"
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_bytes = await file.read()
    try:
        result = await process_upload(session, file.filename, file_bytes)
        return {
            "filename": result["filename"],
            "summary": result["summary"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
