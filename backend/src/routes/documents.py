from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.agents.document_processor import (
    classify_document,
    create_job,
    extract_financial_data,
    get_job,
    map_extraction_to_profile,
    merge_into_profile,
)
from src.agents.edgar_client import lookup_company, map_to_profile, pull_financials
from src.schemas import CompanyProfile
from src.session_store import session_store

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "image/png",
    "image/jpeg",
}


class UploadResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    category: str | None = None
    error: str | None = None


class JobResultResponse(BaseModel):
    job_id: str
    category: str | None = None
    extraction: dict | None = None
    profile_fields: dict | None = None


class ConfirmRequest(BaseModel):
    session_id: str


class ConfirmResponse(BaseModel):
    profile: CompanyProfile
    merged_fields: list[str]


class EdgarLookupResponse(BaseModel):
    company_info: dict | None = None
    financials: dict | None = None
    profile_preview: CompanyProfile | None = None
    error: str | None = None


class EdgarConfirmRequest(BaseModel):
    session_id: str
    ticker: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    # Validate file type
    content_type = file.content_type or ""
    file_name = file.filename or "unknown"

    # Allow by extension if content_type is generic
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    ext_types = {"pdf": "application/pdf", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "csv": "text/csv", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}

    if content_type not in ALLOWED_TYPES and ext not in ext_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    if ext in ext_types and content_type not in ALLOWED_TYPES:
        content_type = ext_types[ext]

    # Read and validate size
    raw_data = await file.read()
    if len(raw_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")

    job = create_job(file_name, content_type, raw_data)

    # Process synchronously for MVP (could be async background task)
    try:
        classify_document(job)
        extract_financial_data(job)
        if job.category and job.extraction:
            job.profile_fields = map_extraction_to_profile(job.category, job.extraction)
    except Exception as e:
        job.status = "error"
        job.error = str(e)

    if job.status == "error":
        raise HTTPException(status_code=422, detail=job.error or "Document processing failed")

    return UploadResponse(job_id=job.id, status=job.status)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        category=job.category,
        error=job.error,
    )


@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str) -> JobResultResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Job not complete (status: {job.status})")

    return JobResultResponse(
        job_id=job.id,
        category=job.category,
        extraction=job.extraction,
        profile_fields=job.profile_fields,
    )


@router.post("/confirm/{job_id}", response_model=ConfirmResponse)
async def confirm_and_merge(job_id: str, request: ConfirmRequest) -> ConfirmResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.extraction:
        raise HTTPException(status_code=400, detail="Job has no extraction results")

    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    merged_profile = merge_into_profile(session.profile, job.category or "custom", job.extraction)
    session.profile = merged_profile
    session.uploaded_docs.append({"name": job.file_name, "category": job.category, "job_id": job.id})

    merged_fields = list((job.profile_fields or {}).keys())

    return ConfirmResponse(profile=merged_profile, merged_fields=merged_fields)


@router.get("/edgar/{ticker}", response_model=EdgarLookupResponse)
async def edgar_lookup(ticker: str) -> EdgarLookupResponse:
    company_info = lookup_company(ticker)
    if company_info is None:
        return EdgarLookupResponse(error=f"Company not found for ticker: {ticker}")

    financial_data = pull_financials(ticker)
    if "error" in financial_data:
        return EdgarLookupResponse(
            company_info=company_info,
            error=financial_data["error"],
        )

    preview_profile = map_to_profile(company_info, financial_data)

    return EdgarLookupResponse(
        company_info=company_info,
        financials=financial_data.get("financials"),
        profile_preview=preview_profile,
    )


@router.post("/edgar/confirm", response_model=ConfirmResponse)
async def edgar_confirm(request: EdgarConfirmRequest) -> ConfirmResponse:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    company_info = lookup_company(request.ticker)
    if company_info is None:
        raise HTTPException(status_code=404, detail=f"Company not found: {request.ticker}")

    financial_data = pull_financials(request.ticker)

    if "error" in financial_data:
        raise HTTPException(status_code=400, detail=financial_data["error"])

    merged_profile = map_to_profile(company_info, financial_data, base_profile=session.profile)
    session.profile = merged_profile
    session.uploaded_docs.append({"name": f"SEC EDGAR: {request.ticker.upper()}", "category": "10k_annual"})

    merged_fields = list(financial_data.get("financials", {}).keys())

    return ConfirmResponse(profile=merged_profile, merged_fields=merged_fields)
