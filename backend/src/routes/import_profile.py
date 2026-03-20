from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from src.migrations import CURRENT_VERSION, SUPPORTED_VERSIONS, migrate, needs_migration
from src.schemas import CompanyGraph, CompanyProfile, OleoExport
from src.session_store import session_store

router = APIRouter(prefix="/api/import", tags=["import"])

MAX_NODES = 20


class ValidateRequest(BaseModel):
    data: dict


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
    needs_migration: bool
    preview: dict | None = None


class LoadRequest(BaseModel):
    data: dict
    target_session_id: str | None = None
    mode: str = "replace"  # replace | side_by_side


class LoadResponse(BaseModel):
    session_id: str
    profile: CompanyProfile
    graph: CompanyGraph


def _validate_oleo(data: dict) -> tuple[list[str], list[str]]:
    """Validate an .oleo data dict. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Format field
    if data.get("format") != "onyx-leopard-export":
        errors.append("Missing or invalid 'format' field — expected 'onyx-leopard-export'")

    # 2. Version
    version = data.get("format_version", "")
    if version not in SUPPORTED_VERSIONS:
        errors.append(f"Unsupported format_version '{version}'. Supported: {', '.join(sorted(SUPPORTED_VERSIONS))}")

    # 3. Required top-level fields
    for field in ("profile", "graph"):
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors, warnings

    # 4. Checksum — must use same serialization as _build_export in export.py
    checksum = data.get("checksum", "")
    if checksum:
        try:
            temp_export = OleoExport(**{**data, "checksum": ""})
            payload = temp_export.model_dump_json(exclude={"checksum"})
            expected = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
            if checksum != expected:
                warnings.append("Checksum mismatch — file may have been modified")
        except Exception:
            warnings.append("Could not verify checksum")

    # 5. Node count
    graph_data = data.get("graph", {})
    nodes = graph_data.get("nodes", [])
    if len(nodes) > MAX_NODES:
        errors.append(f"Node count {len(nodes)} exceeds maximum of {MAX_NODES}")

    # 6. Validate edge references
    node_ids = {n.get("id") for n in nodes}
    edges = graph_data.get("edges", [])
    for i, edge in enumerate(edges):
        if edge.get("source") not in node_ids:
            errors.append(f"Edge {i}: source '{edge.get('source')}' references unknown node")
        if edge.get("target") not in node_ids:
            errors.append(f"Edge {i}: target '{edge.get('target')}' references unknown node")

    # 7. Financial sanity checks
    profile_data = data.get("profile", {})
    financials = profile_data.get("financials", {})
    revenue = financials.get("annual_revenue", 0)
    if revenue < 0:
        errors.append("Negative annual_revenue is not valid")
    gross_margin = financials.get("gross_margin", 0)
    if gross_margin < 0 or gross_margin > 1:
        warnings.append(f"Gross margin {gross_margin} outside expected range [0, 1]")

    return errors, warnings


@router.post("/validate", response_model=ValidateResponse)
async def validate_import(request: ValidateRequest) -> ValidateResponse:
    data = request.data
    errors, warnings = _validate_oleo(data)

    requires_migration = False
    version = data.get("format_version", "")
    if version in SUPPORTED_VERSIONS and version != CURRENT_VERSION:
        requires_migration = True
        warnings.append(f"File version {version} will be migrated to {CURRENT_VERSION}")

    preview = None
    if not errors:
        profile_data = data.get("profile", {})
        identity = profile_data.get("identity", {})
        preview = {
            "name": identity.get("name", "Unknown"),
            "industry": identity.get("industry", ""),
            "stage": identity.get("company_stage", ""),
            "node_count": len(data.get("graph", {}).get("nodes", [])),
            "has_simulation": data.get("simulation_snapshot") is not None,
        }

    return ValidateResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        needs_migration=requires_migration,
        preview=preview,
    )


@router.post("/load", response_model=LoadResponse)
async def load_import(request: LoadRequest) -> LoadResponse:
    data = request.data
    errors, _ = _validate_oleo(data)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Migrate if needed
    if needs_migration(data.get("format_version", CURRENT_VERSION)):
        data = migrate(data)

    # Parse into models
    try:
        profile = CompanyProfile.model_validate(data["profile"])
        graph = CompanyGraph.model_validate(data["graph"])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    # Create or replace session
    if request.mode == "replace" and request.target_session_id:
        session = session_store.get_session(request.target_session_id)
        if session:
            session.profile = profile
            session.graph = graph
            return LoadResponse(
                session_id=session.id,
                profile=profile,
                graph=graph,
            )

    # Create new session
    session = session_store.create_session(profile=profile, graph=graph)

    return LoadResponse(
        session_id=session.id,
        profile=profile,
        graph=graph,
    )
