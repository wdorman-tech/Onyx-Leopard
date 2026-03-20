from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.graph_generator import compute_sim_params, generate_company_graph
from src.agents.parser import parse_company
from src.agents.questionnaire import PHASE_ORDER, questionnaire_agent
from src.schemas import CompanyGraph, CompanyProfile, Department
from src.session_store import session_store

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


class StartResponse(BaseModel):
    session_id: str
    question: str
    phase: str
    fields_targeted: list[str]
    completion_estimate: float
    profile: CompanyProfile


class RespondRequest(BaseModel):
    session_id: str
    message: str


class RespondResponse(BaseModel):
    question: str
    phase: str
    fields_targeted: list[str]
    completion_estimate: float
    profile: CompanyProfile
    graph: CompanyGraph | None = None


class SkipPhaseRequest(BaseModel):
    session_id: str
    current_phase: str


class CompleteRequest(BaseModel):
    session_id: str


class CompleteResponse(BaseModel):
    session_id: str
    profile: CompanyProfile
    graph: CompanyGraph


class QuickStartRequest(BaseModel):
    description: str


class QuickStartResponse(BaseModel):
    session_id: str
    profile: CompanyProfile
    graph: CompanyGraph


@router.post("/start", response_model=StartResponse)
async def start_onboarding() -> StartResponse:
    session = session_store.create_session()

    result = questionnaire_agent.start(session.profile)

    session.conversation_history.append({
        "role": "assistant",
        "content": result.get("question", ""),
    })

    return StartResponse(
        session_id=session.id,
        question=result.get("question", ""),
        phase=result.get("phase", "identity"),
        fields_targeted=result.get("fields_targeted", []),
        completion_estimate=result.get("completion_estimate", 0.0),
        profile=session.profile,
    )


@router.post("/respond", response_model=RespondResponse)
async def respond_onboarding(request: RespondRequest) -> RespondResponse:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Add user message to history
    session.conversation_history.append({
        "role": "user",
        "content": request.message,
    })

    current_phase = session.current_phase
    uploaded_doc_names = [d.get("name", "") for d in session.uploaded_docs]

    result = questionnaire_agent.respond(
        user_message=request.message,
        profile=session.profile,
        phase=current_phase,
        conversation_history=session.conversation_history,
        uploaded_docs=uploaded_doc_names,
    )

    # Update profile — log validation errors but don't crash
    updated_data = result.get("updated_profile", {})
    if updated_data:
        try:
            session.profile = CompanyProfile.model_validate(updated_data)
        except Exception:
            pass  # LLM returned invalid profile update; keep existing

    # Track phase on session
    phase = result.get("phase", current_phase)
    session.current_phase = phase
    session.conversation_history.append({
        "role": "assistant",
        "content": result.get("question", ""),
        "phase": phase,
    })

    # Generate graph if we have enough data (identity.name at minimum)
    graph = None
    if session.profile.identity.name:
        try:
            graph = generate_company_graph(session.profile, use_llm_prompts=False)
            session.graph = graph
        except Exception:
            pass

    return RespondResponse(
        question=result.get("question", ""),
        phase=phase,
        fields_targeted=result.get("fields_targeted", []),
        completion_estimate=result.get("completion_estimate", 0.0),
        profile=session.profile,
        graph=graph,
    )


@router.post("/skip-phase", response_model=RespondResponse)
async def skip_phase(request: SkipPhaseRequest) -> RespondResponse:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    next_phase = questionnaire_agent.skip_phase(request.current_phase)

    result = questionnaire_agent.respond(
        user_message="[User skipped this phase]",
        profile=session.profile,
        phase=next_phase,
        conversation_history=session.conversation_history,
        uploaded_docs=[d.get("name", "") for d in session.uploaded_docs],
    )

    session.conversation_history.append({
        "role": "assistant",
        "content": result.get("question", ""),
        "phase": next_phase,
    })

    graph = None
    if session.profile.identity.name:
        try:
            graph = generate_company_graph(session.profile, use_llm_prompts=False)
            session.graph = graph
        except Exception:
            pass

    return RespondResponse(
        question=result.get("question", ""),
        phase=next_phase,
        fields_targeted=result.get("fields_targeted", []),
        completion_estimate=result.get("completion_estimate", 0.0),
        profile=session.profile,
        graph=graph,
    )


@router.post("/complete", response_model=CompleteResponse)
async def complete_onboarding(request: CompleteRequest) -> CompleteResponse:
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Compute simulation parameters
    session.profile.sim_params = compute_sim_params(session.profile)

    # Generate final graph with LLM-powered agent prompts
    graph = generate_company_graph(session.profile, use_llm_prompts=True)
    session.graph = graph

    return CompleteResponse(
        session_id=session.id,
        profile=session.profile,
        graph=graph,
    )


@router.post("/quick-start", response_model=QuickStartResponse)
async def quick_start(request: QuickStartRequest) -> QuickStartResponse:
    """Quick start: parse a company description into a profile + graph."""
    # Use existing parser to get a graph
    graph = parse_company(request.description)

    # Build a minimal profile from the graph
    profile = CompanyProfile()
    profile.identity.name = graph.name
    profile.identity.description = graph.description

    # Extract org data from graph nodes
    departments = []
    total_headcount = 0
    total_budget = 0.0
    for node in graph.nodes:
        if node.type == "department":
            departments.append(Department(
                id=node.id,
                name=node.label,
                headcount=int(node.metrics.get("headcount", 0)),
                budget=node.metrics.get("budget", 0.0),
            ))
            total_headcount += int(node.metrics.get("headcount", 0))
            total_budget += node.metrics.get("budget", 0.0)

    profile.organization.departments = departments
    profile.organization.total_headcount = total_headcount or int(
        graph.global_metrics.get("total_headcount", 0)
    )

    # Revenue
    revenue = graph.global_metrics.get("revenue", 0.0)
    profile.financials.annual_revenue = revenue

    # Compute sim params
    profile.sim_params = compute_sim_params(profile)

    # Create session
    session = session_store.create_session(profile=profile, graph=graph)

    return QuickStartResponse(
        session_id=session.id,
        profile=profile,
        graph=graph,
    )
