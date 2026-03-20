from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.schemas import CompanyGraph, CompanyProfile


@dataclass
class ProfileSession:
    id: str
    profile: CompanyProfile
    graph: CompanyGraph | None = None
    uploaded_docs: list[dict] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    current_phase: str = "identity"


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ProfileSession] = {}

    def create_session(
        self,
        profile: CompanyProfile | None = None,
        graph: CompanyGraph | None = None,
    ) -> ProfileSession:
        session_id = str(uuid.uuid4())
        session = ProfileSession(
            id=session_id,
            profile=profile or CompanyProfile(),
            graph=graph,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ProfileSession | None:
        return self._sessions.get(session_id)

    def update_profile(self, session_id: str, profile: CompanyProfile) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.profile = profile
        return True

    def update_graph(self, session_id: str, graph: CompanyGraph) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.graph = graph
        return True

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


session_store = SessionStore()
