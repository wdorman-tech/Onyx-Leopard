from __future__ import annotations

import asyncio
import copy
import uuid

from src.schemas import CompanyGraph
from src.simulation.engine import SimulationEngine
from src.simulation.state import StepResult


class SimulationSession:
    def __init__(self, engine: SimulationEngine):
        self.id = str(uuid.uuid4())
        self.engine = engine
        self.playing = False
        self.speed: float = 2.0  # seconds per tick
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # starts unpaused
        self._stop = False

    def play(self) -> None:
        self.playing = True
        self._pause_event.set()

    def pause(self) -> None:
        self.playing = False
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop = True
        self._pause_event.set()  # unblock if paused

    def set_speed(self, multiplier: float) -> None:
        speed_map = {1: 2.0, 2: 1.0, 5: 0.4, 10: 0.2}
        self.speed = speed_map.get(int(multiplier), 2.0)

    async def wait_if_paused(self) -> bool:
        """Wait while paused. Returns False if stopped."""
        await self._pause_event.wait()
        return not self._stop


SPEED_MAP = {1: 2.0, 2: 1.0, 5: 0.4, 10: 0.2}


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SimulationSession] = {}

    def create_session(self, graph: CompanyGraph, max_ticks: int = 50, outlook: str = "normal") -> SimulationSession:
        engine = SimulationEngine(graph, max_ticks=max_ticks, outlook=outlook)
        session = SimulationSession(engine)
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> SimulationSession | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.stop()

    def fork_session(self, session_id: str) -> SimulationSession | None:
        original = self.get_session(session_id)
        if original is None:
            return None
        graph_copy = original.engine.state.graph.model_copy(deep=True)
        new_engine = SimulationEngine(
            graph_copy,
            max_ticks=original.engine.state.max_ticks,
            outlook=original.engine.state.outlook,
        )
        new_engine.state.tick = original.engine.state.tick
        new_session = SimulationSession(new_engine)
        self._sessions[new_session.id] = new_session
        return new_session


session_manager = SessionManager()
