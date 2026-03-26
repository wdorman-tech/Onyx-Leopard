from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from src.simulation.growth import GrowthEngine
from src.simulation.market.engine import MarketEngine
from src.simulation.unified import UnifiedEngine
from src.simulation.unified_models import UnifiedStartConfig

if TYPE_CHECKING:
    pass


class SimulationSession:
    def __init__(
        self,
        engine: GrowthEngine | MarketEngine | UnifiedEngine,
        mode: str = "growth",
    ):
        self.id = str(uuid.uuid4())
        self.engine = engine
        self.mode = mode
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
        self._pause_event.set()

    def set_speed(self, multiplier: float) -> None:
        speed_map = {1: 2.0, 2: 1.0, 5: 0.4, 10: 0.2, 50: 0.02}
        self.speed = speed_map.get(int(multiplier), 2.0)

    async def wait_if_paused(self) -> bool:
        """Wait while paused. Returns False if stopped."""
        await self._pause_event.wait()
        return not self._stop


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SimulationSession] = {}

    def create_session(
        self,
        max_ticks: int = 0,
        industry: str = "restaurant",
        mode: str = "growth",
        preset: str | None = None,
        unified_config: UnifiedStartConfig | None = None,
    ) -> SimulationSession:
        if mode == "market":
            from src.simulation.market.presets import MARKET_PRESETS

            if preset not in MARKET_PRESETS:
                raise ValueError(f"Unknown market preset: {preset}")
            params = MARKET_PRESETS[preset].params
            engine = MarketEngine(params=params, max_ticks=max_ticks)
            session = SimulationSession(engine, mode="market")
        elif mode == "unified":
            config = unified_config or UnifiedStartConfig(max_ticks=max_ticks)
            engine = UnifiedEngine(config=config)
            session = SimulationSession(engine, mode="unified")
        else:
            engine = GrowthEngine(max_ticks=max_ticks, industry=industry)
            session = SimulationSession(engine, mode="growth")
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> SimulationSession | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.stop()


session_manager = SessionManager()
