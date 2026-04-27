from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class OdeSystem(Protocol):
    @property
    def n_vars(self) -> int: ...

    def rhs(self, t: float, y: np.ndarray, *args: object) -> np.ndarray: ...


class Renderer(Protocol):
    def update_state(self, state: dict) -> None: ...


@dataclass(slots=True)
class TickResult:
    state: dict
    events: list[dict] = field(default_factory=list)
