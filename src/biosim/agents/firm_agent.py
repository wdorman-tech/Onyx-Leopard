from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from biosim.agents.mesa_model import BioSimModel


class BioFirmAgent(mesa.Agent):
    """Thin handle into StateArrays -- all computation is vectorized in TickEngine."""

    def __init__(self, model: BioSimModel, state_index: int) -> None:
        super().__init__(model)
        self.state_index = state_index

    @property
    def is_alive(self) -> bool:
        return bool(self.model.state_manager.state.alive[self.state_index])

    @property
    def name(self) -> str:
        return self.model.state_manager.state.company_names[self.state_index]

    def step(self) -> None:
        pass
