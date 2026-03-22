from __future__ import annotations

import mesa

from biosim.agents.firm_agent import BioFirmAgent
from biosim.engine.state_manager import StateManager
from biosim.engine.tick import TickEngine
from biosim.types.config import BioConfig, SimConfig


class BioSimModel(mesa.Model):
    """Mesa model that wraps the BioSim simulation engine."""

    def __init__(
        self,
        bio_config: BioConfig | None = None,
        sim_config: SimConfig | None = None,
    ) -> None:
        super().__init__()
        self.bio_config = bio_config or BioConfig()
        self.sim_config = sim_config or SimConfig()
        self.state_manager = StateManager(self.sim_config)
        self.tick_engine = TickEngine(self.bio_config, self.sim_config)
        self.agents_list: list[BioFirmAgent] = []
        self.last_snapshot: dict = {}

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "total_market_size": _total_market_size,
                "n_active_companies": _n_active_companies,
            },
            agent_reporters={
                "firm_size": _agent_firm_size,
                "cash": _agent_cash,
            },
        )

    def add_company(self, name: str, color: str, **kwargs: str) -> BioFirmAgent:
        idx = self.state_manager.add_company(name, color, **kwargs)
        agent = BioFirmAgent(self, idx)
        self.agents_list.append(agent)
        return agent

    def step(self) -> None:
        """Execute one simulation tick."""
        self.last_snapshot = self.tick_engine.step(self.state_manager.state)
        self.state_manager.tick_count += 1
        self.state_manager.record_snapshot()
        self.datacollector.collect(self)


def _total_market_size(m: BioSimModel) -> float:
    indices = m.state_manager.state.active_indices()
    if len(indices) > 0:
        return float(m.state_manager.state.firm_size[indices].sum())
    return 0.0


def _n_active_companies(m: BioSimModel) -> int:
    return len(m.state_manager.state.active_indices())


def _agent_firm_size(a: BioFirmAgent) -> float:
    if a.model.state_manager.state.alive[a.state_index]:
        return float(a.model.state_manager.state.firm_size[a.state_index])
    return 0.0


def _agent_cash(a: BioFirmAgent) -> float:
    if a.model.state_manager.state.alive[a.state_index]:
        return float(a.model.state_manager.state.cash[a.state_index])
    return 0.0
