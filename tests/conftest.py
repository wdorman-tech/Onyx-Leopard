import pytest

from biosim.engine.state_manager import StateManager
from biosim.engine.tick import TickEngine
from biosim.types.config import BioConfig, SimConfig


@pytest.fixture
def sim_config() -> SimConfig:
    return SimConfig()


@pytest.fixture
def bio_config() -> BioConfig:
    return BioConfig()


@pytest.fixture
def state_manager(sim_config: SimConfig) -> StateManager:
    return StateManager(sim_config)


@pytest.fixture
def tick_engine(bio_config: BioConfig, sim_config: SimConfig) -> TickEngine:
    return TickEngine(bio_config, sim_config)


@pytest.fixture
def populated_state_manager(state_manager: StateManager) -> StateManager:
    """StateManager pre-loaded with 3 companies of different sizes."""
    state_manager.add_company("AlphaCorp", "#E74C3C", size="large")
    state_manager.add_company("BetaInc", "#3498DB", size="medium")
    state_manager.add_company("GammaLtd", "#2ECC71", size="small")
    return state_manager
