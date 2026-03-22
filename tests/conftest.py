import numpy as np
import pytest

from biosim.types.config import BioConfig, SimConfig
from biosim.types.state import StateArrays

SEED_COMPANIES = [
    ("Alpha Corp", "#E74C3C", {"cash": 1_000_000, "firm_size": 100, "growth_rate": 0.05}),
    ("Beta Inc", "#3498DB", {"cash": 500_000, "firm_size": 50, "growth_rate": 0.08}),
    ("Gamma Ltd", "#2ECC71", {"cash": 2_000_000, "firm_size": 200, "growth_rate": 0.02}),
]


@pytest.fixture()
def state_arrays() -> StateArrays:
    sa = StateArrays(max_capacity=10)
    for name, color, params in SEED_COMPANIES:
        sa.add_company(name, color, params)
    return sa


@pytest.fixture()
def bio_config() -> BioConfig:
    return BioConfig()


@pytest.fixture()
def sim_config() -> SimConfig:
    return SimConfig()


@pytest.fixture()
def tick_engine(bio_config, sim_config):
    from biosim.engine.tick import TickEngine

    return TickEngine(bio_config, sim_config)


@pytest.fixture()
def state_manager(sim_config):
    from biosim.engine.state_manager import StateManager

    return StateManager(sim_config)


@pytest.fixture()
def populated_state_manager(sim_config):
    from biosim.engine.state_manager import StateManager

    sm = StateManager(sim_config)
    sm.add_company("Alpha Corp", "#E74C3C", size="large")
    sm.add_company("Beta Inc", "#3498DB", size="medium")
    sm.add_company("Gamma Ltd", "#2ECC71", size="small")
    return sm


@pytest.fixture
def n_agents():
    return 5


@pytest.fixture
def default_params(n_agents):
    return {
        "firm_size": np.array([10.0, 20.0, 5.0, 15.0, 8.0]),
        "cash": np.array([1e6, 2e6, 5e5, 1.5e6, 8e5]),
        "growth_rate": np.array([0.05, 0.03, 0.08, 0.04, 0.06]),
        "carrying_capacity": np.array([100.0, 200.0, 50.0, 150.0, 80.0]),
        "revenue": np.array([1e5, 2e5, 5e4, 1.5e5, 8e4]),
        "fixed_costs": np.array([5e4, 1e5, 2.5e4, 7.5e4, 4e4]),
        "variable_cost_rate": np.array([1000.0, 800.0, 1200.0, 900.0, 1100.0]),
        "tfp": np.ones(5),
        "capital": np.array([1e6, 2e6, 5e5, 1.5e6, 8e5]),
        "labor": np.array([50.0, 100.0, 25.0, 75.0, 40.0]),
        "alpha": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        "beta": np.array([0.7, 0.7, 0.7, 0.7, 0.7]),
    }
