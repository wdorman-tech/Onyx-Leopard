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
