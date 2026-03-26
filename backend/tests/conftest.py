"""Shared fixtures for market simulation tests."""

from __future__ import annotations

import pytest

from src.simulation.market.engine import MarketEngine
from src.simulation.market.models import AgentParams, AgentState, MarketParams
from src.simulation.market.presets import MARKET_PRESETS


@pytest.fixture
def default_params() -> MarketParams:
    return MarketParams()


@pytest.fixture
def two_agents() -> list[AgentState]:
    """Two identical alive agents for unit testing."""
    return [
        AgentState(
            id="agent-0",
            params=AgentParams(name="A", r=0.05, margin=0.25, f=100.0, eta_m=0.2, eta_q=0.1, tau_k=30),
            revenue=500.0,
            cash=10_000.0,
            capacity=1_000.0,
            quality=1.0,
            marketing=30.0,
            q_target=1.0,
            m_target=30.0,
            k_target=1_000.0,
        ),
        AgentState(
            id="agent-1",
            params=AgentParams(name="B", r=0.05, margin=0.25, f=100.0, eta_m=0.2, eta_q=0.1, tau_k=30),
            revenue=500.0,
            cash=10_000.0,
            capacity=1_000.0,
            quality=1.0,
            marketing=30.0,
            q_target=1.0,
            m_target=30.0,
            k_target=1_000.0,
        ),
    ]


@pytest.fixture
def price_war_engine() -> MarketEngine:
    return MarketEngine(MARKET_PRESETS["price-war"].params, max_ticks=500, seed=42)


@pytest.fixture
def innovation_engine() -> MarketEngine:
    return MarketEngine(MARKET_PRESETS["innovation-race"].params, max_ticks=500, seed=42)


@pytest.fixture
def monopoly_engine() -> MarketEngine:
    return MarketEngine(MARKET_PRESETS["monopoly"].params, max_ticks=500, seed=42)


@pytest.fixture
def commodity_engine() -> MarketEngine:
    return MarketEngine(MARKET_PRESETS["commodity"].params, max_ticks=500, seed=42)
