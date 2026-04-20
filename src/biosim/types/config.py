from __future__ import annotations

from pydantic import BaseModel


class BioConfig(BaseModel):
    """Feature flags controlling which biological subsystems are active."""

    competition: bool = True
    cell_cycle: bool = False
    apoptosis: bool = False
    mapk: bool = False
    fba: bool = False
    replicator: bool = False


class AgentConfig(BaseModel):
    """Configuration for the CAMEL-AI agent decision system."""

    enabled: bool = False
    claude_api_key_env: str = "ANTHROPIC_API_KEY"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    novelty_threshold: float = 0.15
    max_concurrent_llm_calls: int = 5
    semantic_cache_enabled: bool = True
    cost_budget_per_run: float = 5.0


class SimConfig(BaseModel):
    """Top-level simulation parameters."""

    max_companies: int = 50
    tick_interval_ms: int = 1000
    max_speed: int = 20
    insolvent_ticks_to_death: int = 3
    default_carrying_capacity: float = 100.0
    growth_division_threshold: float = 0.8
    agent_config: AgentConfig = AgentConfig()
