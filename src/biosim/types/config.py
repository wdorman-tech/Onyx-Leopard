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


class SimConfig(BaseModel):
    """Top-level simulation parameters."""

    max_companies: int = 50
    tick_interval_ms: int = 1000
    max_speed: int = 20
    insolvent_ticks_to_death: int = 3
    default_carrying_capacity: float = 100.0
    growth_division_threshold: float = 0.8
