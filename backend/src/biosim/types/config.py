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

    # Multinomial-logit market-share exponents: s_i ∝ q_i^β * m_i^α.
    # Defaults match unified_v2's `MultiCompanySimV2` constructor defaults so
    # the engine and the math layer produce identical share allocations.
    share_alpha: float = 1.0  # marketing/size exponent
    share_beta: float = 1.5   # quality/tfp exponent


class SimConfig(BaseModel):
    """Top-level simulation parameters."""

    max_companies: int = 50
    tick_interval_ms: int = 1000
    max_speed: int = 20
    insolvent_ticks_to_death: int = 3
    default_carrying_capacity: float = 100.0
    growth_division_threshold: float = 0.8
