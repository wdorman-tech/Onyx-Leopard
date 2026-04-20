"""Data models for Monte Carlo simulation runs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParameterVariation(BaseModel):
    """A single parameter to vary across Monte Carlo runs."""

    param: str  # dot-path: "params.tam_0", "params.alpha", "constants.starting_cash"
    low: float
    high: float


class MonteCarloConfig(BaseModel):
    """Configuration for a Monte Carlo batch."""

    industry: str = "restaurant"
    num_runs: int = Field(default=10, ge=2, le=100)
    ticks_per_run: int = Field(default=1825, ge=100)  # 5 years default
    num_companies: int = Field(default=4, ge=1, le=20)
    start_mode: str = "identical"
    parameter_variations: list[ParameterVariation] = Field(default_factory=list)
    sample_interval: int = Field(default=30, ge=1)  # record metrics every N ticks
    seed: int = 42


class CompanySummary(BaseModel):
    """Per-company summary from a single run."""

    name: str
    # Stable aggregation key — equals `name` for seeded companies (whose names
    # are deterministic per seed), and "entrant" for companies spawned mid-run.
    # Without this, every random entrant name would create a separate bucket
    # in cross-run aggregates with only 1 data point, making them meaningless.
    archetype: str
    alive: bool
    final_cash: float
    final_share: float
    final_locations: int
    peak_cash: float
    min_cash: float
    ticks_survived: int


class SimulationResult(BaseModel):
    """Result from a single Monte Carlo run."""

    run_index: int
    seed: int
    varied_params: dict[str, float] = Field(default_factory=dict)
    final_tick: int = 0
    companies: list[CompanySummary] = Field(default_factory=list)
    sampled_ticks: list[int] = Field(default_factory=list)
    sampled_tam: list[float] = Field(default_factory=list)
    sampled_hhi: list[float] = Field(default_factory=list)


class MonteCarloReport(BaseModel):
    """Statistical summary across all Monte Carlo runs."""

    num_runs: int
    ticks_per_run: int
    varied_parameters: list[str]
    results: list[SimulationResult]
    # Aggregate statistics
    survival_rates: dict[str, float] = Field(default_factory=dict)  # company_name → % survived
    mean_final_cash: dict[str, float] = Field(default_factory=dict)
    std_final_cash: dict[str, float] = Field(default_factory=dict)
    mean_final_share: dict[str, float] = Field(default_factory=dict)
    mean_hhi: float = 0.0
    std_hhi: float = 0.0
