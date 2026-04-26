"""Monte Carlo runner for v2 — seed/stance/shocks driven, not industry-slug.

Replaces `monte_carlo.py`'s industry-spec-driven configuration with the v2
contract: each run is parameterized by `(seed, stance, shock_schedule, rng_seed)`.

Design notes:

  * **Archetype is `stance.archetype`** (not company name). Cross-run
    aggregation buckets results by stance archetype, which is the unit of
    research output v2 cares about ("does bootstrap beat venture_growth in
    a recession?"). Multiple runs with the same stance archetype but
    different RNG seeds aggregate cleanly.

  * **Latin Hypercube sampling** over a list of parameter perturbations
    expressed as dotted paths into the seed (e.g., `"base_price"`,
    `"competitor_density"`). Each MC run samples one point in the
    [low, high] hypercube and overrides those seed fields.

  * **Stance is sampled per archetype**, not perturbed via LHS. Stance
    attributes are too entangled to vary independently — sampling fresh
    stances per run via `sample_stance(archetype, rng)` keeps each run a
    valid, internally-consistent persona.

  * **Shock schedule** is per-run via `ShockScheduler` with its own RNG
    seed derived from the run seed. Lambda overrides are configurable per
    MC batch (e.g., "stress test with 2x market_crash arrivals").

  * **Heuristic-only by default** — LLM tiers can be enabled by passing
    a `transcript_dir` (each run gets its own transcript file). Default is
    no LLM, so MC sweeps are free.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.simulation.library_loader import NodeLibrary, get_library
from src.simulation.replay import CostTracker, Transcript
from src.simulation.seed import (
    ARCHETYPES as SEED_ARCHETYPES,
    CompanySeed,
    sample_seed_for_archetype,
)
from src.simulation.shocks import ShockScheduler
from src.simulation.stance import (
    ARCHETYPES as STANCE_ARCHETYPES,
    CeoStance,
    sample_stance,
)
from src.simulation.unified_v2 import (
    CompanyAgentV2,
    MultiCompanySimV2,
)

log = logging.getLogger(__name__)


# ─── Models ────────────────────────────────────────────────────────────────


class SeedVariation(BaseModel):
    """A single field on `CompanySeed` to vary across MC runs.

    `field` is the attribute name on `CompanySeed` (e.g., "base_price",
    "competitor_density", "tam"). Numeric fields only.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    low: float
    high: float


class MonteCarloConfigV2(BaseModel):
    """Configuration for a v2 MC batch.

    `seed_archetype` and `stance_archetype` together define the persona being
    swept. To compare archetypes, run multiple batches with different
    archetype selections.
    """

    model_config = ConfigDict(extra="forbid")

    seed_archetype: str = Field(
        default="small_team",
        description=f"One of: {', '.join(SEED_ARCHETYPES)}",
    )
    stance_archetype: str = Field(
        default="bootstrap",
        description=f"One of: {', '.join(STANCE_ARCHETYPES)}",
    )
    num_runs: int = Field(default=10, ge=1, le=200)
    num_companies_per_run: int = Field(default=1, ge=1, le=10)
    ticks_per_run: int = Field(default=365, ge=10)
    parameter_variations: list[SeedVariation] = Field(default_factory=list)
    shock_lambdas: dict[str, float] = Field(
        default_factory=dict,
        description="Override Poisson arrival rates per shock type. Empty = use defaults.",
    )
    sample_interval: int = Field(default=30, ge=1)
    base_seed: int = 42
    transcript_dir: str | None = Field(
        default=None,
        description="If set, each run records its CEO transcript here. Default: heuristic-only.",
    )
    cost_ceiling_per_run_usd: float = Field(default=5.0, ge=0.0)
    initial_supplier_types: list[str] = Field(default_factory=list)
    initial_revenue_streams: list[str] = Field(default_factory=list)
    initial_cost_centers: list[str] = Field(default_factory=list)
    pin_economics_model: str | None = Field(
        default=None,
        description="If set, force seed.economics_model to this value (overrides archetype roulette).",
    )
    tam_initial: float = Field(default=1_000_000.0, ge=1.0)


class CompanyResultV2(BaseModel):
    """Per-company outcome from one MC run."""

    company_id: str
    archetype: str  # stance archetype — the aggregation key
    alive: bool
    final_cash: float
    final_revenue: float
    final_satisfaction: float
    spawned_node_count: int
    location_count: int
    ticks_survived: int


class RunResultV2(BaseModel):
    """Single MC run result."""

    run_index: int
    rng_seed: int
    varied_params: dict[str, float] = Field(default_factory=dict)
    final_tick: int
    companies: list[CompanyResultV2]
    sampled_ticks: list[int] = Field(default_factory=list)
    sampled_tam: list[float] = Field(default_factory=list)
    sampled_total_revenue: list[float] = Field(default_factory=list)
    shock_arrivals: int = 0


class MonteCarloReportV2(BaseModel):
    """Aggregated stats across all MC runs."""

    num_runs: int
    ticks_per_run: int
    seed_archetype: str
    stance_archetype: str
    varied_fields: list[str]
    results: list[RunResultV2]
    survival_rate: float = 0.0
    mean_final_cash: float = 0.0
    std_final_cash: float = 0.0
    mean_final_revenue: float = 0.0
    mean_locations: float = 0.0
    mean_shocks_per_run: float = 0.0


# ─── Sampling ──────────────────────────────────────────────────────────────


def _latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Standard LHS in [0, 1]^n_dims with one sample per stratum per dim."""
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[perm[i], dim] = (i + rng.uniform()) / n_samples
    return samples


def _build_seed_for_run(
    config: MonteCarloConfigV2,
    library: NodeLibrary,
    rng: random.Random,
    lhs_row: np.ndarray,
) -> tuple[CompanySeed, dict[str, float]]:
    """Sample a CompanySeed for this run; apply LHS variations + ref overrides."""
    seed = sample_seed_for_archetype(config.seed_archetype, rng=rng)

    overrides: dict[str, Any] = {}
    if config.pin_economics_model is not None:
        overrides["economics_model"] = config.pin_economics_model
    if config.initial_supplier_types:
        overrides["initial_supplier_types"] = list(config.initial_supplier_types)
    if config.initial_revenue_streams:
        overrides["initial_revenue_streams"] = list(config.initial_revenue_streams)
    if config.initial_cost_centers:
        overrides["initial_cost_centers"] = list(config.initial_cost_centers)

    varied: dict[str, float] = {}
    for dim, var in enumerate(config.parameter_variations):
        value = var.low + lhs_row[dim] * (var.high - var.low)
        if not hasattr(seed, var.field):
            raise ValueError(
                f"SeedVariation.field {var.field!r} is not a CompanySeed attribute"
            )
        # Coerce to int if the seed field is int-typed
        current = getattr(seed, var.field)
        if isinstance(current, int):
            value_typed: float | int = int(round(value))
        else:
            value_typed = round(value, 4)
        overrides[var.field] = value_typed
        varied[var.field] = float(value_typed)

    if overrides:
        seed = seed.model_copy(update=overrides)

    # Cross-validate. Hard-fails per V2 spec — no silent fallback.
    library.validate_seed(seed)

    return seed, varied


# ─── Single-run execution ──────────────────────────────────────────────────


async def run_single_v2(
    run_index: int,
    config: MonteCarloConfigV2,
    library: NodeLibrary,
    rng_seed: int,
    lhs_row: np.ndarray,
) -> RunResultV2:
    """Execute a single v2 simulation. Async because MultiCompanySimV2 is async.

    Each run gets its own RNG (deterministic from `rng_seed`), its own
    `ShockScheduler` (also deterministic), and either a per-run transcript
    (if `transcript_dir` is set) or no transcript (heuristic-only).
    """
    rng = random.Random(rng_seed)
    seed, varied = _build_seed_for_run(config, library, rng, lhs_row)

    # One stance per run — sampled fresh, archetype-locked
    stance = sample_stance(config.stance_archetype, rng=rng)

    # Transcript wiring (off by default — heuristic-only sweeps cost $0)
    transcript = None
    if config.transcript_dir:
        Path(config.transcript_dir).mkdir(parents=True, exist_ok=True)
        transcript_path = Path(config.transcript_dir) / f"run_{run_index:04d}.jsonl"
        transcript = Transcript(path=transcript_path, mode="record")

    # Build N companies (default 1)
    companies: list[CompanyAgentV2] = []
    for i in range(config.num_companies_per_run):
        # Each company uses a fresh shock scheduler (independent timelines)
        scheduler = ShockScheduler(
            rng_seed=rng.randint(0, 2**31 - 1),
            lambdas=dict(config.shock_lambdas),
        )
        cost_tracker = CostTracker(ceiling_usd=config.cost_ceiling_per_run_usd)
        company = CompanyAgentV2(
            seed=seed,
            stance=stance,
            library=library,
            sim_id=f"mc-{run_index}",
            company_id=f"co-{run_index}-{i}",
            rng=random.Random(rng.randint(0, 2**31 - 1)),
            transcript=transcript,
            cost_tracker=cost_tracker,
            shock_scheduler=scheduler,
        )
        companies.append(company)

    sim = MultiCompanySimV2(
        sim_id=f"mc-{run_index}",
        companies=companies,
        max_ticks=config.ticks_per_run,
        tam_initial=config.tam_initial,
    )

    # Sampling buffers
    sampled_ticks: list[int] = []
    sampled_tam: list[float] = []
    sampled_total_revenue: list[float] = []
    total_arrivals = 0

    while not sim.is_complete:
        step_result = await sim.step()
        if step_result["tick"] % config.sample_interval == 0:
            sampled_ticks.append(step_result["tick"])
            sampled_tam.append(step_result["tam"])
            total_rev = sum(
                r.daily_revenue
                for r in step_result.get("results", [])
            )
            sampled_total_revenue.append(total_rev)
            for r in step_result.get("results", []):
                total_arrivals += len(r.arriving_shocks)

    # Build per-company summary
    company_results: list[CompanyResultV2] = []
    for c in companies:
        location_count = sum(
            count
            for k, count in c.spawned_nodes.items()
            if k in c.library.nodes and c.library.nodes[k].category == "location"
        )
        company_results.append(CompanyResultV2(
            company_id=c.company_id,
            archetype=c.stance.archetype,
            alive=c.alive,
            final_cash=round(c.cash, 2),
            final_revenue=round(c.daily_revenue, 2),
            final_satisfaction=round(c.satisfaction, 4),
            spawned_node_count=sum(c.spawned_nodes.values()),
            location_count=location_count,
            ticks_survived=c.tick,
        ))

    return RunResultV2(
        run_index=run_index,
        rng_seed=rng_seed,
        varied_params=varied,
        final_tick=sim.tick,
        companies=company_results,
        sampled_ticks=sampled_ticks,
        sampled_tam=sampled_tam,
        sampled_total_revenue=sampled_total_revenue,
        shock_arrivals=total_arrivals,
    )


# ─── Runner ────────────────────────────────────────────────────────────────


class MonteCarloRunnerV2:
    """Orchestrates N v2 simulation runs with seed-field LHS variation."""

    def __init__(
        self,
        config: MonteCarloConfigV2,
        library: NodeLibrary | None = None,
    ) -> None:
        self.config = config
        self.library = library or get_library()
        self.results: list[RunResultV2] = []
        self.completed_count: int = 0
        self.is_complete: bool = False

        # Pre-generate per-run RNG seeds and LHS rows (deterministic).
        np_rng = np.random.default_rng(config.base_seed)
        n = config.num_runs
        d = max(1, len(config.parameter_variations))
        if config.parameter_variations:
            self._lhs = _latin_hypercube(n, d, np_rng)
        else:
            self._lhs = np.zeros((n, 0))
        self._run_seeds: list[int] = [
            int(np_rng.integers(0, 2**31)) for _ in range(n)
        ]

    @property
    def total_runs(self) -> int:
        return self.config.num_runs

    async def run_sequential(self) -> MonteCarloReportV2:
        """Run all simulations sequentially (debugging / small N)."""
        self.results = []
        self.completed_count = 0
        for i in range(self.config.num_runs):
            lhs_row = self._lhs[i] if self._lhs.shape[1] > 0 else np.zeros(1)
            result = await run_single_v2(
                run_index=i,
                config=self.config,
                library=self.library,
                rng_seed=self._run_seeds[i],
                lhs_row=lhs_row,
            )
            self.results.append(result)
            self.completed_count += 1
            log.info(
                "MC v2 run %d/%d complete (cash=%.2f, alive=%d)",
                i + 1, self.config.num_runs,
                result.companies[0].final_cash if result.companies else 0,
                sum(1 for c in result.companies if c.alive),
            )
        self.is_complete = True
        return self.build_report()

    def build_report(self) -> MonteCarloReportV2:
        """Aggregate stats across all runs.

        Single-archetype batches: stats represent that archetype directly.
        For cross-archetype comparison, run multiple batches.
        """
        all_companies = [c for r in self.results for c in r.companies]
        if not all_companies:
            return MonteCarloReportV2(
                num_runs=0,
                ticks_per_run=self.config.ticks_per_run,
                seed_archetype=self.config.seed_archetype,
                stance_archetype=self.config.stance_archetype,
                varied_fields=[v.field for v in self.config.parameter_variations],
                results=[],
            )

        cash_values = [c.final_cash for c in all_companies]
        revenue_values = [c.final_revenue for c in all_companies]
        location_values = [c.location_count for c in all_companies]
        survival = sum(1 for c in all_companies if c.alive) / len(all_companies)

        return MonteCarloReportV2(
            num_runs=len(self.results),
            ticks_per_run=self.config.ticks_per_run,
            seed_archetype=self.config.seed_archetype,
            stance_archetype=self.config.stance_archetype,
            varied_fields=[v.field for v in self.config.parameter_variations],
            results=self.results,
            survival_rate=round(survival, 4),
            mean_final_cash=round(float(np.mean(cash_values)), 2),
            std_final_cash=round(float(np.std(cash_values)), 2),
            mean_final_revenue=round(float(np.mean(revenue_values)), 2),
            mean_locations=round(float(np.mean(location_values)), 2),
            mean_shocks_per_run=round(
                float(np.mean([r.shock_arrivals for r in self.results])), 2
            ),
        )


__all__ = [
    "CompanyResultV2",
    "MonteCarloConfigV2",
    "MonteCarloReportV2",
    "MonteCarloRunnerV2",
    "RunResultV2",
    "SeedVariation",
    "run_single_v2",
]
