"""Monte Carlo simulation runner — execute N parallel simulations with parameter variation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.simulation.monte_carlo_models import (
    CompanySummary,
    MonteCarloConfig,
    MonteCarloReport,
    SimulationResult,
)
from src.simulation.unified import UnifiedEngine
from src.simulation.unified_models import UnifiedParams, UnifiedStartConfig

log = logging.getLogger(__name__)


def _latin_hypercube_samples(
    n_samples: int,
    n_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Latin Hypercube Samples in [0, 1]^n_dims."""
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples[perm[i], dim] = (i + rng.uniform()) / n_samples
    return samples


def _set_nested_param(params: UnifiedParams, path: str, value: float) -> None:
    """Set a parameter on UnifiedParams by dot-path.

    Supports flat names ('tam_0') and dotted paths ('params.tam_0', 'foo.bar.baz').
    A leading 'params.' segment is stripped since params is the root.
    Raises ValueError if any segment doesn't resolve to an attribute.
    """
    parts = path.split(".")
    if parts and parts[0] == "params":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid parameter path: '{path}'")
    obj: object = params
    for segment in parts[:-1]:
        if not hasattr(obj, segment):
            raise ValueError(
                f"Invalid parameter path '{path}': no attribute '{segment}' on {type(obj).__name__}"
            )
        obj = getattr(obj, segment)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise ValueError(
            f"Invalid parameter path '{path}': no attribute '{leaf}' on {type(obj).__name__}"
        )
    setattr(obj, leaf, value)


def generate_configs(
    mc_config: MonteCarloConfig,
) -> list[tuple[UnifiedStartConfig, dict[str, float], int]]:
    """Generate N configs with Latin Hypercube Sampling over parameter ranges."""
    rng = np.random.default_rng(mc_config.seed)
    n = mc_config.num_runs
    variations = mc_config.parameter_variations

    if variations:
        lhs = _latin_hypercube_samples(n, len(variations), rng)
    else:
        lhs = np.zeros((n, 0))

    configs = []
    for i in range(n):
        run_seed = int(rng.integers(0, 2**31))
        varied: dict[str, float] = {}

        params = UnifiedParams()
        for dim, var in enumerate(variations):
            value = var.low + lhs[i, dim] * (var.high - var.low)
            _set_nested_param(params, var.param, value)
            varied[var.param] = round(value, 4)

        config = UnifiedStartConfig(
            industry=mc_config.industry,
            num_companies=mc_config.num_companies,
            start_mode=mc_config.start_mode,
            max_ticks=mc_config.ticks_per_run,
            params=params,
        )
        configs.append((config, varied, run_seed))

    return configs


def run_single(
    run_index: int,
    config: UnifiedStartConfig,
    varied_params: dict[str, float],
    seed: int,
    sample_interval: int,
) -> SimulationResult:
    """Execute a single simulation run and collect results."""
    engine = UnifiedEngine(config=config, seed=seed)

    # Snapshot how many companies exist before any tick — anything appended
    # after this is a market-spawned entrant that should aggregate to a
    # separate "entrant" bucket rather than its own (random) name key.
    initial_company_count = len(engine.companies)

    # Track per-company stats
    peak_cash: dict[str, float] = {}
    min_cash: dict[str, float] = {}
    death_tick: dict[str, int] = {}

    sampled_ticks: list[int] = []
    sampled_tam: list[float] = []
    sampled_hhi: list[float] = []

    for c in engine.companies:
        peak_cash[c.state.name] = c.state.cash
        min_cash[c.state.name] = c.state.cash

    while not engine.is_complete:
        result = engine.tick()
        tick = result["tick"]

        # Track per-company extremes
        for agent_data in result.get("agents", []):
            name = agent_data["name"]
            cash = agent_data["cash"]
            if name in peak_cash:
                peak_cash[name] = max(peak_cash[name], cash)
                min_cash[name] = min(min_cash[name], cash)
            else:
                peak_cash[name] = cash
                min_cash[name] = cash
            if not agent_data["alive"] and name not in death_tick:
                death_tick[name] = tick

        # Sample at intervals
        if tick % sample_interval == 0:
            sampled_ticks.append(tick)
            sampled_tam.append(result.get("tam", 0.0))
            sampled_hhi.append(result.get("hhi", 1.0))

    # Build company summaries
    companies = []
    for c in engine.companies:
        archetype = c.state.name if c.index < initial_company_count else "entrant"
        companies.append(CompanySummary(
            name=c.state.name,
            archetype=archetype,
            alive=c.alive,
            final_cash=round(c.state.cash, 2),
            final_share=round(c.share, 6),
            final_locations=c.location_count(),
            peak_cash=round(peak_cash.get(c.state.name, 0), 2),
            min_cash=round(min_cash.get(c.state.name, 0), 2),
            ticks_survived=death_tick.get(c.state.name, engine.tick_num),
        ))

    return SimulationResult(
        run_index=run_index,
        seed=seed,
        varied_params=varied_params,
        final_tick=engine.tick_num,
        companies=companies,
        sampled_ticks=sampled_ticks,
        sampled_tam=sampled_tam,
        sampled_hhi=sampled_hhi,
    )


class MonteCarloRunner:
    """Orchestrates N parallel simulation runs with parameter variation."""

    def __init__(self, mc_config: MonteCarloConfig) -> None:
        self.mc_config = mc_config
        self.configs = generate_configs(mc_config)
        self.results: list[SimulationResult] = []
        self.completed_count = 0
        self.is_complete = False

    @property
    def total_runs(self) -> int:
        return self.mc_config.num_runs

    def run_all(self, max_workers: int = 4) -> MonteCarloReport:
        """Run all simulations using thread pool. Returns final report."""
        self.results = []
        self.completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single,
                    i,
                    config,
                    varied,
                    seed,
                    self.mc_config.sample_interval,
                ): i
                for i, (config, varied, seed) in enumerate(self.configs)
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                self.completed_count += 1
                log.info("Monte Carlo run %d/%d complete", self.completed_count, self.total_runs)

        self.results.sort(key=lambda r: r.run_index)
        self.is_complete = True
        return self.build_report()

    def run_sequential(self) -> MonteCarloReport:
        """Run all simulations sequentially (for testing/debugging)."""
        self.results = []
        self.completed_count = 0

        for i, (config, varied, seed) in enumerate(self.configs):
            result = run_single(i, config, varied, seed, self.mc_config.sample_interval)
            self.results.append(result)
            self.completed_count += 1

        self.is_complete = True
        return self.build_report()

    def build_report(self) -> MonteCarloReport:
        """Compute aggregate statistics across all completed runs."""
        if not self.results:
            return MonteCarloReport(
                num_runs=0,
                ticks_per_run=self.mc_config.ticks_per_run,
                varied_parameters=[v.param for v in self.mc_config.parameter_variations],
                results=[],
            )

        # Aggregate by stable archetype key, not display name. Spawned entrants
        # all bucket under "entrant" so their stats represent the cohort rather
        # than per-run-unique random names with sample size 1.
        all_archetypes: set[str] = set()
        for r in self.results:
            for c in r.companies:
                all_archetypes.add(c.archetype)

        survival_rates: dict[str, float] = {}
        mean_final_cash: dict[str, float] = {}
        std_final_cash: dict[str, float] = {}
        mean_final_share: dict[str, float] = {}

        for archetype in sorted(all_archetypes):
            cash_values = []
            share_values = []
            alive_count = 0
            total_count = 0

            for r in self.results:
                for c in r.companies:
                    if c.archetype == archetype:
                        cash_values.append(c.final_cash)
                        share_values.append(c.final_share)
                        if c.alive:
                            alive_count += 1
                        total_count += 1

            if total_count > 0:
                survival_rates[archetype] = alive_count / total_count
                mean_final_cash[archetype] = round(float(np.mean(cash_values)), 2)
                std_final_cash[archetype] = round(float(np.std(cash_values)), 2)
                mean_final_share[archetype] = round(float(np.mean(share_values)), 6)

        # HHI stats
        final_hhis = []
        for r in self.results:
            if r.sampled_hhi:
                final_hhis.append(r.sampled_hhi[-1])

        return MonteCarloReport(
            num_runs=len(self.results),
            ticks_per_run=self.mc_config.ticks_per_run,
            varied_parameters=[v.param for v in self.mc_config.parameter_variations],
            results=self.results,
            survival_rates=survival_rates,
            mean_final_cash=mean_final_cash,
            std_final_cash=std_final_cash,
            mean_final_share=mean_final_share,
            mean_hhi=round(float(np.mean(final_hhis)), 6) if final_hhis else 0.0,
            std_hhi=round(float(np.std(final_hhis)), 6) if final_hhis else 0.0,
        )
