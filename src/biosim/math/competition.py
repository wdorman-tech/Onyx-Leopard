"""Stub -- real implementation in Unit 2."""

import numpy as np


def step_competition(
    populations: np.ndarray,
    growth_rates: np.ndarray,
    carrying_capacities: np.ndarray,
    alpha: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    n = len(populations)
    dp = np.zeros(n)
    for i in range(n):
        competition_sum = np.dot(alpha[i], populations) / max(carrying_capacities[i], 1e-10)
        dp[i] = growth_rates[i] * populations[i] * (1 - competition_sum)
    return np.maximum(populations + dp * dt, 0.0)


def build_competition_matrix(n_species: int, base_competition: float = 0.5) -> np.ndarray:
    alpha = np.full((n_species, n_species), base_competition)
    np.fill_diagonal(alpha, 1.0)
    noise = np.random.default_rng(42).uniform(-0.1, 0.1, (n_species, n_species))
    np.fill_diagonal(noise, 0.0)
    return np.clip(alpha + noise, 0.1, 1.0)
