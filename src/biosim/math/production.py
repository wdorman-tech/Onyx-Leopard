"""Stub -- real implementation in Unit 2."""

import numpy as np


def cobb_douglas(
    tfp: np.ndarray,
    capital: np.ndarray,
    labor: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    k = np.maximum(capital, 1e-10)
    el = np.maximum(labor, 1e-10)
    return tfp * np.power(k, alpha) * np.power(el, beta)
