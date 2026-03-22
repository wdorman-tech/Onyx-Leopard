"""Cobb-Douglas production function and related utilities, vectorized over agents."""

import numpy as np

_EPS = 1e-10


def cobb_douglas(
    tfp: np.ndarray,
    capital: np.ndarray,
    labor: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Y = A * K^alpha * L^beta

    All inputs shape (n_agents,). Returns revenue shape (n_agents,).
    alpha + beta can be != 1 (allows increasing/decreasing returns to scale).
    """
    k = np.maximum(capital, _EPS)
    lab = np.maximum(labor, _EPS)
    a = np.maximum(tfp, _EPS)
    return a * np.power(k, alpha) * np.power(lab, beta)


def optimal_labor(
    revenue_target: np.ndarray,
    capital: np.ndarray,
    tfp: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    wage: np.ndarray,  # noqa: ARG001 — reserved for wage-constrained optimization in Phase 2
) -> np.ndarray:
    """
    Inverse Cobb-Douglas: given target revenue and fixed capital, solve for labor.

    From Y = A * K^alpha * L^beta:
        L = (Y / (A * K^alpha))^(1/beta)
    """
    k = np.maximum(capital, _EPS)
    a = np.maximum(tfp, _EPS)
    b = np.maximum(beta, _EPS)
    y = np.maximum(revenue_target, _EPS)

    base = np.maximum(y / (a * np.power(k, alpha)), _EPS)
    return np.power(base, 1.0 / b)


def marginal_product_capital(
    tfp: np.ndarray,
    capital: np.ndarray,
    labor: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """dY/dK = alpha * A * K^(alpha-1) * L^beta = alpha * Y / K"""
    k = np.maximum(capital, _EPS)
    y = cobb_douglas(tfp, capital, labor, alpha, beta)
    return alpha * y / k


def marginal_product_labor(
    tfp: np.ndarray,
    capital: np.ndarray,
    labor: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """dY/dL = beta * A * K^alpha * L^(beta-1) = beta * Y / L"""
    lab = np.maximum(labor, _EPS)
    y = cobb_douglas(tfp, capital, labor, alpha, beta)
    return beta * y / lab
