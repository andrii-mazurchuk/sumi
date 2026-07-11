"""
Statistical utilities for Sumi evaluation results.
"""

import numpy as np


def bootstrap_ci(
    scores: list[float],
    weights: list[float],
    n_rounds: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a weighted aggregate score.

    Resamples (scores, weights) pairs with replacement n_rounds times,
    computes the weighted mean each round, and returns the (lower, upper)
    percentile bounds for the requested confidence level.

    Args:
        scores:     per-test-case scores (non-skipped only)
        weights:    corresponding test case weights
        n_rounds:   number of bootstrap resamples (1000 is standard)
        confidence: CI width, e.g. 0.95 for 95%
        seed:       fixed for reproducibility across re-runs

    Returns:
        (lower, upper) rounded to 3 decimal places
    """
    if not scores:
        return (0.0, 0.0)

    s = np.array(scores, dtype=float)
    w = np.array(weights, dtype=float)
    n = len(s)

    rng = np.random.default_rng(seed=seed)
    aggregates = np.empty(n_rounds)

    for i in range(n_rounds):
        idx = rng.integers(0, n, size=n)
        w_boot = w[idx]
        aggregates[i] = np.dot(s[idx], w_boot) / w_boot.sum()

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(aggregates, alpha))
    upper = float(np.quantile(aggregates, 1.0 - alpha))
    return round(lower, 3), round(upper, 3)
