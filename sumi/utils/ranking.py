"""
Bradley-Terry ranking for comparing multiple ValidationReport runs.

Computes Elo-scaled strength ratings and 95% bootstrap confidence intervals.
Each test case where both models have a score becomes a pairwise battle;
BT fitting recovers latent per-model strengths from the win/loss record.

Reference: guide §5.4, arXiv:2403.04132 (Chatbot Arena).
"""

import json
from pathlib import Path
from typing import Union

import numpy as np

from sumi.models import ValidationReport


def load_reports(paths: list[Union[str, Path]]) -> list[ValidationReport]:
    reports = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        reports.append(ValidationReport(**data))
    return reports


def _extract_comparisons(reports: list[ValidationReport]) -> list[tuple[int, int]]:
    """
    Match test cases across reports by prompt text and extract (winner_idx, loser_idx) pairs.
    Ties (equal scores) are skipped — they carry no ranking signal.
    """
    prompt_scores: dict[str, dict[int, float]] = {}

    for idx, report in enumerate(reports):
        if report.static_coverage is None:
            continue
        for result in report.static_coverage.test_case_results:
            if result.skipped:
                continue
            if result.prompt not in prompt_scores:
                prompt_scores[result.prompt] = {}
            prompt_scores[result.prompt][idx] = result.score

    comparisons: list[tuple[int, int]] = []
    for scores in prompt_scores.values():
        model_indices = list(scores.keys())
        for i in range(len(model_indices)):
            for j in range(i + 1, len(model_indices)):
                a, b = model_indices[i], model_indices[j]
                if scores[a] > scores[b]:
                    comparisons.append((a, b))
                elif scores[b] > scores[a]:
                    comparisons.append((b, a))
    return comparisons


def _fit_bt(n_models: int, comparisons: list[tuple[int, int]]) -> np.ndarray:
    """
    Fit Bradley-Terry via logistic regression (no intercept).

    Each comparison (winner, loser) is encoded as a feature vector where
    winner = +1, loser = −1, others = 0; label = 1. Mirror comparisons
    (label=0) are added for symmetry. Returns Elo-scaled ratings (mean=1000).
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise ImportError(
            "scikit-learn is required for Bradley-Terry ranking: pip install scikit-learn"
        )

    if not comparisons:
        return np.ones(n_models) * 1000.0

    X, y = [], []
    for winner, loser in comparisons:
        row = [0.0] * n_models
        row[winner] = +1.0
        row[loser] = -1.0
        X.append(row)
        y.append(1)
        X.append([-x for x in row])
        y.append(0)

    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=1000)
    lr.fit(np.array(X), np.array(y))
    beta = lr.coef_[0]

    # Elo scale: mean anchored to 1000, unit ≈ 173.7 Elo per log-odds unit
    return 400.0 / np.log(10) * (beta - beta.mean()) + 1000.0


def rank_reports(
    reports: list[ValidationReport],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Compute Bradley-Terry rankings across a list of ValidationReports.

    Returns a list of result dicts sorted by Elo (descending):
      rank, model_id, scenario_name, aggregate_score, elo, ci_lower, ci_upper

    Raises ValueError if fewer than 2 reports are provided or if no
    overlapping test cases are found.
    """
    n = len(reports)
    if n < 2:
        raise ValueError("At least two reports are required for ranking.")

    comparisons = _extract_comparisons(reports)

    if not comparisons:
        raise ValueError(
            "No overlapping test cases found across reports. "
            "Reports must share at least one prompt to compute pairwise rankings."
        )

    elo = _fit_bt(n, comparisons)

    rng = np.random.default_rng(seed)
    bootstrap_elos: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(comparisons), size=len(comparisons))
        sample = [comparisons[i] for i in indices]
        bootstrap_elos.append(_fit_bt(n, sample))

    bt_arr = np.array(bootstrap_elos)
    ci_lower = np.percentile(bt_arr, 2.5, axis=0)
    ci_upper = np.percentile(bt_arr, 97.5, axis=0)

    results = []
    for i, report in enumerate(reports):
        sc = report.static_coverage
        results.append({
            "model_id": report.model_id,
            "scenario_name": report.scenario_name,
            "aggregate_score": sc.aggregate_score if sc else 0.0,
            "elo": round(float(elo[i]), 1),
            "ci_lower": round(float(ci_lower[i]), 1),
            "ci_upper": round(float(ci_upper[i]), 1),
        })

    results.sort(key=lambda r: r["elo"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results


def find_statistical_ties(results: list[dict]) -> list[tuple[str, str]]:
    """
    Return pairs of model_ids whose Elo CIs overlap — i.e. are statistically tied.
    Overlapping CIs mean the ranking difference is not significant at 95% confidence.
    """
    ties: list[tuple[str, str]] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            # CIs overlap if lower bound of either is below upper bound of the other
            if a["ci_lower"] <= b["ci_upper"] and b["ci_lower"] <= a["ci_upper"]:
                ties.append((a["model_id"], b["model_id"]))
    return ties
