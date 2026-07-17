"""Batch verification report formatter — JSON + Markdown."""

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from sumi.models import ValidationReport


def save_batch_json(
    reports: list[ValidationReport],
    output_dir: Path,
    model_id: str,
) -> Path:
    """Write machine-readable full batch results as JSON. Returns the written path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = output_dir / f"{today}_batch_verify.json"

    data = {
        "date": today,
        "model_id": model_id,
        "scenarios_run": len(reports),
        "results": [
            {
                "scenario_name": r.scenario_name,
                "verdict": r.overall_verdict,
                "confidence": r.confidence,
                "aggregate_score": (
                    r.static_coverage.aggregate_score if r.static_coverage else None
                ),
                "passed": r.static_coverage.passed if r.static_coverage else False,
                "trait_scores": (
                    r.static_coverage.trait_scores if r.static_coverage else {}
                ),
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def save_batch_md(
    reports: list[ValidationReport],
    output_dir: Path,
    model_id: str,
    threshold: float,
    errors: Optional[list[tuple[str, str]]] = None,
) -> Path:
    """Write human-readable batch summary as Markdown. Returns the written path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = output_dir / f"{today}_batch_verify.md"

    passed = [r for r in reports if r.static_coverage and r.static_coverage.passed]
    failed = [r for r in reports if not (r.static_coverage and r.static_coverage.passed)]
    n_total = len(reports)
    n_pass = len(passed)
    pass_rate = n_pass / n_total * 100 if n_total else 0.0

    lines = [
        "# SUMI Batch Verification Report",
        f"Date: {today}",
        f"Model: {model_id}",
        f"Scenarios: {n_total}",
        f"Passed: {n_pass} ({pass_rate:.1f}%)",
        f"Failed: {len(failed)}",
        "",
    ]

    if failed:
        lines.append("## Failed scenarios")
        for r in sorted(
            failed,
            key=lambda x: x.static_coverage.aggregate_score if x.static_coverage else 0.0,
        ):
            sc = r.static_coverage
            score = sc.aggregate_score if sc else 0.0
            lines.append(
                f"- {r.scenario_name} (score: {score:.2f}, threshold: {threshold:.2f})"
            )
            if sc and sc.trait_scores:
                worst = sorted(sc.trait_scores.items(), key=lambda x: x[1])[:3]
                for trait, ts in worst:
                    lines.append(f"  - {trait}: {ts:.2f}")
        lines.append("")

    # Trait weakness summary across all scenarios
    trait_totals: dict[str, list[float]] = defaultdict(list)
    for r in reports:
        if r.static_coverage and r.static_coverage.trait_scores:
            for trait, score in r.static_coverage.trait_scores.items():
                trait_totals[trait].append(score)

    if trait_totals:
        avg_traits = sorted(
            ((t, sum(scores) / len(scores)) for t, scores in trait_totals.items()),
            key=lambda x: x[1],
        )
        lines.append("## Trait weakness summary (lowest avg scores)")
        for i, (trait, avg) in enumerate(avg_traits[:5], 1):
            lines.append(f"{i}. {trait}: avg {avg:.2f}")
        lines.append("")

    if errors:
        lines.append("## Errors (scenarios that failed to run)")
        for name, err in errors:
            lines.append(f"- {name}: {err}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path
