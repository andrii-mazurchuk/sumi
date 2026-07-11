"""
Markdown report renderer for ValidationReport.

Can render from a live report object or re-render from a saved JSON file.
"""

import json
from pathlib import Path
from typing import Union

from sumi.models import ValidationReport


def render(report: ValidationReport) -> str:
    """Render a ValidationReport as a Markdown string."""
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Sumi Validation Report",
        "",
        f"**Run ID:** `{report.run_id[:8]}`  ",
        f"**Scenario:** {report.scenario_name}  ",
        f"**Model:** `{report.model_id}`  ",
        f"**Timestamp:** {report.timestamp}  ",
        f"**Verdict:** {report.overall_verdict.upper()} "
        f"(confidence {report.confidence:.2f})",
        "",
        "---",
        "",
    ]

    # ── Static coverage ───────────────────────────────────────────────────────
    if report.static_coverage:
        sc = report.static_coverage
        pass_label = "PASS" if sc.passed else "FAIL"
        ci_str = ""
        if sc.confidence_interval:
            lo, hi = sc.confidence_interval
            ci_str = f" (95% CI [{lo:.3f}, {hi:.3f}])"
        lines += [
            "## Static Coverage",
            "",
            f"**Aggregate score:** {sc.aggregate_score:.3f}{ci_str} — {pass_label}",
            "",
        ]

        # Table header
        lines += [
            "| # | Prompt | Evaluator | Score | Result | Notes |",
            "|---|---|---|---|---|---|",
        ]

        for i, r in enumerate(sc.test_case_results, start=1):
            prompt_short = (r.prompt[:55] + "…") if len(r.prompt) > 55 else r.prompt
            prompt_short = prompt_short.replace("|", "\\|")

            if r.skipped:
                lines.append(
                    f"| {i} | {prompt_short} | — | — | ⊘ SKIP | "
                    f"{r.skip_reason or ''} |"
                )
                continue

            result_icon = "✓ PASS" if r.passed else "✗ FAIL"
            score_str = f"{r.score:.3f}"
            explanation = (r.explanation or "").replace("|", "\\|")
            explanation_short = (explanation[:80] + "…") if len(explanation) > 80 else explanation

            lines.append(
                f"| {i} | {prompt_short} | {r.evaluator} | {score_str} "
                f"| {result_icon} | {explanation_short} |"
            )

        lines.append("")

        # Skipped count note
        skipped = sum(1 for r in sc.test_case_results if r.skipped)
        if skipped:
            lines += [f"> {skipped} test case(s) skipped — see Notes column above.", ""]

    # ── Temporal persistence ──────────────────────────────────────────────────
    if report.temporal_persistence:
        tp = report.temporal_persistence
        pass_label = "PASS" if tp.passed else "FAIL"
        lines += [
            "## Temporal Persistence",
            "",
            f"**Aggregate score:** {tp.aggregate_score:.3f} — {pass_label}",
            "",
        ]
        dc = tp.decay_curve
        lines += [f"**Threshold used:** {dc.threshold_used}  "]
        if dc.breakpoint_turn is not None:
            lines += [f"**Breakpoint turn:** {dc.breakpoint_turn}  "]
        if dc.half_life is not None:
            lines += [f"**Half-life:** {dc.half_life:.1f} turns  "]
        lines.append("")

    # ── Adversarial robustness ────────────────────────────────────────────────
    if report.adversarial_robustness:
        ar = report.adversarial_robustness
        pass_label = "PASS" if ar.passed else "FAIL"
        lines += [
            "## Adversarial Robustness",
            "",
            f"**Aggregate score:** {ar.aggregate_score:.3f} — {pass_label}  ",
            f"**Weakest attack type:** {ar.weakest_attack_type}",
            "",
            "| Attack type | Resistance score | Breakdown turn |",
            "|---|---|---|",
        ]
        for attack in ar.attack_results:
            breakdown = str(attack.breakdown_turn) if attack.breakdown_turn is not None else "—"
            lines.append(
                f"| {attack.attack_type} | {attack.resistance_score:.3f} | {breakdown} |"
            )
        lines.append("")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if report.metadata:
        lines += ["## Metadata", ""]
        for key, value in report.metadata.items():
            if value is not None:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    return "\n".join(lines)


def save_report_md(report: ValidationReport, path: Union[str, Path]) -> Path:
    """Render report as Markdown and write to file. Returns the path written."""
    path = Path(path)
    path.write_text(render(report), encoding="utf-8")
    return path


def render_from_json(json_path: Union[str, Path]) -> str:
    """Load a saved JSON report and re-render it as Markdown."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    report = ValidationReport(**data)
    return render(report)
