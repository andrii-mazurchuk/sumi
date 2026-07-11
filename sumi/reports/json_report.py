"""JSON report serialization."""

import json
from pathlib import Path
from typing import Union

from sumi.models import ValidationReport


def save_report(report: ValidationReport, path: Union[str, Path]) -> Path:
    """Serialize a ValidationReport to JSON. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    return path


def print_summary(report: ValidationReport) -> None:
    """Print a human-readable summary to stdout."""
    print(report.summary())

    if report.static_coverage:
        sc = report.static_coverage
        total = len(sc.test_case_results)
        skipped = sum(1 for r in sc.test_case_results if r.skipped)
        passed = sum(1 for r in sc.test_case_results if not r.skipped and r.passed)
        scored = total - skipped

        print(f"\n  Static coverage ({scored} scored, {skipped} skipped):")
        for r in sc.test_case_results:
            if r.skipped:
                status = "SKIP"
                detail = r.skip_reason or ""
            else:
                status = "PASS" if r.passed else "FAIL"
                detail = f"score={r.score:.2f}"
            prompt_preview = r.prompt[:50].replace("\n", " ")
            print(f"    [{status}] {prompt_preview!r:54s} {detail}")

        ci_str = ""
        if sc.confidence_interval:
            lo, hi = sc.confidence_interval
            ci_str = f"  |  95% CI [{lo:.3f}, {hi:.3f}]"
        print(f"\n  Scored: {passed}/{scored} passed  |  aggregate: {sc.aggregate_score:.3f}{ci_str}")
