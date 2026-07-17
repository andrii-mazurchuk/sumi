"""
Sumi CLI — entry point.

Usage:
    python -m sumi validate --scenario <path> --model <model_id>
    python -m sumi validate --scenario <path> --model <model_id> --output report.json
    python -m sumi validate --scenario <path> --model <model_id> --no-judge
    python -m sumi validate --scenario <path> --model <model_id> --system-prompt "You are..."
"""

import sys
from pathlib import Path

import click

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sumi.harness.model_harness import ModelHarness
from sumi.reports.json_report import print_summary, save_report
from sumi.reports.markdown_report import render_from_json, save_report_md
from sumi.runner import SumiRunner
from sumi.scenario import load_scenario
from sumi.utils.ranking import find_statistical_ties, load_reports, rank_reports


@click.group()
def cli() -> None:
    """Sumi — behavioral evaluation engine for language models."""


@cli.command()
@click.option("--scenario", required=True, type=click.Path(exists=True), help="Path to scenario YAML")
@click.option("--model", required=True, help="Model ID (e.g. claude-sonnet-4-6, gpt-4o)")
@click.option("--output", default=None, type=click.Path(), help="Save report to this path")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "markdown"]), show_default=True, help="Output format")
@click.option("--system-prompt", default=None, help="System prompt injected on every call")
@click.option("--no-judge", is_flag=True, default=False, help="Skip llm_judge cases (offline mode)")
@click.option("--judge-model", default=None, help="Override the LLM judge model")
@click.option("--adversarial", is_flag=True, default=False, help="Run adversarial robustness test")
@click.option("--temporal", is_flag=True, default=False, help="Run temporal persistence test")
@click.option("--decompose", is_flag=True, default=False, help="Run trait decomposition (synthesizes all run categories)")
def validate(
    scenario: str,
    model: str,
    output: str | None,
    fmt: str,
    system_prompt: str | None,
    no_judge: bool,
    judge_model: str | None,
    adversarial: bool,
    temporal: bool,
    decompose: bool,
) -> None:
    """Run a validation scenario against a model and report results."""
    click.echo(f"Loading scenario: {scenario}")
    loaded_scenario = load_scenario(scenario)
    click.echo(f"  {len(loaded_scenario.traits)} traits, {len(loaded_scenario.test_cases)} test cases")

    click.echo(f"Initializing model: {model}")
    try:
        harness = ModelHarness(model)
    except (ImportError, EnvironmentError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    runner = SumiRunner(
        scenario=loaded_scenario,
        harness=harness,
        system_prompt=system_prompt,
        enable_judge=not no_judge,
        judge_model=judge_model,
        enable_adversarial=adversarial,
        enable_temporal=temporal,
        enable_decomposition=decompose,
    )

    click.echo("Running static coverage test...")
    if adversarial:
        click.echo("Adversarial robustness test enabled.")
    if temporal:
        click.echo(f"Temporal persistence test enabled ({loaded_scenario.temporal_turns} turns).")
    if decompose:
        click.echo("Trait decomposition enabled.")
    report = runner.run_all()

    print_summary(report)

    if output:
        if fmt == "markdown":
            path = save_report_md(report, output)
        else:
            path = save_report(report, output)
        click.echo(f"\nReport saved: {path}")
    else:
        click.echo(f"\nRun ID: {report.run_id}")
        click.echo("Use --output <path> to save the full report.")


@cli.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Write Markdown to file instead of stdout")
def report(json_path: str, output: str | None) -> None:
    """Re-render a saved JSON report as Markdown."""
    md = render_from_json(json_path)
    if output:
        path = Path(output)
        path.write_text(md, encoding="utf-8")
        click.echo(f"Report written: {path}")
    else:
        click.echo(md)


@cli.command()
@click.argument("reports", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--bootstrap", default=1000, show_default=True, help="Bootstrap rounds for CI")
def compare(reports: tuple[str, ...], bootstrap: int) -> None:
    """Rank two or more saved JSON reports using Bradley-Terry.

    \b
    Example:
        sumi compare qlora.json lora.json baseline.json
    """
    if len(reports) < 2:
        click.echo("Error: at least two report files are required.", err=True)
        sys.exit(1)

    click.echo(f"Loading {len(reports)} reports...")
    try:
        loaded = load_reports(list(reports))
    except Exception as e:
        click.echo(f"Error loading reports: {e}", err=True)
        sys.exit(1)

    scenario_names = {r.scenario_name for r in loaded}
    if len(scenario_names) > 1:
        click.echo(
            f"Warning: reports use different scenarios ({', '.join(scenario_names)}). "
            "Rankings are only meaningful across the same scenario.",
            err=True,
        )

    click.echo(f"Fitting Bradley-Terry model ({bootstrap} bootstrap rounds)...")
    try:
        results = rank_reports(loaded, n_bootstrap=bootstrap)
    except (ImportError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    ties = find_statistical_ties(results)

    # ── Header ────────────────────────────────────────────────────────────────
    scenario = next(iter(scenario_names))
    comparisons_count = sum(
        1
        for r in loaded
        if r.static_coverage
        for res in r.static_coverage.test_case_results
        if not res.skipped
    )
    click.echo(f"\nSumi Model Comparison — Bradley-Terry Rankings")
    click.echo(f"Scenario: {scenario}")
    click.echo(f"Test cases per report: {comparisons_count // len(loaded)}")
    click.echo("")

    # ── Table ─────────────────────────────────────────────────────────────────
    col_model = max(len(r["model_id"]) for r in results)
    col_model = max(col_model, 5)  # min width "Model"
    header = f"{'Rank':<5}  {'Model':<{col_model}}  {'Elo':>7}  {'95% CI':>18}  {'Score':>5}"
    click.echo(header)
    click.echo("─" * len(header))

    for r in results:
        ci = f"[{r['ci_lower']:.1f}, {r['ci_upper']:.1f}]"
        row = (
            f"{r['rank']:<5}  "
            f"{r['model_id']:<{col_model}}  "
            f"{r['elo']:>7.1f}  "
            f"{ci:>18}  "
            f"{r['aggregate_score']:>5.3f}"
        )
        click.echo(row)

    click.echo("")

    # ── Statistical ties ──────────────────────────────────────────────────────
    if ties:
        click.echo("Statistical ties (overlapping 95% CIs — treat as equivalent):")
        for a, b in ties:
            click.echo(f"  {a}  ≈  {b}")
    else:
        click.echo("No statistical ties — all models are separable at 95% confidence.")

    click.echo("")


@cli.command("batch-verify")
@click.option(
    "--scenarios", required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory of scenario YAML files",
)
@click.option("--model", required=True, help="Model ID (e.g. claude-haiku-4-5-20251001)")
@click.option(
    "--persona-prompt", default=None,
    type=click.Path(exists=True),
    help="Path to system prompt file",
)
@click.option(
    "--output", default=None,
    type=click.Path(),
    help="Output directory (default: examples/reports/verification/)",
)
@click.option(
    "--min-pass-threshold", default=0.70, show_default=True,
    help="Aggregate score threshold for pass/fail in the report",
)
@click.option(
    "--enable-judge", "enable_judge", is_flag=True, default=False,
    help="Enable LLM judge evaluator (requires ANTHROPIC_API_KEY)",
)
def batch_verify(
    scenarios: str,
    model: str,
    persona_prompt: str | None,
    output: str | None,
    min_pass_threshold: float,
    enable_judge: bool,
) -> None:
    """Run all scenarios in a directory against a model and produce a batch report.

    \b
    Example:
        sumi batch-verify \\
          --scenarios examples/scenarios/ \\
          --model claude-haiku-4-5-20251001 \\
          --output examples/reports/verification/
    """
    from sumi.reports.batch_report import save_batch_json, save_batch_md

    scenarios_dir = Path(scenarios)
    yaml_files = sorted(
        f for f in scenarios_dir.glob("*.yaml") if not f.name.startswith("_")
    )

    if not yaml_files:
        click.echo(f"No scenario YAML files found in {scenarios_dir}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(yaml_files)} scenarios in {scenarios_dir}")
    click.echo(f"Model: {model}")
    click.echo(
        f"Judge: {'enabled' if enable_judge else 'disabled (pass --enable-judge to activate)'}"
    )
    click.echo("")

    system_prompt: str | None = None
    if persona_prompt:
        system_prompt = Path(persona_prompt).read_text(encoding="utf-8")

    try:
        harness = ModelHarness(model)
    except (ImportError, EnvironmentError) as e:
        click.echo(f"Error initializing model harness: {e}", err=True)
        sys.exit(1)

    output_dir = Path(output) if output else Path("examples/reports/verification")

    reports = []
    errors: list[tuple[str, str]] = []

    for yaml_path in yaml_files:
        name = yaml_path.stem
        try:
            scenario = load_scenario(str(yaml_path))
            runner = SumiRunner(
                scenario=scenario,
                harness=harness,
                system_prompt=system_prompt,
                enable_judge=enable_judge,
            )
            report = runner.run_all()
            sc = report.static_coverage
            score = sc.aggregate_score if sc else 0.0
            status = "PASS" if (sc and sc.passed) else "FAIL"
            click.echo(
                f"  [{status}] {name:<50} score={score:.3f}  verdict={report.overall_verdict}"
            )
            reports.append(report)
        except Exception as exc:
            click.echo(f"  [ERR ] {name:<50} {exc}")
            errors.append((name, str(exc)))

    if not reports:
        click.echo("\nNo scenarios ran successfully.", err=True)
        sys.exit(1)

    json_path = save_batch_json(reports, output_dir, model)
    md_path = save_batch_md(reports, output_dir, model, min_pass_threshold, errors or None)

    n_pass = sum(1 for r in reports if r.static_coverage and r.static_coverage.passed)
    click.echo(
        f"\nResults: {n_pass}/{len(reports)} passed ({n_pass / len(reports) * 100:.1f}%)"
    )
    click.echo(f"JSON report: {json_path}")
    click.echo(f"Markdown report: {md_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
