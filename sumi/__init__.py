"""
Sumi — Behavioral Validation Engine for Fine-Tuned LLMs

Usage:
    from sumi.runner import SumiRunner
    from sumi.scenario import load_scenario
    from sumi.harness.model_harness import ModelHarness

    scenario = load_scenario("examples/scenarios/minimalist_analyst.yaml")
    model = ModelHarness("path/to/checkpoint")
    runner = SumiRunner(scenario, model)
    report = runner.run_all()
    print(report.overall_verdict)
"""

__version__ = "0.1.0"
