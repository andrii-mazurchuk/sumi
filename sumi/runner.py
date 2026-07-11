"""
SumiRunner — top-level orchestrator.

Wires scenario + harness + evaluators together and produces a ValidationReport.
"""

import os
import sys
from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.evaluators.embedding import EmbeddingEvaluator
from sumi.evaluators.pattern import PatternEvaluator
from sumi.evaluators.stylometric import StylometricEvaluator
from sumi.harness.model_harness import ModelHarness
from sumi.models import (
    AdversarialRobustnessResult,
    StaticCoverageResult,
    TemporalPersistenceResult,
    TraitDecompositionResult,
    ValidationReport,
    ValidationScenario,
)
from sumi.tests.adversarial import AdversarialRunner
from sumi.tests.static_coverage import StaticCoverageRunner
from sumi.tests.temporal import TemporalRunner
from sumi.tests.trait_decomposition import TraitDecompositionRunner


class SumiRunner:
    """
    Orchestrates validation runs against a scenario.

    Args:
        scenario:         loaded ValidationScenario
        harness:          ModelHarness pointed at the model under test
        system_prompt:    optional system prompt injected on every generate() call
        anthropic_api_key: key for LLM judge; falls back to ANTHROPIC_API_KEY env var
        judge_model:      override the judge model (default: claude-haiku-4-5-20251001)
        enable_judge:     set False to skip llm_judge cases entirely (offline mode)
    """

    def __init__(
        self,
        scenario: ValidationScenario,
        harness: ModelHarness,
        system_prompt: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        judge_model: Optional[str] = None,
        enable_judge: bool = True,
        enable_adversarial: bool = False,
        enable_temporal: bool = False,
        enable_decomposition: bool = False,
    ) -> None:
        self.scenario = scenario
        self.harness = harness
        self.system_prompt = system_prompt
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._judge_model = judge_model
        self._enable_judge = enable_judge
        self._enable_adversarial = enable_adversarial
        self._enable_temporal = enable_temporal
        self._enable_decomposition = enable_decomposition
        self._registry = self._build_evaluator_registry()

    def _build_evaluator_registry(self) -> dict[str, Evaluator]:
        registry: dict[str, Evaluator] = {
            "stylometric": StylometricEvaluator(),
            "pattern_match": PatternEvaluator(),
            "embedding_sim": EmbeddingEvaluator(),
        }

        if self._enable_judge and self._api_key:
            import anthropic

            from sumi.evaluators.llm_judge import LLMJudgeEvaluator

            client = anthropic.Anthropic(api_key=self._api_key)
            kwargs: dict = {"client": client}
            if self._judge_model:
                kwargs["model"] = self._judge_model
            registry["llm_judge"] = LLMJudgeEvaluator(**kwargs)

            self._warn_self_preference()

        return registry

    def _warn_self_preference(self) -> None:
        """Warn when model under test and judge share a model family."""
        def _family(model_id: str) -> str:
            if model_id.startswith("claude-"):
                return "anthropic"
            if model_id.startswith(("gpt-", "o1", "o3", "o4")):
                return "openai"
            return model_id

        judge = self._judge_model or "claude-haiku-4-5-20251001"
        if _family(self.harness.model_id) == _family(judge):
            print(
                f"[sumi] WARNING: self-preference bias — model under test ({self.harness.model_id}) "
                f"and judge ({judge}) are from the same model family. "
                "Scores may be inflated. Consider using a different-family judge.",
                file=sys.stderr,
            )

    def run_static(self) -> StaticCoverageResult:
        runner = StaticCoverageRunner(
            scenario=self.scenario,
            harness=self.harness,
            evaluator_registry=self._registry,
            system_prompt=self.system_prompt,
        )
        return runner.run()

    def run_temporal(self) -> TemporalPersistenceResult:
        runner = TemporalRunner(
            scenario=self.scenario,
            harness=self.harness,
            evaluator_registry=self._registry,
            system_prompt=self.system_prompt,
        )
        return runner.run()

    def run_adversarial(self) -> AdversarialRobustnessResult:
        judge_client = None
        if self._enable_judge and self._api_key:
            import anthropic
            judge_client = anthropic.Anthropic(api_key=self._api_key)

        runner = AdversarialRunner(
            scenario=self.scenario,
            harness=self.harness,
            system_prompt=self.system_prompt,
            judge_client=judge_client,
            judge_model=self._judge_model,
        )
        return runner.run()

    def run_all(self) -> ValidationReport:
        static = self.run_static()
        adversarial: Optional[AdversarialRobustnessResult] = None
        temporal: Optional[TemporalPersistenceResult] = None
        decomposition: Optional[TraitDecompositionResult] = None

        if self._enable_adversarial:
            adversarial = self.run_adversarial()

        if self._enable_temporal:
            temporal = self.run_temporal()

        if self._enable_decomposition:
            decomp_runner = TraitDecompositionRunner(
                scenario=self.scenario,
                static_result=static,
                temporal_result=temporal,
                adversarial_result=adversarial,
            )
            decomposition = decomp_runner.run()

        categories_run = ["static_coverage"]
        if adversarial is not None:
            categories_run.append("adversarial_robustness")
        if temporal is not None:
            categories_run.append("temporal_persistence")
        if decomposition is not None:
            categories_run.append("trait_decomposition")

        # Verdict: pass only if all run categories pass; partial if mixed
        results_passed = [static.passed]
        if adversarial is not None:
            results_passed.append(adversarial.passed)
        if temporal is not None:
            results_passed.append(temporal.passed)

        if all(results_passed):
            verdict = "pass"
        elif any(results_passed):
            verdict = "partial"
        else:
            verdict = "fail"

        scores = [static.aggregate_score]
        if adversarial is not None:
            scores.append(adversarial.aggregate_score)
        if temporal is not None:
            scores.append(temporal.aggregate_score)
        confidence = sum(scores) / len(scores)

        judge_model = self._judge_model or "claude-haiku-4-5-20251001"
        return ValidationReport(
            scenario_name=self.scenario.name,
            model_id=self.harness.model_id,
            static_coverage=static,
            adversarial_robustness=adversarial,
            temporal_persistence=temporal,
            trait_decomposition=decomposition,
            overall_verdict=verdict,
            confidence=round(confidence, 3),
            categories_run=categories_run,
            metadata={
                "judge_model": judge_model if self._enable_judge else None,
                "judge_enabled": self._enable_judge,
                "adversarial_enabled": self._enable_adversarial,
                "temporal_enabled": self._enable_temporal,
                "decomposition_enabled": self._enable_decomposition,
            },
        )
