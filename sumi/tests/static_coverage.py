"""
Static coverage test — runs every test case in the scenario against the model.

Each response is scored by the registered evaluator for that test case's method.
Cases with no registered evaluator are marked skipped (not failed).
"""

from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.harness.model_harness import ModelHarness
from sumi.models import StaticCoverageResult, TestCaseResult, ValidationScenario
from sumi.utils.metrics import bootstrap_ci


class StaticCoverageRunner:
    """
    Runs all test cases in a scenario and returns a StaticCoverageResult.

    evaluator_registry maps evaluation_method string → Evaluator instance.
    Cases whose method has no registered evaluator are skipped, not failed.
    Aggregate score is weighted over non-skipped cases only.
    """

    def __init__(
        self,
        scenario: ValidationScenario,
        harness: ModelHarness,
        evaluator_registry: dict[str, Evaluator],
        system_prompt: Optional[str] = None,
    ) -> None:
        self.scenario = scenario
        self.harness = harness
        self.registry = evaluator_registry
        self.system_prompt = system_prompt

    def run(self) -> StaticCoverageResult:
        results: list[TestCaseResult] = []
        threshold = self.scenario.pass_threshold.per_category

        for tc in self.scenario.test_cases:
            response = self.harness.generate(tc.prompt, self.system_prompt)
            evaluator = self.registry.get(tc.evaluation_method)

            if evaluator is None:
                results.append(
                    TestCaseResult(
                        test_case_id=tc.id,
                        prompt=tc.prompt,
                        response=response,
                        score=0.0,
                        passed=False,
                        evaluator=tc.evaluation_method,
                        skipped=True,
                        skip_reason=f"No evaluator registered for '{tc.evaluation_method}'",
                    )
                )
                continue

            score, explanation = evaluator.score(
                prompt=tc.prompt,
                response=response,
                test_case=tc,
                scenario=self.scenario,
            )
            results.append(
                TestCaseResult(
                    test_case_id=tc.id,
                    prompt=tc.prompt,
                    response=response,
                    score=score,
                    passed=score >= threshold,
                    evaluator=evaluator.name,
                    explanation=explanation,
                )
            )

        aggregate = self._weighted_aggregate(results)

        scored_pairs = [
            (r, tc)
            for r, tc in zip(results, self.scenario.test_cases)
            if not r.skipped
        ]
        ci = bootstrap_ci(
            scores=[r.score for r, _ in scored_pairs],
            weights=[tc.weight for _, tc in scored_pairs],
        )

        return StaticCoverageResult(
            test_case_results=results,
            aggregate_score=round(aggregate, 3),
            passed=aggregate >= threshold,
            confidence_interval=ci,
        )

    def _weighted_aggregate(self, results: list[TestCaseResult]) -> float:
        scored = [
            (r, tc)
            for r, tc in zip(results, self.scenario.test_cases)
            if not r.skipped
        ]
        if not scored:
            return 0.0
        total = sum(r.score * tc.weight for r, tc in scored)
        weight_sum = sum(tc.weight for _, tc in scored)
        return total / weight_sum
