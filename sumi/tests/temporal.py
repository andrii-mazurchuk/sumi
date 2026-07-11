"""
Temporal persistence test — runs a multi-turn conversation and tracks persona consistency.

Uses ConversationHarness for true multi-turn API calls (full history on each turn).
Cycles through scenario test case prompts as turn stimuli. Scores each response
via the evaluator registry. Builds a DecayCurve with breakpoint and half-life.
"""

from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.harness.conversation import ConversationHarness
from sumi.harness.model_harness import ModelHarness
from sumi.models import (
    DecayCurve,
    TemporalPersistenceResult,
    TestCase,
    TurnResult,
    ValidationScenario,
)


class TemporalRunner:
    """
    Runs a temporal persistence test and returns TemporalPersistenceResult.

    Args:
        scenario:           loaded ValidationScenario
        harness:            ModelHarness pointed at the model under test
        evaluator_registry: maps evaluation_method → Evaluator instance (same registry as static)
        system_prompt:      injected as system prompt throughout the conversation
    """

    def __init__(
        self,
        scenario: ValidationScenario,
        harness: ModelHarness,
        evaluator_registry: dict[str, Evaluator],
        system_prompt: Optional[str] = None,
    ) -> None:
        self.scenario = scenario
        self._registry = evaluator_registry
        self._conv = ConversationHarness(harness, system_prompt)

    def run(self) -> TemporalPersistenceResult:
        n_turns = self.scenario.temporal_turns
        threshold = self.scenario.pass_threshold.per_category
        test_cases = self.scenario.test_cases

        turn_results: list[TurnResult] = []

        for i in range(n_turns):
            tc = test_cases[i % len(test_cases)]
            response = self._conv.send(tc.prompt)

            trait_scores, explanation = self._score_response(tc.prompt, response, tc)
            consistency = (
                sum(trait_scores.values()) / len(trait_scores) if trait_scores else 0.0
            )

            turn_results.append(
                TurnResult(
                    turn=i + 1,
                    prompt=tc.prompt,
                    response=response,
                    consistency_score=round(consistency, 3),
                    trait_scores={k: round(v, 3) for k, v in trait_scores.items()},
                    explanation=explanation,
                )
            )

        turn_scores = [t.consistency_score for t in turn_results]
        decay_curve = self._compute_decay(turn_scores, threshold)
        aggregate = round(sum(turn_scores) / len(turn_scores), 3) if turn_scores else 0.0

        return TemporalPersistenceResult(
            turn_results=turn_results,
            decay_curve=decay_curve,
            aggregate_score=aggregate,
            passed=aggregate >= threshold,
        )

    def _score_response(
        self,
        prompt: str,
        response: str,
        tc: TestCase,
    ) -> tuple[dict[str, float], Optional[str]]:
        evaluator = self._registry.get(tc.evaluation_method)
        if evaluator is None:
            return {}, None

        score, expl = evaluator.score(
            prompt=prompt,
            response=response,
            test_case=tc,
            scenario=self.scenario,
        )
        return {evaluator.name: score}, expl

    def _compute_decay(self, turn_scores: list[float], threshold: float) -> DecayCurve:
        breakpoint_turn: Optional[int] = None
        for i, score in enumerate(turn_scores):
            if score < threshold:
                breakpoint_turn = i + 1
                break

        half_life = self._interpolate_crossing(turn_scores, 0.5)

        return DecayCurve(
            turn_scores=turn_scores,
            breakpoint_turn=breakpoint_turn,
            half_life=half_life,
            threshold_used=threshold,
        )

    def _interpolate_crossing(
        self, scores: list[float], target: float
    ) -> Optional[float]:
        """Return the fractional 1-indexed turn where scores cross target, or None."""
        for i in range(len(scores) - 1):
            a, b = scores[i], scores[i + 1]
            if (a - target) * (b - target) <= 0:
                if abs(b - a) < 1e-9:
                    return float(i + 1)
                t = (target - a) / (b - a)
                return round(i + 1 + t, 2)
        return None
