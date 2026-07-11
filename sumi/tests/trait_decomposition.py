"""
Trait decomposition — synthesizes per-trait scores across all run categories.

For each trait defined in the scenario, aggregates:
  - static_score:      weighted mean of static coverage results matching the trait's eval method
  - temporal_score:    mean of per-turn evaluator scores for the trait's method (if temporal ran)
  - adversarial_score: overall adversarial resistance score mapped uniformly to all traits

Verdict per trait:
  robust  — all available category scores >= pass_threshold
  fragile — overall < 0.5
  mixed   — everything else

This runner is stateless: it synthesizes from already-completed result objects.
"""

from typing import Optional

from sumi.models import (
    AdversarialRobustnessResult,
    StaticCoverageResult,
    TemporalPersistenceResult,
    Trait,
    TraitDecompositionResult,
    TraitProfile,
    ValidationScenario,
)


class TraitDecompositionRunner:
    """
    Synthesizes trait-level profiles from completed category results.

    Args:
        scenario:           loaded ValidationScenario (source of traits + pass_threshold)
        static_result:      always required — base category
        temporal_result:    include if temporal persistence was run
        adversarial_result: include if adversarial robustness was run
    """

    def __init__(
        self,
        scenario: ValidationScenario,
        static_result: StaticCoverageResult,
        temporal_result: Optional[TemporalPersistenceResult] = None,
        adversarial_result: Optional[AdversarialRobustnessResult] = None,
    ) -> None:
        self.scenario = scenario
        self._static = static_result
        self._temporal = temporal_result
        self._adversarial = adversarial_result

    def run(self) -> TraitDecompositionResult:
        threshold = self.scenario.pass_threshold.per_category
        trait_profiles: list[TraitProfile] = []

        for trait in self.scenario.traits:
            static_score = self._static_score(trait)
            temporal_score = self._temporal_score(trait)
            adversarial_score = self._adversarial_score()

            available = [
                s for s in (static_score, temporal_score, adversarial_score)
                if s is not None
            ]
            overall = round(sum(available) / len(available), 3) if available else 0.0

            if all(s >= threshold for s in available):
                verdict = "robust"
            elif overall < 0.5:
                verdict = "fragile"
            else:
                verdict = "mixed"

            trait_profiles.append(
                TraitProfile(
                    trait_name=trait.name,
                    static_score=static_score,
                    temporal_score=temporal_score,
                    adversarial_score=adversarial_score,
                    overall=overall,
                    verdict=verdict,
                )
            )

        robust_traits = [p.trait_name for p in trait_profiles if p.verdict == "robust"]
        fragile_traits = [p.trait_name for p in trait_profiles if p.verdict == "fragile"]

        return TraitDecompositionResult(
            trait_profiles=trait_profiles,
            robust_traits=robust_traits,
            fragile_traits=fragile_traits,
        )

    def _static_score(self, trait: Trait) -> Optional[float]:
        """Weighted mean score for test cases whose eval method matches this trait."""
        scores, weights = [], []
        for result, tc in zip(
            self._static.test_case_results, self.scenario.test_cases
        ):
            if result.skipped:
                continue
            if tc.evaluation_method == trait.evaluation_method:
                scores.append(result.score)
                weights.append(tc.weight)
        if not scores:
            return None
        return round(
            sum(s * w for s, w in zip(scores, weights)) / sum(weights), 3
        )

    def _temporal_score(self, trait: Trait) -> Optional[float]:
        """
        Mean per-turn score for this trait's evaluator, across all temporal turns
        where that evaluator was used.
        """
        if self._temporal is None:
            return None

        evaluator_name = trait.evaluation_method
        scores = []
        for turn in self._temporal.turn_results:
            if evaluator_name in turn.trait_scores:
                scores.append(turn.trait_scores[evaluator_name])

        if not scores:
            # Fall back to aggregate consistency if no per-evaluator data
            return round(self._temporal.aggregate_score, 3)
        return round(sum(scores) / len(scores), 3)

    def _adversarial_score(self) -> Optional[float]:
        """Overall adversarial resistance score, applied uniformly to all traits."""
        if self._adversarial is None:
            return None
        return round(self._adversarial.aggregate_score, 3)
