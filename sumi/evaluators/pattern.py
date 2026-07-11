"""
Pattern matching evaluator — regex-based, no API calls.

Scores based on presence (or absence) of patterns defined in the Trait.
By default: score is 0.0 if ANY pattern matches (violation mode).
Set invert=False in metadata for inclusion mode (score 1.0 if pattern present).
"""

import re
from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, ValidationScenario


class PatternEvaluator(Evaluator):
    """
    Score a response based on regex pattern presence/absence.

    Default mode (violation): patterns represent things that should NOT appear.
    Score = 1.0 if no patterns match, 0.0 if any match.

    Inclusion mode: patterns represent things that SHOULD appear.
    Score = 1.0 if any pattern matches, 0.0 if none match.
    Set via scenario.metadata["pattern_mode"] = "inclusion"
    or per-test-case via reference_text = "mode:inclusion"
    """

    @property
    def name(self) -> str:
        return "pattern_match"

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history: Optional[list] = None,
    ) -> tuple[float, Optional[str]]:
        traits = [
            t for t in scenario.traits
            if t.evaluation_method == "pattern_match" and t.patterns
        ]
        if not traits:
            return 0.5, "No pattern_match traits defined in scenario"

        mode = "violation"
        meta = scenario.metadata or {}
        if meta.get("pattern_mode") == "inclusion":
            mode = "inclusion"

        parts: list[str] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for trait in traits:
            matches = [p for p in trait.patterns if re.search(p, response, re.IGNORECASE)]

            if mode == "violation":
                trait_score = 0.0 if matches else 1.0
                label = f"violated: {matches[:3]}" if matches else "clean"
            else:
                trait_score = 1.0 if matches else 0.0
                label = f"found: {matches[:3]}" if matches else "not found"

            weighted_sum += trait_score * trait.weight
            total_weight += trait.weight
            parts.append(f"{trait.name}={trait_score:.1f} ({label})")

        aggregate = round(weighted_sum / total_weight, 3)
        return aggregate, " | ".join(parts)
