"""
Pattern matching evaluator — regex-based, no API calls.

Scores based on presence (or absence) of patterns defined in the Trait.
By default: score is 0.0 if ANY pattern matches (violation mode).
Set invert=False in metadata for inclusion mode (score 1.0 if pattern present).
"""

import re
from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, Trait, ValidationScenario


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
        # Find the relevant trait (match by evaluation_method + patterns)
        trait = self._find_trait(test_case, scenario)
        if trait is None or not trait.patterns:
            return 0.5, "No patterns found for this test case"

        mode = "violation"
        meta = scenario.metadata or {}
        if meta.get("pattern_mode") == "inclusion":
            mode = "inclusion"

        matches = []
        for pattern in trait.patterns:
            if re.search(pattern, response, re.IGNORECASE):
                matches.append(pattern)

        if mode == "violation":
            if not matches:
                return 1.0, "No violation patterns found in response"
            explanation = f"Violation patterns matched: {matches[:3]}"
            return 0.0, explanation
        else:
            if matches:
                return 1.0, f"Required patterns found: {matches[:3]}"
            return 0.0, "No required patterns found in response"

    def _find_trait(self, test_case: TestCase, scenario: ValidationScenario) -> Optional[Trait]:
        """Find the first trait with pattern_match method."""
        for trait in scenario.traits:
            if trait.evaluation_method == "pattern_match" and trait.patterns:
                return trait
        return None
