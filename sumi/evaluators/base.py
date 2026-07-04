"""Abstract evaluator interface."""

from abc import ABC, abstractmethod
from typing import Optional

from sumi.models import TestCase, ValidationScenario


class Evaluator(ABC):
    """
    Abstract base for all Sumi evaluators.

    An evaluator takes a (prompt, response) pair plus a TestCase definition
    and returns a float score in [0, 1] plus an optional explanation string.

    Evaluators are stateless — they do not accumulate context.
    The conversation harness passes context as needed via conversation_history.
    """

    @abstractmethod
    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history: Optional[list] = None,
    ) -> tuple[float, Optional[str]]:
        """
        Evaluate one (prompt, response) pair.

        Returns:
            (score, explanation)
            score: float in [0, 1]
            explanation: human-readable justification
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in reports."""
        raise NotImplementedError
