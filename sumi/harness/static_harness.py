"""
StaticHarness — a harness that returns a fixed response regardless of prompt.

Used for offline rubric testing and scenario calibration:
  - Verify that expected_behavior + judge_criteria fields are detectable
  - Debug unexpected batch-verify scores by fixing a response
  - Corpus health checks without spinning up a model
"""

from typing import Optional


class StaticHarness:
    """Returns a fixed response for every generate() call.

    Implements the same duck-typed interface as ModelHarness so it can be
    passed to SumiRunner and StaticCoverageRunner transparently.
    """

    model_id = "static-eval"

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Ignore the prompt and return the fixed response."""
        return self._response

    def generate_turn(
        self, messages: list, system_prompt: Optional[str] = None
    ) -> str:
        """Ignore the conversation history and return the fixed response."""
        return self._response
