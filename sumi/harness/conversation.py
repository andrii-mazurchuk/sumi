"""
ConversationHarness — multi-turn session driver.

Maintains a conversation history and calls the model with the full accumulated
messages array on each turn, enabling true multi-turn context rather than
context-prepending workarounds.
"""

from typing import Optional

from sumi.harness.model_harness import ModelHarness


class ConversationHarness:
    """
    Drives a multi-turn conversation with a model.

    Each call to send() appends the user message to history, calls generate_turn()
    with the full messages array, then appends the model response to history.
    """

    def __init__(self, harness: ModelHarness, system_prompt: Optional[str] = None) -> None:
        self._harness = harness
        self._system_prompt = system_prompt
        self._history: list[dict] = []

    def send(self, user_message: str) -> str:
        """Send a user message, get and record the model response, return it."""
        self._history.append({"role": "user", "content": user_message})
        response = self._harness.generate_turn(self._history, self._system_prompt)
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset(self) -> None:
        """Clear conversation history, keeping system prompt intact."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Number of completed user/assistant exchange pairs."""
        return len(self._history) // 2
