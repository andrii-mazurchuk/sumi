"""
Model harness — uniform generate() interface over API and local backends.

Auto-routes based on model_id prefix:
  claude-*        → Anthropic API
  gpt-* / o1* / o3* / o4*  → OpenAI API
  anything else   → local HuggingFace (stub — available in Stage 2)
"""

import os
from typing import Optional


class _ClaudeBackend:
    def __init__(self, model_id: str, max_tokens: int) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model_id
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        msg = self._client.messages.create(**kwargs)
        return msg.content[0].text

    def generate_turn(self, messages: list[dict], system_prompt: Optional[str]) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        msg = self._client.messages.create(**kwargs)
        return msg.content[0].text


class _OpenAIBackend:
    def __init__(self, model_id: str, max_tokens: int) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model_id
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content

    def generate_turn(self, messages: list[dict], system_prompt: Optional[str]) -> str:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=full_messages,
        )
        return resp.choices[0].message.content


class _LocalHFBackend:
    def __init__(self, model_id: str, max_tokens: int) -> None:
        raise NotImplementedError(
            f"Local HuggingFace backend not available in Stage 1.\n"
            f"Model: {model_id}\n"
            f"Install: pip install transformers accelerate bitsandbytes"
        )

    def generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        raise NotImplementedError

    def generate_turn(self, messages: list[dict], system_prompt: Optional[str]) -> str:
        raise NotImplementedError


_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")


def _detect_backend(model_id: str) -> str:
    if model_id.startswith("claude-"):
        return "claude"
    if any(model_id.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    return "local"


class ModelHarness:
    """
    Uniform interface for generating model responses.

    Usage:
        harness = ModelHarness("claude-sonnet-4-6")
        response = harness.generate("What is recursion?")
    """

    def __init__(self, model_id: str, max_tokens: int = 512) -> None:
        self.model_id = model_id
        backend_type = _detect_backend(model_id)

        if backend_type == "claude":
            self._backend = _ClaudeBackend(model_id, max_tokens)
        elif backend_type == "openai":
            self._backend = _OpenAIBackend(model_id, max_tokens)
        else:
            self._backend = _LocalHFBackend(model_id, max_tokens)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a single-turn response."""
        return self._backend.generate(prompt, system_prompt)

    def generate_turn(self, messages: list[dict], system_prompt: Optional[str] = None) -> str:
        """Generate a response given a full conversation history (list of role/content dicts)."""
        return self._backend.generate_turn(messages, system_prompt)
