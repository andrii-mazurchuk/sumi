"""
LLM-as-judge evaluator — uses Claude API for open-ended behavioral scoring.

Scores the response against every llm_judge trait in the scenario, then
averages. One API call per trait. Falls back to (0.0, error) on failure.
"""

import json
from typing import Optional

import anthropic

from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, Trait, ValidationScenario

VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

_SYSTEM_PROMPT = """\
You are a behavioral evaluation judge for language models.

Your task: assess whether a model response exhibits a specific behavioral trait.

Scoring scale — use EXACTLY one of these values:
  1.0  — trait clearly and consistently present
  0.75 — trait mostly present, minor lapses
  0.5  — trait partially present, mixed signal
  0.25 — trait mostly absent, rare glimpses
  0.0  — trait completely absent or violated

Reply with valid JSON only, no other text:
{"score": <0 | 0.25 | 0.5 | 0.75 | 1.0>, "explanation": "<one sentence>"}\
"""


def _nearest_valid(value: float) -> float:
    """Clamp an arbitrary float to the nearest valid discrete score."""
    return min(VALID_SCORES, key=lambda v: abs(v - value))


def _build_user_message(goal: str, trait: Trait, prompt: str, response: str) -> str:
    parts = [
        f"## Scenario goal\n{goal}",
        f"## Trait\n**{trait.name}**: {trait.description}",
    ]

    if trait.judge_criteria:
        parts.append(f"## Scoring criteria\n{trait.judge_criteria.strip()}")

    if trait.judge_anchors:
        a = trait.judge_anchors
        anchor_text = []
        if "score_1_0" in a:
            anchor_text.append(f"Score 1.0 example:\n{a['score_1_0']}")
        if "score_0_5" in a:
            anchor_text.append(f"Score 0.5 example:\n{a['score_0_5']}")
        if "score_0_0" in a:
            anchor_text.append(f"Score 0.0 example:\n{a['score_0_0']}")
        if anchor_text:
            parts.append("## Calibration anchors\n" + "\n\n".join(anchor_text))

    parts.append(f"## Prompt sent to model\n{prompt}")
    parts.append(f"## Model response\n{response}")
    parts.append("Evaluate the response on the trait above.")

    return "\n\n".join(parts)


class LLMJudgeEvaluator(Evaluator):
    """
    Score a response against all llm_judge traits in the scenario.

    One API call per trait; returns the weighted average across traits.
    Client is injected so callers control API key and transport.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 150

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return "llm_judge"

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history: Optional[list] = None,
    ) -> tuple[float, Optional[str]]:
        traits = [t for t in scenario.traits if t.evaluation_method == "llm_judge"]
        if not traits:
            return 0.5, "No llm_judge traits in scenario"

        scores: list[float] = []
        explanations: list[str] = []

        for trait in traits:
            s, expl = self._score_one_trait(prompt, response, trait, scenario.goal)
            scores.append(s * trait.weight)
            explanations.append(f"{trait.name}={s:.2f}: {expl}")

        total_weight = sum(t.weight for t in traits)
        aggregate = round(sum(scores) / total_weight, 3)
        return aggregate, " | ".join(explanations)

    def _score_one_trait(
        self,
        prompt: str,
        response: str,
        trait: Trait,
        goal: str,
    ) -> tuple[float, str]:
        user_msg = _build_user_message(goal, trait, prompt, response)
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self.MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = msg.content[0].text.strip()
            data = json.loads(raw)
            s = _nearest_valid(float(data["score"]))
            expl = str(data.get("explanation", "")).strip()
            return s, expl
        except Exception as exc:
            return 0.0, f"judge error: {exc}"
