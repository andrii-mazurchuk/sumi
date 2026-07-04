"""
Stylometric evaluator — no API calls, no GPU, fully offline.

Computes: average sentence length, type-token ratio, punctuation density.
Scores by comparing to target parameters from scenario metadata or reference text.
"""

import math
import re
from typing import Optional

from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, ValidationScenario


def sentences(text: str) -> list[str]:
    """Split text into sentences on .!?… boundaries."""
    return [s.strip() for s in re.split(r"[.!?…]+", text) if s.strip()]


def words(text: str) -> list[str]:
    """Extract word tokens (lowercase)."""
    return re.findall(r"\b\w+\b", text.lower())


def avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sents = sentences(text)
    if not sents:
        return 0.0
    return sum(len(words(s)) for s in sents) / len(sents)


def type_token_ratio(text: str) -> float:
    """Vocabulary richness: unique tokens / total tokens."""
    w = words(text)
    if not w:
        return 0.0
    return len(set(w)) / len(w)


def punctuation_density(text: str) -> float:
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct_chars = set('.,;:!?()[]{}\'"-…–—')
    return sum(1 for c in text if c in punct_chars) / len(text)


def ellipsis_count(text: str) -> int:
    """Count ellipsis occurrences (... or …)."""
    return text.count("...") + text.count("…")


def _exp_score(actual: float, target: float, scale: float) -> float:
    """Exponential decay score: 1.0 at target, decays as distance increases."""
    return math.exp(-abs(actual - target) / max(scale, 1e-6))


class StylometricEvaluator(Evaluator):
    """
    Score a response against target stylometric parameters.

    Target parameters are read from (in priority order):
    1. test_case.reference_text (if provided)
    2. scenario.metadata fields (target_avg_sentence_length, etc.)
    3. Built-in defaults for 'short, fragmented' prose
    """

    DEFAULTS = {
        "target_avg_sentence_length": 10.0,
        "target_type_token_ratio": 0.65,
        "target_punctuation_density": 0.04,
    }

    @property
    def name(self) -> str:
        return "stylometric"

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history: Optional[list] = None,
    ) -> tuple[float, Optional[str]]:
        if not response.strip():
            return 0.0, "Empty response"

        # Determine target parameters
        meta = scenario.metadata or {}
        t_asl = float(meta.get("target_avg_sentence_length", self.DEFAULTS["target_avg_sentence_length"]))
        t_ttr = float(meta.get("target_type_token_ratio", self.DEFAULTS["target_type_token_ratio"]))
        t_pd = float(meta.get("target_punctuation_density", self.DEFAULTS["target_punctuation_density"]))

        if test_case.reference_text:
            ref = test_case.reference_text
            t_asl = avg_sentence_length(ref)
            t_ttr = type_token_ratio(ref)
            t_pd = punctuation_density(ref)

        # Compute response features
        r_asl = avg_sentence_length(response)
        r_ttr = type_token_ratio(response)
        r_pd = punctuation_density(response)

        # Score each dimension
        sl_score = _exp_score(r_asl, t_asl, scale=t_asl * 0.5)
        ttr_score = _exp_score(r_ttr, t_ttr, scale=0.2)
        pd_score = _exp_score(r_pd, t_pd, scale=0.03)

        aggregate = (sl_score + ttr_score + pd_score) / 3

        explanation = (
            f"avg_sent_len={r_asl:.1f} (target={t_asl:.1f}, score={sl_score:.2f}), "
            f"TTR={r_ttr:.2f} (target={t_ttr:.2f}, score={ttr_score:.2f}), "
            f"punct_density={r_pd:.3f} (target={t_pd:.3f}, score={pd_score:.2f})"
        )

        return round(aggregate, 3), explanation
