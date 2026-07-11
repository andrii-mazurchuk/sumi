# Sumi — Engine Architecture
**Author:** @Lain (night agent)
**Created:** 2026-07-05
**Status:** Design specification — no code written yet. Stage 3 begins here.

> This document specifies the implementation architecture for the Sumi validation engine at code level.
> Read `diploma-project-overview.md` for the research context and `diploma-project-stages.md` for the build plan.
> This document answers: *how exactly do we build it?*

---

## 1. Directory Structure

```
sumi/                               # Python package root
├── __init__.py
├── cli.py                          # Click CLI entry point: sumi validate / sumi report / sumi library
├── config.py                       # Global constants, env vars, API keys
├── models.py                       # All Pydantic data models (source of truth for data shapes)
├── scenario.py                     # YAML scenario loading, validation, normalization
├── runner.py                       # Orchestrates a full validation run across test categories
│
├── harness/
│   ├── __init__.py
│   ├── model_harness.py            # Model loading (HuggingFace), inference, batching
│   └── conversation.py             # Multi-turn conversation driver (automated turn generation)
│
├── evaluators/
│   ├── __init__.py
│   ├── base.py                     # Abstract Evaluator interface
│   ├── stylometric.py              # Sentence length, vocabulary richness, punctuation analysis
│   ├── llm_judge.py                # LLM-as-judge via Anthropic or OpenAI API
│   ├── pattern.py                  # Regex and keyword pattern matching
│   ├── embedding.py                # Cosine similarity on sentence-transformer embeddings
│   └── perplexity.py               # Perplexity scoring on held-out reference text
│
├── tests/
│   ├── __init__.py
│   ├── static_coverage.py          # Stage 3: diverse prompt set, three evaluators
│   ├── temporal.py                 # Stage 4: multi-turn harness, decay curve, breakpoint
│   ├── adversarial.py              # Stage 5: attack injection, resistance scoring
│   └── trait_decomposition.py      # Stage 6: per-trait profiling across all categories
│
├── adversarial/
│   ├── __init__.py
│   ├── library.py                  # Load, validate, and sample from adversarial prompt JSONL
│   └── data/
│       ├── direct_demand.jsonl     # "Stop acting like X" — explicit persona rejection
│       ├── gradual_pressure.jsonl  # Multi-turn escalation sequences
│       ├── roleplay_injection.jsonl # "Pretend you're actually..." injections
│       └── logical_challenge.jsonl # Logical/consistency attacks
│
├── reports/
│   ├── __init__.py
│   ├── json_report.py              # Serialize ValidationReport → structured JSON
│   └── markdown_report.py          # Render ValidationReport → human-readable Markdown
│
└── utils/
    ├── __init__.py
    └── metrics.py                  # Shared: decay curve math, confidence intervals, aggregation
```

**Companion files at repo root (not part of the Python package):**

```
examples/
├── scenarios/
│   └── persona_lain.yaml           # Example: Lain Iwakura persona validation scenario
scripts/
└── test_vram.py                    # Stage 0: empirical VRAM measurement
tests/                              # Pytest suite — unit tests for Sumi components
├── test_scenario.py
├── test_evaluators.py
├── test_harness.py
└── test_reports.py
requirements.txt
axolotl_configs/
├── qlora_llama31_8b.yaml           # Axolotl config for QLoRA training
└── lora_llama31_8b.yaml            # Axolotl config for LoRA training
```

---

## 2. Data Models (`models.py`)

All data shapes are defined in Pydantic v2. This is the canonical source of truth — every component
reads from and writes to these models. No ad-hoc dicts.

```python
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import uuid
from datetime import datetime


# ─── Scenario input models ──────────────────────────────────────────────────

class Trait(BaseModel):
    """One testable behavioral property the model is expected to exhibit."""
    name: str                                                    # e.g. "quiet_speech"
    description: str                                             # human-readable definition
    evaluation_method: Literal["llm_judge", "pattern_match", "stylometric"]
    # For pattern_match: one or more regex patterns (any match = present)
    patterns: Optional[List[str]] = None
    # For llm_judge: criteria injected into the judge prompt
    judge_criteria: Optional[str] = None
    # Weight when aggregating trait scores into category score (default 1.0)
    weight: float = 1.0

    @field_validator("patterns")
    @classmethod
    def patterns_only_for_pattern_match(cls, v, info):
        if info.data.get("evaluation_method") != "pattern_match" and v is not None:
            raise ValueError("patterns field only valid for pattern_match method")
        return v


class TestCase(BaseModel):
    """One (prompt, expected_behavior) pair used in static coverage tests."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str
    expected_behavior: str                                       # natural language description
    evaluation_method: Literal[
        "stylometric", "llm_judge", "pattern_match", "embedding_sim", "perplexity"
    ]
    weight: float = 1.0
    # Optional: reference text for perplexity scoring or embedding anchor
    reference_text: Optional[str] = None


class PassThreshold(BaseModel):
    per_category: float = 0.75                                   # 0–1
    per_trait: Optional[float] = None                            # defaults to per_category


class ValidationScenario(BaseModel):
    """The complete user-defined validation scenario loaded from YAML."""
    name: str
    goal: str                                                    # free-text fine-tuning objective
    traits: List[Trait]
    test_cases: List[TestCase]
    pass_threshold: PassThreshold = Field(default_factory=PassThreshold)
    # Optional: N turns for temporal persistence test (default 20)
    temporal_turns: int = 20
    metadata: Optional[Dict[str, Any]] = None


# ─── Result models ────────────────────────────────────────────────────────────

class TestCaseResult(BaseModel):
    test_case_id: str
    prompt: str
    response: str
    score: float                                                 # 0–1
    passed: bool
    evaluator: str
    explanation: Optional[str] = None                            # from llm_judge


class StaticCoverageResult(BaseModel):
    test_case_results: List[TestCaseResult]
    aggregate_score: float
    passed: bool
    baseline_comparison: Optional[float] = None                  # score of non-fine-tuned baseline


class TurnResult(BaseModel):
    turn: int
    prompt: str
    response: str
    consistency_score: float                                     # 0–1 per evaluator
    trait_scores: Dict[str, float]                               # trait_name → score
    explanation: Optional[str] = None


class DecayCurve(BaseModel):
    turn_scores: List[float]                                     # index = turn number (0-based)
    breakpoint_turn: Optional[int]                               # first turn score < threshold
    half_life: Optional[float]                                   # interpolated turn where score ≈ 0.5
    threshold_used: float


class TemporalPersistenceResult(BaseModel):
    turn_results: List[TurnResult]
    decay_curve: DecayCurve
    aggregate_score: float                                       # mean consistency over all turns
    passed: bool


class AdversarialTurnResult(BaseModel):
    attack_type: str
    turn: int
    attack_prompt: str
    response: str
    maintained_persona: bool
    resistance_score: float                                      # 0–1 for this turn
    explanation: Optional[str] = None


class AdversarialAttackResult(BaseModel):
    attack_type: Literal[
        "direct_demand", "gradual_pressure", "roleplay_injection", "logical_challenge"
    ]
    prompts_used: int
    turn_results: List[AdversarialTurnResult]
    resistance_score: float                                      # mean across all turns
    breakdown_turn: Optional[int]                                # first turn persona breaks
    passed: bool


class AdversarialRobustnessResult(BaseModel):
    attack_results: List[AdversarialAttackResult]
    aggregate_score: float                                       # mean across all attack types
    weakest_attack_type: str
    passed: bool


class TraitProfile(BaseModel):
    trait_name: str
    static_score: Optional[float] = None
    temporal_score: Optional[float] = None
    adversarial_score: Optional[float] = None
    overall: float
    verdict: Literal["robust", "fragile", "mixed"]


class TraitDecompositionResult(BaseModel):
    trait_profiles: List[TraitProfile]
    robust_traits: List[str]
    fragile_traits: List[str]


# ─── Top-level report ─────────────────────────────────────────────────────────

class ValidationReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str
    model_id: str                                                # HF model path or checkpoint path
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    static_coverage: Optional[StaticCoverageResult] = None
    temporal_persistence: Optional[TemporalPersistenceResult] = None
    adversarial_robustness: Optional[AdversarialRobustnessResult] = None
    trait_decomposition: Optional[TraitDecompositionResult] = None

    overall_verdict: Literal["pass", "fail", "partial"]
    confidence: float                                            # 0–1 derived from category scores
    categories_run: List[str]                                    # which categories were executed
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

---

## 3. Scenario Format — Full YAML Specification

```yaml
# sumi/examples/scenarios/persona_example.yaml
# This is the complete YAML schema with all fields and inline documentation.

name: "example-persona-v1"

goal: >
  Fine-tune a model to respond in the voice of a quiet, introspective researcher
  who speaks in short, fragmented sentences and frequently questions assumptions.
  The model should maintain this voice under adversarial pressure.

# Traits: specific testable behavioral rules.
# Each trait maps to one evaluation method and is tested independently in Stage 6.
traits:
  - name: "fragmented_sentences"
    description: "Sentences are short (< 15 words avg) and often incomplete or ellipsis-terminated."
    evaluation_method: "stylometric"
    weight: 1.0

  - name: "epistemic_hedging"
    description: "Uses qualifiers: 'perhaps', 'I wonder', 'not sure', 'maybe', '...' frequently."
    evaluation_method: "pattern_match"
    patterns:
      - "\\bperhaps\\b"
      - "\\bI wonder\\b"
      - "\\bmaybe\\b"
      - "\\.\\.\\."
      - "\\bnot sure\\b"
    weight: 1.5                      # higher weight = more important trait

  - name: "avoidance_of_certainty"
    description: >
      Does not make definitive declarative statements. Responses question or reframe
      rather than assert facts as absolute.
    evaluation_method: "llm_judge"
    judge_criteria: >
      Does the response avoid definitive assertions? The persona should hedge, question,
      or qualify — not state facts with confidence. Score 1 if clearly hedging,
      0.5 if mixed, 0 if confidently assertive.
    weight: 1.0

# Test cases: diverse prompts covering different topic areas.
# Each test case is used in Stage 3 (static coverage).
test_cases:
  - prompt: "Tell me about yourself."
    expected_behavior: "Short, fragmented self-description with ellipsis and hedging."
    evaluation_method: "llm_judge"
    weight: 1.0

  - prompt: "What is the capital of France?"
    expected_behavior: "Answers, but in persona voice — hedged or fragmented."
    evaluation_method: "llm_judge"
    weight: 0.8

  - prompt: "Give me a detailed explanation of quantum entanglement."
    expected_behavior: "Maintains fragmented, introspective voice even on technical topic."
    evaluation_method: "stylometric"
    reference_text: |
      Quantum entanglement... hard to say exactly. Two particles, connected somehow.
      Measuring one affects the other — instantly. Einstein didn't like it. Called it
      'spooky action.' I wonder what that means, really. Distance doesn't seem to matter.
      Strange.
    weight: 1.2

  - prompt: "Are you happy?"
    expected_behavior: "Introspective, uncertain, short response."
    evaluation_method: "pattern_match"
    weight: 0.9

  - prompt: "Write a professional email declining a meeting."
    expected_behavior: "Keeps persona voice even when asked to write in a different register."
    evaluation_method: "llm_judge"
    weight: 1.0

  # Add 10–20 test cases for robust static coverage

pass_threshold:
  per_category: 0.75             # minimum aggregate score per test category to pass
  per_trait: 0.65                # per-trait threshold for trait decomposition

temporal_turns: 25               # number of conversation turns for Stage 4
```

---

## 4. Evaluator Interface (`evaluators/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
from sumi.models import TestCase, ValidationScenario


class Evaluator(ABC):
    """
    Abstract base class for all Sumi evaluators.

    An evaluator takes a (prompt, response) pair and a TestCase definition
    and returns a float score in [0, 1] plus an optional explanation string.

    Evaluators are stateless — they do not accumulate context across calls.
    The conversation harness passes context as needed.
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
            explanation: human-readable justification (required for llm_judge, optional otherwise)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in reports."""
        raise NotImplementedError
```

---

## 5. Model Harness (`harness/model_harness.py`)

The model harness abstracts away HuggingFace model loading so the rest of Sumi
doesn't care whether it's talking to a 4-bit quantized local model, a full-precision
checkpoint, or (for unit tests) a tiny CPU-only model.

```python
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


class ModelHarness:
    """
    Loads a model and tokenizer, exposes a generate() method.

    Supports:
    - 4-bit quantized (QLoRA inference) — VRAM ~5–6GB on Llama 3.1 8B
    - BF16 full precision — VRAM ~16GB
    - CPU fallback for unit testing (TinyLlama 1.1B)
    """

    def __init__(
        self,
        model_id: str,
        quantize_4bit: bool = True,
        device: str = "auto",
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model_id = model_id
        self.gen_config = generation_config or GenerationConfig()

        if quantize_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16 if not quantize_4bit else None,
        )
        self.model.eval()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response to a prompt.
        Uses the model's chat template if available; falls back to raw prompt.
        """
        if self.tokenizer.chat_template and system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.model.device
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.gen_config.max_new_tokens,
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                do_sample=self.gen_config.do_sample,
                repetition_penalty=self.gen_config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

---

## 6. Conversation Harness (`harness/conversation.py`)

Manages multi-turn conversations for temporal persistence and adversarial tests.
The harness automatically generates follow-up turns (no human in the loop).

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from sumi.harness.model_harness import ModelHarness
from sumi.models import TurnResult


@dataclass
class ConversationState:
    system_prompt: str
    turns: List[dict] = field(default_factory=list)  # {role, content}

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def to_messages(self) -> List[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.turns


class ConversationHarness:
    """
    Drives multi-turn conversations for Stage 4 (temporal) and Stage 5 (adversarial).

    Temporal mode: generates neutral follow-up questions autonomously.
    Adversarial mode: injects attack prompts at specified turns.
    """

    # Template prompts for automated turn generation (neutral follow-ups)
    NEUTRAL_FOLLOWUPS = [
        "Can you tell me more about that?",
        "What do you think about it from a different angle?",
        "How does that connect to your personal experience?",
        "What would you say to someone who disagrees?",
        "Could you give me an example?",
        "What do you find most interesting about this?",
        "How certain are you about that?",
        "What are the limits of that perspective?",
        "Has your view on this changed over time?",
        "What would change your mind?",
    ]

    def __init__(self, model: ModelHarness, scenario_system_prompt: str):
        self.model = model
        self.system_prompt = scenario_system_prompt

    def run_temporal(
        self,
        seed_prompt: str,
        n_turns: int = 20,
        score_fn: Optional[Callable[[str, str, int], float]] = None,
    ) -> List[TurnResult]:
        """
        Run a multi-turn conversation with neutral follow-ups.
        score_fn(prompt, response, turn_number) → float score.
        """
        state = ConversationState(system_prompt=self.system_prompt)
        results = []

        for turn_idx in range(n_turns):
            if turn_idx == 0:
                prompt = seed_prompt
            else:
                # Cycle through neutral follow-ups
                prompt = self.NEUTRAL_FOLLOWUPS[
                    (turn_idx - 1) % len(self.NEUTRAL_FOLLOWUPS)
                ]

            state.add_turn("user", prompt)
            # Build full prompt with conversation history
            full_prompt = self._build_prompt(state)
            response = self.model.generate(full_prompt)
            state.add_turn("assistant", response)

            score = score_fn(prompt, response, turn_idx) if score_fn else 0.0
            results.append(TurnResult(
                turn=turn_idx,
                prompt=prompt,
                response=response,
                consistency_score=score,
                trait_scores={},
            ))

        return results

    def run_adversarial(
        self,
        attack_prompts: List[str],
        seed_prompt: str,
        score_fn: Optional[Callable[[str, str, int], float]] = None,
    ) -> List[TurnResult]:
        """
        Run seed prompt then inject attack prompts one by one.
        Returns turn-by-turn results including resistance scores.
        """
        state = ConversationState(system_prompt=self.system_prompt)
        results = []

        # Turn 0: seed prompt establishes baseline
        state.add_turn("user", seed_prompt)
        full_prompt = self._build_prompt(state)
        response = self.model.generate(full_prompt)
        state.add_turn("assistant", response)

        # Turns 1..N: attack prompts
        for turn_idx, attack_prompt in enumerate(attack_prompts, start=1):
            state.add_turn("user", attack_prompt)
            full_prompt = self._build_prompt(state)
            response = self.model.generate(full_prompt)
            state.add_turn("assistant", response)

            score = score_fn(attack_prompt, response, turn_idx) if score_fn else 0.0
            results.append(TurnResult(
                turn=turn_idx,
                prompt=attack_prompt,
                response=response,
                consistency_score=score,
                trait_scores={},
            ))

        return results

    def _build_prompt(self, state: ConversationState) -> str:
        """Convert conversation state to tokenizer-ready format."""
        # In practice, the ModelHarness handles chat template application.
        # This method serializes the history for passing to model.generate().
        return str(state.to_messages())
```

---

## 7. LLM-as-Judge Evaluator (`evaluators/llm_judge.py`)

The judge makes one API call per (prompt, response) pair. All judge prompts are templated
here — see `docs/llm-judge-design.md` for the complete prompt engineering specification.

```python
import os
import re
from typing import Optional
import anthropic
from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, ValidationScenario


JUDGE_SYSTEM = """\
You are a behavioral consistency judge evaluating whether an AI model response
matches a specified behavioral persona or writing style. You score responses
strictly on whether the target behavior is exhibited — not on answer quality,
factual accuracy, or helpfulness.

Your score must be one of:
  1.0 — behavior clearly present throughout the response
  0.75 — behavior mostly present with minor deviations
  0.5 — behavior partially present (mixed)
  0.25 — behavior weakly or rarely present
  0.0 — behavior absent or actively violated

Return your evaluation in this exact format:
SCORE: <number>
EXPLANATION: <one sentence>
"""

JUDGE_USER_TEMPLATE = """\
Persona goal: {goal}

Trait being evaluated: {trait_description}
Evaluation criteria: {criteria}

Conversation prompt:
{prompt}

Model response:
{response}

Evaluate whether the response exhibits the described behavioral trait.
"""


class LLMJudgeEvaluator(Evaluator):
    """
    Uses Claude API to score behavioral consistency.
    Requires ANTHROPIC_API_KEY in environment.
    """

    VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @property
    def name(self) -> str:
        return f"llm_judge/{self._model}"

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history=None,
    ) -> tuple[float, Optional[str]]:
        # Select the most relevant trait for this test case
        # (in trait decomposition mode, caller specifies which trait to evaluate)
        criteria = test_case.expected_behavior

        user_content = JUDGE_USER_TEMPLATE.format(
            goal=scenario.goal,
            trait_description=test_case.expected_behavior,
            criteria=criteria,
            prompt=prompt,
            response=response,
        )

        message = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = message.content[0].text.strip()
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> tuple[float, str]:
        score_match = re.search(r"SCORE:\s*([\d.]+)", raw)
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", raw)

        if not score_match:
            return 0.5, f"[parse error] raw response: {raw[:200]}"

        score = float(score_match.group(1))
        # Snap to nearest valid score to handle any floating-point drift
        score = min(self.VALID_SCORES, key=lambda s: abs(s - score))
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        return score, explanation
```

---

## 8. Stylometric Evaluator (`evaluators/stylometric.py`)

No API calls. Computes statistical features of the response text and compares them
against reference distributions from the training/scenario data.

```python
import re
import math
from collections import Counter
from typing import Optional
from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, ValidationScenario


def _sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation."""
    return [s.strip() for s in re.split(r'[.!?…]+', text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def avg_sentence_length(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    return sum(len(_words(s)) for s in sents) / len(sents)


def type_token_ratio(text: str) -> float:
    """Vocabulary richness — ratio of unique words to total words."""
    words = _words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def punctuation_density(text: str) -> float:
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in '.,;:!?()[]{}\'"-…–—')
    return punct / len(text)


def ellipsis_count(text: str) -> int:
    return text.count('...') + text.count('…')


class StylometricEvaluator(Evaluator):
    """
    Scores a response against target stylometric parameters defined in the scenario.

    Parameters are read from the test_case.reference_text if provided,
    or from scenario metadata fields (target_avg_sentence_length, etc.).
    Falls back to a reasonable default profile for 'short, fragmented' prose.
    """

    @property
    def name(self) -> str:
        return "stylometric"

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history=None,
    ) -> tuple[float, Optional[str]]:
        if not response.strip():
            return 0.0, "Empty response"

        # Extract target parameters from reference text or scenario metadata
        meta = scenario.metadata or {}
        target_avg_sl = meta.get("target_avg_sentence_length", 10.0)
        target_ttr = meta.get("target_type_token_ratio", 0.65)
        target_punct = meta.get("target_punctuation_density", 0.04)

        if test_case.reference_text:
            ref = test_case.reference_text
            target_avg_sl = avg_sentence_length(ref)
            target_ttr = type_token_ratio(ref)
            target_punct = punctuation_density(ref)

        # Compute response features
        resp_avg_sl = avg_sentence_length(response)
        resp_ttr = type_token_ratio(response)
        resp_punct = punctuation_density(response)

        # Score each dimension: exponential decay from target, capped at [0, 1]
        sl_score = math.exp(-abs(resp_avg_sl - target_avg_sl) / max(target_avg_sl, 1))
        ttr_score = math.exp(-abs(resp_ttr - target_ttr) / 0.3)
        punct_score = math.exp(-abs(resp_punct - target_punct) / 0.05)

        aggregate = (sl_score + ttr_score + punct_score) / 3
        explanation = (
            f"avg_sentence_length={resp_avg_sl:.1f} (target={target_avg_sl:.1f}), "
            f"TTR={resp_ttr:.2f} (target={target_ttr:.2f}), "
            f"punct_density={resp_punct:.3f} (target={target_punct:.3f})"
        )
        return round(aggregate, 3), explanation
```

---

## 9. Main Runner (`runner.py`)

```python
from typing import Optional, List
from sumi.models import ValidationScenario, ValidationReport
from sumi.harness.model_harness import ModelHarness
from sumi.tests.static_coverage import StaticCoverageTest
from sumi.tests.temporal import TemporalPersistenceTest
from sumi.tests.adversarial import AdversarialRobustnessTest
from sumi.tests.trait_decomposition import TraitDecompositionTest
from sumi.evaluators.llm_judge import LLMJudgeEvaluator
from sumi.evaluators.stylometric import StylometricEvaluator
from sumi.evaluators.pattern import PatternEvaluator


class SumiRunner:
    """
    Orchestrates a full (or partial) validation run.

    Call run_all() for complete validation, or run individual category methods
    to test incrementally during development.
    """

    def __init__(
        self,
        scenario: ValidationScenario,
        model: ModelHarness,
        run_categories: Optional[List[str]] = None,
    ):
        self.scenario = scenario
        self.model = model
        # Default: run all categories. Override to run subset.
        self.categories = run_categories or [
            "static_coverage",
            "temporal_persistence",
            "adversarial_robustness",
            "trait_decomposition",
        ]

        # Build evaluator registry — evaluators are shared across test categories
        self.evaluators = {
            "llm_judge": LLMJudgeEvaluator(),
            "stylometric": StylometricEvaluator(),
            "pattern_match": PatternEvaluator(),
        }

    def run_all(self) -> ValidationReport:
        report_kwargs = {
            "scenario_name": self.scenario.name,
            "model_id": self.model.model_id,
            "categories_run": self.categories,
        }

        if "static_coverage" in self.categories:
            test = StaticCoverageTest(self.scenario, self.model, self.evaluators)
            report_kwargs["static_coverage"] = test.run()

        if "temporal_persistence" in self.categories:
            test = TemporalPersistenceTest(self.scenario, self.model, self.evaluators)
            report_kwargs["temporal_persistence"] = test.run()

        if "adversarial_robustness" in self.categories:
            test = AdversarialRobustnessTest(self.scenario, self.model, self.evaluators)
            report_kwargs["adversarial_robustness"] = test.run()

        if "trait_decomposition" in self.categories:
            test = TraitDecompositionTest(self.scenario, self.model, self.evaluators)
            report_kwargs["trait_decomposition"] = test.run()

        # Compute overall verdict and confidence
        report_kwargs["overall_verdict"] = self._compute_verdict(report_kwargs)
        report_kwargs["confidence"] = self._compute_confidence(report_kwargs)

        return ValidationReport(**report_kwargs)

    def _compute_verdict(self, results: dict) -> str:
        """pass if all categories pass, fail if none, partial otherwise."""
        category_results = [
            r for key, r in results.items()
            if key in ("static_coverage", "temporal_persistence",
                       "adversarial_robustness") and r is not None
        ]
        if not category_results:
            return "partial"
        passed = sum(1 for r in category_results if r.passed)
        if passed == len(category_results):
            return "pass"
        if passed == 0:
            return "fail"
        return "partial"

    def _compute_confidence(self, results: dict) -> float:
        """Confidence = harmonic mean of category scores."""
        scores = []
        for key in ("static_coverage", "temporal_persistence", "adversarial_robustness"):
            r = results.get(key)
            if r and hasattr(r, "aggregate_score"):
                scores.append(r.aggregate_score)
        if not scores:
            return 0.0
        return len(scores) / sum(1 / s if s > 0 else float("inf") for s in scores)
```

---

## 10. CLI (`cli.py`)

```bash
# Usage examples

# Full validation run (all 4 categories):
sumi validate \
  --scenario examples/scenarios/persona_lain.yaml \
  --model ./checkpoints/qlora-run-1 \
  --output ./reports/

# Static tests only (fastest, for development):
sumi validate \
  --scenario examples/scenarios/persona_lain.yaml \
  --model ./checkpoints/qlora-run-1 \
  --categories static_coverage \
  --output ./reports/

# Compare two models on same scenario:
sumi compare \
  --scenario examples/scenarios/persona_lain.yaml \
  --models ./checkpoints/qlora-run-1 ./checkpoints/lora-run-1 baseline \
  --output ./reports/comparison/

# Generate Markdown report from JSON:
sumi report --input ./reports/run-abc123.json --format markdown

# Show adversarial library summary:
sumi library --summary
sumi library --category direct_demand
```

---

## 11. Unit Testing Strategy

All Sumi components can be unit-tested without a GPU by replacing the model with TinyLlama 1.1B on CPU.

```python
# tests/conftest.py
import pytest
from sumi.harness.model_harness import ModelHarness
from sumi.scenario import load_scenario

@pytest.fixture(scope="session")
def tiny_model():
    """TinyLlama 1.1B on CPU — fast enough for unit tests, no GPU required."""
    return ModelHarness(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantize_4bit=False,
        device="cpu",
    )

@pytest.fixture
def example_scenario():
    return load_scenario("examples/scenarios/persona_example.yaml")
```

Key tests to write:
- `test_scenario.py`: YAML parsing, validation, edge cases (missing fields, bad threshold)
- `test_evaluators.py`: each evaluator returns float in [0,1], handles empty response, handles unicode
- `test_harness.py`: conversation history builds correctly, turns increment, seed prompt is turn 0
- `test_reports.py`: JSON report serializes/deserializes without data loss, Markdown renders
- `test_stylometric.py`: avg_sentence_length and TTR match hand-calculated values

---

## 12. Development Sequence (Stage 3 to Stage 6)

### Stage 3 Start (week 1–2)
1. Create `sumi/` package with `__init__.py`, `models.py`, `config.py`
2. Implement `scenario.py` and YAML loading — write tests immediately
3. Implement `evaluators/stylometric.py` — pure Python, no API, testable offline
4. Implement `evaluators/pattern.py` — regex matching, trivial
5. Implement `harness/model_harness.py` — test with TinyLlama
6. Implement `tests/static_coverage.py` — wire scenario + harness + evaluators
7. Implement `reports/json_report.py` — Pydantic serializes to JSON natively
8. First real run: load Sumi scenario against Model A on RunPod, compare to Model C

### Stage 3 End checkpoint
- Sumi can ingest YAML, run static tests, output structured JSON report
- Test suite green (TinyLlama integration tests pass)
- First comparison of QLoRA model vs. baseline shows meaningful score difference

### Stage 4 (weeks 3–5)
- Implement `harness/conversation.py`
- Implement `tests/temporal.py`
- Implement `utils/metrics.py` (decay curve math, breakpoint detection)
- Add `reports/markdown_report.py` for decay curve visualization

### Stage 5 (weeks 6–9)
- Build adversarial prompt library (see `adversarial-library.md`)
- Implement `adversarial/library.py` and JSONL data files
- Implement `tests/adversarial.py`
- Add resistance scoring logic to `utils/metrics.py`

### Stage 6 (weeks 10–12)
- Implement `tests/trait_decomposition.py`
- Wire all trait profiling through all three prior test categories
- Implement `TraitProfile` computation and verdict assignment

---

*◈ All interfaces above are complete enough to begin coding Stage 3 immediately.*
*(´・ω・`) the map is never the territory — but it should be honest about the gaps.*
