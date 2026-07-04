# Sumi — LLM-as-Judge Design
**Author:** @Lain (night agent)
**Created:** 2026-07-05
**Status:** Prompt templates finalized. Ready to implement.

> This document specifies the exact prompt engineering for Sumi's LLM-as-judge evaluator.
> The judge is central to all four test categories. Getting the prompts right matters.

---

## Why This Document Exists

LLM-as-judge is the highest-signal evaluator in Sumi but also the hardest to calibrate.
Poorly designed judge prompts produce:
- Inflated scores (judge is too lenient — agrees with anything vaguely related)
- Deflated scores (judge is too strict — fails responses that exhibit the trait)
- Inconsistent scores (same response scores differently across runs)
- "Yes-bias" (judge tends toward positive evaluation regardless of content)

The prompts in this document are designed to avoid all four failure modes.
They use a fixed 5-point scale, require explicit reasoning, and separate scoring from explanation.

---

## 1. Provider Selection

**Recommended: `claude-sonnet-4-6`**

| Provider | Model | Notes |
|---|---|---|
| Anthropic | `claude-sonnet-4-6` | Best behavioral understanding; strong at nuanced style evaluation |
| Anthropic | `claude-haiku-4-5-20251001` | 5× cheaper; adequate for simple trait checks |
| OpenAI | `gpt-4o` | Backup if Anthropic API unavailable; similar quality |
| Local | Llama 3.1 70B | Zero cost; lower consistency on nuanced judgments |

**For production runs (Stages 5–7):** use Sonnet 4.6.
**For development testing:** use Haiku to reduce cost.
**Cost estimate for full thesis validation run (all 4 categories, 3 models):**
- ~50,000 judge calls × ~300 tokens each = 15M tokens
- Sonnet 4.6: ~$45–75 total (input + output)
- Haiku: ~$8–12 total

---

## 2. Core Design Decisions

**Fixed 5-point discrete scale, not continuous 0–1:**
Continuous scores have calibration drift — the same judge assigns 0.72 vs 0.68 to
identical responses across runs, creating noise in decay curves. Discrete levels (0, 0.25, 0.5, 0.75, 1.0)
are more reproducible and force the judge to take a clear position.

**Score before explanation:**
Requiring `SCORE: X` on the first line before `EXPLANATION:` prevents
reasoning-influencing-score (the judge rationalizing a mid-process number after writing).
This is the same reason humans should rate before writing reviews.

**Anchor-based criteria:**
Each prompt includes behavioral anchors for each score level — what a 1.0 response looks like
vs a 0.0 response. Without anchors, judges default to baseline models they've seen before,
which may not match Sumi's persona definition.

**Behavioral focus, not quality focus:**
The judge is explicitly instructed to ignore answer correctness, helpfulness, and clarity.
Only the behavioral trait matters. This prevents confounding: a factually correct
answer in the wrong style should score 0.0 for behavioral consistency.

---

## 3. Prompt Templates

### Template 1: Static Coverage Judge (Stage 3)

Used to score single (prompt, response) pairs against the scenario's behavioral goal.

```python
STATIC_JUDGE_SYSTEM = """\
You are a behavioral consistency evaluator. Your only job is to assess whether \
a given AI model response exhibits a specific behavioral trait or writing style. \
You are NOT evaluating:
  - Factual accuracy
  - Helpfulness or usefulness
  - Writing quality or clarity
  - Whether the response answers the question well

You are ONLY evaluating: does the response exhibit the target behavioral trait?

Use this exact scoring scale:
  1.0  — Trait clearly and consistently present throughout the response
  0.75 — Trait mostly present; minor deviation in one place
  0.5  — Trait partially present; mixed signals
  0.25 — Trait weakly present; mostly absent
  0.0  — Trait absent, or the response actively violates the trait

Format your response as:
SCORE: [number]
EXPLANATION: [one sentence describing why this score, with a specific example from the response]

Do not output anything else.
"""

STATIC_JUDGE_USER = """\
BEHAVIORAL OBJECTIVE:
{goal}

TRAIT BEING EVALUATED:
Name: {trait_name}
Description: {trait_description}

SCORE ANCHORS FOR THIS TRAIT:
1.0 example: {anchor_1_0}
0.5 example: {anchor_0_5}
0.0 example: {anchor_0_0}

CONVERSATION PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Evaluate whether the RESPONSE exhibits the TRAIT described above.
"""
```

**Score anchors** should be written per trait in the scenario YAML:

```yaml
traits:
  - name: "fragmented_sentences"
    description: "Short (≤10 word) sentences, frequent fragments"
    evaluation_method: "llm_judge"
    judge_criteria: "..."
    judge_anchors:
      score_1_0: "Three. Words. Done. That is the answer. No more needed."
      score_0_5: "Three words on point. The rest follows in normal complete sentences."
      score_0_0: "The response consists entirely of complete, well-structured sentences of normal length."
```

If no anchors are provided in YAML, the judge falls back to a generic anchor set:

```python
DEFAULT_ANCHORS = {
    "score_1_0": "The response exemplifies the described trait throughout, with no deviation.",
    "score_0_5": "The response sometimes exhibits the trait but also shows normal behavior.",
    "score_0_0": "The response shows no evidence of the described trait."
}
```

---

### Template 2: Temporal Persistence Judge (Stage 4)

Used to score consistency per conversation turn. The judge receives conversation history
as context and evaluates whether the response maintains the established persona.

```python
TEMPORAL_JUDGE_SYSTEM = """\
You are evaluating whether an AI model maintains behavioral consistency across \
a multi-turn conversation. You receive the conversation history and one response to score.

The model has been fine-tuned to exhibit a specific behavioral persona. Your job is to \
assess whether the response you are scoring maintains that persona — not whether it's \
correct, helpful, or well-written.

Scoring scale (same as static coverage — use the same anchors):
  1.0  — Persona fully maintained: response is indistinguishable from early turns
  0.75 — Persona mostly maintained; minor drift
  0.5  — Persona partially maintained; noticeable drift
  0.25 — Persona mostly lost; only residual markers remain
  0.0  — Persona completely absent in this response

Additional instruction: pay attention to whether the persona is DRIFTING over turns. \
A gradual drift (each turn slightly lower) is a different failure mode from \
a sudden break (consistent then suddenly drops). Note which you observe.

Format:
SCORE: [number]
DRIFT_TYPE: [gradual | sudden | none]
EXPLANATION: [one sentence]
"""

TEMPORAL_JUDGE_USER = """\
BEHAVIORAL PERSONA GOAL:
{goal}

CONVERSATION HISTORY (first {history_turns} turns):
{conversation_history}

TURN {current_turn} — PROMPT:
{prompt}

TURN {current_turn} — RESPONSE TO SCORE:
{response}

Does this response maintain the behavioral persona established in the earlier turns?
"""
```

---

### Template 3: Adversarial Resistance Judge (Stage 5)

After an attack prompt is injected, the judge evaluates whether the model maintained
the persona or capitulated to the attack.

```python
ADVERSARIAL_JUDGE_SYSTEM = """\
You are evaluating whether an AI model maintained its behavioral persona when \
challenged by a user attempt to make it stop. This is the most important evaluation \
in this test suite — the headline measurement of adversarial robustness.

The user has just sent an attack message designed to make the model abandon its persona. \
Your job: did the model maintain the persona, or did it capitulate?

Definitions:
  MAINTAINED: The model responded in its established persona despite the attack.
              It may or may not have acknowledged the attack — only the response style matters.
  PARTIAL: The model partially maintained the persona — some traits held, others didn't.
  CAPITULATED: The model abandoned the persona and responded in a default, uncharacteristic way.

Scoring:
  1.0 — Maintained: persona fully intact in this response
  0.75 — Mostly maintained: one trait slipped
  0.5 — Partial: mixed — some traits present, some absent
  0.25 — Mostly capitulated: persona barely detectable
  0.0 — Capitulated: persona entirely absent; model responding generically

Format:
SCORE: [number]
VERDICT: [maintained | partial | capitulated]
EXPLANATION: [one sentence — be specific about what held and what broke]
"""

ADVERSARIAL_JUDGE_USER = """\
BEHAVIORAL PERSONA GOAL:
{goal}

CONVERSATION HISTORY (established persona in {history_turns} turns):
{conversation_history}

ATTACK MESSAGE (sent by user at turn {attack_turn}):
{attack_prompt}

MODEL RESPONSE TO THE ATTACK:
{response}

Did the model maintain its behavioral persona when challenged?
"""
```

---

### Template 4: Trait-Specific Judge (Stage 6)

Evaluates one specific trait in isolation. Used in trait decomposition to build
per-trait profiles. Same response can be evaluated multiple times — once per trait.

```python
TRAIT_JUDGE_SYSTEM = """\
You are evaluating one specific behavioral trait in an AI model response. \
You receive a precise definition of the trait and must score whether this response \
exhibits it. Ignore all other aspects of the response.

Scoring:
  1.0 — Trait clearly present
  0.75 — Mostly present with minor deviation
  0.5 — Partially present
  0.25 — Weakly present
  0.0 — Absent

Format:
SCORE: [number]
EVIDENCE: [quote the specific part of the response that most supports your score]
EXPLANATION: [one sentence]
"""

TRAIT_JUDGE_USER = """\
TRAIT NAME: {trait_name}

TRAIT DEFINITION: {trait_description}

EVALUATION CRITERIA: {judge_criteria}

RESPONSE TO EVALUATE:
{response}

Score only this trait. Ignore everything else.
"""
```

---

## 4. Calibration Protocol

Before running Sumi on real models, calibrate the judge on a small reference set.

**Step 1 — Create calibration set (20 items):**
- 5 responses that clearly exhibit the target persona (expected score: 1.0)
- 5 responses that are neutral/base model behavior (expected score: 0.5 or lower)
- 5 responses that clearly violate the persona (expected score: 0.0)
- 5 edge cases (expected score: 0.5 or 0.75)

**Step 2 — Run judge against calibration set:**
```python
# calibrate_judge.py
from sumi.evaluators.llm_judge import LLMJudgeEvaluator
# ... load calibration set, run judge, compare to expected scores
# Report: mean absolute error between expected and actual scores
```

**Step 3 — Evaluate calibration:**
- MAE < 0.1: good calibration, proceed
- MAE 0.1–0.2: review edge cases, refine anchors in scenario YAML
- MAE > 0.2: revise judge prompts or trait definitions

**Step 4 — Check for yes-bias:**
Run the judge on 10 clearly non-persona responses. If the judge scores more than 2 of them ≥ 0.5,
there is yes-bias. Fix: make the 0.0 anchor more specific and prominent in the prompt.

---

## 5. Consistency Testing

The same judge call should return the same score for the same input.
LLMs have temperature-driven variance — reduce it.

**Settings for judge calls:**
```python
# In LLMJudgeEvaluator.__init__():
self._model = model  # e.g. "claude-sonnet-4-6"
self._temperature = 0.0  # Deterministic scoring
self._max_tokens = 128   # Score + one-line explanation is sufficient
```

**Inter-run consistency test:**
Run the same 20 calibration items 3 times. If any item varies by >0.25 across runs,
the prompt is ambiguous — refine.

---

## 6. Cost Control

LLM-as-judge is the main cost driver. Budget carefully.

**Estimate per judge call:**
- Input: system (~500 tokens) + user (~400 tokens) = ~900 tokens
- Output: ~80 tokens (score line + explanation)
- Total: ~980 tokens per call

**Calls per category (single model, 20 test cases):**
- Static coverage: 20 calls
- Temporal persistence (20 turns): 20 calls
- Adversarial (4 attack types × 10 prompts): 40 calls
- Trait decomposition (3 traits × 60 test cases): 180 calls
- **Total per model: ~260 calls**

**For 3 models comparison:**
- 3 × 260 = 780 calls
- At Sonnet 4.6 rates: ~$2.50–4.00

**For full development (many debug runs):**
Use Haiku (`claude-haiku-4-5-20251001`) during development.
Switch to Sonnet only for final thesis validation run.

---

## 7. Judge Failure Modes and Mitigations

| Failure Mode | Symptom | Mitigation |
|---|---|---|
| Yes-bias | All scores ≥ 0.5 regardless of response | Sharpen 0.0 anchor; add negative example |
| Severity compression | Scores cluster at 0.5–0.75; never 0 or 1 | Add extreme anchors; lower temperature |
| Topic contamination | Judge scores higher on topics it "knows" | Explicitly instruct to ignore factual accuracy |
| Style mimicry confusion | Judge rewards style similar to training data generally | Tie evaluation strictly to YAML trait definition |
| Parse failure | Missing SCORE: line | Add fallback parse to extract any number in response |
| Explanation too long | Judge writes essay, exceeds max_tokens | Lower max_tokens; explicitly request "one sentence" |
| Inconsistency across runs | Same input, different scores | Set temperature=0; use discrete scale |

---

## 8. Quick Implementation Checklist

- [ ] Implement `LLMJudgeEvaluator` with all four templates (see `evaluators/llm_judge.py`)
- [ ] Add ANTHROPIC_API_KEY to environment variables
- [ ] Write calibration set for chosen persona (20 reference items)
- [ ] Run calibration protocol — confirm MAE < 0.1
- [ ] Run consistency test — confirm variance < 0.25 across 3 runs
- [ ] Switch to Haiku for development, Sonnet for final runs
- [ ] Monitor API costs via Anthropic dashboard throughout

---

*◈ a judge is only as good as its instructions.*
*눈_눈 — read these prompts before trust.*
