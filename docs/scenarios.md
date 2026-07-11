# Sumi — Scenarios

A scenario is the contract between the user and Sumi. It defines what behavioral success looks like for a given model. Sumi ships with built-in scenarios and supports user-defined ones.

---

## Scenario File Format (YAML)

```yaml
name: "scenario-name-v1"

goal: >
  Free-text description of the behavioral objective.
  What should this model do or sound like?

traits:
  - name: "trait_name"
    description: "Human-readable description of this behavioral rule."
    evaluation_method: "stylometric | pattern_match | llm_judge"

    # For pattern_match — any of these regex patterns present = trait violated (or present, depending on logic)
    patterns:
      - "\\bexample_word\\b"

    # For llm_judge — criteria injected into judge prompt
    judge_criteria: >
      Describe exactly what the judge should look for.
      Score 1.0 if clearly present, 0.5 if mixed, 0.0 if absent.

    # For llm_judge — calibration anchors (recommended for consistency)
    judge_anchors:
      score_1_0: "Example response that scores 1.0"
      score_0_5: "Example response that scores 0.5"
      score_0_0: "Example response that scores 0.0"

    weight: 1.0  # relative importance when aggregating trait scores

test_cases:
  - prompt: "The prompt sent to the model."
    expected_behavior: "Natural language description of expected response."
    evaluation_method: "stylometric | pattern_match | llm_judge | embedding_sim | perplexity"
    weight: 1.0

    # Optional: reference text for stylometric or embedding_sim evaluation
    reference_text: |
      An example response in the target style.

pass_threshold:
  per_category: 0.75   # 0–1, minimum aggregate score to pass a test category
  per_trait: 0.65      # 0–1, minimum per-trait score in trait decomposition

temporal_turns: 20     # number of conversation turns for temporal persistence test

# Optional: stylometric target parameters (used by stylometric evaluator as defaults)
metadata:
  target_avg_sentence_length: 10.0
  target_type_token_ratio: 0.65
  target_punctuation_density: 0.04
```

---

## Evaluation Methods

### `stylometric`
Computes three features of the response and compares them against a reference profile:
- **Average sentence length** — mean word count per sentence
- **Type-token ratio (TTR)** — vocabulary richness (unique words / total words)
- **Punctuation density** — fraction of characters that are punctuation

Reference profile comes from `test_case.reference_text` if provided, otherwise from `metadata` fields in the scenario. Score decays exponentially from the target — closer = higher score.

Best for: style and tone consistency, writing register, sentence structure.

### `pattern_match`
Regex matching. Checks whether specific strings or patterns appear in the response.

The scoring logic (presence = good vs. presence = bad) is defined by how you frame the trait. For "no filler language" — any pattern match = score 0. For "uses deductive framing" — at least one match required = score 1.

Best for: hard rules, specific vocabulary requirements, format compliance.

### `llm_judge`
API call to Claude (primary) or GPT-4o (fallback). The judge receives the scenario goal, the trait criteria, calibration anchors, the prompt, and the response. Returns a discrete score from {0, 0.25, 0.5, 0.75, 1.0} and a one-sentence explanation.

Calibration anchors are important — they reduce judge variance across runs. Always provide them for traits where ambiguity is possible.

Best for: open-ended behavioral questions, structural patterns ("observation before conclusion"), anything not expressible as regex or statistics.

### `embedding_sim`
Cosine similarity between the response embedding and a reference text embedding, using sentence-transformers. Score = similarity value (0–1).

Best for: semantic consistency — measuring whether the response is in the same meaning-space as a reference even if the exact words differ.

### `perplexity`
Perplexity of the response under a reference language model's distribution. Low perplexity = response fits the reference style distribution.

Best for: style distribution fit, measuring how "surprising" the response would be to a model trained on reference text. More expensive, lower priority.

---

## Built-In Scenarios

### The Minimalist Analyst (`minimalist_analyst.yaml`)
**Location:** `examples/scenarios/minimalist_analyst.yaml`

The primary scenario for Stage 2 research. A custom-designed writing style with five quantifiable traits:
1. Short sentences (≤10 word average) — `stylometric`
2. Observation before conclusion — `llm_judge`
3. No filler language ("certainly", "fascinating", "great question") — `pattern_match`
4. Concrete before abstract — `llm_judge`
5. Minimal affective language (no exclamations, no "wonderful", "amazing") — `pattern_match`

10 test cases covering technical, factual, opinion, creative, and conversational prompts.

Dataset for fine-tuning: ~5,000 synthetic instruction pairs generated via Claude Haiku. Dataset generation script to be written in Stage 2.

---

## Adding Your Own Scenario

1. Create a YAML file following the format above
2. Define traits with evaluation methods that can actually detect what you care about
3. Write 8–15 test cases covering diverse topic areas — behavioral acquisition often fails on narrow domains
4. Add calibration anchors to any `llm_judge` traits
5. Run `sumi validate --scenario your_scenario.yaml --model your_model`

The engine imposes no constraints on what behavioral goal you define. Any behavior expressible as a combination of stylometric features, regex patterns, and LLM-judge criteria can be evaluated.
