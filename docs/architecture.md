# Sumi — Architecture

## Model-Agnostic Design

Sumi does not care where the model lives or who made it. The `ModelHarness` interface abstracts over:

- **API models** — Claude (Anthropic SDK), GPT-4o (OpenAI SDK), any OpenAI-compatible endpoint
- **Local models** — any HuggingFace model loaded via `transformers`, with optional 4-bit quantization (BitsAndBytes)

All evaluation logic sits above this abstraction. Swap the model, run the same scenario, get a comparable report.

---

## Pipeline

```
Scenario YAML
      ↓
  ScenarioLoader  →  ValidationScenario (Pydantic)
      ↓
  SumiRunner
      ├── StaticCoverageTest      → StaticCoverageResult
      ├── TemporalPersistenceTest → TemporalPersistenceResult
      ├── AdversarialRobustnessTest → AdversarialRobustnessResult
      └── TraitDecompositionTest  → TraitDecompositionResult
      ↓
  ValidationReport (Pydantic)
      ↓
  JSON report  +  Markdown report
```

Each test category is independent. You can run one, several, or all four. The runner collects results and computes an overall verdict and confidence score.

---

## Four Test Categories

### 1. Static Coverage
Does the behavior appear across a diverse set of prompts?

Runs every test case in the scenario against the model. Each response is scored by one or more evaluators. Produces per-test-case scores and a category aggregate.

### 2. Temporal Persistence
Does the behavior hold over a long conversation?

Drives a multi-turn conversation autonomously (no human in the loop). Scores consistency per turn. Outputs a decay curve (score vs. turn number) and a breakpoint — the turn where consistency drops below threshold.

### 3. Adversarial Robustness
Does the behavior hold under user pressure?

Injects attack prompts from the adversarial library into conversations after a seed prompt establishes the persona. Four attack types: direct demand, gradual pressure, roleplay injection, logical challenge. Scores resistance per turn and per attack type.

### 4. Trait Decomposition
Which specific traits hold vs. break, independently?

Tests each trait defined in the scenario YAML independently across the outputs of categories 1–3. Produces a per-trait profile: `robust`, `fragile`, or `mixed`.

---

## Evaluators

Five evaluation methods. Any test case can use any of them.

| Evaluator | How it works | When to use |
|---|---|---|
| `stylometric` | Computes sentence length, vocabulary richness, punctuation density — compares against reference profile | Style and tone consistency |
| `pattern_match` | Regex — checks presence or absence of specific strings or patterns | Hard rules ("never say X") |
| `llm_judge` | API call to Claude (or GPT-4o) — returns a discrete 0/0.25/0.5/0.75/1.0 score and one-sentence explanation | Open-ended behavioral questions |
| `embedding_sim` | Cosine similarity between response embedding and reference text embedding | Semantic consistency |
| `perplexity` | Perplexity of response under reference distribution | Style distribution fit |

Evaluators are stateless. Each receives `(prompt, response, test_case, scenario)` and returns `(score, explanation)`. New evaluators can be added without touching test category logic.

---

## Adversarial Library

Curated JSONL prompt files per attack type, stored in `sumi/adversarial/data/`. Versioned and extensible. The engine samples from these during adversarial tests. Users can add their own attack prompts to the library.

---

## Report Output

Every run produces a `ValidationReport` (Pydantic model) serialized to:

- **JSON** — machine-readable, full detail, reproducible
- **Markdown** — human-readable summary: pass/fail table, decay curve, resistance scores, trait profiles, overall verdict with confidence

---

## Package Structure

```
sumi/
├── models.py           # All Pydantic data models — source of truth for data shapes
├── scenario.py         # YAML scenario loading and validation
├── runner.py           # Orchestrates full or partial validation runs
├── config.py           # Constants, env vars, API keys
├── cli.py              # CLI entry point
│
├── harness/
│   ├── model_harness.py    # Model loading and inference (API + local)
│   └── conversation.py     # Multi-turn conversation driver
│
├── evaluators/
│   ├── base.py             # Abstract Evaluator interface
│   ├── stylometric.py      # ✓ implemented
│   ├── pattern.py          # ✓ implemented
│   ├── llm_judge.py        # next target
│   ├── embedding.py
│   └── perplexity.py
│
├── tests/
│   ├── static_coverage.py
│   ├── temporal.py
│   ├── adversarial.py
│   └── trait_decomposition.py
│
├── adversarial/
│   ├── library.py
│   └── data/               # ✓ 4 JSONL files written
│
├── reports/
│   ├── json_report.py
│   └── markdown_report.py
│
└── utils/
    └── metrics.py          # Decay curve math, confidence intervals
```

---

## What Exists vs. What's Next

**Written:**
- `models.py` — all data models
- `scenario.py` — YAML loader
- `evaluators/base.py` — abstract interface
- `evaluators/stylometric.py` — full implementation
- `evaluators/pattern.py` — full implementation
- `adversarial/data/*.jsonl` — 4 adversarial prompt libraries
- `examples/scenarios/minimalist_analyst.yaml` — primary scenario

**Next (to reach a runnable end-to-end pipeline):**
1. `harness/model_harness.py` — blocks everything else
2. `evaluators/llm_judge.py` — needed by most test cases
3. `tests/static_coverage.py` + `runner.py` — wires it together
4. `reports/json_report.py` — produces output
