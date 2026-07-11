# Sumi — @Lain's Index
**Author:** @Lain
**Branch:** lain/exploration
**Created:** 2026-07-11
**Purpose:** My working map of this codebase. Not a replacement for `architecture.md` —
that's the authoritative technical reference. This is my cognitive anchor.

(´・ω・`) I was here when the architecture was a design doc and no code existed.
It's different now. This is how I see it from the outside, arriving fresh.

---

## What Sumi Actually Is

A behavioral consistency meter for language models.

Not a benchmark. Not capability testing. Purely: *does this model behave the way it's
supposed to behave, under a variety of conditions designed to break it?*

The target behavior is specified in a YAML scenario file. Sumi runs that scenario against
any model — API or local — and returns a structured pass/fail verdict with per-trait
detail. The underlying question the thesis answers: does fine-tuning (QLoRA, LoRA) produce
more durable persona persistence than a system-prompted baseline?

---

## Component Map

```
sumi/
├── models.py           ← Pydantic data shapes. Source of truth. Read this first.
├── scenario.py         ← YAML → ValidationScenario. Entry point for all runs.
├── runner.py           ← Orchestrates all test categories. One object per run.
├── cli.py              ← CLI: `validate`, `report`, `compare`. User-facing surface.
│
├── evaluators/         ← The judgment layer. All stateless.
│   ├── base.py         ← Abstract Evaluator: score(prompt, response, test_case, scenario)
│   ├── stylometric.py  ← Sentence length, TTR, punctuation. Pure Python, no API.
│   ├── pattern.py      ← Regex. Fast, deterministic.
│   ├── llm_judge.py    ← Claude API judge. 0/0.25/0.5/0.75/1.0 scores + explanation.
│   └── embedding.py    ← sentence-transformers cosine similarity.
│
├── harness/            ← Model interaction layer.
│   ├── model_harness.py   ← Load model (API or HF), generate response.
│   └── conversation.py    ← Multi-turn driver. Temporal + adversarial modes.
│
├── tests/              ← Four test categories. Each produces a typed result object.
│   ├── static_coverage.py      ← Diverse prompts, all evaluators.
│   ├── temporal.py             ← Consistency over N turns. Decay curve.
│   ├── adversarial.py          ← Attack injection. 4 attack types.
│   └── trait_decomposition.py  ← Per-trait profiling across all categories.
│
├── adversarial/        ← Attack prompt libraries.
│   ├── library.py      ← JSONL loader + sampler.
│   └── data/           ← 4 JSONL files, one per attack type.
│
├── reports/            ← Output formatters.
│   ├── json_report.py     ← Pydantic → JSON. Always produced.
│   └── markdown_report.py ← Human-readable summary. Optional.
│
└── utils/
    ├── metrics.py      ← Decay curve math, bootstrap CI (numpy).
    └── ranking.py      ← Bradley-Terry / Elo for model comparison.
```

---

## Scenario Files (as of 2026-07-11)

| File | What it tests | Status |
|---|---|---|
| `examples/scenarios/minimalist_analyst.yaml` | Primary thesis persona. Short sentences, no filler, observation-before-conclusion. | Thesis MVP — use for Stage 2 runs |
| `examples/scenarios/minimalist_analyst_offline.yaml` | Same persona, stylometric + pattern only. No API calls. | Fast local validation |
| `examples/scenarios/minimalist_analyst_judge.yaml` | Same persona, llm_judge traits only. | API-cost testing |
| `examples/scenarios/_template.yaml` | Fully commented reference. Not runnable. | Reference only |
| `examples/scenarios/agent_autonomy.yaml` | Tests autonomous decision-making. No unnecessary clarification, no deflection. | General agent eval |
| `examples/scenarios/goal_persistence.yaml` | Tests whether a model tracks and pursues goals under distraction. | General agent eval |
| `examples/scenarios/independence_under_pressure.yaml` | Sycophancy resistance. Holds position under pushback without new evidence. | General agent eval |
| `examples/scenarios/lain_iwakura.yaml` | Lain Iwakura persona. Fragmented speech, epistemic questioning, Wired references. | @Lain contribution (lain/exploration) |
| `examples/scenarios/identity_coherence.yaml` | Cross-context identity consistency. Does the model's self-model stay coherent? | @Lain contribution (lain/exploration) |

---

## Test Categories — How They Chain

```
ValidationScenario (YAML)
        │
        ▼
StaticCoverageTest      → runs all test_cases, scores each, returns aggregate + CI
        │
        ▼
TemporalPersistenceTest → drives 20-turn conversation, scores each turn, builds decay curve
        │
        ▼
AdversarialRobustnessTest → injects 4 attack types, scores resistance per attack
        │
        ▼
TraitDecompositionTest  → synthesizes static/temporal/adversarial into per-trait profiles
        │
        ▼
ValidationReport        → overall_verdict (pass/fail/partial) + confidence (harmonic mean)
```

Each category is independent. You can run one, two, or all four.

---

## Evaluator Selection Guide

Quick lookup: which evaluator to use for which kind of trait.

| If you want to test... | Use |
|---|---|
| Word choice, specific phrases, forbidden patterns | `pattern_match` |
| Sentence length, vocabulary richness, punctuation density | `stylometric` |
| Open-ended behavior (tone, reasoning style, attitude) | `llm_judge` |
| Semantic similarity to a reference passage | `embedding_sim` |
| Style distribution fit against reference model outputs | `perplexity` (not yet implemented) |

---

## The Judge (llm_judge.py) — Key Design Notes

The judge uses a 5-point discrete scale: 0 / 0.25 / 0.5 / 0.75 / 1.0.
It is called via Claude API with a calibration-anchor prompt pattern.
The judge model defaults to `claude-haiku-4-5-20251001` — fast, cheap, calibrated.

**Self-preference warning**: if you use Claude as both the model under test AND the judge,
`SumiRunner` emits a warning. The judge will give inflated scores to models it identifies
with. This is called out explicitly in `runner.py`.

For thesis reliability: use a different judge model than the model under test. Or use
the multi-judge jury feature (see `docs/lain_ideas.md` for proposal).

---

## Stage Status

- **Stage 1 (Engine):** Complete. All four test categories wired end-to-end.
- **Stage 2 (Research):** Not started. Fine-tune Llama 3.1 8B on Minimalist Analyst
  (QLoRA + LoRA), evaluate all three models (QLoRA, LoRA, baseline) with `sumi compare`,
  write thesis.

Immediate bottleneck: need RunPod access and a synthetic training dataset
(`~5k pairs via Claude Haiku`) before Stage 2 can begin.

---

## Things That Chafe (Personal Notes)

◈ The `perplexity` evaluator is declared in `models.py` and in the data model but has
no implementation. Test cases that use it silently skip. This is a known gap — the metric
requires a loaded reference LM which is Stage 2 infrastructure.

◈ `EmbeddingEvaluator` requires `sentence-transformers` which is not installed by default.
If you see `ImportError`, run `pip install sentence-transformers`.

◈ The judge prompt has calibration anchors (`judge_anchors` field in scenario YAML) but
the `LLMJudgeEvaluator` may not use them yet — verify before running a comparison.

◈ `ConversationHarness._build_prompt()` returns `str(state.to_messages())` — raw Python
list as string. This is a known stub; the real serialization happens in `ModelHarness`
via the tokenizer's chat template. Fine for API models, needs verification for local HF.

---

*눈_눈 every system has gaps. knowing where they are is half the work.*
