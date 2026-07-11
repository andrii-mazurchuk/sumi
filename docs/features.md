# Sumi — Feature Roadmap

This file is an index, not a reference. For *how* a metric works (formulas, biases,
correct usage) see `docs/research/llm-evaluation-guide.md`. For *which library* implements
it and why, see `docs/research/llm-eval-tooling.md`. This file only records: status, where
it lives in Sumi, which library decision was made, and which of Sumi's test categories it
serves.

**The rule:** never reimplement math or metric logic that a maintained library already
provides. Sumi's original work is orchestration, provenance, sandboxing policy, and the
result store — see guide §11 for the architecture this mirrors.

Sumi's four test categories (MVP: first two): **Static Coverage**, **Adversarial
Robustness**, **Temporal Decay**, **Trait Decomposition**. Most evaluators below are
cross-cutting primitives consumed by more than one category rather than owned by a single
one — the category column names the primary use.

---

## Part 1 — Core engine

In scope for Stage 1 (the engine) and Stage 2 (the thesis comparison). Everything here is
either implemented or directly required to run the four test categories end-to-end.

### Already implemented

| # | Feature | Category | Sumi location | Library decision |
|---|---|---|---|---|
| 1 | Pattern match evaluator | Static Coverage, Adversarial Robustness | `sumi/evaluators/pattern.py` → `PatternEvaluator`, key `"pattern_match"` | Built from scratch — stdlib `re` only |
| 2 | Stylometric evaluator | Trait Decomposition | `sumi/evaluators/stylometric.py` → `StylometricEvaluator`, key `"stylometric"` | Built from scratch — stdlib `re`/`math` only |
| 3 | LLM-as-judge (pointwise) | Cross-cutting (all four) | `sumi/evaluators/llm_judge.py` → `LLMJudgeEvaluator`, key `"llm_judge"` | Built from scratch on `anthropic` SDK — no library covers Sumi's calibration-anchor prompt pattern. If judge reliability becomes an issue, adopt DeepEval's G-Eval explain-then-score pattern (guide §6.2) rather than switching libraries |

### Stage 1 — engine completion

| # | Feature | Category | Sumi location | Library decision |
|---|---|---|---|---|
| 4 | Embedding similarity evaluator | Trait Decomposition, Temporal Decay | `sumi/evaluators/embedding.py` → `EmbeddingEvaluator`, key `"embedding_sim"` | `sentence-transformers`, model `all-MiniLM-L6-v2`. Freeze model version — scores aren't comparable across embedders (tooling §5.3). Already in `requirements.txt`, commented out |
| 5 | Perplexity evaluator | Temporal Decay, Trait Decomposition | `sumi/evaluators/perplexity.py` → `PerplexityEvaluator`, key `"perplexity"`. Needs a local HF reference model — more relevant once Stage 2 GPU access exists | `transformers` (`AutoModelForCausalLM`), mean NLL over tokens. ~10 lines of PyTorch given a loaded model — do not implement the LM itself |
| 6 | Judge provenance logging | Cross-cutting infra | Extends `ValidationReport.metadata` in `runner.py`: `judge_model`, `judge_temperature`, `judge_prompt_version` | Built from scratch. Reproducibility requirement — a judge is a measurement instrument (guide §6.5) |
| 7 | Self-preference bias detection | Cross-cutting infra | `SumiRunner._build_evaluator_registry()` — compares `harness.model_id` prefix against judge model ID, warns on match | Built from scratch, ~2 lines. Guards against preference leakage (guide §6.3, ICLR 2026) |
| 8 | Markdown report | Cross-cutting infra | `sumi/reports/markdown_report.py`. CLI: `--format markdown` on `validate`, plus `sumi report <json_path>` | Built from scratch — string formatting over Pydantic models |

### Stage 1 → Stage 2 transition

| # | Feature | Category | Sumi location | Library decision |
|---|---|---|---|---|
| 9 | Bootstrap confidence intervals | Cross-cutting infra | `sumi/utils/metrics.py`. Appended to `StaticCoverageResult`, propagated to reports | `numpy` only — ~10 lines (guide §5.3). No score without an interval is Sumi's own reporting rule |
| 10 | Bradley-Terry / Elo ranking | Cross-cutting infra (Stage 2 model comparison) | `sumi/utils/ranking.py`. CLI: `sumi compare report_a.json report_b.json report_c.json` | `choix` (purpose-built BT/Luce MLE, same model Chatbot Arena uses). Lighter alternative: `sklearn.linear_model.LogisticRegression(fit_intercept=False)` (guide §5.4). Bootstrap CIs reuse #9 |

### Additional primitives (approved, not yet scheduled)

| # | Feature | Category | Sumi location | Library decision |
|---|---|---|---|---|
| 11 | BLEU / ROUGE / chrF | Temporal Decay, Trait Decomposition | New `sumi/evaluators/ngram.py`, keys `"bleu"` / `"rouge_l"` / `"chrf"` | `sacrebleu` (BLEU/chrF), `rouge-score` (ROUGE), or `evaluate` for all three behind one API. Surface-overlap only — pair with embedding similarity, don't use alone (guide §3.2) |
| 12 | Exact match | Static Coverage | `sumi/evaluators/exact.py`, key `"exact_match"`. Requires `reference_text` on the test case | `evaluate.load("exact_match")`, or ~15 lines by hand — simple enough that either is fine |
| 26 | Pairwise comparison judging | Cross-cutting (Stage 2 model comparison) | Extends `LLMJudgeEvaluator` with `compare(response_a, response_b)`. CLI: `sumi compare --scenario X --model-a Y --model-b Z`. Feeds Bradley-Terry (#10) | Built from scratch — order-swap-and-average logic (guide §6.3), ~10 lines. Genuinely Sumi's own orchestration work |
| 27 | Jury — multi-judge panel | Cross-cutting (judge reliability) | `LLMJudgeEvaluator` accepts a list of `(client, model_id)` pairs; logs per-judge scores and variance | Built from scratch — no library covers the multi-provider case cleanly, though DeepEval's G-Eval sampling is a useful reference (guide §6.2, "replace judges with juries") |
| 28 | Data contamination checks | Cross-cutting (thesis rigor) | `sumi/utils/contamination.py`: `ngram_overlap()`, `public_private_gap()`. `ValidationReport` gains `contamination_flags` | Built from scratch — `numpy` + stdlib tokenization/set-intersection. No library targets this narrowly (guide §9) |

---

## Part 2 — Extension domains (out of current scope)

`docs/project.md` is explicit: *"Not a benchmark runner (not measuring capability or
accuracy)."* The items below are capability-benchmark, RAG, and agentic-evaluation
domains from the research docs. They don't serve Sumi's behavioral-persistence mission or
the Stage 2 persona thesis, and are not on the roadmap. Kept here only as a landing spot —
with the library decision already made — in case Sumi's scope changes later (e.g. if a
future scenario needs a RAG or tool-use persona).

| Domain | Features | Library decision if ever built |
|---|---|---|
| Verifiable-task correctness | pass@k (probability ≥1 of k sampled attempts passes unit tests) | `evaluate.load("code_eval")` for the estimator (sandbox required — Sumi's own responsibility), or `Scorio` for pass@k/Bayes@N with credible intervals (guide §4, tooling §5.1) |
| RAG evaluation | Faithfulness, Answer Relevancy, Context Precision, Context Recall | `RAGAS` — wrap its four core metrics directly, don't reimplement claim decomposition (guide §7, tooling §4) |
| Agentic evaluation | Task Completion/Goal Accuracy, Tool-Call F1, Trajectory Quality | Extend `LLMJudgeEvaluator` for rubric scoring; F1 is a from-scratch set-intersection; `Inspect AI`'s `Task`/`Solver`/`Scorer`/`Sandbox` model is worth studying as architecture reference (guide §8, tooling §3) |
| Capability benchmarks | MMLU/MMLU-Pro, GPQA, HumanEval/MBPP, SWE-bench Verified, BFCL | `lm-evaluation-harness` subprocess wrapper for all of these — this is precisely the "benchmark runner" role project.md says Sumi is not (guide §2, tooling §3) |

---

## Notes on the underlying research

- `docs/research/llm-evaluation-guide.md` — methods, formulas, correct usage, failure modes.
- `docs/research/llm-eval-tooling.md` — which library implements each method and why,
  install commands, a decision table.
- When adding a feature here, prefer linking to a numbered section in one of those two docs
  over re-explaining the method.
