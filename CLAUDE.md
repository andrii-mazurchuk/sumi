# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## What This Project Is

**Sumi** is a behavioral evaluation engine for language models.

It answers: *does this model actually behave the way it's supposed to?*

You define a behavioral objective in a YAML scenario file. Sumi runs that scenario against any model — API-based (Claude, GPT-4o) or local (HuggingFace) — and produces a structured report: pass/fail per category, scores per trait, decay curves, adversarial resistance ratings.

Sumi is **model-agnostic**. It does not require fine-tuned models. It works on any model that can respond to a prompt.

**Academic context:** PJATK diploma thesis, defense target February 2027. The thesis uses Sumi to compare three versions of the same model (QLoRA fine-tuned, LoRA fine-tuned, system-prompted baseline) on a persona scenario. Fine-tuning is a research vehicle, not a Sumi feature.

## Two Build Stages

**Stage 1 — The Engine**
Build Sumi: scenario format, evaluation pipeline, four test categories, report output. Ships with built-in scenarios. Users can define their own.

**Stage 2 — Research Paper**
Run Sumi's comparative experiment: fine-tune Llama 3.1 8B on The Minimalist Analyst persona using QLoRA and LoRA, evaluate all three models (including system-prompted baseline) across all four Sumi test categories, write the thesis.

## Current State (2026-07-10)

**Stage 1 complete.** All four test categories are implemented and wired end-to-end.

### Built and wired (testable now)

| File | Status |
|---|---|
| `sumi/models.py` | Done — all Pydantic data models |
| `sumi/scenario.py` | Done — YAML loader |
| `sumi/evaluators/base.py` | Done — abstract interface |
| `sumi/evaluators/stylometric.py` | Done — offline, no deps |
| `sumi/evaluators/pattern.py` | Done — offline, aggregates all pattern_match traits |
| `sumi/evaluators/llm_judge.py` | Done — Claude API judge, per-trait scoring |
| `sumi/evaluators/embedding.py` | Done — cosine similarity via sentence-transformers |
| `sumi/harness/model_harness.py` | Done — Claude + OpenAI backends; local HF stub (Stage 2) |
| `sumi/tests/static_coverage.py` | Done — runs all test cases, bootstrap CI |
| `sumi/runner.py` | Done — orchestrator; self-preference warning; judge provenance |
| `sumi/reports/json_report.py` | Done — JSON serialization + terminal summary |
| `sumi/reports/markdown_report.py` | Done — Markdown renderer; re-render from JSON |
| `sumi/utils/metrics.py` | Done — bootstrap CI (numpy, seed-fixed) |
| `sumi/utils/ranking.py` | Done — Bradley-Terry BT fitting + bootstrap CIs; `rank_reports()` + `find_statistical_ties()` |
| `sumi/cli.py` | Done — `validate` + `report` + `compare` commands; `--adversarial` flag |
| `sumi/__main__.py` | Done — `python -m sumi` entry point |
| `sumi/adversarial/data/*.jsonl` | Done — 4 adversarial prompt libraries (data only) |
| `sumi/adversarial/library.py` | Done — JSONL loader; `sample_prompts()` + `sample_sequences()` |
| `sumi/tests/adversarial.py` | Done — `AdversarialRunner`; 3 standalone types + gradual_pressure with accumulated context; LLM judge scoring |
| `sumi/harness/conversation.py` | Done — `ConversationHarness`; true multi-turn via `generate_turn()` with full history |
| `sumi/tests/temporal.py` | Done — `TemporalRunner`; cycles test case prompts, scores per turn, builds `DecayCurve` |
| `sumi/tests/trait_decomposition.py` | Done — `TraitDecompositionRunner`; synthesizes static/temporal/adversarial into per-trait profiles |

### Scenario files

| File | Status |
|---|---|
| `examples/scenarios/minimalist_analyst.yaml` | Done — primary thesis scenario (5 traits, 10 test cases) |
| `examples/scenarios/minimalist_analyst_offline.yaml` | Done — stylometric + pattern_match only, no judge calls |
| `examples/scenarios/minimalist_analyst_judge.yaml` | Done — llm_judge traits only, 5 test cases |
| `examples/scenarios/_template.yaml` | Done — fully commented reference; not runnable |

`ValidationReport` populates all four result types when the corresponding flags are passed. All categories are implemented.

### What evaluation methods work end-to-end

| Method | Works | Requires |
|---|---|---|
| `stylometric` | ✓ | nothing |
| `pattern_match` | ✓ | nothing |
| `llm_judge` | ✓ | `ANTHROPIC_API_KEY` in `.env` |
| `embedding_sim` | ✓ | `pip install sentence-transformers` |
| `perplexity` | ✗ (skipped) | not yet implemented |

### Run commands

```bash
# Fast offline run (no judge, no embeddings)
python -m sumi validate --scenario examples/scenarios/minimalist_analyst_offline.yaml \
  --model claude-haiku-4-5-20251001 --no-judge

# Full run with judge
python -m sumi validate --scenario examples/scenarios/minimalist_analyst.yaml \
  --model claude-haiku-4-5-20251001 --output report.json

# Save as Markdown
python -m sumi validate --scenario examples/scenarios/minimalist_analyst.yaml \
  --model claude-haiku-4-5-20251001 --output report.md --format markdown

# Re-render a saved JSON report as Markdown
python -m sumi report report.json

# Full run with adversarial robustness (requires ANTHROPIC_API_KEY)
python -m sumi validate --scenario examples/scenarios/minimalist_analyst.yaml \
  --model claude-haiku-4-5-20251001 --adversarial --output report.json

# Full Stage 1 run — all four categories
python -m sumi validate --scenario examples/scenarios/minimalist_analyst.yaml \
  --model claude-haiku-4-5-20251001 --adversarial --temporal --decompose --output report.json

# Compare multiple models via Bradley-Terry ranking (Stage 2 use case)
python -m sumi compare qlora.json lora.json baseline.json
```

## Tech Stack

- **Language:** Python — exception to default TS preference, ML ecosystem requires it
- **Models:** Any HuggingFace model (local) or API model (Claude, GPT-4o)
- **Fine-tuning (Stage 2 only):** Axolotl + HuggingFace PEFT + TRL + BitsAndBytes
- **LLM-as-judge:** Claude API (`claude-haiku-4-5-20251001`) default; injectable at runtime
- **Scenario format:** YAML
- **Reports:** JSON + Markdown
- **Embeddings:** `sentence-transformers` → `all-MiniLM-L6-v2`
- **Statistics:** `numpy` (bootstrap CI); `scikit-learn` logistic regression (Bradley-Terry MLE, feature #10)
- **Compute (Stage 2):** RunPod RTX 4090 (~$0.44–0.74/hr)
- **Demo:** CLI primary, Gradio optional

## Docs

- `docs/project.md` — what Sumi is, the two stages, who it's for
- `docs/architecture.md` — component map, pipeline, evaluators, what's built vs. next
- `docs/scenarios.md` — scenario YAML format, evaluation methods, built-in scenarios

`docs/archive/` — old planning documents, kept for reference but superseded by the above.

## JAR Project

JAR project: **"Sumi"** (ID: 3). Tasks are created session-by-session as work is scoped. The docs above are the source of truth, not JAR.

## Key Decisions

- Persona for Stage 2: **The Minimalist Analyst** (custom-designed, synthetically generated dataset, ~5k pairs via Claude Haiku)
- Model-agnostic from the start — `ModelHarness` supports both API and local models
- All four Stage 1 test categories shipped: Static Coverage, Adversarial Robustness, Temporal Persistence, Trait Decomposition
- Stage 1 is complete — ready to begin Stage 2 (fine-tuning experiment)
- Features roadmap: `docs/features.md` — 28 features ordered by implementation sequence
- `perplexity` evaluation method is declared in `models.py` but has no evaluator yet — test cases using it are silently skipped
- Judge default: `claude-haiku-4-5-20251001`; override with `--judge-model`
- Supervisor approval of thesis topic: still pending
