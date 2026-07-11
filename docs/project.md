# Sumi — Project

**What it is:** A behavioral evaluation engine for language models.

Sumi answers the question: *does this model actually behave the way it's supposed to?*

You define what "supposed to" means in a scenario file. Sumi runs that scenario against any model and produces a structured behavioral report — pass/fail per category, scores per trait, decay curves, adversarial resistance ratings.

---

## The Core Problem

Standard model evaluation measures capability: can the model answer questions correctly, solve tasks, follow instructions. That's not what Sumi measures.

Sumi measures **behavioral persistence**: does the model maintain a specific behavioral profile across varied prompts, over long conversations, and under direct user pressure trying to break it. These are different questions and existing tools don't answer them.

---

## What Sumi Is Not

- Not a benchmark runner (not measuring capability or accuracy)
- Not a fine-tuning tool (does not train models)
- Not tied to any specific model or provider — works with any model that can respond to a prompt

---

## Two Build Stages

### Stage 1 — The Engine

Build Sumi itself: the scenario format, the evaluation pipeline, the test categories, the report output.

At the end of Stage 1, Sumi can:
- Load a scenario YAML
- Point at any model (API or local)
- Run all four test categories
- Output a structured JSON + Markdown behavioral report

The engine ships with a set of built-in scenarios. Users can also define and run their own.

### Stage 2 — Research Paper

Use Sumi to run a comparative experiment: three versions of the same model (QLoRA fine-tuned, LoRA fine-tuned, system-prompted baseline), all targeting the same persona, all evaluated by Sumi across all four test categories.

The paper answers: which fine-tuning method produces the most behaviorally robust model, and across which specific traits?

Stage 2 exists to validate Sumi's usefulness in a real research context and to produce the academic thesis (PJATK diploma, defense target: February 2027). Fine-tuning is a vehicle for demonstrating Sumi — it is not part of the engine.

---

## Target Users

1. **Researchers and engineers** who fine-tuned a model and want to measure what behavioral change actually happened
2. **Anyone** who defined a behavioral objective for a model (via fine-tuning, system prompt, or RLHF) and wants to know how robustly it holds
3. **Thesis/academic context** — the fine-tuning experiment in Stage 2 is the primary demonstration case
