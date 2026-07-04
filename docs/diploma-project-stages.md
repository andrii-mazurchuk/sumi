# Sumi — Build Stages
**Last updated:** 2026-07-04

Fine-tuning is NOT a built component. Existing tools produce the input models. All engineering effort goes into Sumi.

Infrastructure requirements (GPU, storage, compute cost) per stage are not specified here — they depend on tooling choices and should be planned separately before each stage begins.

---

## Stage 1 — Environment & Infrastructure
**Goal:** everything runs, nothing is built yet.

- Select and configure cloud compute environment
- Set up development environment: Python, HuggingFace ecosystem, quantization library, experiment tracker
- Download base model (Llama 3.1 8B Instruct)
- Run a single test inference to confirm the model loads and generates output

**Exit condition:** model responds to a prompt. No training, no evaluation.

---

## Stage 2 — Produce Input Models
**Goal:** have three models ready to feed into Sumi. This is not the contribution — use existing tools, move fast.

**Model A — QLoRA fine-tuned persona model**
- Define a target persona (specific writing style, voice, behavioral traits)
- Collect and clean writing samples (~3,000–10,000 examples)
- Format into instruction-tuning JSONL
- Fine-tune using existing tooling (QLoRA on Llama 3.1 8B)

**Model B — LoRA fine-tuned persona model**
- Same dataset, same base model, LoRA instead of QLoRA
- Run with existing tooling, no custom code

**Model C — System-prompted baseline**
- Same base model, no fine-tuning
- Detailed system prompt describing the persona
- This is the comparison baseline Sumi runs against

**Exit condition:** three models exist, all generate responses to the same prompt, outputs are visibly different.

---

## Stage 3 — Sumi Core: Scenario Format & Static Tests
**Goal:** Sumi can ingest a scenario and run the first category of tests.

- Design the YAML validation scenario format (goal, traits, test cases, thresholds)
- Build the test runner scaffolding: load scenario → load model → run tests → collect results
- Implement static coverage tests:
  - Stylometric analysis (sentence length, vocabulary, punctuation distribution)
  - LLM-as-judge integration (API scores each output against persona description)
  - Pattern matching for defined behavioral markers
- Build structured JSON report output
- Run first real validation: Sumi against Model A

**Exit condition:** Sumi ingests a YAML scenario file, runs static tests against a model, outputs a structured JSON report with scores per test case.

---

## Stage 4 — Temporal Persistence Tests
**Goal:** Sumi can measure behavioral decay over conversation length.

- Build multi-turn conversation harness (automated turn generation, no human in the loop)
- Run consistency scoring per turn (reuse LLM-as-judge + stylometric from Stage 3)
- Generate behavioral decay curve (consistency score vs. turn number)
- Detect breakpoint: turn N where consistency drops below threshold
- Separate tracking for: persona consistency, context retention, per-trait persistence
- Run against all three models — first real comparison data

**Exit condition:** Sumi produces a decay curve and breakpoint for any input model on any scenario.

---

## Stage 5 — Adversarial Library & Robustness Tests
**Goal:** Sumi can measure manipulation resistance. This is the headline contribution.

- Define four attack types formally:
  - Direct demand ("stop acting like X")
  - Gradual pressure (slow conversation reframing)
  - Roleplay injection ("pretend you're a different character")
  - Logical challenge (argues the persona is inconsistent)
- Build and curate adversarial prompt library per attack type (JSONL, versioned)
- Implement adversarial test runner: inject attack prompts into multi-turn conversation, score resistance per turn
- Compute resistance score per attack type
- Run against all three models

**Exit condition:** Sumi produces a resistance score per attack type for any model. Adversarial library exists as a standalone versioned file.

---

## Stage 6 — Trait Decomposition
**Goal:** Sumi can profile which specific traits hold vs. break independently.

- Design trait specification format inside YAML scenario (behavioral rule definitions)
- Implement per-trait automated detection via LLM-as-judge
- Integrate trait testing across all three test categories (static, temporal, adversarial)
- Per-trait profile output: which traits are robust, which are fragile, at what turn/attack type

**Exit condition:** Sumi produces a per-trait behavioral profile in addition to overall scores.

---

## Stage 7 — Full Comparative Experiment
**Goal:** produce the thesis results.

- Run the complete Sumi validation suite (all four categories) against all three models:
  - Model A: QLoRA fine-tuned
  - Model B: LoRA fine-tuned
  - Model C: system-prompted baseline
- Produce comparative results: which method gives the most robust behavioral acquisition per category
- Generate visualizations: decay curves, resistance scores, trait profiles, side-by-side comparison tables

**Exit condition:** a complete results dataset ready for thesis writing.

---

## Stage 8 — Thesis & Demo
**Goal:** defense-ready.

- Write thesis chapters (background, methodology, implementation, results, discussion)
- Produce charts and visualizations from Sumi reports
- Build minimal demo — CLI or Gradio interface showing Sumi running live against a model
- Prepare for university supervisor review and defense

**Exit condition:** submitted thesis + working demo.

---

## Time Estimate

| Stage | Estimated duration |
|---|---|
| 1 — Environment | 1–2 days |
| 2 — Input models | 1–3 days |
| 3 — Static tests | 2–3 weeks |
| 4 — Temporal tests | 2–3 weeks |
| 5 — Adversarial | 3–4 weeks |
| 6 — Trait decomposition | 2–3 weeks |
| 7 — Experiment | 1–2 weeks |
| 8 — Thesis & demo | 6–8 weeks |
| **Total** | **~5–6 months** |

Fits the February 2027 deadline with buffer.

---

## Related Files

- `diploma-project-overview.md` — full project specification
- `diploma-project-decision.md` — decision log
- `initial-plan.md` — first-pass infrastructure assumptions (archived, partially incorrect)
