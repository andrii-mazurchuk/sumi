# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sumi** is a behavioral validation engine for fine-tuned LLMs — an academic diploma project (PJATK, thesis defense target: February 2027).

**Core question:** Can efficient fine-tuning (LoRA, QLoRA, prompt tuning) specialize a small open-source LLM for behavioral goals, and can the robustness of that specialization be measured through structured behavioral persistence tests including adversarial resistance?

**Sumi is the core contribution.** Fine-tuning uses existing tools (Axolotl, HuggingFace PEFT); Sumi is the novel validation engine built on top.

## Architecture

### Two Components

**1. Fine-Tuning Pipeline** (not the contribution — uses existing tools)
- Base model: Llama 3.1 8B Instruct
- Methods compared: QLoRA, LoRA, Prompt tuning
- Tools: Axolotl, HuggingFace PEFT + TRL, BitsAndBytes
- Tracking: Weights & Biases
- Compute: Cloud GPU — RunPod RTX 4090 (JarvisLabs is defunct as of Q1 2026)

**2. Sumi Validation Engine** (the core contribution)
- Ingests user-defined YAML validation scenarios
- Runs four test categories against fine-tuned models
- Outputs structured JSON reports + rendered Markdown

### Sumi's Four Test Categories

| Category | What it measures |
|---|---|
| **Static Coverage** | Does the behavior appear at all? (stylometric analysis, LLM-as-judge, pattern matching) |
| **Temporal Persistence** | Does it hold over long conversations? (decay curves, breakpoint detection) |
| **Adversarial Robustness** | Does it hold under user pressure? (4 attack types: direct demand, gradual pressure, roleplay injection, logical challenge) |
| **Trait Decomposition** | Which specific traits hold vs. break independently? |

### Validation Scenario Format (YAML)
```yaml
goal: what fine-tuning was supposed to achieve
traits:
  - behavioral rules (machine-testable)
test_cases:
  - prompt: ...
    expected_behavior: ...
    evaluation_method: stylometric | llm_judge | pattern_match | embedding_sim | perplexity
pass_threshold:
  per_category: 0.75
```

### Evaluation Methods
- **Stylometric classifier** — style/tone consistency
- **LLM-as-judge** — Claude or GPT-4 API for open-ended evaluation
- **Pattern matching / regex** — format compliance
- **Embedding similarity** — semantic consistency
- **Perplexity** — distribution fit on held-out text

### Output
Structured JSON + Markdown reports: pass/fail per category, behavioral decay curves, resistance scores per attack type, per-trait profiles, overall verdict with confidence.

## Tech Stack

- **Language:** Python (compute/ML side) — this is the exception to default TS preference
- **Fine-tuning:** HuggingFace PEFT + TRL, Axolotl
- **Quantization:** BitsAndBytes
- **Tracking:** Weights & Biases
- **Evaluation:** Custom Sumi engine + HuggingFace Evaluate
- **LLM-as-judge:** Claude API (`claude-sonnet-4-6`) or GPT-4
- **Scenario format:** YAML
- **Demo:** Gradio or CLI
- **Compute:** Cloud GPU — RunPod RTX 4090 (community cloud, ~$0.44–0.74/hr)

## Build Stages

The project has 8 stages. Current status: **Stage 0 (Planning complete, no code yet).**

1. Environment & Infrastructure — cloud GPU, Python env, HF/PEFT/BitsAndBytes/W&B, test inference
2. Produce Input Models — generate QLoRA, LoRA, and system-prompted baseline models
3. **Sumi Core: Scenario Format & Static Tests** — YAML parser, test runner, stylometric + LLM-as-judge + pattern matching, JSON reports
4. Temporal Persistence Tests — multi-turn harness, decay curves, breakpoint detection
5. Adversarial Library & Robustness Tests — 4 attack types, curated JSONL prompt library, resistance scoring
6. Trait Decomposition — per-trait automated detection and profiles
7. Full Comparative Experiment — run complete suite, visualizations, comparative analysis
8. Thesis & Demo — write thesis, charts, CLI/Gradio demo, defense prep

## Key Decisions Still Open

- University supervisor approval of scope
- Number of personas to validate (1 MVP minimum)
- LLM-as-judge provider: Claude vs. GPT-4 vs. local model
- Demo interface: CLI vs. Gradio web UI
- Persona selection: see `docs/persona-selection.md` — recommendation is Custom Persona (Minimalist Analyst)

## JAR Project & Task Tracking

This repository corresponds to a JAR project named **"Musub Sumy"** (same name as this folder). All implementation tasks are tracked there as a hierarchical task architecture.

Before starting any implementation work, check the JAR project for the current task breakdown, priorities, and progress. Use the `jar` skill or MCP tools (`mcp__jar__project_list`, `mcp__jar__project_tasks`) to query the task hierarchy. Follow the JAR task structure during implementation — it represents the agreed execution plan.

## Docs

All planning documents are in `docs/`:

**Architecture and specification:**
- `diploma-project-overview.md` — full project specification (vision, architecture, research question)
- `diploma-project-stages.md` — detailed stage breakdown with exit conditions per stage
- `sumi-engine-architecture.md` — **code-level design** (Python module structure, interfaces, data models — start here for Stage 3)

**Decision records:**
- `diploma-project-decision.md` — decision log: what was decided, rejected, and why
- `persona-selection.md` — **persona choice research**: 4 options analyzed, recommendation with academic rationale
- `initial-plan.md` — **archived first-pass assumptions** (infrastructure, GPU, provider choices); partially incorrect — read to understand history, not as a guide

**Implementation guides:**
- `development-plan.md` — operational plan: hardware, software, where to test, where to deploy, stage roadmap
- `adversarial-library.md` — **adversarial prompt library**: 53 seed prompts across 4 attack types with JSONL schema
- `llm-judge-design.md` — **LLM-as-judge prompt engineering**: 4 templates, calibration protocol, cost estimates

**Archived:**
- `pc-build-report.md` — redirect stub, content in `initial-plan.md`

Read `diploma-project-overview.md` and `diploma-project-stages.md` for the research context.
Read `sumi-engine-architecture.md` before writing any code.
Read `persona-selection.md` before Stage 2 (persona decision is a blocker for dataset collection).
