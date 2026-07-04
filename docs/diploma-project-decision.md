# Sumi — Decision Log
**Last updated:** 2026-07-04

---

## 2026-06-23 — Initial Direction Set

Combined Direction 3 (fine-tuning method comparison) + Direction 1 (persona/style fine-tuning) into one unified project.

Working title at the time: "Efficient fine-tuning of open-source LLMs for persona and style specialization: a comparative study of LoRA, QLoRA, and prompt tuning on consumer hardware."

## 2026-06-28 — Project Named: Sumi

The project evolved from a fine-tuning comparison study into a behavioral validation engine for fine-tuned LLMs. The fine-tuning pipeline remains, but the core contribution is now Sumi — the engine that validates whether a model robustly learned what it was trained on.

**Key decisions made:**

- Validation engine is the primary contribution; fine-tuning pipeline is the subject of validation
- Four test categories: static coverage, temporal persistence, adversarial robustness, trait decomposition
- Adversarial test library is in scope and built as part of the project
- Automated trait detection via LLM-as-judge is in scope
- Validation scenario format: user-defined YAML
- Base model: Llama 3.1 8B Instruct
- Methods compared: QLoRA, LoRA, prompt tuning
- Primary use case: persona and writing style specialization

**Rejected directions:**
- RAG integration (separate concern, not fine-tuning)
- Real-time training monitor (solved by existing tools like W&B)
- Training from scratch (no compute justification)
- Drone swarms / robotics (original assigned topic, abandoned)

## 2026-07-04 — Infrastructure Assumptions Invalidated

First-pass infrastructure decisions (recorded in `initial-plan.md`) turned out to be incorrect before implementation began:

- **JarvisLabs** was the originally planned cloud provider — it froze new signups in Q1 2026 and is no longer available
- **A100 80GB** was specified as the required GPU for all stages — this was over-specified; an RTX 4090 (24GB) is sufficient for QLoRA and LoRA on an 8B model

**Decision:** Infrastructure planning is separated from the project specification. Main docs describe what Sumi is and does. Per-stage infrastructure requirements (GPU tier, storage, provider, cost) are to be planned by a dedicated agent with current market information before each stage begins.

---

## Related Files

- `diploma-project-overview.md` — full current specification
- `diploma-project-stages.md` — stage breakdown with exit conditions
- `initial-plan.md` — first-pass assumptions, archived with notes on what was wrong
