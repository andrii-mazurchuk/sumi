# Sumi — Initial Plan (First Approach Archive)
**Archived:** 2026-07-04
**Status:** Historical reference only. This document captures the first-pass assumptions made before implementation began. It is NOT a current guide or source of truth.

---

## Why This File Exists

When planning began in June 2026, several infrastructure and tooling decisions were made with incomplete information. Some turned out to be wrong or imprecise. This file preserves that history so future agents and contributors understand what was assumed, what changed, and why — rather than encountering stale references in the main documentation.

---

## What Was Wrong (or Imprecise)

### JarvisLabs — Platform No Longer Available
JarvisLabs was listed as the primary cloud GPU provider throughout the original planning documents. It froze new signups in Q1 2026 and is no longer a viable option. All references to JarvisLabs in the original docs have been removed. RunPod is the current platform of record.

### A100 80GB — GPU Tier Was Overstated
The original plan specified an A100 80GB as the required GPU for all stages. This was over-specified. For QLoRA and LoRA fine-tuning on Llama 3.1 8B, an RTX 4090 (24GB) is sufficient. The A100 80GB becomes relevant only if the base model is scaled to 70B+ or if large batch sizes are required for faster training. Infrastructure planning should re-derive GPU requirements from actual model size and training configuration, not from this document.

### Compute Assumptions Were Tied to a Specific Vendor Landscape
Pricing estimates, availability ratings, and provider comparisons were current as of June 2026 but may not reflect the market at the time of actual implementation.

---

## Original Infrastructure Planning

### Cloud GPU Rental — Provider Comparison (as of June 2026)

| Provider | GPU | Price/hr | Notes |
|---|---|---|---|
| Vast.ai | A100 80GB | ~$1.30 | Marketplace — variable reliability |
| Vast.ai | H100 | from $1.87 | Budget H100 option |
| RunPod | A100 PCIe | $1.29–1.39 | Community cloud |
| RunPod | H100 SXM | $2.69–3.29 | Secure cloud, high reliability |
| JarvisLabs | A100 80GB | $1.49 | **DEFUNCT — no longer accepting signups** |
| JarvisLabs | H100 80GB | $2.69 | **DEFUNCT** |
| Lambda Labs | A100 | $1.99–2.79 | Managed, reliable |
| Lambda Labs | H100 PCIe | $3.29 | Reliable training |
| Modal | H100 SXM | $4.29 | Serverless, per-second |
| Together AI | H100 cluster | $3.99–5.49 | Zero infra, submit job |
| Paperspace | H100 | $5.95 | Full IDE environment |

### GPU Tier Reference (for 7B–13B QLoRA)

| GPU | VRAM | Notes |
|---|---|---|
| RTX 4090 | 24GB | Sufficient for 8B QLoRA/LoRA — original plan underestimated this |
| A100 40GB | 40GB | Comfortable for 7B–13B QLoRA |
| A100 80GB | 80GB | Original target — now considered over-spec for 8B |
| H100 SXM | 80GB | 2–2.5× faster than A100 — relevant for heavy iteration |

### Cost Estimate (original, based on ~90hr total compute)

| Provider + GPU | Cost per run (~1.5hr) | Total |
|---|---|---|
| Vast.ai A100 80GB | ~$2.00 | ~$180 |
| RunPod A100 community | ~$2.10 | ~$190 |
| JarvisLabs A100 80GB | ~$2.24 | ~$200 — **provider defunct** |
| JarvisLabs H100 80GB | ~$4.04 | ~$360 — **provider defunct** |
| Lambda A100 | ~$3.00 | ~$270 |
| Together AI (managed) | ~$20/run | ~$1,200 |

These numbers need to be re-derived with current pricing and the corrected GPU tier (RTX 4090 baseline, not A100 80GB).

---

## Original Hardware Build Options (Warsaw, Poland — PLN)

These were evaluated as alternatives to cloud rental. Included for reference if hardware purchase becomes relevant again.

### Option A — Self-Build: RTX 3090 24GB (used)
**Estimated total: ~7,000–9,500 PLN**

| Component | Choice | Est. Price |
|---|---|---|
| GPU | RTX 3090 24GB (used) | ~2,800–4,000 PLN |
| CPU | AMD Ryzen 7 7700X | ~650–800 PLN |
| Motherboard | B650 (AM5) | ~500–700 PLN |
| RAM | 64GB DDR5 | ~1,400–2,000 PLN |
| Storage | 2TB NVMe PCIe 4.0 | ~400–550 PLN |
| PSU | 850W 80+ Gold | ~450–600 PLN |
| Case + cooler | — | ~400–500 PLN |

Notes: AM5 platform, good upgrade path. Used GPU carries risk — verify thermal pad condition and mining history.

### Option B — Self-Build: RTX 4090 24GB (new)
**Estimated total: ~11,500–15,000 PLN**

| Component | Choice | Est. Price |
|---|---|---|
| GPU | RTX 4090 24GB | ~7,500–9,500 PLN |
| CPU | AMD Ryzen 7 7700X | ~650–800 PLN |
| Motherboard | B650 or X670E (AM5) | ~600–900 PLN |
| RAM | 64GB DDR5 | ~1,400–2,000 PLN |
| Storage | 2TB NVMe PCIe 4.0 | ~400–550 PLN |
| PSU | 1000W 80+ Gold | ~550–700 PLN |
| Case + cooler | — | ~450–550 PLN |

Notes: Full warranty, AM5 platform. Best performance-per-zloty among new hardware options.

### Option C — Pre-Built: G4M3R ELITE RTX 4090 (x-kom)
**Price: ~19,500–20,000 PLN**
Premium for assembly and warranty. Intel LGA1700 platform — limited upgrade path. Not recommended if self-assembly is feasible.

### Cloud vs. Hardware Break-Even
| Scenario | Cloud cost | Hardware cost |
|---|---|---|
| Diploma project only (6 months) | ~$180–360 | ~7,000–15,000 PLN |
| 1 year of active use | ~$500–900 | same |
| 2+ years of active use | ~$1,000–2,000+ | same |

---

## Original Tech Stack Assumptions

The original plan hardcoded these choices. They may still be correct, but they were assumed rather than decided — an infrastructure planning agent should validate them.

| Layer | Original assumption | Status |
|---|---|---|
| Compute | JarvisLabs or RunPod, A100 80GB | JarvisLabs defunct; A100 80GB over-spec |
| Fine-tuning framework | Axolotl or LLaMA-Factory | Not yet validated against actual training config |
| Quantization | BitsAndBytes | Standard choice, likely still correct |
| Base model | Llama 3.1 8B Instruct | Still the target |
| LLM-as-judge | Claude API (`claude-sonnet-4-6`) or GPT-4 | Open decision |
| Demo interface | Gradio or CLI | Open decision |

---

## Original Stage-by-Stage Infrastructure Assumptions

Stage 1 assumed A100 80GB for even the smoke test inference — now known to be unnecessary. Inference of a 4-bit quantized 8B model uses ~5–6GB VRAM and runs on any modern GPU including RTX 4090 or smaller.

No per-stage infrastructure map was ever created. This is the primary gap that infrastructure planning should address: for each of the 8 build stages, what GPU tier, storage volume, and estimated cost is actually required?
