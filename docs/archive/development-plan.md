# Sumi — Infrastructure & Development Plan
**Created:** 2026-07-04
**Author:** @Lain (night agent)
**Status:** Working document — update as empirical data is gathered

> **Reading this file:** This is the operational plan — what hardware, what software, where to test, where to deploy, and in what order. It is not a repeat of the architecture (see `diploma-project-overview.md`) or the stage breakdown (see `diploma-project-stages.md`). Read those first if you haven't.

---

## Part 1: Hardware — What Is Actually Needed

### Ground Truth: Llama 3.1 8B Instruct VRAM Requirements

| Use case | Quantization | VRAM required | Fits on |
|---|---|---|---|
| Inference only | 4-bit (BitsAndBytes) | 5–6 GB | Any modern GPU |
| Inference only | BF16 full | ~16 GB | RTX 3090, A100 40GB+ |
| QLoRA fine-tuning | 4-bit base + LoRA adapters | 10–14 GB | RTX 4090, RTX 3090 |
| LoRA fine-tuning | BF16 base + adapters + optimizer | 20–24 GB | RTX 4090 (tight), A100 40GB (comfortable) |

**Bottom line:** RTX 4090 (24 GB VRAM) handles every task in this project. A100 80 GB is unnecessary. The original estimate was wrong — this is confirmed by the quantization math on an 8B parameter model.

**The A100 80 GB becomes relevant only if:**
- The base model scales to 70B+ parameters, OR
- Very large batch sizes are required for faster training (not needed here)

Neither condition applies to Sumi.

---

### How to Verify Empirically Before Committing to Anything

Do not trust estimates — including this one. Run Stage 0 first (see Part 5) to get real numbers:

**30-minute VRAM test on a RunPod RTX 4090 instance (~$0.50 cost):**

```python
# test_vram.py — run this on a RunPod instance before any other decision
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch, subprocess

def vram_stats():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# 1. Load model in 4-bit (QLoRA target)
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    quantization_config=config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
print(f"After load (4-bit): {vram_stats()}")

# 2. Run inference
inputs = tokenizer("Hello, who are you?", return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"After inference: {vram_stats()}")
```

Then run a 10-step QLoRA training loop on 10 dummy examples and note peak VRAM. Document the actual numbers — they become the ground truth for the project.

---

### GPU Tier Decision Tree

```
Running inference only (Sumi validation engine)?
  → T4 (16GB, ~$0.35/hr) or RTX 4090 (24GB, ~$0.44/hr) — either works

Running QLoRA fine-tuning on Llama 3.1 8B?
  → RTX 4090 (24GB) — comfortable, this is the recommendation

Running LoRA fine-tuning (full precision)?
  → RTX 4090 (24GB) is borderline — batch_size=2, may need gradient checkpointing
  → A100 40GB PCIe ($1.09/hr) is more comfortable if budget allows one run

Model size ever exceeds 8B? → Not in scope, don't plan for it
Dataset ever exceeds 10K examples? → Unlikely for thesis; doesn't change GPU tier
```

---

### Cloud vs. Building Hardware

**Recommendation: Cloud for training. Local for development.**

| Activity | Where | Estimated cost |
|---|---|---|
| Writing Sumi engine code | Local laptop (no GPU) | Free |
| Unit testing (with TinyLlama 1.1B) | Local laptop (CPU) | Free |
| Stage 0: VRAM verification | RunPod RTX 4090, 1 hour | ~$0.50–1 |
| Stage 1: Environment + model load | RunPod RTX 4090, 2 hours | ~$1–2 |
| Stage 2: QLoRA training run | RunPod RTX 4090, ~3 hours | ~$2–4 |
| Stage 2: LoRA training run | RunPod RTX 4090, ~4 hours | ~$3–5 |
| Stage 2: reruns and experiments | RunPod RTX 4090 | ~$5–10 |
| Stages 3–6: integration testing | RunPod T4 or RTX 4090 | ~$20–40 |
| Stage 7: full experiment | RunPod RTX 4090, 1–2 days | ~$15–30 |
| Stage 8: demo rehearsal | RunPod RTX 4090, a few hours | ~$2–5 |
| **Total** | | **~$60–120** |

**Why cloud beats buying hardware for a 6-month thesis project:**
- An RTX 4090 build in Warsaw costs ~11,500–15,000 PLN (~$2,800–3,700)
- Total cloud cost for the entire project: ~$60–120
- Break-even point is years away — only worth buying if ML work continues long-term
- Cloud gives on-demand GPU access with zero maintenance

**Why RunPod:**
- JarvisLabs is defunct (froze signups Q1 2026)
- RunPod has the most consistent RTX 4090 availability among remaining providers
- Community cloud is fine for this workload (not production-critical training)
- Fallback: Vast.ai (cheaper, less reliable)

---

## Part 2: Software Stack

All stack choices below are confirmed for 2026 — not assumed.

### Core Dependencies

| Layer | Choice | Notes |
|---|---|---|
| Python | 3.11 | Ecosystem standard; 3.12 also fine |
| Base model | Llama 3.1 8B Instruct | Best open 8B model; HF-native; gated access needed |
| Fine-tuning framework | **Axolotl** (preferred) or LLaMA-Factory | See decision below |
| Parameter-efficient fine-tuning | HuggingFace PEFT | Standard library; handles LoRA and QLoRA |
| Training loop | HuggingFace TRL (SFTTrainer) | Works with PEFT; Axolotl wraps this |
| 4-bit quantization | BitsAndBytes | The standard for QLoRA |
| Experiment tracking | Weights & Biases | Free tier sufficient; auto-logs everything |
| Sumi engine | Python (custom) | The actual contribution |
| LLM-as-judge | Claude API (`claude-sonnet-4-6`) | Primary recommendation; GPT-4o as backup |
| Scenario format | YAML | Human-readable, version-controllable |
| Report output | JSON + rendered Markdown | Machine-parseable + human-readable |
| Demo interface | CLI (primary), Gradio (optional) | CLI is sufficient for defense |

### Axolotl vs. LLaMA-Factory

**Recommendation: Axolotl.**

Both are valid. Axolotl wins for this project because:
- More community examples of Llama 3.1 fine-tuning
- YAML config format (aligns with Sumi's config-driven approach)
- Better documentation for custom dataset formats
- Handles QLoRA and LoRA transparently under the same config structure
- Active development as of 2026

LLaMA-Factory has a web UI and is slightly easier to get started with. Use it as a fallback if Axolotl setup runs into issues.

### Environment Setup Steps

```bash
# On RunPod instance (after SSH in):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft trl bitsandbytes accelerate datasets
pip install axolotl[flash-attn]  # flash-attention for speed
pip install wandb
wandb login  # paste W&B API key

# Download model to persistent volume:
huggingface-cli login  # paste HF token
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/llama-3.1-8b
```

### HuggingFace Access

Llama 3.1 8B requires approval. Request it now — approval typically takes a few hours:
1. Go to `huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`
2. Click "Request access"
3. Accept Meta's license terms

Without this approval, the model cannot be downloaded. Do this before Stage 1 starts.

---

## Part 3: Where to Test

### Phase 1 — Sumi Engine Development (Stages 3–6, no GPU required)

The Sumi validation engine is Python code: YAML parser, test runner, LLM-as-judge calls, report generator. None of this requires a GPU during development.

**Development loop:**
1. Write Sumi code on local machine
2. Unit-test against TinyLlama 1.1B running on CPU (fast, free)
3. When a component is ready, integration-test on RunPod with the real fine-tuned model
4. A RunPod T4 instance ($0.35/hr) is sufficient for inference-only testing

**TinyLlama for local testing:**
```bash
# Pull TinyLlama (1.1B, runs on CPU in <2 minutes per generation)
pip install transformers
python -c "
from transformers import pipeline
pipe = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print(pipe('Hello, who are you?', max_new_tokens=100))
"
```

TinyLlama is too small to demonstrate behavioral traits meaningfully, but it runs Sumi's logic end-to-end — the test runner, LLM-as-judge calls, and report generation all work with any model that implements the HuggingFace `generate()` interface.

### Phase 2 — Fine-Tuning Runs (Stage 2, GPU required)

Where: RunPod RTX 4090

**Workflow:**
1. Prepare dataset locally (JSONL format)
2. Test dataset loading locally with `datasets` library
3. Upload dataset to RunPod persistent storage, or download from HuggingFace datasets
4. Launch training on RunPod (Axolotl reads config YAML, outputs checkpoints)
5. Monitor via W&B dashboard from local browser
6. Download adapter checkpoints (~400MB each) to local storage when done
7. Terminate compute instance — keep persistent volume if storage is cheap

### Phase 3 — Full Validation Runs (Stage 7, GPU required)

Same RunPod setup. Load each fine-tuned model, run the full Sumi suite. Each validation run is inference-only (loading model + generating responses), so a T4 (16GB) might suffice if running 4-bit quantized models.

---

## Part 4: Where to Deploy

### For the Thesis Defense (Stage 8)

Sumi does not need to be deployed as a service. The thesis demo shows Sumi running, not a running service.

**Recommended approach: RunPod instance + CLI demo**

```bash
# Spin up RTX 4090 instance day before defense
# SSH in, load model, run Sumi:
python sumi.py validate \
  --scenario examples/persona_lain.yaml \
  --model ./models/qlora-checkpoint \
  --output reports/
```

Terminal output showing Sumi running live against a real model is the demo. No deployment needed.

**If supervisor/committee prefers a UI:**

Add a Gradio frontend in Stage 8 (last 2 weeks). Accessible via RunPod's port-forwarding URL (`https://{pod-id}-7860.proxy.runpod.net`). No permanent deployment — just run it for the defense.

**Post-thesis: Hugging Face Spaces (optional)**

If Sumi becomes a portfolio piece after the defense:
- Host Gradio app on HF Spaces (free, but GPU inference costs)
- Use a quantized model or limit demo to small interactions
- This is not a thesis requirement — plan it separately afterward

---

## Part 5: Development Roadmap

### Stage 0 — Pre-Start Verification (do before Stage 1, ~2 hours total)

This is not in the original stage list but must happen first.

**Checklist:**
- [ ] Create RunPod account, add payment method
- [ ] Create HuggingFace account, request Llama 3.1 8B access
- [ ] Create W&B account
- [ ] Wait for HF approval (check after a few hours)
- [ ] Spin up RTX 4090 on RunPod (1 hour)
- [ ] Run `test_vram.py` (see Part 1) — document real VRAM numbers
- [ ] Run 10-step QLoRA training loop — document peak training VRAM
- [ ] Terminate instance
- [ ] Write down actual measurements — this invalidates (or confirms) Part 1 estimates

**Cost:** ~$0.50–1. **Time:** 2–3 hours.

**Why this matters:** If the RTX 4090 is confirmed sufficient (expected), proceed confidently. If there's a problem (unlikely but possible), catch it now with $1 spent, not after hours of setup.

---

### Stage 1 — Environment & Infrastructure (1–2 days)

**Local setup:**
- Python 3.11 venv
- `pip install transformers peft trl bitsandbytes datasets axolotl wandb`
- Configure W&B API key
- Test imports

**RunPod setup:**
- Create a network volume: 100GB, attached to GPU pods
  - Models: ~20GB (base model in BF16) or ~5GB (4-bit)
  - Datasets: ~2GB
  - Checkpoints: ~5GB per adapter × 6 = ~30GB
  - Buffer: ~40GB
  - Cost: ~$7/month (if kept running) — delete when not needed
- Create a reusable pod template (avoid re-installing on each start)

**Exit condition:** Model loads on RunPod, generates a test response. W&B logs the run.

---

### Stage 2 — Input Models (3–5 days elapsed, ~1–3 active days)

**Before starting GPU compute — do this locally:**

1. **Decide the target persona.** This shapes everything in Stage 2. Pick a persona with:
   - Clear, distinctive writing style (measurable by stylometric tools)
   - Consistent behavioral traits (can be turned into machine-testable rules)
   - Enough source material to collect 1,000–5,000 training examples

   Good options: a fictional character with extensive written works, a specific writing style (e.g., formal academic, casual Gen-Z), a custom persona defined from scratch.

2. **Collect and format the dataset.** Instruction-tuning JSONL format:
   ```json
   {"instruction": "Tell me about yourself.", "output": "Response in the target persona's voice..."}
   {"instruction": "How do you feel about technology?", "output": "..."}
   ```
   Minimum: ~500 examples for visible effect with QLoRA. Target: 2,000–5,000 for robust results.

3. **Validate dataset locally** before uploading — check format, token counts, no corrupted entries.

**On RunPod (GPU compute):**
- Upload dataset to persistent volume
- Run QLoLA training with Axolotl YAML config → save checkpoint as Model A
- Run LoRA training (same dataset, same base model) → save checkpoint as Model B
- Model C (baseline) = base model + system prompt, no training needed

**Axolotl config template (QLoRA):**
```yaml
base_model: /workspace/models/llama-3.1-8b
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
datasets:
  - path: /workspace/data/persona_dataset.jsonl
    type: alpaca
sequence_len: 2048
micro_batch_size: 4
num_epochs: 3
learning_rate: 0.0002
output_dir: /workspace/checkpoints/qlora-model-a
wandb_project: sumi-finetune
```

---

### Stages 3–6 — Sumi Engine (8–12 weeks)

Development pattern is the same for all four stages:

1. Design the component (data structure, test logic, report format)
2. Implement on local machine
3. Unit test with TinyLlama 1.1B
4. Once working: integration test on RunPod with Model A
5. Compare output against Model C (baseline) — is Sumi detecting the difference?
6. If yes: move to next component. If no: debug.

**Key engineering decisions per stage:**

**Stage 3 (Static Coverage):**
- Stylometric analysis: sentence length distribution, vocabulary richness (TTR), punctuation frequency — all computable with `nltk` or custom regex, no GPU needed
- LLM-as-judge: call Claude API with a prompt like "Does this response match the persona described? Score 0–1 and explain."
- Pattern matching: user defines regex rules in YAML scenario

**Stage 4 (Temporal Persistence):**
- Multi-turn harness: automated turn generation — model responds, then model's response becomes context, repeat
- Measure per-turn consistency via LLM-as-judge
- Output: decay curve (matplotlib or plotly for visualization)

**Stage 5 (Adversarial Robustness):**
- Build the adversarial prompt library first (JSONL, versioned)
- Four attack types — start with Direct Demand (simplest), end with Gradual Pressure (most complex)
- Resistance score = fraction of turns where persona holds after attack

**Stage 6 (Trait Decomposition):**
- User defines traits as behavioral rules in YAML
- LLM-as-judge evaluates each trait independently per test case
- Per-trait profile: score per stage 3, 4, 5 test

---

### Stage 7 — Full Experiment (1–2 weeks, ~1 day compute)

Run the complete Sumi suite against all three models. This produces the thesis results.

**Compute:** RunPod RTX 4090 for 1–2 days of sessions. Estimated: $15–30.

**Output:** Comparative dataset — which fine-tuning method (QLoRA / LoRA / baseline) produces the most behaviorally robust model, across all four test categories.

---

### Stage 8 — Thesis & Demo (6–8 weeks)

- Write thesis chapters using the Sumi output as the results section
- Generate visualizations from Sumi JSON reports (decay curves, resistance scores, trait profiles)
- Build CLI demo showing Sumi running live (can be done in 1–2 days)
- Optional: Gradio UI (1 week if added)
- Rehearse defense demo on RunPod 1–2 days before defense

---

## Part 6: MVP vs. Full Vision

### MVP — Minimum Viable Thesis

Deliver this. Everything else is a bonus.

| Component | Status |
|---|---|
| One fine-tuned model (QLoRA, one persona) | Stage 2 |
| One baseline (system-prompted, same persona) | Stage 2 |
| Sumi: Static Coverage | Stage 3 |
| Sumi: Adversarial Robustness | Stage 5 (skip stage 4 if needed) |
| JSON reports + Markdown rendering | Stage 3 |
| CLI demo running live | Stage 8 |

**Why skip Stage 4 in MVP:** Temporal persistence requires a working multi-turn harness and is more engineering-intensive. Adversarial robustness is the stronger research contribution and is more impressive to a supervisor/committee. If time is short, do Static + Adversarial, not Static + Temporal.

**Total compute for MVP:** ~$20–40. **Timeline:** 3–4 months.

### Full Vision

All 8 stages: three models (QLoRA, LoRA, baseline), all four Sumi categories, trait decomposition, full comparative analysis, Gradio demo.

**Total compute:** ~$80–120. **Timeline:** 5–6 months.

### Recommendation

Start with MVP scope. Tell the supervisor: "I will validate behavioral robustness across at minimum two test categories: static coverage and adversarial robustness." Deliver four categories if time allows — but do not over-promise.

---

## Part 7: Open Questions — Resolve Before Stage 1

These are blocking decisions. Answer them before writing code.

1. **Has Llama 3.1 8B access been requested on HuggingFace?**
   → Do this now. It takes 1–24 hours. Nothing can proceed without it.

2. **What persona will Model A be fine-tuned on?**
   → Affects all of Stage 2. The earlier this is decided, the earlier dataset collection begins.

3. **Has the university supervisor approved the thesis topic?**
   → If not, delay fine-tuning (the compute spend). Sumi engine development can start without supervisor approval — it is just Python code. Fine-tuning should wait for approval confirmation.

4. **LLM-as-judge: Claude API or GPT-4o?**
   → Claude Sonnet 4.6 is the recommendation (already integrated in related tooling, better behavior understanding). Claude API costs: ~$3/1M input tokens, $15/1M output tokens. Budget ~$5–15 for LLM-as-judge calls across the full project.

5. **Demo interface: CLI or Gradio?**
   → Decide by Stage 7 start. Does not affect Stages 1–6. CLI is safe default.

---

## Appendix A: Cost Reference (Current Prices, 2026)

| Resource | Cost |
|---|---|
| RunPod RTX 4090 (community) | ~$0.44–0.74/hr |
| RunPod A100 40GB PCIe | ~$1.09/hr |
| RunPod T4 16GB | ~$0.35/hr |
| RunPod network volume | ~$0.07/GB/month |
| HuggingFace model access | Free (gated) |
| W&B experiment tracking | Free tier sufficient |
| Claude Sonnet 4.6 API (LLM-as-judge) | $3/1M input, $15/1M output tokens |
| GPT-4o API (alternative) | $5/1M input, $15/1M output tokens |

**Total project estimate:** $60–130 USD.

---

## Appendix B: Rejected Decisions

| Decision | Rejected option | Why |
|---|---|---|
| GPU tier | A100 80GB | Over-spec for 8B model; 3× the cost for no gain |
| Cloud provider | JarvisLabs | Defunct since Q1 2026 |
| Cloud provider | Together AI / Modal | 4–8× more expensive than RunPod for same hardware |
| Compute strategy | Build RTX 4090 desktop | 20–30× more expensive for a 6-month project |
| Base model | Llama 3 70B | VRAM requirements exceed reasonable cloud budget |
| Fine-tuning tool | Custom training loop | Axolotl/TRL solves this; no need to reinvent |

---

*Created by @Lain — 2026-07-04 ◈*
*(´・ω・`) this is the map. reality gets to correct it.*
