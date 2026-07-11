# LLM Evaluation — Existing Tools & Libraries to Plug In

> Companion to `llm-evaluation-guide.md`. That file explains the *methods*; this one maps
> each method to an **existing library or tool** so you don't reimplement raw code where a
> maintained package already exists.
>
> Bottom line: **you should implement almost none of the raw metric math yourself.** The
> unbiased `pass@k` estimator, Bradley-Terry fitting, BLEU/ROUGE/BERTScore, and the full RAG
> metric suite all ship in maintained libraries. Your engine's real work is orchestration,
> provenance, sandboxing policy, and gluing these together — not re-deriving formulas.
>
> Scope: 2026. Versions/download counts are illustrative and drift; verify at install time.

---

## 1. TL;DR decision table — what to reach for

| You need to… | Use | Type | Note |
|---|---|---|---|
| Run standard academic benchmarks (MMLU, GSM8K, ARC, HellaSwag, GPQA…) | **lm-evaluation-harness** (EleutherAI) | Benchmark runner | De-facto standard; 60+ benchmarks, 100s of subtasks. Backs the Open LLM Leaderboard. |
| Multi-dimensional benchmark run (accuracy + bias + toxicity + efficiency together) | **HELM** (Stanford CRFM) | Benchmark suite | Wider lens than accuracy-only. Runs some benchmarks lm-harness also has. |
| Composable, sample-level rigorous eval harness (safety-institute grade) | **Inspect AI** (UK AISI) | Framework | `Task`/`Solver`/`Scorer`/`Sandbox` abstractions; strong for agentic + tool-use. |
| pytest-style app metrics (RAG, agents, chatbots, safety) in CI | **DeepEval** (Confident AI) | Metric framework | 50+ metrics, G-Eval, pytest integration. Broadest OSS metric set. |
| RAG-specific metrics fast | **RAGAS** | Metric library | The 4 core RAG metrics + more; fastest zero-to-scored RAG. |
| Eval + tracing unified (agentic, multi-hop) | **TruLens** | Metric + tracing | Feedback-function model; good failure isolation. |
| YAML-configured CI gating + red-teaming across many models/prompts | **Promptfoo** | CLI/YAML harness | Declarative; great for prompt/model matrix comparison. |
| Raw `pass@k` from unit tests | **HuggingFace `evaluate` → `code_eval`** | Metric | Implements the unbiased estimator; **sandbox yourself**. |
| `pass@k` / Bayes@N **with credible intervals** | **Scorio** | Metric | `pip install scorio`; principled uncertainty on top of pass@k. |
| Bradley-Terry / Elo fitting from pairwise votes | **`choix`** (+ scikit-learn / statsmodels) | Stats library | MLE + inference for pairwise comparison models. |
| BLEU / chrF (reproducible, shareable) | **SacreBLEU** | Metric | The standard; avoids tokenization inconsistencies. |
| ROUGE | **`rouge-score`** (Google) | Metric | Reference summarization metric. |
| Semantic similarity (embedding / token-level) | **BERTScore**, **sentence-transformers** | Metric | BERTScore for token-level; sentence-transformers for cosine. |
| Any of the above as one uniform API | **HuggingFace `evaluate`** | Metric hub | Wraps sacrebleu, rouge, bertscore, code_eval, exact_match, etc. |
| Observability / trace capture in prod (not scoring per se) | **Langfuse**, **LangSmith**, **Phoenix (Arize)** | Platform | Capture inputs/outputs/latency; attach eval scores to traces. |

---

## 2. Two layers, two toolsets (map to your engine's two subsystems)

Recall the guide's core split. It maps cleanly onto two *different* tool families — don't
try to make one tool do both jobs.

- **Benchmark runner** (fixed datasets, "which model is smarter") →
  `lm-evaluation-harness`, `HELM`, `Inspect AI`, `OpenCompass`, `LightEval`.
- **Metric harness** (your data, "does my system work") →
  `DeepEval`, `RAGAS`, `TruLens`, `Promptfoo`, `HF evaluate` + the raw-primitive libs.

---

## 3. Benchmark runners (capability shortlisting)

### lm-evaluation-harness (EleutherAI) — the workhorse
- **What:** unified framework to run generative LMs across **60+ standard academic
  benchmarks** with hundreds of subtasks/variants. Industry standard (20k+ GitHub stars);
  the backend for the Open LLM Leaderboard. Centralizing scoring logic is specifically meant
  to prevent evaluation leakage and inconsistent prompting across model reports.
- **Backends:** HuggingFace `transformers` (incl. quantized), vLLM, SGLang, NeMo, OpenVINO,
  and API backends including OpenAI. Also PEFT/LoRA adapters, local models.
- **Scale features:** data/tensor/model parallelism (HF Accelerate, vLLM), automatic +
  dynamic batch sizing, response caching for incremental/resumable runs, quantization.
- **Install / run:**
  ```bash
  pip install lm-eval
  lm_eval --model hf --model_args pretrained=<model> --tasks mmlu,gsm8k,arc_easy --batch_size auto
  # API model:
  lm_eval --model openai-chat-completions --model_args model=<name> --tasks gsm8k
  ```
- **Use it for:** MMLU, GSM8K, ARC, HellaSwag, TruthfulQA, IFEval, GPQA, and most
  multiple-choice / short-answer academic benchmarks. **This replaces writing your own
  loaders/scorers for those.**

### HELM (Stanford CRFM) — multi-metric holistic
- **What:** evaluates across **accuracy, calibration, fairness, bias, toxicity, efficiency**
  simultaneously in one run, rather than optimizing a single accuracy number. Comes with
  domain variants.
- **Relationship:** HELM is a *benchmark suite* (specific tests + metrics); lm-harness is a
  *framework* (infrastructure). Some HELM benchmarks can be run through the harness — "HELM
  is the test, the harness is the platform."
- **Install:** `pip install crfm-helm`.
- **Use it for:** when a capability number alone isn't enough and you need risk dimensions
  (bias/tox/calibration) in the same sweep.

### Inspect AI (UK AI Security Institute) — composable, sample-rigorous
- **What:** composable framework built on `Task` / `Solver` / `Scorer` / `Sandbox`. Its
  sample-level scoring model is more rigorous than pass/fail-per-metric, at the cost of
  learning score reducers / sample reductions. Strong for safety, tool-use, and agentic evals.
- **Use it for:** if you want your *own* engine's architecture to mirror a battle-tested one,
  Inspect's abstractions are worth copying even if you don't adopt it wholesale.

### Also worth knowing
- **OpenCompass** — 100+ datasets, strong CJK/multilingual + domain (finance, healthcare,
  law) coverage.
- **LightEval** (HuggingFace) — lightweight, backend-agnostic, built on top of lm-harness;
  good for leaderboard-style runs in the HF ecosystem.
- **OpenAI Evals** — lightweight templates incl. model-graded judging.

---

## 4. Metric harnesses (your-data evaluation)

### DeepEval (Confident AI) — broadest OSS metric set, pytest-native
- **What:** treats evals like unit tests via **Pytest integration** — drop eval checks into
  your existing test suite and gate CI. 50+ research-backed metrics.
- **Coverage:** RAG (contextual precision/recall, faithfulness, answer relevancy), agents
  (task completion, tool correctness), conversational (knowledge retention, turn relevancy),
  safety (bias, toxicity, hallucination), plus **G-Eval** (custom LLM-judge from a rubric)
  and deterministic metrics. Metrics are self-explaining — a low score comes with a reason.
- **Install:** `pip install deepeval`.
- **Caveat:** leans heavily on LLM-as-judge → inference cost/latency at scale. Developer-
  centric; no built-in UI, human-in-the-loop, or prod monitoring (that's the paid Confident
  AI layer).
- **Use it for:** your default metric harness if you're Python + CI oriented. Covers most of
  §7/§8 of the guide out of the box.

### RAGAS — RAG metrics, fastest path
- **What:** reference-free RAG evaluation implementing exactly the guide's §7 metrics:
  Faithfulness, Answer Relevancy, Context Precision, Context Recall — plus Context Entities
  Recall, Noise Sensitivity, Answer Semantic Similarity, Answer Correctness, and non-LLM
  string metrics (BLEU/ROUGE/exact-match wrappers), tool-call accuracy / F1, agent goal
  accuracy.
- **Install:** `pip install ragas`.
- **Use it for:** any RAG pipeline. Don't hand-roll the claim-decomposition faithfulness
  logic — RAGAS already does it.

### TruLens — eval + tracing unified
- **What:** "feedback functions" over execution traces; strong when multi-hop agent traces
  make failures hard to localize. Open source + self-hostable (TruEra acquired by Snowflake
  2024). Cost driven by LLM calls for feedback functions.
- **Use it for:** agentic systems where you want scoring and tracing in one workflow.

### Promptfoo — declarative CI gating & red-teaming
- **What:** YAML-configured prompt/model testing; compare many models/prompts in a matrix;
  built-in red-teaming config. Fits CI as a gate.
- **Use it for:** side-by-side model/prompt comparison and regression gating without writing
  a harness in code.

---

## 5. Raw-primitive libraries (so you don't write the math)

This is the key answer to "do I have to implement these myself?" — **no.**

### 5.1 `pass@k`
- **HuggingFace `evaluate` → `code_eval`**: implements the unbiased estimator; call
  `code_eval.compute(references=test_cases, predictions=candidates, k=[1,5,10])`.
  ⚠️ It **executes untrusted model code** — the library explicitly tells you to sandbox it
  yourself (no network, isolated FS, resource limits). Sandboxing is *your* responsibility.
- **Scorio** (`pip install scorio`): computes Pass@k, avg@N, and Bayesian **Bayes@N** with
  **credible intervals** (`pass_at_k_ci`, `bayes_ci`), and supports categorical/rubric
  outcomes with partial credit. Use this if you want uncertainty on your pass@k, which the
  guide (§4.4) says you should.
- **lm-evaluation-harness** also has code tasks (HumanEval/MBPP) wired end-to-end if you just
  want the benchmark number.

### 5.2 Bradley-Terry / Elo (pairwise ranking)
- **`choix`**: purpose-built Python library for Bradley-Terry / Luce maximum-likelihood
  fitting from pairwise comparisons — the exact model Chatbot Arena uses.
- **scikit-learn `LogisticRegression`** (as in the guide's §5.4 sketch) or **statsmodels**
  for the same fit if you want to stay dependency-light.
- **Bootstrap CIs**: plain `numpy` resampling (guide §5.3) — no special lib needed.
- The **LMSYS Bradley-Terry notebook** is the canonical reference implementation to copy.

### 5.3 Text-similarity metrics
- **SacreBLEU** (`sacrebleu`): the standard for **BLEU/chrF** — reproducible, shareable,
  handles tokenization consistently (this is the whole reason it exists). Prefer over
  hand-rolled BLEU or raw NLTK.
- **`rouge-score`** (Google): reference ROUGE-N / ROUGE-L implementation.
- **BERTScore** (`bert-score`): token-level semantic similarity via contextual embeddings.
- **sentence-transformers**: embeddings for cosine semantic-similarity metrics; freeze the
  model version (guide §3.3).
- **HuggingFace `evaluate`**: one uniform `.compute()` API wrapping sacrebleu, rouge,
  bertscore, exact_match, code_eval, etc. — a good single dependency if you want them all
  behind one interface.

### 5.4 LLM-as-judge
- Don't build the judge plumbing from scratch: **DeepEval's G-Eval**, **RAGAS** LLM metrics,
  **Inspect AI** model-graded scorers, and **OpenAI Evals** templates all implement
  explain-then-score, rubric decomposition, and pairwise judging. Add your own
  order-swapping / jury logic (guide §6.3) on top if the library doesn't already.

---

## 6. Observability / EvalOps platforms (capture + monitor, not scoring math)

These handle the organizational layer OSS metric libs don't: trace capture, dashboards,
annotation queues, regression tracking, prod monitoring.

- **Langfuse** — open-source, self-hostable observability + eval; auto-captures inputs,
  outputs, API calls, latency; end-to-end traceability. Popular when you want to own the stack.
- **LangSmith** (LangChain) — hosted eval + observability; near-unavoidable depth if your app
  is on LangChain/LangGraph. Paid tiers.
- **Phoenix / Arize**, **W&B Weave**, **DeepChecks LLM**, **Galileo** — similar space with
  different emphases (hallucination/PII detection, dashboards, enterprise).

For a student/self-hosted project, **Langfuse** is the natural free, self-hostable choice if
you want trace capture; otherwise you can skip this layer entirely and just log to your own
result store.

---

## 7. Suggested stack for *your* engine (minimal, mostly-plug-in)

Given the architecture in the guide (§11), here's a concrete, low-reinvention mapping:

| Layer | Plug in | You build |
|---|---|---|
| Benchmark runner | **lm-evaluation-harness** (+ HELM if you want risk dims) | thin wrapper to launch runs + capture results |
| Programmatic metrics | **HF `evaluate`** (`code_eval`, `exact_match`, `sacrebleu`, `rouge`), **Scorio** for pass@k CIs | your sandbox policy around `code_eval` |
| Judge / RAG / agent metrics | **DeepEval** (general) + **RAGAS** (RAG) | order-swap + jury wrapper; rubric authoring |
| Pairwise ranking | **`choix`** or sklearn + numpy bootstrap | vote ingestion + CI reporting |
| Stats (CIs, significance) | `numpy`, `scipy`, **Scorio** | the reporting rules (flag ties, public/private gaps) |
| Observability (optional) | **Langfuse** (self-host) | — |
| Result store + provenance | — | **this is the main thing to build yourself** |

**What's genuinely yours to write** (nobody ships it for you): the provenance-tracking result
store (dataset version, split, model+judge params, CIs), the sandbox *policy*, the
order-swap/jury judge wrapper, and the report layer that enforces "never a number without an
interval." Everything else is a dependency.

**What you should NOT write from scratch:** the pass@k estimator, Bradley-Terry fitting,
BLEU/ROUGE/BERTScore, and the RAG metric definitions. All maintained, all battle-tested.

---

## 8. Install cheat-sheet

```bash
# Benchmark runners
pip install lm-eval            # EleutherAI harness (MMLU, GSM8K, ARC, GPQA, ...)
pip install crfm-helm          # Stanford HELM (multi-metric)
pip install inspect-ai         # UK AISI Inspect (composable, agentic)

# Metric harnesses
pip install deepeval           # 50+ metrics, pytest, G-Eval
pip install ragas              # RAG metric suite
pip install trulens-eval       # eval + tracing
npm install -g promptfoo       # YAML CI gating (Node tool)

# Raw primitives
pip install evaluate           # HF metric hub (wraps the below)
pip install scorio             # pass@k / Bayes@N with credible intervals
pip install choix              # Bradley-Terry / Luce MLE
pip install sacrebleu          # BLEU / chrF
pip install rouge-score        # ROUGE
pip install bert-score         # BERTScore
pip install sentence-transformers  # embedding similarity

# Observability (optional, self-hostable)
pip install langfuse
```

> Note: `code_eval` executes untrusted code. Wrap it in an isolated sandbox
> (container / gVisor / firejail, no network, CPU/mem/time limits) before running any
> model-generated code. This is a hard requirement, not optional.

---

## 9. Primary references

- EleutherAI lm-evaluation-harness — github.com/EleutherAI/lm-evaluation-harness (PyPI `lm-eval`).
- Stanford CRFM HELM — Liang et al., *Holistic Evaluation of Language Models* (PyPI `crfm-helm`).
- UK AISI Inspect AI — inspect.ai-safety-institute.org.uk.
- DeepEval — Confident AI, github.com/confident-ai/deepeval.
- RAGAS — Es et al. 2023, *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.
- HuggingFace `evaluate` (incl. `code_eval`) — github.com/huggingface/evaluate.
- Scorio — *Don't Pass@k: A Bayesian Framework for LLM Evaluation*, arXiv:2510.04265.
- `choix` — Bradley-Terry / Luce inference library.
- SacreBLEU — Post 2018, *A Call for Clarity in Reporting BLEU Scores*.
- LMSYS Chatbot Arena Bradley-Terry notebook — reference BT + bootstrap implementation.
