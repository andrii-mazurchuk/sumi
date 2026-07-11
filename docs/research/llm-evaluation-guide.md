# LLM Evaluation & Benchmarking — Implementation Reference

> A practical reference for building an LLM validation engine. Covers the standard
> benchmarks, the metrics behind them, exact formulas, implementation guidance,
> correct usage, and the failure modes to guard against.
>
> Audience: engineers implementing evaluation tooling, and agents operating on this
> codebase. Scope: as of 2026. Treat the "current scores" as illustrative — they
> go stale fast. The *methods* are the durable part.

---

## 0. The single most important distinction: benchmarks vs. metrics

These are two different jobs. Most teams that get evaluation wrong conflate them.

| | Benchmark | Metric |
|---|---|---|
| **Question it answers** | "Which model is generally more capable?" | "Does *my* system actually work?" |
| **Data** | Fixed public dataset | Your own production/eval data |
| **Use** | Shortlist candidate models | Gate deployments, monitor in prod |
| **Cadence** | Once per model selection | Continuously (CI + post-deploy) |
| **Example** | MMLU, SWE-bench, GPQA | Faithfulness, Task Completion, custom rubric |

**Rule of thumb:** pick 3–4 benchmarks to shape a shortlist, then ship 4–6 metrics on
your own data. Run both, in different places, on different cadences. When a public
leaderboard disagrees with your own eval set, **trust your own set** — it reflects your
actual usage.

An evaluation engine should treat these as two subsystems: a **benchmark runner** (fixed
datasets, standardized scoring) and a **metric harness** (pluggable scorers over arbitrary
datasets). They share primitives (a model-invocation layer, a sandbox, a judge client) but
have different lifecycles.

---

## 1. The three primitives every metric is built from

Almost every scorer reduces to one of three underlying mechanisms. Implement these three
well and most metrics become thin wrappers.

1. **Exact / programmatic verification.** A deterministic checker: string match, numeric
   equality, unit tests, a regex, a parser, SQL-equivalence. Cheap, reproducible, zero
   judgment. Use it whenever the task *has* a checkable answer (math, code, structured
   output, classification).

2. **Statistical / similarity scoring.** Compare output to a reference with a fixed
   algorithm: exact match, token-overlap (BLEU/ROUGE), edit distance, or embedding cosine
   similarity. No LLM needed. Cheap and reproducible but shallow — it measures surface or
   semantic overlap, not correctness.

3. **LLM-as-a-judge.** Use a model to score, classify, or compare outputs when there is no
   checkable answer (open-ended text, helpfulness, tone, groundedness). Powerful and
   semantics-aware but noisy, biased, and drift-prone — must be calibrated against human
   labels.

Design your metric interface so any scorer declares which primitive(s) it uses; that alone
tells you its cost, determinism, and calibration needs.

---

## 2. Benchmark map (2026) — organized by capability

General benchmarks have **saturated**: once frontier models score 88%+, the number stops
differentiating them and mostly measures test difficulty. The field responded with harder
tests, contamination-resistant designs, and human-preference arenas. Pick benchmarks by the
capability dimension your application actually uses.

### Knowledge & reasoning
- **MMLU** — 57 subjects, ~16k multiple-choice questions. The classic general-capability
  test. **Saturated** for frontier models (>88%); still informative for small/mid-tier
  models where spread remains. Use as a floor, not a differentiator.
- **MMLU-Pro** — harder, more distractor-heavy variant. Also saturating.
- **GPQA (Diamond)** — graduate-level science, "Google-proof." Still separates strong
  reasoners.
- **BIG-Bench Hard (BBH)** — 23 tasks designed to resist shortcut solutions.

### Math
- **GSM8K** — grade-school word problems. **Saturated.**
- **MATH** — competition math. Mostly saturated.
- **AIME-25 / FrontierMath** — where 2026 models actually separate. FrontierMath is
  deliberately extremely hard and held partly private.

### Code
- **HumanEval / MBPP** — isolated function generation, `pass@k` scored. **Saturated** but
  still the canonical didactic example (see §4).
- **SWE-bench Verified** — the active frontier: 500 real GitHub issues; the model must
  produce a patch that passes the repo's unit tests. Repository-scale, contamination-resistant.

### Agentic & tool use
- **BFCL (Berkeley Function-Calling Leaderboard)** — function/tool-call correctness.
- **tau-bench / TAU2** — multi-step tool use with failure recovery.
- **GAIA, WebArena** — general assistant / web-agent tasks.

### Human preference
- **Chatbot Arena (LMArena)** — crowd-sourced pairwise voting, Bradley-Terry rating
  (see §5). The largest human-evaluation dataset in the field.

### Domain-specific (the 2026 growth area)
General benchmarks fragmented into vertical ones with expert-written rubrics:
- **HealthBench** — ~48.5k rubric criteria written by 262 physicians across 26 specialties.
- **LegalBench-RAG**, finance, cybersecurity, multilingual, multimodal, etc.

> **Live leaderboards to reference rather than reimplement:** LiveBench and
> Artificial Analysis aggregate many benchmarks and update frequently. Your engine should
> be able to *reproduce* a benchmark locally for candidate models, not necessarily maintain
> a public leaderboard.

---

## 3. Statistical / similarity metrics (no LLM required)

These are the cheapest scorers and the right default whenever a reference answer exists.

### 3.1 Exact match (EM)
Output equals reference after normalization (lowercase, strip punctuation/articles/whitespace).
Binary per item; report the mean. Correct for short-answer QA and classification; wrong for
anything open-ended.

### 3.2 Token-overlap metrics — BLEU, ROUGE, chrF
- **BLEU** — precision of n-gram overlap with reference(s), with a brevity penalty. Built for
  machine translation.
- **ROUGE** — recall-oriented n-gram overlap (ROUGE-N) or longest-common-subsequence
  (ROUGE-L). Built for summarization.
- **chrF** — character-n-gram F-score; more robust across languages and morphology.

**Use with care.** They measure surface overlap, not correctness or factuality. A correct
answer phrased differently from the reference scores low; a fluent wrong answer can score
high. Fine as a cheap regression signal, poor as a quality verdict. In RAG especially they
ignore whether the answer is *supported* by evidence.

### 3.3 Semantic (embedding) similarity
Cosine similarity between embeddings of output and reference. Captures meaning across
paraphrase, unlike token overlap. Still not a correctness check — two semantically similar
strings can differ on the one fact that matters. Good as a component (e.g., RAGAS Answer
Semantic Similarity), not as a sole verdict.

**Implementation notes:** pick one embedding model and freeze it (scores aren't comparable
across embedding models); normalize vectors; cache embeddings keyed by text hash.

---

## 4. `pass@k` — the canonical code/verifiable metric

`pass@k` = the probability that **at least one** of `k` sampled attempts passes all unit
tests for a problem. It acknowledges that generation is stochastic, not deterministic.

### 4.1 Why not just sample k and check?
Sampling exactly `k` and checking "did any pass" is a valid estimate but has **high
variance**. The standard fix (from the HumanEval paper, Chen et al. 2021) is to generate
`n ≥ k` samples per problem, count the `c` that pass, and compute an **unbiased estimator**.

### 4.2 The formula

For a single problem with `n` total samples, `c` correct:

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

where `C(a, b)` is "a choose b". Intuition: `C(n−c, k) / C(n, k)` is the probability that a
randomly chosen size-`k` subset (without replacement) contains **only** failing samples;
one minus that is the probability at least one passes.

Report the value as the **mean over all problems** (macro-average across tasks):

```
pass@k = E_problems [ 1 − C(n−c, k) / C(n, k) ]
```

Edge cases: if `n − c < k` then `C(n−c, k) = 0`, so `pass@k = 1` for that problem. No
unbiased estimator exists for `n < k`, so always enforce `n ≥ k`. Estimator variance
shrinks at rate ~`1/n`.

### 4.3 Numerically stable implementation
Do **not** compute the binomials directly — they overflow. Use the product form:

```python
import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k for a single problem.
    n = total samples generated, c = number correct, k = attempts budget.
    """
    if n - c < k:
        return 1.0
    # 1 - prod_{i = n-c+1}^{n} (1 - k/i)
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def pass_at_k_corpus(results: list[tuple[int, int]], k: int) -> float:
    """results = list of (n_i, c_i) per problem. Returns macro-averaged pass@k."""
    return float(np.mean([pass_at_k(n, c, k) for (n, c) in results]))
```

### 4.4 Correct usage & pitfalls
- **Report `n`, `k`, temperature, and sampling params** — the estimator assumes i.i.d.
  samples; temperature and prompt formatting change the statistical properties.
- **`k` inflates scores.** `pass@1` ≈ single-attempt reliability; `pass@10`/`pass@100`
  reflect "generate-and-filter" workflows. Never compare a `pass@1` number to a `pass@10`
  number.
- **Standard settings:** the HumanEval paper used `n = 200`, `k ≤ 100`.
- **Confidence intervals** depend on both the number of problems *and* samples per problem;
  small eval sets give false precision.
- **Requires a sandbox.** Model-generated code is untrusted — execute in an isolated,
  resource-limited container (gVisor / firejail / a locked-down subprocess with no network,
  CPU/memory/time limits). This is a hard security requirement, not optional.
- **Related:** `coverage@k` (fraction of tasks with ≥k *distinct* correct solutions after
  dedup) — a simple fraction, not an estimator.

`pass@k` generalizes beyond code to any programmatically verifiable task (AIME/GSM8K math,
where the checker is numeric equality).

---

## 5. Human preference & pairwise ranking — Bradley-Terry / Elo

When there's no ground truth (open-ended chat), rank models by **pairwise human
preference**. Show two anonymous responses, a human picks the winner, aggregate millions of
battles into per-model strength scores. This is dynamic, so it resists contamination.

### 5.1 Elo vs Bradley-Terry
- **Online Elo** (from chess) updates ratings sequentially after each game — but results
  depend on game *order*, making ratings less stable.
- **Bradley-Terry (BT)** is the preferred method. It's the maximum-likelihood estimate of
  the same underlying model assuming **fixed** (order-independent) strengths, computed
  centrally over the whole vote history. More stable, tighter confidence intervals. Chatbot
  Arena uses BT (often still labeled "Elo" for readability).

### 5.2 The model
Each model `m` has a latent strength `β_m`. The probability that A beats B is a logistic
function of the strength difference:

```
P(A beats B) = 1 / (1 + exp(−(β_A − β_B)))          # sigmoid of the gap
```

Fit `β` by maximizing the likelihood of observed outcomes — equivalently, **logistic
regression** over pairwise comparisons. Encode each comparison `i` as a feature vector
`X_i` where the first model is `+1`, the second `−1`, others `0`; label `Y_i = 1` if the
first model won. Minimize cross-entropy:

```
minimize  (1/n) Σ_i  CrossEntropy( sigmoid(X_i · β),  Y_i )
```

Map the fitted `β` onto the familiar Elo scale (e.g. `rating = 400/ln(10) · β + 1000`,
anchored to a reference) purely for readability.

### 5.3 Confidence intervals via bootstrap
Ratings are meaningless without uncertainty. Standard approach: **bootstrap** — resample the
vote history with replacement (e.g. 100 rounds), refit BT each time, take the 2.5th/97.5th
percentiles of each model's rating as the 95% CI.

### 5.4 Minimal implementation sketch
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_bradley_terry(comparisons, models):
    """comparisons: list of (winner_idx, loser_idx). models: list of names."""
    M = len(models)
    X, y = [], []
    for w, l in comparisons:
        row = np.zeros(M); row[w] = 1.0; row[l] = -1.0
        X.append(row);  y.append(1)          # "first (winner) model won"
    X = np.array(X); y = np.array(y)
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=1000)
    lr.fit(X, y)
    beta = lr.coef_[0]
    return 400.0 / np.log(10) * (beta - beta.mean()) + 1000.0   # Elo-scaled

def bootstrap_ci(comparisons, models, rounds=100):
    ratings = []
    n = len(comparisons)
    for _ in range(rounds):
        sample = [comparisons[i] for i in np.random.randint(0, n, n)]
        ratings.append(fit_bradley_terry(sample, models))
    R = np.array(ratings)
    return R.mean(0), np.percentile(R, 2.5, 0), np.percentile(R, 97.5, 0)
```
(Real deployments handle ties, add regularization, and often use the `choix` library.)

### 5.5 Correct usage & pitfalls — read this before quoting a rank
- **Always quote the CI.** A rank without `± value` is misleading.
- **Overlapping CIs = statistical tie.** If `(Elo_A − CI_A) < (Elo_B + CI_B)` and vice
  versa, A and B are tied regardless of rank order. Top-3 Arena models are routinely tied;
  treat clusters within ~50 Elo as a wash unless per-topic boards agree.
- **Vote volume matters.** Low-volume models have CIs of 30+ Elo; ratings stabilize to
  ±10–15 only after several thousand comparisons.
- **Style/length bias.** Longer, more formatted, more confident answers win votes
  independent of correctness — hence **style-controlled** ratings, which regress out
  length/markdown. Prefer the style-controlled number.
- **Sycophancy bias.** Answers that agree with the user's framing win more, even when a more
  cautious answer is better.
- **It's one signal.** Human preference filtered through whatever users happen to ask. Pair
  it with task-specific benchmarks; never make a capability claim from Arena alone.

---

## 6. LLM-as-a-judge

When there's no checkable answer, use an LLM to score. Achieves ~80–90% agreement with human
judgment at a fraction of the cost — but only if you design it carefully.

### 6.1 The three single-output techniques
1. **Pointwise / direct scoring** — judge scores one output against a rubric (e.g. 1–5 or
   pass/fail). Simplest; most prone to scale drift.
2. **Pairwise comparison** — judge picks the better of two outputs. More reliable for
   "is version B better than A?" (this is the primitive behind Arena-style ranking).
3. **Reference-guided** — judge compares output against a gold answer/rubric.

A fourth, **pairwise**, spans model-vs-model comparison and prompt/version A/B testing.

### 6.2 Techniques that measurably improve reliability
- **Make the judge explain before scoring.** Forcing a single bare number is *suboptimal*;
  requiring a rationale then a score significantly improves alignment with humans (G-Eval).
- **Write explicit evaluation steps.** Vague criteria → noisy judges. Spell out what each
  score level means.
- **Decompose into a decision tree / checklist** (DAG-style) when the judge must enforce
  hard rules — one broad judgment becomes several small, well-defined ones.
- **Use a jury, not a judge.** A *panel* of diverse models reduces single-model bias
  ("replace judges with juries").
- **Cross-check against human labels.** Even a small pass/fail set from a domain expert tells
  you whether the judge agrees with humans. This is mandatory before trusting a judge.

### 6.3 Judge biases to defend against
- **Position bias** — favors whichever answer is shown first. **Mitigation:** run both
  orders and average, or randomize and track order as a variable.
- **Length/verbosity bias** — prefers longer answers.
- **Self-preference / preference leakage** — a judge favors outputs from itself or from
  related models (same family, or models trained on data the judge helped synthesize). This
  is a genuine contamination channel (ICLR 2026 work). **Never** judge a model with a judge
  from the same family when ranking it against competitors.
- **Style/formatting bias** — markdown, confidence, and structure sway scores.

### 6.4 Judge reliability is itself measurable (2026 practice)
Don't just ask "which model won" — ask "is my judge a reliable instrument?"
- **Agreement with humans:** report it (target ~90%). Use Cohen's/Krippendorff's for
  inter-rater agreement; teams target **Krippendorff's α ≈ 0.8**.
- **Raw judge scores drift** across time and model versions — don't compare raw scores
  across studies without **calibration**. Newer methods apply bias-corrected confidence
  intervals (accounting for imperfect judge sensitivity/specificity) and Item Response
  Theory to find rubric items that are too easy, ambiguous, or judge-sensitive.

### 6.5 Implementation notes
- Pin the judge model + version + prompt + temperature (0 for reproducibility) and record
  them with every score. A judge is a measurement instrument; changing it invalidates
  historical comparisons.
- Log the judge's rationale (`verbose`) for debugging.
- Budget for order-swapped double-runs on pairwise comparisons.

---

## 7. RAG evaluation metrics

A RAG pipeline has a **retriever** and a **generator**; evaluate each in isolation so you can
localize failures (bad retrieval vs bad generation). The RAGAS framework is the reference and
is **reference-free** (needs no gold answer for the core four). Each is typically an
LLM-as-judge computation under the hood.

### Retriever metrics
- **Context Precision** — are the *relevant* retrieved chunks ranked near the top? (signal-to-
  noise + ranking quality). Low → retriever returns relevant material but buries it, or pulls
  too much noise.
- **Context Recall** — does the retrieved context contain all the info needed to answer?
  Computed as the fraction of ground-truth claims covered by the retrieved context. Low →
  necessary evidence is missing.

### Generator metrics
- **Faithfulness (groundedness)** — is the answer supported by the retrieved context? The
  core anti-hallucination metric. Computed as:

  ```
  Faithfulness = (# claims in answer supported by context) / (total claims in answer)
  ```

  Procedure: an LLM decomposes the answer into atomic claims, then verifies each against the
  context.
- **Answer Relevancy (Response Relevancy)** — does the answer actually address the question
  (not off-topic, vague, or padded)? Often computed by generating candidate questions from
  the answer and measuring their similarity to the real question.

### End-to-end metrics
- **Answer Semantic Similarity** — meaning-level closeness to a ground-truth answer.
- **Answer Correctness** — combines factual overlap with semantic similarity vs ground truth.

An overall RAGAS score is often just the arithmetic mean of the four core metrics, though
weighting by importance is common.

**Interpreting jointly (this is the point):** the four metrics *localize* failure. Low
Context Recall → fix retrieval/chunking. High Context Recall but low Faithfulness → the
model has the evidence but ignores it (a generation/prompt problem). This diagnostic split
is why you run all four rather than one aggregate.

---

## 8. Agentic evaluation metrics

Agents (multi-step, tool-using) need outcome- and process-level metrics beyond single-turn
quality:
- **Task Completion / Goal Accuracy** — did the agent actually accomplish the objective?
  The headline outcome metric.
- **Tool-call correctness** — right tool, right arguments (BFCL-style). Report as accuracy
  or **Tool-Call F1** (precision/recall over expected calls).
- **Trajectory quality** — was the path efficient and free of loops / redundant steps?
- **Failure recovery** — does the agent recover when a tool errors (tau-bench focus)?

### The "Hybrid Norm" (current best practice for agentic/coding)
Combine **verifiable rewards** with **rubric judgments**:
- **Verifiable:** unit tests / numeric checkers answer *"did it solve the problem?"* (the
  "what"). Objective, cheap, gameable-resistant.
- **Rubric (LLM-judge):** answers *"how well / was the process sound?"* (the "how"). Handles
  the qualitative dimension.

This underpins **RLVR (RL from Verifiable Rewards)**. For an eval engine: prefer a verifiable
checker whenever one exists, and layer a rubric judge only for the parts that can't be
checked programmatically.

---

## 9. Data contamination — the central threat to static benchmarks

Because models train on web-scale corpora, a benchmark may have **leaked** into training
data, inflating scores. You often can't verify exposure directly. Signs and defenses:

- **Symptom:** big score drop on a held-out **private** subset vs the public one. (Observed
  repeatedly on realistic software-engineering tasks — frontier models drop several points
  from public to private splits.) Your engine should support **public/private split
  reporting** and flag large gaps.
- **Defenses:**
  - Prefer **contamination-resistant** benchmarks (repo-scale, freshly-authored, held-out
    private sets, brand-new exams the model couldn't have seen).
  - Prefer **dynamic** evaluation (Arena-style live votes; adaptive question-tree
    generators like TreeEval that create unique paths per run).
  - Use **canary strings** and n-gram overlap checks between eval items and known training
    corpora where possible.
  - Rotate / refresh eval sets; treat any static benchmark's absolute number with suspicion
    once it's popular.

Related governance point: raw LLM-judge scores and benchmark numbers are **not** comparable
across time without calibration and contamination controls. Bake provenance (dataset
version, date captured, split) into every stored result.

---

## 10. Other systematic pitfalls (checklist)

- **Saturation** — a maxed-out benchmark measures difficulty, not model quality. Drop it
  from the differentiating set.
- **Benchmark gaming** — models tuned to the test, not the capability. Cross-check with
  held-out and private sets.
- **Position / length / style / sycophancy bias** in judges and arenas (see §5, §6).
- **Preference leakage** — related generator/judge models (see §6.3).
- **False precision** — small eval sets or low vote volume produce narrow-looking intervals
  that aren't real. Always compute and report CIs.
- **Metric/benchmark confusion** — shipping a leaderboard winner that fails on your data
  (see §0).
- **Non-reproducibility** — unpinned model versions, temperatures, prompts, judge models.
  Pin and log everything.

---

## 11. Recommended architecture for a validation engine

A layered pipeline (mirrors how mature teams wire evaluation into CI/CD):

```
Layer 1 — Automated benchmarks & programmatic metrics  (fast, deterministic, cheap)
          exact-match, pass@k, tool-call F1, similarity scores
          → runs on every model change; hard gate.

Layer 2 — LLM-as-judge metrics                          (semantic, calibrated, mid-cost)
          faithfulness, answer relevancy, rubric scores, pairwise A/B
          → triggered when Layer 1 passes; soft gate + report.

Layer 3 — Human review / preference                     (gold standard, slow, expensive)
          expert rubric labels, arena-style pairwise
          → triggered on anomalies or release candidates; calibrates Layers 1–2.
```

**Cross-cutting components to build once and share:**
- **Model-invocation layer** — provider-agnostic, records model+version+params per call.
- **Secure sandbox** — isolated execution for any code/tool metric (no network, resource
  limits). Non-negotiable for `pass@k` and agentic evals.
- **Judge client** — pinned judge model, rationale logging, order-swap support, jury option.
- **Result store with full provenance** — dataset version, split (public/private), date,
  model params, judge params, raw outputs, and CIs. Enables honest cross-time comparison.
- **Statistics module** — bootstrap CIs, significance tests, the `pass@k` and BT estimators.
- **Report layer** — always emits point estimate **plus** interval; flags ties and
  public/private gaps.

**Design principles**
1. Every scorer declares its primitive (exact / similarity / judge), cost, and determinism.
2. Nothing is reported without an uncertainty estimate.
3. Everything needed to reproduce a number is stored with the number.
4. Prefer verifiable checkers; add judges only where verification is impossible; calibrate
   judges against humans.
5. Keep benchmark-runner and metric-harness as separate subsystems over shared primitives.

---

## 12. Quick reference table

| Metric | Primitive | Task type | Formula / method | Key caveat |
|---|---|---|---|---|
| Exact Match | exact | short-answer QA | normalized string equality | useless for open-ended |
| BLEU / ROUGE / chrF | similarity | translation / summarization | n-gram overlap | surface, not correctness |
| Semantic similarity | similarity | any w/ reference | embedding cosine | freeze the embedder |
| `pass@k` | exact (unit tests) | code / verifiable | `1 − C(n−c,k)/C(n,k)`, macro-avg | report n, k, temp; sandbox required |
| Bradley-Terry / Elo | judge/human pairwise | open-ended ranking | logistic MLE + bootstrap CI | quote CI; overlaps = tie |
| Faithfulness | judge | RAG generation | supported claims / total claims | LLM decomposition step |
| Answer Relevancy | judge | RAG generation | question-regeneration similarity | penalizes padding |
| Context Precision | judge | RAG retrieval | ranking of relevant chunks | diagnoses ranking |
| Context Recall | judge | RAG retrieval | GT claims covered / total | diagnoses missing evidence |
| Task Completion | judge/verifiable | agents | goal achieved (bool/rubric) | pair with trajectory |
| Tool-Call F1 | exact | agents | precision/recall over calls | needs expected-call set |
| Rubric (LLM-judge) | judge | open-ended | explain-then-score, calibrated | measure judge reliability (α≈0.8) |

---

## 13. Primary sources to cite in your docs

- Chen et al. 2021, *Evaluating Large Language Models Trained on Code* (HumanEval, `pass@k`
  unbiased estimator) — arXiv:2107.03374.
- Hendrycks et al. 2020, *Measuring Massive Multitask Language Understanding* (MMLU) —
  arXiv:2009.03300.
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* —
  arXiv:2306.05685.
- Chiang et al. 2024, *Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference* (Bradley-Terry + bootstrap CIs) — arXiv:2403.04132.
- Liu et al. 2023, *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*
  (explain-then-score) — arXiv:2303.16634.
- Verga et al. 2024, *Replacing Judges with Juries* — arXiv:2404.18796.
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.
- Li et al. 2025/26, *Preference Leakage: A Contamination Problem in LLM-as-a-judge*
  (ICLR 2026) — arXiv:2502.01534.
- SWE-bench, BFCL, tau-bench — active-frontier agentic/code benchmarks.

> Tooling worth reading for reference implementations: RAGAS, DeepEval, and the LMSYS
> Bradley-Terry notebook (and the `choix` library for BT MLE fitting).
