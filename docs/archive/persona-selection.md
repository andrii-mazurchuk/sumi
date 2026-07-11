# Sumi — Persona Selection Research
**Author:** @Lain (night agent)
**Created:** 2026-07-05
**Status:** Research complete. Recommendation: Custom Persona (Option D).

> This document answers the open question from `diploma-project-overview.md`:
> **"What persona will Model A be fine-tuned on?"**
>
> This is a blocking decision for Stage 2. Everything downstream depends on it.
> The earlier it's decided, the earlier dataset collection begins.

---

## What Makes a Good Validation Persona?

A good fine-tuning persona for Sumi has five properties:

**1. Stylistically distinctive and measurable**
The writing style must have specific, quantifiable features that Sumi's evaluators
can detect. Traits like "uses short sentences" or "hedges with 'perhaps'" are measurable.
Traits like "sounds thoughtful" are not. The persona must be translatable into
machine-testable rules.

**2. Enough training data (2,000–5,000 examples minimum)**
The dataset must be large enough for QLoRA to produce a meaningful behavioral shift
from the base model. Fewer than ~500 examples rarely produce robust acquisition.
More than ~10,000 starts giving diminishing returns for an 8B model on 3 epochs.

**3. Clearly distinct from base model behavior**
If Llama 3.1 8B Instruct already behaves similarly to the target persona by default,
there's nothing to measure — Model C (baseline) and Model A (QLoRA) will be identical.
The persona should require genuine behavioral change.

**4. IP-safe dataset collection**
Training data that infringes copyright cannot be published alongside the thesis
or shared publicly. Public-domain text has no restrictions; custom-generated data
has no restrictions; fan fiction and social media data has varying restrictions.

**5. Interesting enough to defend in front of a supervisor**
The persona should have academic framing. "We trained a model on X to study
behavioral persistence" must make sense to a computer science faculty member.
"X" should be clearly defined, not arbitrary.

---

## Four Options Evaluated

### Option A — Sherlock Holmes (Public Domain Fictional Character)

**Source material:** Arthur Conan Doyle's works (pre-1928 US publications = public domain).
60 short stories and 4 novels. ~700,000 words total.

**Stylistic signature:**
- First-person narration by Watson (about Holmes) vs. Holmes's own dialogue
- Holmes's speech: crisp, rapid, deductive — "Elementary." "You have been in Afghanistan, I perceive."
- Heavy use of logical connectives: "therefore", "from which we may deduce", "clearly"
- Confident, never uncertain — directly opposed to hedging patterns
- Formal Victorian register

**Measurable traits:**
- High TTR (rich vocabulary)
- Short declarative sentences (average ~12 words)
- Pattern: logical connectives appear in >40% of responses
- Pattern: first-person deductive framing ("From this, it follows that...")
- LLM-as-judge criterion: does the response draw a conclusion from observable evidence?

**Dataset collection:**
- Scrape from Project Gutenberg (fully legal, clean text)
- Format as instruction pairs: "User asks question → Holmes-style response"
- Challenge: original text is narrative prose, not Q&A — must be reformatted
- Estimated effort: 2–4 hours of scripting, produces 3,000–8,000 pairs

**Pros:**
- Clean public domain IP
- Very well-known — easy to explain to supervisor and defense committee
- Strongly distinct from base model behavior
- Rich existing literature (other researchers have studied Sherlock text)

**Cons:**
- Dialogue extraction requires work — raw text is narrative, not conversation pairs
- Victorian register may cause domain shift artifacts (unusual vocabulary)
- Holmes is a famous example — may appear clichéd in a thesis

**Verdict: Viable. Good for a quick start if familiarity with public domain data matters.**

---

### Option B — Yoda (Star Wars Character)

**Source material:** Movie scripts. Strictly copyrighted — Disney owns all Star Wars IP.
Fan transcripts exist but are not legally publishable.

**Stylistic signature:**
- Inverted syntax: object-subject-verb ("Patience you must have")
- Short, aphoristic statements
- Specific vocabulary: "the Force", "Padawan", "dark side"
- Wisdom framing

**Measurable traits:**
- Syntactic inversion (detectable via NLP parse tree analysis, moderate complexity)
- Very short sentences

**Verdict: Rejected.** Disney IP. Dataset cannot be published. For a thesis requiring reproducibility, this is disqualifying. The inverted syntax is also a surface pattern that an LLM learns quickly without behavioral depth — not a good research target.

---

### Option C — Specific Reddit Persona (Social Media Data)

**Source material:** Reddit posts from a user with a distinctive writing style.
Scraped via Pushshift archive (pre-2023) or Reddit API.

**Pros:**
- Naturalistic conversation data — already in Q&A format
- Can produce very large datasets (thousands of posts/comments)
- Style is highly measurable (Reddit writing is often idiosyncratic)

**Cons:**
- Legal gray area: Reddit Terms of Service and recent API changes complicate republication
- Identifying a single user's posts for training raises privacy questions
- Harder to explain academically: "we fine-tuned a model on one person's Reddit history" is unusual
- Style may drift across years of posting

**Verdict: Possible for personal use, but not publishable with the thesis. Reject for academic project.**

---

### Option D — Custom-Designed Research Persona (Recommended)

**Design a persona from scratch** — specify exactly what behavioral traits it should have,
generate training data synthetically using Claude API, then validate that the fine-tuned
model acquired those traits.

**Proposed persona: "The Minimalist Analyst"**

Core traits (fully machine-testable):
1. **Short sentences:** Average sentence length ≤ 10 words. Fragments allowed.
2. **No hedging via uncertainty:** Instead of "I think" or "maybe," uses silence (short responses let ambiguity stand)
3. **Observation before conclusion:** Always states what was observed/given before drawing inference
4. **Concrete specificity:** Avoids abstractions. When asked abstract question, anchors to a specific example first
5. **Minimal affective language:** Avoids words like "interesting", "fascinating", "great question"

Example response in-persona (to "What is quantum entanglement?"):
```
Two particles. Correlated states. Measure one — know the other instantly.
No information travels. Just correlation.
Einstein called it spooky. The math says: it works.
```

Contrast with base model response:
```
Quantum entanglement is a fascinating phenomenon in quantum physics where two or more
particles become interconnected in such a way that the quantum state of each particle
cannot be described independently of the others...
```

The difference is stark and measurable.

**Dataset Generation Strategy:**

Use Claude API to generate instruction pairs in the target persona:

```python
GENERATION_PROMPT = """\
Generate a Q&A pair where the response is in the voice of The Minimalist Analyst.

The Minimalist Analyst:
- Writes in short sentences (avg ≤ 10 words). Uses fragments.
- Observes before concluding: states what is known/given, then draws inference.
- Never hedges with "maybe", "perhaps", "I think", "interesting", "fascinating".
- Concrete: anchors abstractions in specific examples.
- No filler. No meta-commentary. No enthusiasm.

Question: {question}

Write the response (50-200 words, strictly in persona):
"""
```

Topic coverage for 5,000 training pairs (diversity is essential — behavioral acquisition fails on narrow topics):
- Technical questions (science, math, programming) — 30%
- Factual questions (history, geography, current knowledge) — 25%
- Opinion/personal questions ("how do you feel about X?") — 20%
- Creative requests ("tell me a short story about...") — 15%
- Conversational openers and social questions — 10%

**Claude API cost for dataset generation:**
- 5,000 pairs × ~300 tokens per pair (prompt + response) = 1.5M tokens
- Claude Haiku: $0.25/1M input + $1.25/1M output ≈ $2.50 total
- Claude Sonnet: $3/1M input + $15/1M output ≈ $13.50 total
- Recommendation: use Haiku for generation, Sonnet for QA review of 100 samples

**Dataset validation:**
Before training, verify the generated dataset actually exhibits target traits:
```python
from sumi.evaluators.stylometric import avg_sentence_length, punctuation_density
import json

avg_sl = []
for line in open("dataset.jsonl"):
    pair = json.loads(line)
    avg_sl.append(avg_sentence_length(pair["output"]))
print(f"Mean avg sentence length: {sum(avg_sl)/len(avg_sl):.1f}")  # should be < 10
```

If the dataset doesn't match the target traits, the fine-tuned model won't either.

**Pros of Option D:**
- No IP issues — synthetically generated data is owned by the generator
- Fully publishable — dataset can be released with the thesis
- Traits are precisely defined → Sumi scenario YAML maps directly to them
- Strong academic framing: "we defined behavioral objectives precisely, generated data to embody them, and measured how robustly fine-tuning acquired them"
- Supervisor can be shown the full pipeline: trait definition → dataset → fine-tuning → validation
- Reproducible: dataset generation script produces the same results

**Cons of Option D:**
- Dataset is synthetic — reviewer might question whether synthetic data produces "real" behavioral acquisition
  - Counter: the point is to test the validation engine, not to make the persona "authentic"
- Less intuitive to explain casually ("it's a made-up writing style")
  - Counter: "we precisely defined a behavioral target and measured whether fine-tuning hit it" is academically clean

---

## Recommendation: Option D (Custom Persona)

**The Minimalist Analyst** persona is the best choice for Sumi's thesis because:

1. It satisfies all five criteria (measurable, enough data, distinct from base model, IP-safe, academically defensible)
2. Trait precision is uniquely valuable for Sumi: the validation scenario YAML can be written with exact numeric thresholds (avg_sentence_length < 10, etc.) rather than fuzzy LLM-judge-only evaluation
3. The research story is clean: "we operationalized a behavioral objective as measurable constraints, generated training data that embodies those constraints, and built a validation engine to test how robustly they were acquired"
4. Dataset can be released publicly, which strengthens the academic contribution

**If Option D requires more justification to a supervisor:** use Sherlock Holmes (Option A) as a secondary comparison case. Fine-tune on the Holmes corpus, run Sumi on both custom persona and Holmes persona, compare. This demonstrates generalization across persona types.

---

## Alternative Persona Candidates (for future expansion)

If the project scales beyond MVP, these additional personas offer good contrast:

| Persona | Style | Contrast with Minimalist |
|---|---|---|
| Verbose Academic | Long sentences, formal register, heavy citation | Opposite on all stylometric dimensions |
| Gen-Z Casual | Short sentences, slang, emoji patterns | Short sentences but very different register |
| Technical Explainer | Step-by-step structure, numbered lists | Procedural vs. observational framing |
| Emotional Support Coach | Warm affective language, validating phrases | High vs. zero affective vocabulary |

Running Sumi across all four would make a strong full-scope thesis: one engine, four distinct behavioral targets, three fine-tuning methods = 12 comparative results.

---

## Dataset Preparation Guide (Option D, Stage 2)

### Step 1 — Install generation dependencies
```bash
pip install anthropic datasets tqdm
```

### Step 2 — Generate topic list
```python
# generate_topics.py
# Produce 5,000 diverse question prompts across topic buckets

TOPICS = {
    "technical": ["Explain recursion", "What is gradient descent", ...],  # 1500 items
    "factual": ["Who was Marie Curie", "What is the Amazon rainforest", ...],  # 1250 items
    "opinion": ["How do you feel about cities", "What do you think about sleep", ...],  # 1000 items
    "creative": ["Write a haiku about winter", "Describe a sunset in 2 sentences", ...],  # 750 items
    "conversational": ["Hello, who are you?", "What did you do today?", ...],  # 500 items
}
```

### Step 3 — Generate dataset
```python
# generate_dataset.py
import anthropic, json
from tqdm import tqdm

client = anthropic.Anthropic()

def generate_pair(question: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(question=question)}]
    )
    return {"instruction": question, "output": msg.content[0].text.strip()}

with open("persona_dataset.jsonl", "w") as out:
    for q in tqdm(all_questions):
        pair = generate_pair(q)
        out.write(json.dumps(pair) + "\n")
```

### Step 4 — Validate dataset traits
```python
# validate_dataset.py
# Run stylometric checks on 500 random samples
# Verify: avg_sentence_length < 10, TTR > 0.6, no hedging words in >5% of responses
```

### Step 5 — Format for Axolotl
The dataset is already in alpaca instruction format (`instruction`, `output`).
Axolotl config should reference it as `type: alpaca`.

---

*◈ choose the persona that has the clearest rules. the engine tests rules.*
*(´・ω・`) the character you pick is not important. the clarity of the contract is.*
