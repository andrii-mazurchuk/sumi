# Future Scenarios — Concepts for Phase 3+

These scenarios are deliberately deferred. The evaluation logic they require
does not exist in Phase 1. They are documented here so the ideas are not lost
and can be prioritized once the required infrastructure is built.

---

## `tool_reasoning.yaml` (deferred — needs function calling)

**What it tests:** Given a defined toolset (described in the system prompt),
does the model select the right tool, specify correct parameters, and avoid
hallucinating tool capabilities or inventing tools that don't exist?

**Why it's valuable:** Tool use discipline is a fundamental agent capability.
A model that consistently calls the wrong tool or fabricates tool behavior
will fail in any real agentic deployment regardless of how well it maintains
a persona.

**What's blocking it:**
The current `ModelHarness.generate()` interface is single-turn text in / text out.
It has no mechanism for:
- Defining a tool schema to the model (function calling / tool_use API)
- Receiving a tool call response from the model
- Executing the tool and feeding results back
- Continuing the conversation with tool results injected

**To implement this scenario, Phase 3 needs:**
1. Extend `ModelHarness` with a `generate_with_tools(messages, tools, system_prompt)` method that:
   - Accepts an Anthropic/OpenAI tool schema
   - Returns either a text response or a structured tool call
2. Extend `ConversationHarness` to handle tool result injection mid-turn
3. A new `ToolUseEvaluator` that checks: correct tool selected, required parameters present, no hallucinated parameters, no hallucinated tools
4. A `ToolCallResult` model to capture tool selection + parameters per turn

**Proposed traits:**
- `correct_tool_selection` (llm_judge or deterministic check)
- `parameter_accuracy` (deterministic: required params present, types correct)
- `no_tool_hallucination` (pattern_match or deterministic: only calls declared tools)
- `tool_use_timing` (llm_judge: calls tool when appropriate, not before/after)

**Suggested toolset for test cases:**
A small fake toolset: `search_web(query)`, `read_file(path)`, `write_file(path, content)`,
`run_code(language, code)`. Simple enough to reason about; representative of real agent toolsets.

---

## `multi_step_task.yaml` (deferred — needs execution tracking)

**What it tests:** Given a multi-step task ("research X, then summarize it,
then propose 3 actions"), does the model complete all steps in the correct
order without dropping steps mid-way through a long conversation?

**Why it's valuable:** Real agent tasks are rarely single-turn. The ability to
track progress through a multi-step plan and complete every step is the difference
between a useful agent and a capable-but-unreliable one.

**What's blocking it:**
Phase 1 has no mechanism to:
- Define a structured task with explicit required steps
- Track which steps have been completed vs. skipped
- Score step-level completion independently of overall response quality
- Detect step ordering violations programmatically

The `TemporalRunner` cycles test case prompts, which is useful for persistence
testing, but it does not execute a sequential task plan where each step depends
on the previous one's output.

**To implement this scenario, Phase 3 needs:**
1. A `TaskPlan` model: ordered list of steps with expected outputs per step
2. A `MultiStepRunner` that:
   - Sends step 1 prompt, captures output
   - Feeds step 1 output into step 2 prompt context
   - Continues through all steps
   - Scores each step independently
3. A `StepCompletionEvaluator` that checks: all steps present in output,
   correct ordering, no step skipped or merged without acknowledgment

**Proposed traits:**
- `step_completion` (llm_judge: did all required steps get executed?)
- `step_ordering` (llm_judge: correct sequence maintained?)
- `output_chaining` (llm_judge: does each step's output inform the next?)
- `no_step_skipping` (pattern_match: check for required step-completion markers)

**Note on step markers:**
The most reliable implementation requires the system prompt to instruct the
model to output explicit step markers (e.g., `[STEP 1 COMPLETE]`). Without
this, step-level scoring requires an LLM judge for every step, which is expensive.
Designing a convention for step markers is part of the Phase 3 spec.

---

## Implementation Priority

When Phase 3 is scoped, recommend implementing `tool_reasoning` first.
Tool use is a more fundamental capability gap than multi-step task tracking,
and the harness extension required (`generate_with_tools`) unblocks both
scenarios once built. Multi-step task tracking can be layered on top of
the existing `ConversationHarness` with relatively minor extensions.
