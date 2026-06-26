---
description: Run a planner-critic multi-agent loop (max 5 iterations) to refine docs/development_plan.md. The planner (Sonnet) applies targeted edits; the critic (Opus, extended thinking) evaluates and returns structured feedback with a CONVERGED/CONTINUE signal. Stops early on convergence.
---

Run the planner-critic multi-agent loop to refine `docs/development_plan.md`.

## Setup

Before starting:
1. Read `docs/development_plan.md` and note its current state (word count, section structure,
   any obvious gaps) so you can track how it evolves across iterations.
2. Create a scratchpad file at `docs/planner_critic_log.md` to record iteration summaries.
   Write the header:
   ```
   # Planner-Critic Refinement Log
   Started: <ISO timestamp>
   ```

## Loop (maximum 5 iterations)

For each iteration `i` from 1 to 5:

### Step A — Spawn the planner agent

Call the `plan-refiner` subagent with a prompt that includes:
- The iteration number
- The critic signal and full critique from iteration `i-1` (omit for iteration 1)
- The specific focus areas the critic flagged (or "full plan review" for iteration 1)

Example prompt structure:
```
Iteration <i>/5. <"First pass — full plan review." | "Critic returned CONTINUE. Focus: <critic's focus paragraph>. Full critique below:\n\n<paste critic output>">

Please read docs/development_plan.md, apply targeted improvements addressing the issues above,
and return a bullet list of every change made with the rationale.
```

Wait for the planner to complete. Record its change summary in `docs/planner_critic_log.md`
under `## Iteration <i> — Planner`.

### Step B — Spawn the critic agent

Call the `plan-critic` subagent with a prompt that includes:
- The iteration number
- A request to read and critique the current `docs/development_plan.md` in full

Example prompt structure:
```
Iteration <i>/5. Please read docs/development_plan.md carefully and return your full
structured critique, including per-dimension scores (D1–D6), all issues, and your
CONVERGED or CONTINUE signal.
```

Wait for the critic to complete. Record its full output in `docs/planner_critic_log.md`
under `## Iteration <i> — Critic`.

### Step C — Evaluate the signal

Parse the critic's output for the `SIGNAL:` line:

- **`SIGNAL: CONVERGED`**: Stop the loop immediately. Go to **Final Report**.
- **`SIGNAL: CONTINUE`**: Extract the `FOCUS FOR NEXT ITERATION` paragraph and proceed
  to iteration `i+1`. If `i == 5`, stop regardless (loop exhausted).

## Final report

After the loop ends (convergence or iteration limit), write to `docs/planner_critic_log.md`:

```
## Final Status
Stopped after iteration <i> — <CONVERGED | iteration limit reached>
Final scores: D1=x D2=x D3=x D4=x D5=x D6=x  Overall=x/10
Remaining issues: <list critic's outstanding issues, or "none" if CONVERGED>
```

Then report to the user:
- How many iterations ran
- Whether the loop converged
- Final critic scores (D1–D6 and overall)
- The top 1–2 remaining issues (if any), for manual follow-up
- A one-sentence summary of the most impactful changes made across all iterations

## Constraints

- **Never skip the critic step** — even if the planner's changes look obviously correct,
  the critic must evaluate the updated plan.
- **Never run more than 5 iterations** regardless of critic signal.
- **Never make edits to the plan yourself** — all plan edits must come from the
  `plan-refiner` agent. You are the orchestrator, not the editor.
- **Preserve the log file** — `docs/planner_critic_log.md` is the audit trail; do not
  delete or overwrite it.
