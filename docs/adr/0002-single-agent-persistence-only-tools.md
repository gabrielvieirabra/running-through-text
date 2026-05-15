# Single agent, persistence-only tools, recommendation is free text

A single LLM agent handles every runner message. The exposed tools are exclusively **write-side** (`register_checkin`, `register_workout`, `update_profile`, `update_coach_note`) and, optionally, deterministic calculators for numeric facts (`get_recent_volume`). **The day's recommendation is the LLM's text response** — not the output of a "training generation" tool.

## Why

The (B1) thesis — ephemeral plan, daily adaptation — has no "plan" entity for tools to mutate. Tools like `create_training_plan` or `adjust_today_workout` from the original PLAN.md imply a (A) periodized model that was explicitly rejected. Multi-agent (the original Phase 4: Planner / Safety / Progress Analyst) is the typical over-engineering trap for AI products: it hurts latency, cost, and debuggability without concrete gain while a single agent still does the job.

## Considered options

- **Multi-agent (Planner + Safety + Progress Analyst)** — rejected for now; only justified if a single agent hits a real quality/scope ceiling.
- **Deterministic plan-generation tool** (e.g. `generate_week_skeleton`) — rejected; contradicts B1 by bringing back a structured plan artifact.
- **Explicit runner confirmation of extraction** ("I extracted pain=6, sleep=bad, ok?") — rejected; adds friction. The runner can correct in following messages.

## Consequences

- Phase 4 of the original roadmap (multi-agent) is removed. May return if concrete pain shows up.
- Agent evals measure **quality of the text response** + **correctness of persistence tool calls**, not "quality of a structured plan".
- If pressure ever builds again for a plan-as-artifact, this decision flips together with 0001 and B1 → B2.
