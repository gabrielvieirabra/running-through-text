# Scenario-based evals, pytest + delta-measuring LLM judge

The eval suite is **plain pytest** with YAML scenario fixtures (profile + history + message). Covers 6 categories: extraction, safety, recommendation coherence, longitudinal coherence, onboarding completeness, cost. **LLM-as-judge** powers qualitative assertions, but only fails the PR gate when the score *drops* against the recorded baseline — absolute score does not pass/fail.

## Why

Product scenarios are few in the MVP (10–30) and heavily domain-specific. Dedicated tools (promptfoo, braintrust) cost plumbing/lock-in for benefits that only kick in at scale (hundreds of scenarios). Pytest already exists for normal code, integrates with CI without a new runtime, and LLM-as-judge is a short Python function. Absolute LLM-judge score is noisy and not reliable enough for gating; delta is stable enough.

## Considered options

- **Promptfoo** — good but over-spec here; migrate later if scenario count explodes.
- **Braintrust / LangSmith** — SaaS, lock-in, recurring cost.
- **OpenAI Evals / inspect_ai** — oriented toward rigorous benchmarks, not product scenarios.
- **Pytest + bespoke harness** *(chosen)*.

## Consequences

- `packages/evals/` with `fixtures/`, `scenarios/`, `judges/llm_judge.py`.
- **Light** suite (~5 deterministic scenarios) runs per PR in ~30s; **heavy** suite (all + judge) runs nightly; **regression** suite when swapping models.
- 10% of judged responses go to periodic human review — judge calibration.
- When scenarios pass 100 or maintaining the harness becomes a burden, revisit with promptfoo.
