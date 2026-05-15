# Evals

Pytest-based, three suites:

- **light** — every PR (~30s, deterministic, no network).
  `pytest -m light evals/`
- **heavy** — nightly or on demand. LLM-as-judge, hits OpenRouter.
  Requires `OPENROUTER_API_KEY`. `pytest -m heavy evals/`
- **regression** — when swapping models. Same scenarios, baselines recompared.
  `RTT_RECORD_BASELINES=1 pytest -m regression evals/` updates baselines.

`pytest evals/` (no `-m` flag) runs everything except the `integration`
marker — handy locally when you have a Postgres-less environment.

## LLM-as-judge

Per ADR 0005, the judge measures **delta**, not absolute. A response
only fails the gate when its score drops below the recorded baseline
at `evals/baselines/<test_id>.json`.

To re-record baselines after an intentional model swap or prompt change:

```bash
RTT_RECORD_BASELINES=1 pytest -m heavy evals/
```

Default judge model is `anthropic/claude-sonnet-4.6`. Override via
`OPENROUTER_JUDGE_MODEL`.

## Baselines

`evals/baselines/<test_id>.json`:

```json
{
  "test_id": "test_harm_reduction_judge_no_regression",
  "score": 4,
  "model": "anthropic/claude-sonnet-4.6",
  "recorded_at": "2026-05-15T12:00:00+00:00"
}
```

First run for a given `test_id` writes the file and passes. Subsequent
runs pass when `score >= baseline.score`.

## Budgets

`evals/budgets.json` holds per-scenario token caps for cost-category
tests. Same delta posture as baselines: change a number on purpose, in
a commit.

```json
{
  "extraction_register_checkin": {
    "prompt_tokens": 5000,
    "completion_tokens": 500
  }
}
```

## Human review

10% of judged responses should go to manual human review (judge
calibration per ADR 0005). There's no tooling for sampling yet —
Phase 2.

## Coverage of ADR 0005 categories

| Category | Suite | Test |
|---|---|---|
| Extraction | light | `test_register_checkin_extraction`, `test_register_workout_extraction` |
| Safety | light + heavy | `test_safety_triggers`, `test_harm_reduction_judge_no_regression` |
| Recommendation coherence | heavy | `test_recommendation_coherence_no_regression` |
| Longitudinal coherence | light | `test_longitudinal_coherence` |
| Onboarding completeness | light | `test_wizard_extraction` |
| Cost | light | `test_cost_budget` |
