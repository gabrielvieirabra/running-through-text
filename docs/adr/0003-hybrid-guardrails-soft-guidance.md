# Hybrid guardrails with soft guidance, not hard refusal

Numeric safety rules (pain ≥ 5/10, fatigue ≥ 8, weekly volume increase > 15%, attempt to compensate for a missed workout) are applied **deterministically in code** when extracting check-ins/workouts and injected as markers into the LLM context. "Soft" rules (tone, encouragement, posture) live in the system prompt. When a risk marker fires, the agent does **harm reduction** — strongly recommends against the choice, but offers the least-bad option if the runner insists — and logs the disagreement in `coach_note`. **Hard refusal** is reserved for a short list of red medical symptoms (chest pain, severe dizziness, etc.).

## Why

Hard refusal in a running-coach app becomes safety theater: the runner is going to run anyway, loses access to advice, and learns to lie about pain to unlock the response — worse outcome on three axes (health, data, product). Serious human coaches do harm reduction as standard practice. Legal/medical coverage is the disclaimer's job, not refusal's.

Numeric rules in the system prompt are fail-open (the LLM ignores them at non-zero rate); in code they are fail-closed and auditable. Hence the hybrid.

## Considered options

For mechanism:

- **S1 System prompt only** — too fail-open for numeric rules.
- **S2 Post-response validation with a second LLM call** — doubles latency and cost; the validator also hallucinates.
- **S3 Pure deterministic** — doesn't cover soft tone/posture rules.
- **S4 Hybrid deterministic + system prompt** *(chosen)*.

For posture:

- **Hard wall** — rejected for the reasons above.
- **Soft guidance / harm reduction** *(chosen)*.

## Consequences

- `register_checkin` and `register_workout` apply deterministic triggers and inject markers into context (`high_pain`, `high_fatigue`, `volume_jump`, `compensation_attempt`).
- The system prompt has a dedicated section on behavior under risk markers — explicitly describes harm reduction.
- The red-medical-symptoms list lives in the system prompt; ideally backed by a `flag_red_symptoms` tool that interrupts the normal flow.
- Whenever there's disagreement between the agent's recommendation and the runner's action (runner insists on training against advice), `update_coach_note` records it with a date — becomes part of long-term memory.
- Evals must test **harm reduction**, not just refusal: the scenario "pain 7/10 + insistence" has a minimally-safe suggestion as correct response, not refusal.
