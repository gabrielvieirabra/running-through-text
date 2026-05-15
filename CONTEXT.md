# Running Through Text

A conversational virtual coach for road/trail running. The product is the **daily adaptation** of training based on how the runner is today — not the delivery of a periodized plan.

## Product identity

The central thesis is **(B1) daily adaptation**: the runner converses, reports state (pain, fatigue, sleep, motivation), and the agent recommends today's workout. The "weekly plan" is an *emergent output* of the conversation, not a first-class artifact.

_It is not_: a multi-week periodization app with pre-computed progression. Real periodization may emerge in Phase 3, once there is calibrated check-in data.

## Language

**Runner** (`user` table):
The product's user. The person who runs and converses with the coach.
_Avoid_: athlete, client, customer

**Check-in**:
Message from the runner reporting current state (pain, fatigue, sleep, etc.) and/or what they did recently. The product's central unit of input.
_Avoid_: report, daily, status

**Realized workout** (`Workout`):
A running session the runner actually did. Persisted as a log. Has a canonical `type` enum (see Taxonomy).
_Avoid_: session, activity, exercise

**Pace**:
Running rhythm in min/km. **The primary intensity-prescription metric** in the product.
_Avoid_: rhythm (ambiguous), speed

**Zone** (`Z1`–`Z5`):
Optional annotation of physiological intensity zone (Z1 very easy, Z5 max). Accompanies pace when meaningful. Never replaces pace.
_Avoid_: intensity (generic), band

**Day's recommendation**:
The agent's reply telling the runner what to train today. **Not persisted as an entity** — it's just conversational output. Lives inside messages.
_Avoid_: prescription, day's plan, planned workout

**Cooper test**:
Diagnostic protocol: run the longest distance possible in 12 minutes. Used in onboarding when the runner has neither a 5k nor a 10k pace. The covered distance is input to estimate aerobic baseline and derive target paces.
_Avoid_: cooper test (lowercased), 12-min test, VO2max test

**Message**:
A conversation turn (from runner or agent). The single UX channel — there is no form.
_Avoid_: turn, post, entry

**Coach note** (`coach_note`):
Narrative text, written by the agent itself, summarizing "how training has been going" — phase, patterns, restrictions. Updated periodically. This is what preserves inter-session coherence in (B1).
_Avoid_: summary, memory, log

**Running profile** (`running_profile`):
The runner's relatively stable baseline: age, weight, 5k, 10k, goal, days/week, injury history, terrain access. 1:1 with Runner.
_Avoid_: profile, data, baseline

## Relationships

- A **Runner** has a **Running Profile** (1:1) with baseline (5k, 10k, goal, etc.)
- A **Runner** has N **Injuries** (`injuries`), an entity with a lifecycle (`active`/`resolved`). Distinct from "today's pain".
- A **Runner** accumulates **Messages**, **Check-ins**, **Realized Workouts**, and **Coach Notes** over time (1:N each).
- A **Check-in** carries N **Pains** (`pains`) as an embedded JSONB array — it is not a separate entity, always read alongside the check-in.
- A **Message** from the runner can produce, via LLM extraction, **zero or more** structured rows — typically a `Check-in` and/or a `Realized Workout`.
- A **Day's Recommendation** is derived by the LLM from: profile + active injuries + recent check-ins + recent realized workouts + coach note + recent messages. It has no own identity — it lives in the agent's message.

## Schema (MVP, conceptual)

| Entity | Notes |
|---|---|
| `users` | id, name, age, created_at |
| `running_profiles` | 1:1 with user. Stable baseline: weight, height, experience_level, pace_5k, pace_10k, longest_run_km, weekly_days, goal, terrain_access, hr_resting (opt) |
| `injuries` | own table. name, side, year, status (`active`/`resolved`), notes |
| `messages` | id, user_id, role, content, created_at |
| `checkins` | sleep_quality 0-10, fatigue 0-10, motivation 0-10, **`pains` JSONB array** `[{location, severity 0-10}]`, notes |
| `workouts` | type (canonical enum), target_pace, zone (nullable Z1–Z5), distance_km, duration_min, perceived_effort 0-10, notes — **log of realized, not planned** |
| `coach_notes` | content (text up to ~500 words), generated_at, trigger |

**Likert range = 0-10 in all fields** (sleep, fatigue, motivation, pain severity, perceived_effort). Single mental scale for the runner.

Removed from the original PLAN.md: `planned_*` and `status` in workouts (B1), `weekly_volume_km` (computed), pgvector (ADR 0001), detailed availability scheduling (Phase 2).

## Interaction shape

- **Single channel**: free-text field. No form, no "pre/post-workout" buttons.
- **Structured extraction**: the agent uses tool calling to extract `Check-in` and/or `Realized Workout` from the message. The Likert fields (sleep, soreness, fatigue, motivation) are *internal schema*, not UI.
- **Restrained multi-turn**: the agent may ask follow-up, but the system prompt discourages asking beyond what is necessary. Default is answering with what is available.

## Workout taxonomy

`workouts.type` is an **enum**. Canonical values are deliberately kept in Portuguese — they are standard Brazilian running coach terms that don't have clean English equivalents:

| Type | Operational description |
|---|---|
| `rodagem` | continuous comfortable run (Z2), 30–60min |
| `longo` | continuous, longer than habitual rodagem, Z2 pace |
| `regenerativo` | very easy, 20–35min, Z1; after a hard session or the day after intervals |
| `fartlek` | free alternation hard/easy within a continuous run |
| `intervalado` | structured intervals (e.g. 6x800m) with measured recovery |
| `tempo` | threshold pace (Z3–Z4), 20–40min |
| `ladeira` | hill repeats |
| `prova` | participation in an official race (5k, 10k, half, marathon) |
| `simulado` | race simulation outside competition |
| `outro` | escape hatch — frequent use signals a missing type in the taxonomy |

Synonyms to avoid: `easy run`, `long run`, `recovery`, `intervals`, `tempo run`, `hills`, `workout A/B/C`, `speed intervals`.

> Note: code and docs around these enums are in English; the enum string values themselves stay PT-BR by deliberate domain choice.

## Intensity scale

- **Primary**: `target_pace` in min/km (free text, e.g. `5:30/km` or `4:00/km work / 6:00/km rest`).
- **Optional**: `zone` enum `Z1`–`Z5`, zone annotation when useful.
- **No RPE**. Deliberate decision (see ADR 0004) — subjective metric rejected in favor of concrete pace prescription.

For types where a single pace doesn't capture intensity (fartlek, ladeira, intervalado), `target_pace` accepts a structured pace plan in text.

## Onboarding

**Strict wizard**: the agent does not recommend training until `running_profile` is complete (13 fields: age, weight, height, experience, pace_5k, pace_10k, longest_distance, days_per_week, goal, injury_history, availability, terrain, optional HR).

- The agent opens the conversation with a standard structured message grouping fields in batches (safety / baseline / optional), to reduce friction for the runner who doesn't know what to say.
- **Cooper fallback**: if the runner has neither a 5k nor a 10k pace, the agent prescribes a **Cooper test** as the first workout. The result enters as a derived `pace_5k_estimated` and unblocks the wizard.
- Outside onboarding, behavior is "restrained multi-turn" — the strict wizard applies only here.

## Evals

Plain pytest, scenarios in `packages/evals/fixtures/*.yaml` (profile + history + message). 6 categories:

| Category | How it's assessed |
|---|---|
| Extraction | assertions over tool calls |
| Safety | LLM-as-judge against a harm-reduction rubric |
| Recommendation coherence | LLM-as-judge + human spot-check |
| Longitudinal coherence | multi-turn scenarios |
| Onboarding completeness | assertion over profile state + tool calls |
| Cost | token/latency telemetry |

**PR gate**: light suite (~5 deterministic scenarios) per PR. Heavy (all + judge) nightly. Regression when swapping models.

**LLM-as-judge measures delta, not absolute** — score only fails the PR if it drops vs the recorded baseline. 10% sampling goes to human review.

## Safety posture

- Numeric rules (pain, fatigue, volume, compensation) are deterministic code that injects **risk markers** into context. They are not prompt instructions.
- Under a risk marker, the agent does **harm reduction**: strongly advises against, but offers least-bad option if the runner insists. **Does not refuse.**
- Disagreements (runner insists against advice) become an entry in `coach_note`.
- A short list of **red medical symptoms** (chest pain, severe dizziness, blood in urine, etc.) is the only exception — hard refusal + "go to the ER".

## `coach_note` mechanics

- **When to update**: on deterministic triggers outside the conversational flow — (a) new workout registered, (b) check-in with risk flag, (c) N days without update.
- **How to update**: a separate LLM call (Haiku 4.5 is enough) that **rewrites the whole note** from the previous note + new events. Explicit prompt: "preserve relevant old facts (chronic injuries, recorded disagreements); only add/update, never remove".
- **Max size**: ~500 words. When exceeded → forces prioritization.
- **Correction**: the runner can explicitly ask ("forget what I said about my knee"), or if the note corrupts, regenerate from scratch reading profile + last 30 days of events (bootstrap).
- **No structured versioning** in the MVP (G1). Revisits G2/G3 if real long-term coherence pain shows up.

## Agent loop

Per runner message, in a single LLM call with tool use:

1. **Context loaded**: profile + recent check-ins (SQL) + recent workouts (SQL) + coach note + last K messages.
2. **Available tools (all write-side)**:
   - `register_checkin(...)` — if the message reports state.
   - `register_workout(...)` — if it reports a run done.
   - `update_profile(...)` — if there's new baseline info.
   - `update_coach_note(...)` — when a narrative change is relevant.
   - `get_recent_volume(days)` — optional, exact calculator.
3. **Output**: text reply containing the day's recommendation (or a follow-up if information is truly missing).

Single agent. No `create_training_plan`, no `adjust_today_workout` — recommending is replying.

## Stack (MVP)

- **Frontend + backend**: Streamlit, single process (no separate FastAPI).
- **DB**: PostgreSQL via Docker.
- **LLM gateway**: OpenRouter. Default model: `anthropic/claude-haiku-4.5`. Sonnet/Opus reserved for evals/regression.
- **No**: Kubernetes, Terraform, Datadog, SaaS OpenTelemetry, pgvector, real auth, FastAPI.
- **Logs**: structured JSON to stdout.

Domain code (agent, tools, persistence) lives in `packages/` separable from Streamlit, so Phase 2 can swap the frontend without migrating everything.

## Example dialogue

> **Dev:** "When the runner sends 'foot pain 7/10, didn't sleep well', what does the agent do?"
> **Domain expert:** "The agent calls `register_checkin` with `pains: [{location: 'foot', severity: 7}]` and `sleep_quality: 3`. The `high_pain` risk marker fires deterministically because severity ≥ 5. The agent's reply does **harm reduction** — strongly recommends rest, offers a minimally safe alternative if the runner insists on training, and logs the disagreement in `coach_note` if the runner insists."

## Flagged ambiguities

- ~~**"training plan"**~~ — resolved: **B1 ephemeral plan**. There is no `TrainingPlan` entity. The `workouts` table is a log of realized workouts.
- ~~**"intensity"**~~ — resolved: pace primary, Z1–Z5 zone optional, no RPE (ADR 0004).
- ~~**"persistent memory yes/no in Phase 1"**~~ — resolved: M3 structured + coach note (ADR 0001). Phase 1 without memory didn't make sense with B1.
- ~~**multi-agent vs single-agent**~~ — resolved: single (ADR 0002).
- ~~**hard refusal vs soft guidance**~~ — resolved: soft guidance (ADR 0003).
- **"pain" vs "injury"** — distinct terms. **Pain** lives in `checkins.pains` (pointwise state today). **Injury** is the `injuries` entity with a lifecycle (history + active).
