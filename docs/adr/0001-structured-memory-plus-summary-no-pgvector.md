# Structured memory + narrative summary, no pgvector

The agent loads, per turn: (a) the runner's profile, (b) recent check-ins and workouts via SQL, (c) a "coach note" — narrative text that the agent itself updates periodically. No RAG, no pgvector.

## Why

A runner's history is small, linear, and temporal — there is no heterogeneous corpus to "search semantically over". Facts (current 5k, last pain episode, weekly km) belong in structured SQL, which is exact and cheap. Inter-session narrative (training phase, patterns, the runner's fears) belongs in the coach note, which preserves coherence without per-turn cost growth.

## Considered options

- **M1 Pure long context**: simple, but cost grows with the runner's history and debugging becomes painful at 1M tokens.
- **M2 Structured + recent window**: cheap, but loses long-term patterns — the agent forgets what it said 30 days ago.
- **M3 Structured + narrative summary** *(chosen)*: covers long-term coherence without growing cost; the note is auditable.
- **M4 RAG over messages with pgvector**: introduces an operational dependency, retrieval is a black box, and the semantic-search use case over a runner's history is weak.

## Consequences

- **pgvector drops out of the stack** proposed in the original PLAN.md. Plain PostgreSQL is enough.
- **New entity**: `coach_notes` — narrative text, timestamp-versioned, updated by the agent.
- Still to resolve: update trigger for the note, drift prevention (hallucination accumulating in the note itself), max size.
