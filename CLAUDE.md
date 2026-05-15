# Running Through Text

Conversational virtual coach for road/trail running. Plan and context live in `PLAN.md` and `CONTEXT.md`; architectural decisions in `docs/adr/`.

## Project language

All project artifacts (docs, ADRs, code, comments, commit messages, issue files) are written in **English**. The only PT-BR content allowed is:
- Domain enum values that are deliberately Brazilian running terms (e.g. `rodagem`, `fartlek`, `longo` — see CONTEXT.md "Workout taxonomy").
- Test fixture content reflecting what a Brazilian runner would actually type.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature-slug>/`. No GitHub Issues in the MVP. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.
