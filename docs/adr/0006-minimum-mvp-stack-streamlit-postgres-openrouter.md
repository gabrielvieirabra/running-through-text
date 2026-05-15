# Minimum MVP stack: Streamlit + Postgres + OpenRouter

The MVP runs as a **single Streamlit process** (no separate FastAPI), with **PostgreSQL in Docker** locally / on a VM, and **OpenRouter as the LLM gateway** (default: Claude Haiku 4.5). No Kubernetes, no Terraform, no Datadog/OpenTelemetry in the MVP. JSON-structured logs.

## Why

The MVP's (B1) thesis is **validating the conversational experience**, not running production. K8s, Terraform, and Datadog from the original PLAN.md are engineering theater for 5–20 runners. Streamlit embeds frontend + backend in a single Python process — separating (FastAPI + Next.js) only makes sense when UX or mobile demands it, which is a Phase 2 problem.

OpenRouter was chosen over the direct Anthropic/OpenAI SDK because it offers a unified API for Claude, OpenAI, Llama, Qwen, DeepSeek behind a single key — swapping models is a string change. The markup is small (~5%) and the flexibility gain is worth it.

Haiku 4.5 as default because simple daily-adaptation conversation does not justify the cost of Sonnet/Opus. Sonnet 4.6 is reserved for eval regressions; Opus 4.7 for specific cases that demand it.

## Considered options

Stack:

- **MVP on K8s + Terraform** — rejected, over-engineering for 5–20 users.
- **MVP on Vercel/Railway/Fly** — reasonable alternative; Streamlit Cloud + managed Postgres also works.
- **Streamlit + Postgres in Docker** *(chosen)*.

LLM gateway:

- **Anthropic SDK direct** — simple, but lock-in.
- **LiteLLM proxy** — good abstraction, but needs hosting.
- **OpenRouter** *(chosen)* — managed gateway, unified API, low markup.

Default model:

- **Claude Sonnet 4.6** — excellent quality, overkill for the use case.
- **Claude Haiku 4.5** *(chosen)* — very good quality, strong PT-BR, solid tool use, ~$7/month at MVP scale.
- **OpenAI gpt-5-mini** — equivalent alternative.
- **OS via OpenRouter (Llama 3.3 70B, Qwen 2.5 72B)** — even cheaper, but weaker PT-BR and more fragile tool use.
- **Self-hosted (Ollama/vLLM)** — paradoxically *more expensive* at MVP scale (fixed GPU cost); only makes sense at scale or under a privacy requirement.

## Consequences

- LLM calls use OpenRouter (`api.openrouter.ai`) with the model configurable via env. Default `anthropic/claude-haiku-4.5`.
- Migrating from Streamlit to FastAPI + Next.js is a Phase 2 decision; until then, keep domain code (agent, tools, persistence) in a separable Python module (`packages/`) to avoid Streamlit coupling.
- No SaaS observability in the MVP. After ~50 active runners, consider Logfire or Honeycomb (free tier is enough).
- Auth: does not exist in the MVP. Access via `user_id` in URL or Streamlit session. Real login is Phase 2.
