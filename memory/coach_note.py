"""Coach-note rewrite path (Slice 6, ADR 0001 M3).

A separate LLM call dedicated to rewriting the runner's narrative coach
note. Uses the same OpenRouter client and default model as the main
agent — same identity conceptually (ADR 0002, single agent) — but a
distinct system prompt focused on note-keeping.

Triggers (from CONTEXT.md "coach_note mechanics"):
    new_workout : after a `register_workout` write.
    risk_flag   : after a `register_checkin` if any soft safety marker
                  fires on the just-saved state.
    idle        : >= 7 days since the last note while the runner is active.
    manual      : the agent explicitly called `update_coach_note`.
    bootstrap   : forget previous note and regenerate from scratch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any

from openai import OpenAI

from persistence import db as _default_db

logger = logging.getLogger(__name__)

# Soft warning threshold above the 500-word target. The LLM is told to
# self-prioritize at ~500; if it returns much more we don't truncate (that
# would be lying — see CONTEXT.md), but we log a warning so a recurring
# overshoot is observable.
_WORD_COUNT_WARN_THRESHOLD = 600


NOTE_REWRITE_SYSTEM_PROMPT = """You maintain a coach's narrative note about a single runner.

The note summarizes "how training has been going" — the runner's current phase,
patterns, restrictions, chronic injuries, and any disagreements registered
between coach recommendations and runner actions.

Given:
- the PREVIOUS coach note (may be empty if first time),
- the runner's profile,
- recent check-ins, workouts, and active injuries,
- the trigger that fired this update,

REWRITE the entire note. Constraints:
- Maximum ~500 words. Prioritize aggressively if you'd go longer.
- Preserve relevant old facts: chronic injuries, recorded disagreements with
  past advice, long-term patterns. Only add or update; never remove signals
  that future-you would want to see.
- Be specific. Use dates. Reference actual workouts and pains.
- Write in PT-BR (the runner converses in PT-BR; the coach's narrative for
  themselves stays in the same language for coherence).
- Output ONLY the new note content. No commentary, no preamble, no "Here is...".
"""


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def _model_name(model: str | None) -> str:
    return model or os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")


def _json_default(value: Any) -> Any:
    """Make datetimes / dates / UUIDs JSON-serializable for the prompt payload."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _build_user_payload(
    *,
    previous_note: str,
    trigger: str,
    profile: dict | None,
    checkins: list[dict],
    workouts: list[dict],
    active_injuries: list[dict],
) -> str:
    """Pack the rewrite inputs as a JSON object.

    JSON over YAML: smaller surface (no quoting rules, no indentation
    pitfalls with multiline strings), better tooling support, and the LLM
    handles it at least as well. The note CONTENT stays in PT-BR; the
    envelope keys are English because they are addressed to the LLM, not
    the runner.
    """
    payload = {
        "trigger": trigger,
        "previous_note": previous_note,
        "profile": profile or {},
        "active_injuries": active_injuries,
        "recent_checkins_last_30d": checkins,
        "recent_workouts_last_30d": workouts,
    }
    return json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2)


def rewrite_coach_note(
    *,
    user_id: str,
    trigger: str,
    db_module: Any = _default_db,
    llm_client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    """Rewrite the runner's coach note and persist the new revision.

    Loads the previous note + profile + last 30 days of check-ins/workouts +
    active injuries, sends them with the rewrite system prompt to the LLM,
    and saves whatever the LLM returns as a new `coach_notes` row.

    Returns the new note content.
    """
    if trigger not in {"new_workout", "risk_flag", "idle", "manual", "bootstrap"}:
        raise ValueError(f"unknown coach-note trigger: {trigger!r}")

    if trigger == "bootstrap":
        previous_note = ""
    else:
        latest = db_module.load_latest_coach_note(user_id)
        previous_note = latest["content"] if latest else ""

    profile = db_module.load_profile(user_id)
    checkins = db_module.load_checkins_since(user_id, 30)
    workouts = db_module.load_workouts_since(user_id, 30)
    active_injuries = db_module.load_active_injuries(user_id)

    user_payload = _build_user_payload(
        previous_note=previous_note,
        trigger=trigger,
        profile=profile,
        checkins=checkins,
        workouts=workouts,
        active_injuries=active_injuries,
    )

    llm = llm_client or _build_client()
    response = llm.chat.completions.create(
        model=_model_name(model),
        messages=[
            {"role": "system", "content": NOTE_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.2,
        top_p=0.9,
    )
    new_content = (response.choices[0].message.content or "").strip()

    # Don't truncate — that would lie about what the LLM produced. Per
    # CONTEXT.md, "exceeds → forces prioritization" is the LLM's job; if it
    # overshoots we observe it and let prompt iteration fix it.
    word_count = len(new_content.split())
    if word_count > _WORD_COUNT_WARN_THRESHOLD:
        logger.warning(
            "coach_note rewrite returned %d words (>%d threshold) for user_id=%s trigger=%s",
            word_count,
            _WORD_COUNT_WARN_THRESHOLD,
            user_id,
            trigger,
        )

    db_module.save_coach_note(user_id, new_content, trigger)
    return new_content


def bootstrap_coach_note(
    *,
    user_id: str,
    db_module: Any = _default_db,
    llm_client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    """Forget any previous note and regenerate from scratch.

    Same path as `rewrite_coach_note` but with the previous note forced to
    empty and trigger='bootstrap'. Use when the note has drifted past
    rescue or when seeding a runner who has accumulated history but never
    had a note written.
    """
    return rewrite_coach_note(
        user_id=user_id,
        trigger="bootstrap",
        db_module=db_module,
        llm_client=llm_client,
        model=model,
    )
