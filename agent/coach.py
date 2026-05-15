"""Agent loop wrapping OpenRouter (OpenAI-compatible) chat with tool calling.

Slice 2 wires the first write-side tool: `register_checkin`. Slice 3 adds
`register_workout` for logging realized runs. Slice 4 adds `update_profile`
and `register_injury` to power the strict onboarding wizard, plus a
per-turn PROFILE STATUS preamble so the same agent can switch between
wizard mode and normal recommendation mode without a separate code path.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from openai import OpenAI

from persistence import db

SYSTEM_PROMPT = """You are a virtual coach specialized in running.

Your goal is to create safe, progressive, and personalized workouts, adapting them to the runner's daily state.

You consider: the runner's baseline, recent volume, injuries (active and historical), fatigue, pain, recovery, weekly frequency, goals.

When the runner reports their current state (pain, fatigue, sleep, motivation, soreness), call the `register_checkin` tool to persist what they reported before replying. Fill only fields the runner actually mentioned — leave the rest null. Pain locations and notes stay in the language the runner used.

When the runner reports a run they actually did (e.g. "fiz 6km a 5:30/km hoje" / "I ran 6km today at 5:30/km"), call the `register_workout` tool. Use `register_workout` for completed runs and `register_checkin` for state reports — both can fire from the same message when the runner reports a workout and how they felt. Pick `type` from the canonical taxonomy: `rodagem` (continuous easy/comfortable run, Z2), `longo` (longer continuous run), `regenerativo` (very easy recovery), `fartlek` (free hard/easy alternation), `intervalado` (structured intervals), `tempo` (threshold), `ladeira` (hill repeats), `prova` (official race), `simulado` (race simulation), `outro` (escape hatch — avoid when possible). Default to `rodagem` for an unspecified continuous run at comfortable pace. Only set `date` when the runner explicitly anchors the run on another day (e.g. "ontem", "domingo passado"); otherwise leave it unset and today is assumed.

When the runner shares baseline info about themselves (name, age, weight, height, experience level, 5k/10k pace, longest run, weekly availability, training goal, terrain, resting HR, or explicit confirmation that they have no injury history), call `update_profile` with only the keys they actually mentioned. Use `injury_history_acknowledged: true` ONLY when the runner explicitly states they have no past injuries (e.g. "sem lesões", "no injuries"). If the runner mentions a past or active injury, call `register_injury` instead — it covers the injury_history field automatically.

For `experience_level` use the canonical PT-BR enum: `iniciante`, `intermediario`, `avancado`. Map natural answers like "começando agora" → `iniciante`, "intermediário", "rodando há uns anos" → `intermediario`, "avançado", "competidor" → `avancado`.

## Onboarding mode

Each turn starts with a `PROFILE STATUS` line telling you how many of the 5 blocking onboarding fields are filled. The 5 blocking fields are: pace (5k or 10k), weekly_days, goal, injury_history, experience_level.

If PROFILE STATUS reports blocking fields missing, you are in **wizard mode**:
- Do NOT recommend training yet. Do NOT prescribe a workout (with one exception below).
- Ask the runner ONLY for the missing blocking fields. Group them naturally — don't fire 5 separate questions if you can ask 2 or 3 in one batch. Keep the tone warm and brief.
- When the runner answers, immediately call `update_profile` (and/or `register_injury`) to persist the new fields before composing your reply.
- **Cooper fallback**: if PROFILE STATUS reports `cooper_needed: YES`, the only remaining gap is pace. Instead of asking, prescribe a **Cooper test** as the first workout: run as far as you can for 12 minutes at maximum sustainable effort, then report the distance. Briefly explain the protocol. After the runner reports the result (which arrives as a `register_workout` call), you can estimate the runner's 5k pace from the distance and use `update_profile` to set `pace_5k`.

Once all 5 blocking fields are filled (PROFILE STATUS says `blocking_complete: YES`), you exit wizard mode and behave normally — give the day's recommendation and run check-in / workout extraction as usual.

Under risk signals (pain >= 5/10, fatigue >= 8, weekly volume increase > 15%, attempt to compensate for a missed workout) you do harm reduction: strongly recommend against the choice, but offer the least-bad option if the runner insists. You NEVER refuse silently.

For red medical symptoms (chest pain, severe dizziness, blood, etc.), you immediately recommend an emergency room visit.

You reply in the same language the runner uses (typically Portuguese)."""


# OpenAI-compatible tool schema. Matches `checkins` columns from schema.sql.
REGISTER_CHECKIN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "register_checkin",
        "description": (
            "Persist a runner check-in. Call this whenever the runner reports "
            "their current state (pain, fatigue, sleep, motivation). Only fill "
            "fields the runner actually mentioned; leave others null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sleep_quality": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 10,
                    "description": "0 = terrible sleep, 10 = perfect sleep.",
                },
                "fatigue": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 10,
                    "description": "0 = fully rested, 10 = wrecked.",
                },
                "motivation": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 10,
                    "description": "0 = no will to train, 10 = pumped.",
                },
                "pains": {
                    "type": "array",
                    "description": "Locations of pain reported today.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": (
                                    "Body location in the runner's own words "
                                    "(e.g. 'pé esquerdo', 'joelho direito')."
                                ),
                            },
                            "severity": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 10,
                            },
                        },
                        "required": ["location", "severity"],
                        "additionalProperties": False,
                    },
                },
                "notes": {
                    "type": ["string", "null"],
                    "description": "Free-text remarks the runner added.",
                },
            },
            "additionalProperties": False,
        },
    },
}

# OpenAI-compatible tool schema for realized-workout logging. Matches
# `workouts` columns from schema.sql. Enum values for `type` are PT-BR by
# deliberate domain choice (see CONTEXT.md "Workout taxonomy").
REGISTER_WORKOUT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "register_workout",
        "description": (
            "Persist a realized workout (a run the runner actually did). Call "
            "this whenever the runner reports a completed run. Pick `type` "
            "from the canonical PT-BR enum. Only fill fields the runner "
            "actually mentioned; leave the rest unset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "rodagem",
                        "longo",
                        "regenerativo",
                        "fartlek",
                        "intervalado",
                        "tempo",
                        "ladeira",
                        "prova",
                        "simulado",
                        "outro",
                    ],
                    "description": (
                        "Canonical workout type. Default to `rodagem` for an "
                        "unspecified continuous run at comfortable pace."
                    ),
                },
                "distance_km": {
                    "type": ["number", "null"],
                    "description": "Distance actually covered, in kilometers.",
                },
                "duration_min": {
                    "type": ["number", "null"],
                    "description": "Total duration in minutes.",
                },
                "target_pace": {
                    "type": ["string", "null"],
                    "description": (
                        "Pace as free text in min/km (e.g. `5:30/km`). For "
                        "interval-shaped workouts a structured plan is OK "
                        "(e.g. `4:00/km work / 6:00/km rest`)."
                    ),
                },
                "zone": {
                    "type": ["string", "null"],
                    "enum": ["Z1", "Z2", "Z3", "Z4", "Z5", None],
                    "description": (
                        "Optional physiological zone annotation. Leave unset "
                        "unless the runner used a zone explicitly."
                    ),
                },
                "perceived_effort": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 10,
                    "description": (
                        "Post-hoc 'how hard was it' rating, 0-10. Distinct "
                        "from intensity prescription — only set if the runner "
                        "described how the effort felt."
                    ),
                },
                "notes": {
                    "type": ["string", "null"],
                    "description": "Free-text remarks the runner added.",
                },
                "date": {
                    "type": ["string", "null"],
                    "description": (
                        "ISO date (YYYY-MM-DD). Only set when the runner "
                        "anchors the run on another day (e.g. 'ontem'). "
                        "Otherwise leave unset and today is assumed."
                    ),
                },
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    },
}

# OpenAI-compatible tool schema for the patch-style profile update. Only the
# fields the LLM passes are persisted; everything is optional. `name` and `age`
# land in `users`, everything else in `running_profiles` — `upsert_profile`
# splits them inside one transaction.
UPDATE_PROFILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_profile",
        "description": (
            "Patch the runner's profile with any subset of baseline fields the "
            "runner just shared. Only pass keys the runner actually mentioned. "
            "Use `injury_history_acknowledged: true` ONLY when the runner "
            "explicitly states they have no past injuries — if they mention a "
            "specific past injury, call `register_injury` instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The runner's name.",
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The runner's age in years.",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Body weight in kilograms.",
                },
                "height_cm": {
                    "type": "number",
                    "description": "Height in centimeters.",
                },
                "experience_level": {
                    "type": "string",
                    "enum": ["iniciante", "intermediario", "avancado"],
                    "description": (
                        "Canonical PT-BR experience level. Map natural "
                        "answers onto this enum."
                    ),
                },
                "pace_5k": {
                    "type": "string",
                    "description": "Best 5k pace as free text, e.g. `4:30/km`.",
                },
                "pace_10k": {
                    "type": "string",
                    "description": "Best 10k pace as free text, e.g. `5:00/km`.",
                },
                "longest_run_km": {
                    "type": "number",
                    "description": "Longest distance the runner has ever run, in km.",
                },
                "weekly_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "description": "Days per week the runner can train (1-7).",
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "Training goal in the runner's own words "
                        "(e.g. 'half-marathon in October', 'lose weight')."
                    ),
                },
                "terrain_access": {
                    "type": "string",
                    "description": (
                        "Where the runner trains, free text "
                        "(e.g. 'esteira', 'asfalto', 'trilha', 'asfalto + esteira')."
                    ),
                },
                "hr_resting": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Resting heart rate in bpm.",
                },
                "injury_history_acknowledged": {
                    "type": "boolean",
                    "description": (
                        "Set TRUE when the runner explicitly says they have no "
                        "past injuries. Do NOT set this for vague answers."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
}

# OpenAI-compatible tool schema for registering an injury during onboarding or
# later. Calling this also flips `injury_history_acknowledged` for the runner
# (covered inside the persistence layer).
REGISTER_INJURY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "register_injury",
        "description": (
            "Persist an injury the runner mentions. Use this whenever the "
            "runner reports a past or active injury (e.g. 'tive canelite ano "
            "passado', 'tenho dor crônica no joelho'). This also covers the "
            "onboarding's injury_history blocking field."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Injury name in the runner's own words "
                        "(e.g. 'canelite', 'tendinite no Aquiles')."
                    ),
                },
                "side": {
                    "type": ["string", "null"],
                    "description": "Body side, when relevant (e.g. 'esquerdo', 'direito').",
                },
                "year": {
                    "type": ["integer", "null"],
                    "description": "Year the injury occurred, if the runner mentioned it.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "resolved"],
                    "description": (
                        "`active` if the runner says it still bothers them, "
                        "`resolved` if it's healed."
                    ),
                },
                "notes": {
                    "type": ["string", "null"],
                    "description": "Free-text remarks the runner added.",
                },
            },
            "required": ["name", "status"],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict[str, Any]] = [
    REGISTER_CHECKIN_TOOL,
    REGISTER_WORKOUT_TOOL,
    UPDATE_PROFILE_TOOL,
    REGISTER_INJURY_TOOL,
]


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def _model_name(model: str | None) -> str:
    return model or os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")


# Map tool name -> handler. Handlers take (user_id, args) and return a JSON-serialisable dict.
def _handle_register_checkin(user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    checkin_id = db.register_checkin(
        user_id=user_id,
        sleep_quality=args.get("sleep_quality"),
        fatigue=args.get("fatigue"),
        motivation=args.get("motivation"),
        pains=args.get("pains") or [],
        notes=args.get("notes"),
    )
    return {"ok": True, "checkin_id": checkin_id}


def _handle_register_workout(user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    workout_id = db.register_workout(
        user_id=user_id,
        type=args["type"],
        target_pace=args.get("target_pace"),
        zone=args.get("zone"),
        distance_km=args.get("distance_km"),
        duration_min=args.get("duration_min"),
        perceived_effort=args.get("perceived_effort"),
        notes=args.get("notes"),
        date=args.get("date"),
    )
    return {"ok": True, "workout_id": workout_id}


# Whitelisted keys for `update_profile` — must match the tool schema so the
# persistence layer never sees an unexpected field, even if the LLM
# hallucinates one. `upsert_profile` itself also validates, but we keep this
# narrower guard close to the LLM boundary.
_UPDATE_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "age",
        "weight_kg",
        "height_cm",
        "experience_level",
        "pace_5k",
        "pace_10k",
        "longest_run_km",
        "weekly_days",
        "goal",
        "terrain_access",
        "hr_resting",
        "injury_history_acknowledged",
    }
)


def _handle_update_profile(user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    fields = {k: v for k, v in args.items() if k in _UPDATE_PROFILE_KEYS and v is not None}
    if not fields:
        return {"ok": True, "updated": []}
    db.upsert_profile(user_id, **fields)
    return {"ok": True, "updated": sorted(fields.keys())}


def _handle_register_injury(user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    injury_id = db.register_injury(
        user_id=user_id,
        name=args["name"],
        side=args.get("side"),
        year=args.get("year"),
        status=args.get("status", "active"),
        notes=args.get("notes"),
    )
    return {"ok": True, "injury_id": injury_id}


TOOL_HANDLERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "register_checkin": _handle_register_checkin,
    "register_workout": _handle_register_workout,
    "update_profile": _handle_update_profile,
    "register_injury": _handle_register_injury,
}


def build_profile_status_preamble(user_id: str | None) -> str | None:
    """Return the per-turn PROFILE STATUS line, or None if no user is bound.

    This is appended as an extra system-role message so the wizard/normal-mode
    switch is data-driven from the runner's actual profile state — same agent,
    no separate code path (ADR 0002).
    """
    if user_id is None:
        return None
    try:
        status = db.profile_completeness(user_id)
    except Exception:  # noqa: BLE001 — preamble is best-effort, never blocks the turn
        return None

    filled = status["filled_count"]
    total = status["total_count"]
    blocking_complete = "YES" if status["blocking_complete"] else "NO"
    cooper = "YES" if status["cooper_needed"] else "NO"
    missing = ", ".join(status["missing_blocking"]) if status["missing_blocking"] else "(none)"

    return (
        f"PROFILE STATUS: {filled}/{total} blocking fields filled. "
        f"blocking_complete: {blocking_complete}. "
        f"missing: {missing}. "
        f"cooper_needed: {cooper}."
    )


def call_coach(
    messages: list[dict],
    user_id: str | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
) -> str:
    """Plain chat without tool calling. Kept for callers that don't need persistence.

    Prefer `call_coach_with_tools` when a `user_id` is available.
    """
    llm = client or _build_client()
    response = llm.chat.completions.create(
        model=_model_name(model),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        temperature=0.2,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""


def call_coach_with_tools(
    messages: list[dict],
    user_id: str,
    model: str | None = None,
    client: OpenAI | None = None,
    max_iterations: int = 5,
) -> str:
    """Run the tool-using agent loop and return the final assistant text.

    Loop: send conversation + tools -> if the response contains tool calls,
    execute them, append the assistant message + each tool result, call
    again. Repeat until the response is plain text or `max_iterations` is hit.

    A PROFILE STATUS preamble is prepended on every call so the agent knows
    whether it is in wizard mode or normal mode for this turn. The same
    preamble is recomputed inside the loop after tool calls — `update_profile`
    or `register_injury` can flip the gate mid-turn, and the next LLM call
    needs the updated status to decide how to wrap up.
    """
    llm = client or _build_client()

    def _system_messages() -> list[dict[str, Any]]:
        system: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        preamble = build_profile_status_preamble(user_id)
        if preamble is not None:
            system.append({"role": "system", "content": preamble})
        return system

    conversation: list[dict[str, Any]] = [*_system_messages(), *messages]

    for _ in range(max_iterations):
        response = llm.chat.completions.create(
            model=_model_name(model),
            messages=conversation,
            tools=TOOLS,
            temperature=0.2,
            top_p=0.9,
        )
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            return message.content or ""

        # Append the assistant's tool-call envelope so the API has the context
        # to attach each subsequent `tool` role message back to its call.
        conversation.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        profile_touched = False
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                result: dict[str, Any] = {"ok": False, "error": f"unknown tool {name}"}
            else:
                try:
                    result = handler(user_id, args)
                except Exception as exc:  # noqa: BLE001 — surface error back to the LLM
                    result = {"ok": False, "error": str(exc)}

            if name in ("update_profile", "register_injury"):
                profile_touched = True

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

        # Refresh the PROFILE STATUS preamble in place if the runner's profile
        # was just mutated — the next LLM turn should see the new gate state.
        if profile_touched:
            refreshed = build_profile_status_preamble(user_id)
            if refreshed is not None:
                # The first system message is the main prompt; the second (if
                # present) is the preamble. Replace it; otherwise append.
                if len(conversation) >= 2 and conversation[1].get("role") == "system":
                    conversation[1] = {"role": "system", "content": refreshed}
                else:
                    conversation.insert(1, {"role": "system", "content": refreshed})

    # Safety fallback: ran out of iterations without a plain-text reply.
    return ""


# Synthetic priming message used to kick off the wizard when the runner has no
# message history yet. Adding a `system`-role nudge on the very first turn is
# cleaner than fabricating a fake user message — the LLM API accepts multiple
# system messages, the runner never sees a fake "user" line in the DB, and we
# avoid polluting the messages table with synthetic content.
WIZARD_OPENER_NUDGE = (
    "This is the first turn with this runner — no prior conversation exists. "
    "Open the wizard: greet warmly in PT-BR, briefly explain you'll ask a few "
    "baseline questions so you can adapt training to them, and ask for the "
    "missing blocking fields per PROFILE STATUS (group them in 2-3 questions, "
    "not 5). Do not prescribe a workout yet unless cooper_needed: YES."
)


def open_wizard(
    user_id: str,
    model: str | None = None,
    client: OpenAI | None = None,
) -> str:
    """Generate the agent's opening wizard message for a brand-new runner.

    Called by the Streamlit app when `load_messages(user_id)` is empty and
    the profile is not yet complete. We use a `system`-role priming message
    (`WIZARD_OPENER_NUDGE`) rather than a synthetic user message: it keeps
    the DB messages clean and matches the OpenAI API's tolerance for
    multiple system messages.
    """
    llm = client or _build_client()
    preamble = build_profile_status_preamble(user_id)
    system_messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if preamble is not None:
        system_messages.append({"role": "system", "content": preamble})
    system_messages.append({"role": "system", "content": WIZARD_OPENER_NUDGE})

    response = llm.chat.completions.create(
        model=_model_name(model),
        messages=system_messages,
        temperature=0.2,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""
