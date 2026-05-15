"""Agent loop wrapping OpenRouter (OpenAI-compatible) chat with tool calling.

Slice 2 wires the first write-side tool: `register_checkin`. Slice 3 adds
`register_workout` for logging realized runs. Injury creation is intentionally
*not* an LLM-callable tool yet — distinguishing "today's pain" from "starting
an injury record" needs more signal than the current agent has; deferred to a
later slice. `register_injury` exists in persistence and can be called by
tests or future tools.
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

TOOLS: list[dict[str, Any]] = [REGISTER_CHECKIN_TOOL, REGISTER_WORKOUT_TOOL]


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


TOOL_HANDLERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "register_checkin": _handle_register_checkin,
    "register_workout": _handle_register_workout,
}


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
    """
    llm = client or _build_client()
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

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

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    # Safety fallback: ran out of iterations without a plain-text reply.
    return ""
