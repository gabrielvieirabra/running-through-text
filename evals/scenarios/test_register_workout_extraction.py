"""Extraction eval: a runner reporting a completed run must trigger
`register_workout` with the right structured args.

The OpenRouter client and the persistence layer are both mocked so this
scenario runs without Postgres and without a real LLM call.

The companion test verifying the Postgres CHECK constraint on `type` is
deliberately not included as a unit test — sqlite would not enforce the
Postgres-style CHECK the same way, so it belongs in a Postgres-backed
integration suite (planned later). The constraint is exercised by the
schema; the agent-side enum in the tool schema and the system prompt are
what this slice owns.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import coach


CANONICAL_WORKOUT_TYPES = {
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
}


def _tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(content: str | None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def test_completed_run_message_triggers_register_workout(monkeypatch):
    # Mocked persistence: capture call args, skip DB.
    register_workout_mock = MagicMock(return_value="workout-uuid-abc")
    monkeypatch.setattr(coach.db, "register_workout", register_workout_mock)

    # Mocked OpenAI client: first turn returns a tool call, second returns text.
    # `rodagem` is the canonical choice for a continuous comfortable run at a
    # single pace — the most likely model output for this message.
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "register_workout",
                    {
                        "type": "rodagem",
                        "distance_km": 6,
                        "target_pace": "5:30/km",
                        "notes": "foi tranquilo",
                    },
                )
            ],
        ),
        _completion(
            content="Boa! 6km a 5:30/km tranquilo — registrei aqui.",
            tool_calls=None,
        ),
    ]

    reply = coach.call_coach_with_tools(
        messages=[
            {
                "role": "user",
                "content": "fiz 6km a 5:30/km hoje, foi tranquilo",
            }
        ],
        user_id="runner-1",
        client=fake_client,
    )

    # The runner sees the final assistant text, not the tool-call envelope.
    assert reply.startswith("Boa!")

    # Persistence got the structured fields from the tool call.
    register_workout_mock.assert_called_once()
    kwargs = register_workout_mock.call_args.kwargs
    assert kwargs["user_id"] == "runner-1"
    assert kwargs["type"] in CANONICAL_WORKOUT_TYPES
    assert kwargs["type"] == "rodagem"
    assert kwargs["distance_km"] == pytest.approx(6)
    assert kwargs["target_pace"] == "5:30/km"
    # Unmentioned fields stay null — the agent should not invent them.
    assert kwargs["duration_min"] is None
    assert kwargs["zone"] is None
    assert kwargs["perceived_effort"] is None
    # Date defaults to today on the SQL side — the LLM didn't anchor a date.
    assert kwargs["date"] is None

    # Loop sent two LLM calls: one to get the tool call, one to get the reply.
    assert fake_client.chat.completions.create.call_count == 2

    # The second call must include the tool result so the LLM can react to it.
    second_call_messages = fake_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    tool_payload = json.loads(tool_messages[0]["content"])
    assert tool_payload["ok"] is True
    assert tool_payload["workout_id"] == "workout-uuid-abc"


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Postgres CHECK constraint on workouts.type — requires a live "
        "Postgres connection; covered by the integration suite, not the "
        "unit-level evals. sqlite would not enforce the same constraint."
    )
)
def test_invalid_workout_type_rejected_by_db_check_constraint():
    """Sanity test for the enum CHECK constraint. Runs against real Postgres only."""
    from persistence import db

    with pytest.raises(Exception):
        db.register_workout(user_id="runner-1", type="not-a-real-type")
