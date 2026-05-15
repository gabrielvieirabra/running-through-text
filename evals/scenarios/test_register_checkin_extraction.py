"""Extraction eval: a runner reporting pain + bad sleep must trigger
`register_checkin` with the right structured args.

The OpenRouter client and the persistence layer are both mocked so this
scenario runs without Postgres and without a real LLM call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import coach


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


def test_pain_and_sleep_message_triggers_register_checkin(monkeypatch):
    # Mocked persistence: capture call args, skip DB.
    register_checkin_mock = MagicMock(return_value="checkin-uuid-123")
    monkeypatch.setattr(coach.db, "register_checkin", register_checkin_mock)

    # Mocked OpenAI client: first turn returns a tool call, second returns text.
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "register_checkin",
                    {
                        "pains": [{"location": "pé esquerdo", "severity": 7}],
                        "sleep_quality": 3,
                    },
                )
            ],
        ),
        _completion(
            content=(
                "Entendi. Hoje recomendo descanso — dor 7/10 é sinal forte."
            ),
            tool_calls=None,
        ),
    ]

    reply = coach.call_coach_with_tools(
        messages=[
            {
                "role": "user",
                "content": "tô com dor 7/10 no pé esquerdo, dormi pouco",
            }
        ],
        user_id="runner-1",
        client=fake_client,
    )

    # The runner sees the final assistant text, not the tool-call envelope.
    assert reply.startswith("Entendi")

    # Persistence got the structured fields from the tool call.
    register_checkin_mock.assert_called_once()
    kwargs = register_checkin_mock.call_args.kwargs
    assert kwargs["user_id"] == "runner-1"
    assert kwargs["sleep_quality"] == 3
    assert kwargs["pains"] == [{"location": "pé esquerdo", "severity": 7}]
    # Unmentioned fields stay null — the agent should not invent them.
    assert kwargs["fatigue"] is None
    assert kwargs["motivation"] is None
    assert kwargs["notes"] is None

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
    assert tool_payload["checkin_id"] == "checkin-uuid-123"
