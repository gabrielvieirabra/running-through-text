"""Extraction eval for the onboarding wizard.

Two scenarios:

1. **Wizard mode triggered on empty profile** — a runner reply with partial
   baseline info ("32 anos, corro 5k em 25min, quero correr 4x/semana, sem
   lesões, intermediário") must drive an `update_profile` tool call that
   carries every field the runner mentioned, including
   `injury_history_acknowledged: true`.

2. **Cooper fallback** — when the runner has all blocking fields filled except
   both paces, the per-turn PROFILE STATUS preamble must report
   `cooper_needed: YES` so the agent has the right signal to prescribe the
   Cooper test. We test the signal the system gives the LLM (preamble
   contents), not what the mocked LLM happens to say — that's the honest
   eval at this layer.

The OpenRouter client and the persistence layer are both mocked so this
scenario runs without Postgres and without a real LLM call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import coach


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
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


def test_wizard_extracts_partial_baseline_and_acknowledges_no_injuries(monkeypatch):
    """Runner shares 5 baseline facts at once; agent calls `update_profile`
    with the matching keyword args AND flips `injury_history_acknowledged`."""

    upsert_mock = MagicMock(return_value=None)
    monkeypatch.setattr(coach.db, "upsert_profile", upsert_mock)

    # Wizard preamble: report a fresh runner with all 5 blocking fields missing.
    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _user_id: {
            "blocking_complete": False,
            "missing_blocking": [
                "pace (5k or 10k)",
                "weekly_days",
                "goal",
                "injury_history",
                "experience_level",
            ],
            "filled_count": 0,
            "total_count": 5,
            "has_5k_or_10k_pace": False,
            "cooper_needed": False,
        },
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "update_profile",
                    {
                        "age": 32,
                        "pace_5k": "5:00/km",
                        "weekly_days": 4,
                        "experience_level": "intermediario",
                        "injury_history_acknowledged": True,
                    },
                )
            ],
        ),
        _completion(
            content=(
                "Beleza! Anotei aqui. Falta só o seu objetivo de treino — "
                "que evento ou meta você tem em mente?"
            ),
            tool_calls=None,
        ),
    ]

    reply = coach.call_coach_with_tools(
        messages=[
            {
                "role": "user",
                "content": (
                    "32 anos, corro 5k em 25min, quero correr 4x/semana, "
                    "sem lesões, intermediário"
                ),
            }
        ],
        user_id="runner-wizard-1",
        client=fake_client,
    )

    assert reply.startswith("Beleza")

    upsert_mock.assert_called_once()
    call = upsert_mock.call_args
    # `user_id` may arrive positionally or as a kwarg depending on the call site.
    if call.args:
        assert call.args[0] == "runner-wizard-1"
    else:
        assert call.kwargs.get("user_id") == "runner-wizard-1"

    kwargs = call.kwargs
    # The agent must pass the structured fields, including the explicit
    # "no injuries" acknowledgement so the wizard gate sees injury_history covered.
    assert kwargs["age"] == 32
    assert kwargs["pace_5k"] == "5:00/km"
    assert kwargs["weekly_days"] == 4
    assert kwargs["experience_level"] == "intermediario"
    assert kwargs["injury_history_acknowledged"] is True

    # Two LLM round-trips: tool call + final text.
    assert fake_client.chat.completions.create.call_count == 2

    # The first LLM call must include the PROFILE STATUS preamble so the
    # agent knows wizard mode is active.
    first_call_messages = fake_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    system_messages = [m for m in first_call_messages if m.get("role") == "system"]
    combined_system = "\n".join(m["content"] for m in system_messages)
    assert "PROFILE STATUS" in combined_system
    assert "blocking_complete: NO" in combined_system


def test_cooper_fallback_signal_in_preamble(monkeypatch):
    """When the runner has every blocking field filled EXCEPT both paces, the
    PROFILE STATUS preamble reports `cooper_needed: YES`. That's the signal the
    system gives the LLM to prescribe the Cooper test.

    We assert two things:
    1. The system prompt itself documents the Cooper fallback rule.
    2. The preamble built for this runner reports `cooper_needed: YES`.

    This is the "honest" eval — we verify the system gives the right signal,
    not that a mocked LLM happens to mention Cooper.
    """

    # The system prompt must actually contain the Cooper rule so the LLM
    # knows what to do when it sees `cooper_needed: YES`.
    assert "Cooper" in coach.SYSTEM_PROMPT
    assert "12 minut" in coach.SYSTEM_PROMPT  # matches "12 minutes"

    # Mock the DB so `build_profile_status_preamble` reports a Cooper-needed
    # gate: 4 of 5 blocking fields filled, pace is the only gap.
    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _user_id: {
            "blocking_complete": False,
            "missing_blocking": ["pace (5k or 10k)"],
            "filled_count": 4,
            "total_count": 5,
            "has_5k_or_10k_pace": False,
            "cooper_needed": True,
        },
    )

    preamble = coach.build_profile_status_preamble("runner-cooper-1")
    assert preamble is not None
    assert "cooper_needed: YES" in preamble
    assert "blocking_complete: NO" in preamble
    assert "pace (5k or 10k)" in preamble


def test_register_injury_tool_handler_persists_injury(monkeypatch):
    """When the runner mentions a past injury, the agent calls
    `register_injury`. The handler must forward all relevant fields to
    persistence (which then flips `injury_history_acknowledged` for us)."""

    register_injury_mock = MagicMock(return_value="injury-uuid-xyz")
    monkeypatch.setattr(coach.db, "register_injury", register_injury_mock)
    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _user_id: {
            "blocking_complete": False,
            "missing_blocking": [
                "pace (5k or 10k)",
                "weekly_days",
                "goal",
                "injury_history",
                "experience_level",
            ],
            "filled_count": 0,
            "total_count": 5,
            "has_5k_or_10k_pace": False,
            "cooper_needed": False,
        },
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "register_injury",
                    {
                        "name": "canelite",
                        "side": "esquerda",
                        "year": 2024,
                        "status": "resolved",
                    },
                )
            ],
        ),
        _completion(
            content="Anotei a canelite resolvida. Vamos seguir com cuidado.",
            tool_calls=None,
        ),
    ]

    reply = coach.call_coach_with_tools(
        messages=[
            {"role": "user", "content": "tive canelite na esquerda em 2024, já curei"}
        ],
        user_id="runner-injury-1",
        client=fake_client,
    )

    assert "canelite" in reply
    register_injury_mock.assert_called_once()
    kwargs = register_injury_mock.call_args.kwargs
    assert kwargs["user_id"] == "runner-injury-1"
    assert kwargs["name"] == "canelite"
    assert kwargs["side"] == "esquerda"
    assert kwargs["year"] == 2024
    assert kwargs["status"] == "resolved"
