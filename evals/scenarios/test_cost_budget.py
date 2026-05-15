"""Cost eval (ADR 0005 — category 6).

Token/latency telemetry. The agent loop's prompt assembly grows over
time (system prompt, profile preamble, safety preamble, memory
preamble) and we want a tight budget so prompt bloat shows up before it
hits production. Budgets live in `evals/budgets.json` and are tracked
the same way as judge baselines — change them deliberately, in a
commit, when the prompt actually moves.

The mocked LLM returns a synthetic `usage` block; we don't measure real
tokenizer output (that's the heavy/regression suite's job once we have
a model under live test). This light test instead verifies the budget
plumbing AND walks the actual extraction scenario, so the
`prompt_tokens` we feed reflects the assembled context shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import coach
from evals.judges import budgets as budgets_module


pytestmark = pytest.mark.light


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion_with_usage(
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _approximate_prompt_tokens(messages: list[dict]) -> int:
    """Rough token estimate: chars / 4. Good enough for a budget gate that
    catches prompt bloat — the absolute number is not load-bearing, the
    delta is. The heavy suite will compare against real tokenizer counts
    when it runs against a live model."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    chars += len(json.dumps(part))
    return chars // 4


def test_extraction_scenario_stays_within_token_budget(monkeypatch):
    """Run the existing checkin-extraction scenario, capture the assembled
    prompt, and assert it stays under the recorded `evals/budgets.json`
    cap. Catches accidental prompt growth before it ships."""

    # Wire DB stubs: complete profile, no recent state, no coach note.
    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _user_id: {
            "blocking_complete": True,
            "missing_blocking": [],
            "filled_count": 5,
            "total_count": 5,
            "has_5k_or_10k_pace": True,
            "cooper_needed": False,
        },
    )
    monkeypatch.setattr(coach.db, "load_recent_checkins", lambda _u, limit=1: [])
    monkeypatch.setattr(coach.db, "load_recent_workouts", lambda _u, limit=30: [])
    monkeypatch.setattr(
        coach.db,
        "load_profile",
        lambda _uid: {
            "name": "Ana",
            "age": 32,
            "experience_level": "intermediario",
            "pace_5k": "5:00/km",
            "pace_10k": None,
            "weekly_days": 4,
            "goal": "meia maratona em outubro",
        },
    )
    monkeypatch.setattr(coach.db, "load_active_injuries", lambda _uid: [])
    monkeypatch.setattr(coach.db, "load_checkins_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_workouts_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_latest_coach_note", lambda _uid: None)
    monkeypatch.setattr(coach.db, "register_checkin", lambda **_kw: "checkin-uuid-1")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion_with_usage(
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
            prompt_tokens=900,
            completion_tokens=60,
        ),
        _completion_with_usage(
            content="Entendi. Hoje recomendo descanso — dor 7/10 é forte.",
            tool_calls=None,
            prompt_tokens=950,
            completion_tokens=80,
        ),
    ]

    coach.call_coach_with_tools(
        messages=[
            {
                "role": "user",
                "content": "tô com dor 7/10 no pé esquerdo, dormi pouco",
            }
        ],
        user_id="runner-cost-1",
        client=fake_client,
    )

    # Compute the assembled prompt size on the first call. We trust the
    # actual assembled messages over the mocked `usage` block — the mock
    # is a placeholder until the heavy suite measures real model output.
    first_call_messages = fake_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    prompt_tokens = _approximate_prompt_tokens(first_call_messages)

    # The mocked completion gives a deterministic completion-tokens number
    # we can pass through to the budget check.
    completion_tokens = 80

    budgets_module.assert_within_budget(
        "extraction_register_checkin",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
