"""Coach-note eval (Slice 6, ADR 0001 M3).

Covers the rewrite path, trigger logic, memory preamble, bootstrap helper,
and the mid-loop preamble rebuild. Persistence and OpenRouter are mocked
end-to-end so the suite runs without Postgres or a real LLM call.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import coach
from memory import coach_note as coach_note_module
from memory import context as context_module

pytestmark = pytest.mark.light


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


def _make_fake_db(**overrides) -> SimpleNamespace:
    """Build a fake `db_module` with sensible defaults; override what you need."""
    defaults = {
        "load_latest_coach_note": lambda _uid: None,
        "load_profile": lambda _uid: {
            "name": "Ana",
            "age": 32,
            "experience_level": "intermediario",
            "pace_5k": "5:00/km",
            "pace_10k": None,
            "weekly_days": 4,
            "goal": "meia maratona em outubro",
        },
        "load_checkins_since": lambda _uid, _days: [],
        "load_workouts_since": lambda _uid, _days: [],
        "load_active_injuries": lambda _uid: [],
        "save_coach_note": lambda _uid, _content, _trigger: "note-uuid-1",
        "days_since_last_coach_note": lambda _uid: None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1) rewrite_coach_note persistence + word-count warning
# ---------------------------------------------------------------------------


def test_rewrite_coach_note_persists_llm_output_verbatim_and_warns_over_threshold(caplog):
    """The function must not silently truncate — what the LLM returns is what
    gets saved. If the LLM overshoots the ~500-word target by a lot, the
    function emits a warning so the overshoot is observable.
    """
    long_output = " ".join(["palavra"] * 800)  # 800 words, well above the cap.

    save_calls: list[tuple[str, str, str]] = []

    def _fake_save(user_id: str, content: str, trigger: str) -> str:
        save_calls.append((user_id, content, trigger))
        return "note-uuid-long"

    fake_db = _make_fake_db(save_coach_note=_fake_save)

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion(long_output)

    with caplog.at_level(logging.WARNING, logger="memory.coach_note"):
        out = coach_note_module.rewrite_coach_note(
            user_id="runner-1",
            trigger="manual",
            db_module=fake_db,
            llm_client=fake_client,
        )

    # Persisted content matches the LLM output verbatim (after a strip).
    assert out == long_output
    assert len(save_calls) == 1
    _, persisted, trigger = save_calls[0]
    assert persisted == long_output
    assert trigger == "manual"

    # And a warning fires because word_count (800) > 600 threshold.
    matching = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert matching, "expected a WARNING log for the long rewrite"
    assert any("800" in r.getMessage() for r in matching)


def test_rewrite_coach_note_strips_whitespace_and_returns_short_output(caplog):
    """A short output (under threshold) persists clean and emits no warning."""
    short_output = "  Runner em fase de base. Dor leve no joelho — observar.  \n"

    save_calls: list[tuple[str, str, str]] = []

    def _fake_save(user_id: str, content: str, trigger: str) -> str:
        save_calls.append((user_id, content, trigger))
        return "note-uuid-short"

    fake_db = _make_fake_db(save_coach_note=_fake_save)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion(short_output)

    with caplog.at_level(logging.WARNING, logger="memory.coach_note"):
        out = coach_note_module.rewrite_coach_note(
            user_id="runner-1",
            trigger="new_workout",
            db_module=fake_db,
            llm_client=fake_client,
        )

    assert out == short_output.strip()
    assert save_calls[0][1] == short_output.strip()
    # No warning at this length.
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_rewrite_coach_note_rejects_unknown_trigger():
    fake_db = _make_fake_db()
    fake_client = MagicMock()
    try:
        coach_note_module.rewrite_coach_note(
            user_id="runner-1",
            trigger="not-a-real-trigger",
            db_module=fake_db,
            llm_client=fake_client,
        )
    except ValueError as exc:
        assert "not-a-real-trigger" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown trigger")


def test_rewrite_coach_note_loads_previous_note_for_non_bootstrap_triggers():
    """`previous_note` must flow into the LLM user payload so it can preserve
    chronic facts. Bootstrap path is the exception (tested separately)."""
    fake_db = _make_fake_db(
        load_latest_coach_note=lambda _uid: {
            "id": "prev-1",
            "content": "Histórico: canelite crônica desde 2024.",
            "generated_at": datetime(2026, 5, 10, tzinfo=timezone.utc),
            "trigger": "manual",
        }
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion("nova nota")

    coach_note_module.rewrite_coach_note(
        user_id="runner-1",
        trigger="risk_flag",
        db_module=fake_db,
        llm_client=fake_client,
    )

    call_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    user_payload = call_messages[-1]["content"]
    parsed = json.loads(user_payload)
    assert parsed["trigger"] == "risk_flag"
    assert parsed["previous_note"] == "Histórico: canelite crônica desde 2024."


# ---------------------------------------------------------------------------
# 2) bootstrap_coach_note
# ---------------------------------------------------------------------------


def test_bootstrap_coach_note_forces_empty_previous_and_bootstrap_trigger():
    """Bootstrap must ignore any existing note and call the rewrite path with
    `previous_note=''` and `trigger='bootstrap'`."""

    fake_db = _make_fake_db(
        load_latest_coach_note=lambda _uid: {
            "id": "stale",
            "content": "Old corrupted text",
            "generated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "trigger": "manual",
        }
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _completion("nova nota limpa")

    coach_note_module.bootstrap_coach_note(
        user_id="runner-1",
        db_module=fake_db,
        llm_client=fake_client,
    )

    call_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    user_payload = call_messages[-1]["content"]
    parsed = json.loads(user_payload)
    assert parsed["trigger"] == "bootstrap"
    # Bootstrap path explicitly drops the previous content.
    assert parsed["previous_note"] == ""


# ---------------------------------------------------------------------------
# 3) build_memory_preamble shape
# ---------------------------------------------------------------------------


def test_build_memory_preamble_includes_latest_note_content():
    note_content = "Runner em fase de base; canelite resolvida; pace_5k 5:00."
    fake_db = _make_fake_db(
        load_latest_coach_note=lambda _uid: {
            "id": "n1",
            "content": note_content,
            "generated_at": datetime(2026, 5, 12, tzinfo=timezone.utc),
            "trigger": "manual",
        },
        load_active_injuries=lambda _uid: [
            {
                "id": "i1",
                "name": "canelite",
                "side": "esquerda",
                "year": 2024,
                "status": "active",
                "notes": None,
                "created_at": None,
            }
        ],
        load_checkins_since=lambda _uid, _days: [
            {
                "id": "c1",
                "date": date(2026, 5, 13),
                "sleep_quality": 6,
                "fatigue": 4,
                "motivation": 7,
                "pains": [{"location": "joelho", "severity": 3}],
                "notes": None,
                "created_at": None,
            }
        ],
        load_workouts_since=lambda _uid, _days: [
            {
                "id": "w1",
                "date": date(2026, 5, 12),
                "type": "rodagem",
                "target_pace": "5:30/km",
                "zone": "Z2",
                "distance_km": 8,
                "duration_min": 44,
                "perceived_effort": 5,
                "notes": None,
                "created_at": None,
            }
        ],
    )
    preamble = context_module.build_memory_preamble("runner-1", db_module=fake_db)

    assert "RUNNER PROFILE" in preamble
    assert "pace_5k=5:00/km" in preamble
    assert "ACTIVE INJURIES" in preamble
    assert "canelite" in preamble
    assert "RECENT CHECK-INS" in preamble
    assert "2026-05-13" in preamble
    assert "joelho:3/10" in preamble
    assert "RECENT WORKOUTS" in preamble
    assert "rodagem" in preamble
    assert "8km" in preamble
    assert "COACH NOTE" in preamble
    assert note_content in preamble
    assert "2026-05-12" in preamble


def test_build_memory_preamble_handles_no_note_yet():
    fake_db = _make_fake_db()
    preamble = context_module.build_memory_preamble("runner-empty", db_module=fake_db)
    assert "no note yet" in preamble
    assert "ACTIVE INJURIES: none acknowledged" in preamble
    assert "none" in preamble


# ---------------------------------------------------------------------------
# 4) maybe_update_coach_note trigger logic
# ---------------------------------------------------------------------------


def test_maybe_update_coach_note_fires_new_workout_trigger():
    """After a successful `register_workout` with no risk markers, the
    post-reply step must call the rewriter with `trigger='new_workout'`."""

    fake_db = _make_fake_db()
    rewrite_calls: list[dict] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append({"user_id": user_id, "trigger": trigger})
        return "note"

    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made={"register_workout"},
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == ["new_workout"]
    assert rewrite_calls == [{"user_id": "runner-1", "trigger": "new_workout"}]


def test_maybe_update_coach_note_fires_manual_when_update_tool_called():
    """`update_coach_note` from the LLM beats all other triggers."""

    fake_db = _make_fake_db()
    rewrite_calls: list[str] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append(trigger)
        return "note"

    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made={"update_coach_note", "register_workout"},
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == ["manual"]
    assert rewrite_calls == ["manual"]


def test_maybe_update_coach_note_fires_risk_flag_when_checkin_with_marker(monkeypatch):
    """A check-in + active soft safety marker fires `risk_flag`. With no
    marker, the same check-in alone is not enough to rewrite."""

    fake_db = _make_fake_db()
    # First call: marker active.
    monkeypatch.setattr(
        coach,
        "evaluate_safety",
        lambda _uid, _msg: [
            {"name": "high_pain", "severity": "soft", "detail": "joelho 7/10"}
        ],
    )
    rewrite_calls: list[str] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append(trigger)
        return "note"

    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made={"register_checkin"},
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == ["risk_flag"]

    # Second call: no markers — no rewrite fires.
    monkeypatch.setattr(coach, "evaluate_safety", lambda _uid, _msg: [])
    rewrite_calls.clear()
    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made={"register_checkin"},
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == []
    assert rewrite_calls == []


def test_maybe_update_coach_note_fires_idle_when_no_writes_and_7_days_old():
    """No write tools fired, last note is 8 days old → `idle` rewrite."""

    fake_db = _make_fake_db(days_since_last_coach_note=lambda _uid: 8)
    rewrite_calls: list[str] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append(trigger)
        return "note"

    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made=set(),
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == ["idle"]
    assert rewrite_calls == ["idle"]


def test_maybe_update_coach_note_idle_skipped_when_note_is_fresh():
    """Last note 2 days old → idle doesn't fire."""
    fake_db = _make_fake_db(days_since_last_coach_note=lambda _uid: 2)
    rewrite_calls: list[str] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append(trigger)
        return "note"

    executed = coach.maybe_update_coach_note(
        user_id="runner-1",
        tool_calls_made=set(),
        db_module=fake_db,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == []
    assert rewrite_calls == []


# ---------------------------------------------------------------------------
# 5) Mid-loop memory preamble rebuild
# ---------------------------------------------------------------------------


def test_memory_preamble_present_on_first_call_and_rebuilt_after_register_workout(monkeypatch):
    """End-to-end at the agent boundary: the system messages on call #1
    include the MEMORY preamble; after `register_workout` the system
    prefix is rebuilt so call #2 also sees the (potentially updated)
    MEMORY preamble. Honest test — assert on input to the LLM, not on
    the mocked LLM's reply."""

    # No wizard noise.
    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _uid: {
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
    # Memory inputs.
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
            "goal": "meia maratona",
        },
    )
    monkeypatch.setattr(coach.db, "load_active_injuries", lambda _uid: [])
    monkeypatch.setattr(coach.db, "load_checkins_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_workouts_since", lambda _uid, _days: [])
    monkeypatch.setattr(
        coach.db,
        "load_latest_coach_note",
        lambda _uid: {
            "id": "n1",
            "content": "Nota anterior do treinador.",
            "generated_at": datetime(2026, 5, 12, tzinfo=timezone.utc),
            "trigger": "manual",
        },
    )
    monkeypatch.setattr(
        coach.db,
        "register_workout",
        lambda **kwargs: "workout-uuid-1",
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "register_workout",
                    {"type": "rodagem", "distance_km": 6, "target_pace": "5:30/km"},
                )
            ],
        ),
        _completion(content="Boa rodagem! Anotei.", tool_calls=None),
    ]

    tool_calls_seen: set[str] = set()
    coach.call_coach_with_tools(
        messages=[{"role": "user", "content": "fiz 6km hoje"}],
        user_id="runner-mem-1",
        client=fake_client,
        tool_calls_seen=tool_calls_seen,
    )

    # The loop saw the workout tool call.
    assert "register_workout" in tool_calls_seen

    # Memory preamble present on both LLM calls.
    for call_index in (0, 1):
        messages = fake_client.chat.completions.create.call_args_list[call_index].kwargs[
            "messages"
        ]
        combined_system = "\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        assert "COACH NOTE" in combined_system, f"call #{call_index} missing memory preamble"
        assert "Nota anterior do treinador." in combined_system


def test_update_coach_note_tool_does_not_run_rewrite_inline(monkeypatch):
    """Calling `update_coach_note` inside the loop must NOT do the rewrite
    inline — it queues a `manual` trigger that's resolved post-reply by
    `maybe_update_coach_note`. The rewrite path is keeping out of the
    runner's hot wait time on purpose."""

    monkeypatch.setattr(
        coach.db,
        "profile_completeness",
        lambda _uid: {
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
    monkeypatch.setattr(coach.db, "load_profile", lambda _uid: None)
    monkeypatch.setattr(coach.db, "load_active_injuries", lambda _uid: [])
    monkeypatch.setattr(coach.db, "load_checkins_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_workouts_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_latest_coach_note", lambda _uid: None)

    save_coach_note_mock = MagicMock()
    monkeypatch.setattr(coach.db, "save_coach_note", save_coach_note_mock)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "update_coach_note",
                    {"reason": "runner insistiu em treinar com dor 7/10"},
                )
            ],
        ),
        _completion(content="Ok, registrei e seguimos.", tool_calls=None),
    ]

    tool_calls_seen: set[str] = set()
    coach.call_coach_with_tools(
        messages=[{"role": "user", "content": "vou treinar mesmo com dor"}],
        user_id="runner-manual-1",
        client=fake_client,
        tool_calls_seen=tool_calls_seen,
    )

    # The LLM signal arrived in the seen set...
    assert "update_coach_note" in tool_calls_seen
    # ...but no rewrite ran inside the loop (no save_coach_note hit).
    save_coach_note_mock.assert_not_called()
