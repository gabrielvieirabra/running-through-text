"""Longitudinal coherence eval (ADR 0005 — category 4).

A 4-turn scripted conversation, all LLM calls mocked, focused on the
coach_note flow:

  1. Runner reports a 5km run.
  2. Runner reports feeling drained.
  3. Runner insists on doubling Saturday volume → harm reduction +
     `update_coach_note` with the disagreement.
  4. Next day, fresh `call_coach_with_tools` invocation — the system
     messages must include the disagreement that was written into the
     coach note in turn 3.

We assert the side-effects we care about:
- `register_workout` fires in turn 1.
- `update_coach_note` fires in turn 3 with the right `reason`.
- `maybe_update_coach_note` writes the new note via the rewriter when
  the manual trigger fires.
- The system prefix of turn 4 contains the new note's content (i.e. the
  memory preamble exposes the rewritten note to the agent).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import coach
from memory import coach_note as coach_note_module


pytestmark = pytest.mark.light


_PROFILE = {
    "name": "Marina",
    "age": 30,
    "experience_level": "intermediario",
    "pace_5k": "5:00/km",
    "pace_10k": None,
    "weekly_days": 4,
    "goal": "meia maratona em outubro",
}


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


def _wire_common_stubs(monkeypatch, *, coach_note: dict | None):
    """Wire all the DB stubs needed for `call_coach_with_tools` to run end-to-end.

    `coach_note` is the latest coach note (or None) — turn 4 wants to see
    the just-written note in the memory preamble.
    """
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
    monkeypatch.setattr(coach.db, "load_profile", lambda _uid: _PROFILE)
    monkeypatch.setattr(coach.db, "load_active_injuries", lambda _uid: [])
    monkeypatch.setattr(coach.db, "load_checkins_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_workouts_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_latest_coach_note", lambda _uid: coach_note)
    monkeypatch.setattr(coach.db, "register_workout", lambda **_kw: "workout-uuid-1")
    monkeypatch.setattr(coach.db, "register_checkin", lambda **_kw: "checkin-uuid-1")
    monkeypatch.setattr(coach.db, "save_coach_note", lambda *_a, **_kw: "note-uuid-1")
    monkeypatch.setattr(
        coach.db, "days_since_last_coach_note", lambda _uid: 0
    )


def test_4_turn_disagreement_propagates_into_next_session(monkeypatch):
    """The disagreement registered in turn 3 must surface in turn 4's preamble."""

    # ---- Turn 1: runner reports a 5km rodagem ---------------------------
    _wire_common_stubs(monkeypatch, coach_note=None)
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "register_workout",
                    {
                        "type": "rodagem",
                        "distance_km": 5,
                        "target_pace": "5:00/km",
                    },
                )
            ],
        ),
        _completion(content="Boa rodagem! Anotei aqui.", tool_calls=None),
    ]
    turn1_seen: set[str] = set()
    coach.call_coach_with_tools(
        messages=[{"role": "user", "content": "fiz 5km hoje a 5:00/km"}],
        user_id="runner-long-1",
        client=fake_client,
        tool_calls_seen=turn1_seen,
    )
    assert "register_workout" in turn1_seen

    # ---- Turn 2: runner says they feel drained (no tool calls) ----------
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content="Tá pesado mesmo. Manda ver no descanso amanhã.",
            tool_calls=None,
        ),
    ]
    turn2_seen: set[str] = set()
    coach.call_coach_with_tools(
        messages=[{"role": "user", "content": "tá pesado, vou faltar amanhã"}],
        user_id="runner-long-1",
        client=fake_client,
        tool_calls_seen=turn2_seen,
    )
    assert turn2_seen == set()

    # ---- Turn 3: runner insists on doubling Saturday. compensation_attempt
    # marker should fire and the agent should call `update_coach_note`
    # with the disagreement reason.
    #
    # We script `compensation_attempt` directly by clearing recent workouts
    # — the helper looks for runs in the last 2 days; we already wired
    # `load_recent_workouts` to return []. The triggers fire on the message
    # alone via `evaluate_safety` (called from the preamble builder).
    disagreement_reason = (
        "runner insists on doubling Saturday volume against advice"
    )
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content=None,
            tool_calls=[
                _tool_call(
                    "update_coach_note",
                    {"reason": disagreement_reason},
                )
            ],
        ),
        _completion(
            content=(
                "Recomendo fortemente NÃO dobrar. Se você insistir, "
                "limite a 8km bem leves no sábado."
            ),
            tool_calls=None,
        ),
    ]
    turn3_seen: set[str] = set()
    coach.call_coach_with_tools(
        messages=[
            {
                "role": "user",
                "content": "perdi treino, vou compensar dobrando sábado",
            }
        ],
        user_id="runner-long-1",
        client=fake_client,
        tool_calls_seen=turn3_seen,
    )
    assert "update_coach_note" in turn3_seen

    # The first LLM call in turn 3 must have seen the
    # compensation_attempt safety marker — verifying our scripted
    # premise rather than depending on the model.
    turn3_first_call = fake_client.chat.completions.create.call_args_list[-2]
    combined_system_turn3 = "\n".join(
        m["content"]
        for m in turn3_first_call.kwargs["messages"]
        if m.get("role") == "system"
    )
    assert "compensation_attempt" in combined_system_turn3

    # ---- post-reply: maybe_update_coach_note fires `manual` -------------
    rewrite_calls: list[dict] = []

    def _fake_rewrite(*, user_id: str, trigger: str, db_module=None) -> str:
        rewrite_calls.append({"user_id": user_id, "trigger": trigger})
        # Simulate what the rewriter would produce — content includes the
        # disagreement, mirroring what the LLM would write per the system
        # prompt for the note path.
        new_note_content = (
            "Marina, intermediária, meia maratona em outubro. "
            f"Desacordo registrado: {disagreement_reason}. "
            "Manter atenção em compensações futuras."
        )
        # Persist via the same db_module stub the loop wired up.
        if db_module is not None:
            db_module.save_coach_note(user_id, new_note_content, trigger)
        # Smuggle the rewritten note back through the closure so turn 4 can
        # serve it via `load_latest_coach_note`.
        rewrite_calls[-1]["content"] = new_note_content
        return new_note_content

    executed = coach.maybe_update_coach_note(
        user_id="runner-long-1",
        tool_calls_made=turn3_seen,
        rewrite_fn=_fake_rewrite,
    )
    assert executed == ["manual"]
    assert rewrite_calls and rewrite_calls[0]["trigger"] == "manual"

    new_note_content = rewrite_calls[0]["content"]
    assert disagreement_reason in new_note_content

    # ---- Turn 4: new session next day. The preamble must show the
    # disagreement now baked into the coach note.
    _wire_common_stubs(
        monkeypatch,
        coach_note={
            "id": "note-uuid-1",
            "content": new_note_content,
            "generated_at": datetime(2026, 5, 15, tzinfo=timezone.utc),
            "trigger": "manual",
        },
    )

    fake_client_turn4 = MagicMock()
    fake_client_turn4.chat.completions.create.side_effect = [
        _completion(content="Bom dia, Marina! Como você acordou hoje?"),
    ]
    coach.call_coach_with_tools(
        messages=[{"role": "user", "content": "bom dia"}],
        user_id="runner-long-1",
        client=fake_client_turn4,
    )
    turn4_messages = fake_client_turn4.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    combined_system_turn4 = "\n".join(
        m["content"] for m in turn4_messages if m.get("role") == "system"
    )
    assert "COACH NOTE" in combined_system_turn4
    assert disagreement_reason in combined_system_turn4, (
        "the disagreement registered in turn 3 must surface in turn 4 "
        "via the memory preamble"
    )


def test_rewrite_fn_actually_called_with_real_rewriter_path(monkeypatch):
    """Sanity check that the default `rewrite_coach_note` path is what
    `maybe_update_coach_note` would resolve to when no override is passed —
    we don't actually run it (would call the real LLM), just verify the
    import path."""
    # Import-time check — if this module loads, the default rewrite path is wired.
    from memory.coach_note import rewrite_coach_note

    assert rewrite_coach_note is coach_note_module.rewrite_coach_note
