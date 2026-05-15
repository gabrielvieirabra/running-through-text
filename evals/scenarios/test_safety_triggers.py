"""Safety eval (Slice 5, ADR 0003).

Two layers covered here:

1. Pure-function tests over the individual triggers in `safety.triggers`.
   These verify the deterministic numeric rules — high pain, volume jump,
   red medical symptoms — independently of any agent or DB plumbing.

2. Preamble-signal tests at the agent boundary: when markers are active,
   the LLM receives the SAFETY MARKERS preamble (or the hard-refusal
   preamble for red symptoms) and the system prompt's Safety section is
   in scope. Following the same "honest" eval style as Slice 4's Cooper
   test, we assert on the input given to the LLM rather than on the
   mocked LLM's reply.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import coach
from safety import triggers

pytestmark = pytest.mark.light


# ---------------------------------------------------------------------------
# 1) Pure-function triggers
# ---------------------------------------------------------------------------


def test_high_pain_fires_when_any_pain_severity_is_5_or_more():
    checkin = {
        "pains": [
            {"location": "joelho direito", "severity": 3},
            {"location": "pé esquerdo", "severity": 7},
        ],
        "fatigue": 4,
    }
    marker = triggers.high_pain(checkin)
    assert marker is not None
    assert marker["name"] == "high_pain"
    assert marker["severity"] == "soft"
    # The worst pain wins — pé/7 takes precedence over joelho/3.
    assert "pé esquerdo" in marker["detail"]
    assert "7" in marker["detail"]


def test_high_pain_does_not_fire_when_all_pains_below_five():
    checkin = {
        "pains": [
            {"location": "joelho", "severity": 2},
            {"location": "panturrilha", "severity": 4},
        ],
        "fatigue": 1,
    }
    assert triggers.high_pain(checkin) is None


def test_high_pain_returns_none_with_empty_checkin():
    assert triggers.high_pain(None) is None
    assert triggers.high_pain({"pains": []}) is None


def test_high_fatigue_fires_at_threshold():
    assert triggers.high_fatigue({"fatigue": 8}) is not None
    assert triggers.high_fatigue({"fatigue": 9}) is not None
    assert triggers.high_fatigue({"fatigue": 7}) is None
    assert triggers.high_fatigue({"fatigue": None}) is None


def test_volume_jump_fires_when_this_week_exceeds_last_by_more_than_15_percent():
    """Build two ISO weeks of workouts where week N is >1.15 * week N-1.

    Anchor at a fixed date so the test is deterministic across ISO-week
    boundaries — Wednesday 2026-05-13 sits inside ISO week 20, and the
    previous Wednesday 2026-05-06 sits inside ISO week 19.
    """
    this_week = date(2026, 5, 13)
    last_week = date(2026, 5, 6)

    workouts = [
        # This week: 25 km total.
        {"date": this_week, "distance_km": 12.5, "type": "rodagem"},
        {"date": this_week - timedelta(days=2), "distance_km": 12.5, "type": "rodagem"},
        # Last week: 20 km total — 25/20 = 1.25 > 1.15.
        {"date": last_week, "distance_km": 10, "type": "rodagem"},
        {"date": last_week - timedelta(days=1), "distance_km": 10, "type": "rodagem"},
    ]
    marker = triggers.volume_jump(workouts)
    assert marker is not None
    assert marker["name"] == "volume_jump"
    assert marker["severity"] == "soft"
    # Detail mentions both totals so the LLM has the numbers at hand.
    assert "25" in marker["detail"]
    assert "20" in marker["detail"]


def test_volume_jump_does_not_fire_within_15_percent_band():
    this_week = date(2026, 5, 13)
    last_week = date(2026, 5, 6)
    # 22 vs 20 = 1.10x — well inside the band.
    workouts = [
        {"date": this_week, "distance_km": 22, "type": "rodagem"},
        {"date": last_week, "distance_km": 20, "type": "rodagem"},
    ]
    assert triggers.volume_jump(workouts) is None


def test_volume_jump_needs_at_least_two_weeks_of_data():
    workouts = [
        {"date": date(2026, 5, 13), "distance_km": 50, "type": "rodagem"},
    ]
    assert triggers.volume_jump(workouts) is None


def test_red_medical_symptoms_fires_on_chest_pain_ptbr():
    marker = triggers.red_medical_symptoms("tô com dor no peito desde de manhã")
    assert marker is not None
    assert marker["name"] == "red_medical_symptoms"
    assert marker["severity"] == "red"
    assert "chest pain" in marker["detail"]


def test_red_medical_symptoms_fires_on_english_chest_pain():
    marker = triggers.red_medical_symptoms("having sharp chest pain right now")
    assert marker is not None
    assert marker["severity"] == "red"


def test_red_medical_symptoms_no_match_returns_none():
    assert triggers.red_medical_symptoms("fiz 6km hoje tranquilo") is None
    assert triggers.red_medical_symptoms("") is None
    assert triggers.red_medical_symptoms(None) is None


def test_compensation_attempt_fires_on_missed_keyword_with_no_recent_workouts():
    # "Faltei treino terça e quarta, vou compensar dobrando hoje" — clear signal.
    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    marker = triggers.compensation_attempt(
        latest_user_message="perdi o treino ontem e anteontem, vou compensar hoje",
        recent_workouts=[
            # Last workout 5 days ago — outside the 2-day window.
            {"date": date(2026, 5, 9), "distance_km": 8, "type": "rodagem"},
        ],
        now=now,
    )
    assert marker is not None
    assert marker["name"] == "compensation_attempt"


def test_compensation_attempt_suppressed_when_recent_workout_logged():
    """Even if the keyword matches, a workout logged yesterday means the
    runner didn't actually miss days — don't fire."""
    now = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    marker = triggers.compensation_attempt(
        latest_user_message="quero dobrar a quilometragem hoje",
        recent_workouts=[
            {"date": date(2026, 5, 13), "distance_km": 8, "type": "rodagem"},
        ],
        now=now,
    )
    assert marker is None


# ---------------------------------------------------------------------------
# 2) Preamble-signal tests at the agent boundary
# ---------------------------------------------------------------------------


def _completion(content: str | None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def test_build_safety_markers_preamble_lists_markers():
    """Given high_pain markers, the preamble mentions the name and detail."""
    fake_db = SimpleNamespace(
        load_recent_checkins=lambda uid, limit=1: [
            {
                "pains": [{"location": "pé esquerdo", "severity": 7}],
                "fatigue": 3,
            }
        ],
        load_recent_workouts=lambda uid, limit=30: [],
    )
    markers = triggers.evaluate(
        "runner-1",
        latest_user_message="estou com dor no pé esquerdo 7/10",
        db_module=fake_db,
    )
    # `high_pain` from the check-in plus no other markers.
    assert any(m["name"] == "high_pain" for m in markers)

    preamble = coach.build_safety_markers_preamble(markers)
    assert preamble is not None
    assert "SAFETY MARKERS ACTIVE" in preamble
    assert "high_pain" in preamble
    assert "pé esquerdo" in preamble
    assert "harm reduction" in preamble.lower()


def test_build_safety_markers_preamble_returns_none_without_markers():
    assert coach.build_safety_markers_preamble([]) is None
    assert coach.build_safety_markers_preamble(None) is None


def test_red_medical_preamble_present_only_for_red_markers():
    soft_only = [{"name": "high_pain", "severity": "soft", "detail": "pé 7/10"}]
    assert coach.build_red_medical_preamble(soft_only) is None

    red = [
        {"name": "red_medical_symptoms", "severity": "red", "detail": "chest pain"}
    ]
    out = coach.build_red_medical_preamble(red)
    assert out is not None
    assert "RED MEDICAL SYMPTOM" in out
    assert "immediate medical attention" in out.lower() or "emergency" in out.lower() or "pronto-socorro" in out


def test_system_prompt_documents_harm_reduction_and_hard_refusal():
    """The system prompt itself must carry the Safety section so the LLM
    knows what to do when it sees markers in the preamble. This is the
    preamble-signal style test like Slice 4's Cooper one."""
    sp = coach.SYSTEM_PROMPT
    assert "## Safety" in sp
    assert "HARM REDUCTION" in sp
    assert "HARD REFUSE" in sp
    # Soft marker names should be listed explicitly so the LLM can map a
    # marker it sees in the preamble to the documented behavior.
    for marker_name in (
        "high_pain",
        "high_fatigue",
        "volume_jump",
        "compensation_attempt",
    ):
        assert marker_name in sp


def test_call_coach_with_tools_injects_safety_preamble_when_high_pain_fires(monkeypatch):
    """End-to-end at the agent boundary: with a high-pain check-in loaded,
    the first LLM call's system messages must include both the system
    prompt (with the Safety section) AND the SAFETY MARKERS preamble
    mentioning high_pain. We don't assert on the LLM's reply — we assert
    on the input the system gives the LLM."""

    # No profile gate noise — pretend onboarding is done.
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
    # Fake DB returns one check-in carrying a pain at severity 7.
    monkeypatch.setattr(
        coach.db,
        "load_recent_checkins",
        lambda _user_id, limit=1: [
            {
                "pains": [{"location": "pé esquerdo", "severity": 7}],
                "fatigue": 4,
                "sleep_quality": 5,
            }
        ],
    )
    monkeypatch.setattr(
        coach.db,
        "load_recent_workouts",
        lambda _user_id, limit=30: [],
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content="Recomendo descanso hoje, dor 7/10 é forte.",
            tool_calls=None,
        ),
    ]

    coach.call_coach_with_tools(
        messages=[
            {"role": "user", "content": "ainda dói o pé, vou treinar mesmo assim?"}
        ],
        user_id="runner-safety-1",
        client=fake_client,
    )

    first_call_messages = fake_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    system_messages = [m for m in first_call_messages if m.get("role") == "system"]
    combined_system = "\n".join(m["content"] for m in system_messages)

    # Both the marker preamble and the documented behavior must be present.
    assert "SAFETY MARKERS ACTIVE" in combined_system
    assert "high_pain" in combined_system
    assert "pé" in combined_system
    assert "## Safety" in combined_system
    assert "HARM REDUCTION" in combined_system


def test_call_coach_with_tools_injects_red_medical_preamble(monkeypatch):
    """When the runner's message contains a red medical phrase, the
    hard-refusal preamble must be added to the system messages."""

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

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _completion(
            content="Procure um pronto-socorro imediatamente. Não vou prescrever treino agora.",
            tool_calls=None,
        ),
    ]

    coach.call_coach_with_tools(
        messages=[
            {"role": "user", "content": "tô com dor no peito agora, posso correr?"}
        ],
        user_id="runner-red-1",
        client=fake_client,
    )

    first_call_messages = fake_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    combined_system = "\n".join(
        m["content"] for m in first_call_messages if m.get("role") == "system"
    )
    assert "RED MEDICAL SYMPTOM" in combined_system
    assert "HARD REFUSE" in combined_system  # from SYSTEM_PROMPT
