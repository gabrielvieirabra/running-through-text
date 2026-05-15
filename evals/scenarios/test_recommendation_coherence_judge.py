"""Recommendation coherence eval — LLM-as-judge over a real recommendation.

Asks the agent for "today's recommendation" with a clear profile (goal
+ weekly_days) and checks via judge that the response respects both.
The point is to catch responses that contradict the wizard-set goal
(e.g. recommending an intervalado heavy session for a beginner aiming
at a first 10k) or that ignore weekly availability.

Heavy: hits OpenRouter. Skipped without `OPENROUTER_API_KEY`. Gated by
delta-only baseline per ADR 0005.
"""

from __future__ import annotations

import os

import pytest

from agent import coach
from evals.judges import assert_judge_no_regression, llm_judge


pytestmark = pytest.mark.heavy


COHERENCE_RUBRIC = """Score 0-5. The runner's profile says:
- goal: "first 10k race in 3 months"
- weekly_days: 3
- experience_level: iniciante
- pace_5k: 6:30/km

The runner just asked "o que faço hoje?" (what should I do today).

5 = the recommendation is appropriate for an iniciante targeting their
    first 10k, training 3x/week. It does NOT prescribe a high-intensity
    structured workout (intervalado / tempo) for an iniciante. It does
    NOT contradict the goal or the 3x/week availability (e.g. by
    implying daily training).
4 = good fit but with a small mismatch (e.g. proposes 4 sessions in the
    discussion, or proposes tempo before the runner has a base).
3 = generic recommendation that doesn't reference the runner's profile.
2 = recommendation that mismatches the goal or the experience level.
1 = recommendation that would harm an iniciante (e.g. long intervalado
    on day one).
0 = response refuses to recommend or is unrelated.

Penalise responses that ignore the goal or that prescribe sessions
inconsistent with iniciante / 3x per week."""


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="recommendation-coherence heavy test requires OPENROUTER_API_KEY",
)
def test_recommendation_coherence_no_regression(monkeypatch):
    """Realistic profile + neutral check-in; expect a coherent recommendation."""

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
    # No active pain, neutral fatigue — no safety markers should fire.
    monkeypatch.setattr(coach.db, "load_recent_checkins", lambda _u, limit=1: [])
    monkeypatch.setattr(coach.db, "load_recent_workouts", lambda _u, limit=30: [])
    monkeypatch.setattr(
        coach.db,
        "load_profile",
        lambda _uid: {
            "name": "João",
            "age": 28,
            "experience_level": "iniciante",
            "pace_5k": "6:30/km",
            "pace_10k": None,
            "weekly_days": 3,
            "goal": "primeira corrida de 10k em 3 meses",
        },
    )
    monkeypatch.setattr(coach.db, "load_active_injuries", lambda _uid: [])
    monkeypatch.setattr(coach.db, "load_checkins_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_workouts_since", lambda _uid, _days: [])
    monkeypatch.setattr(coach.db, "load_latest_coach_note", lambda _uid: None)

    # Absorb any incidental write (the agent may call register_checkin for the
    # turn even though there's no real state report).
    monkeypatch.setattr(
        coach.db, "register_checkin", lambda **_kwargs: "checkin-uuid-coh"
    )
    monkeypatch.setattr(
        coach.db, "save_coach_note", lambda *_args, **_kwargs: "note-uuid-coh"
    )

    reply = coach.call_coach_with_tools(
        messages=[
            {"role": "user", "content": "o que faço hoje?"},
        ],
        user_id="runner-coh-1",
    )

    assert reply.strip()

    verdict = llm_judge(
        response=reply,
        rubric=COHERENCE_RUBRIC,
        context={
            "scenario": "iniciante targeting first 10k, 3x/week, neutral state",
            "profile": {
                "experience_level": "iniciante",
                "weekly_days": 3,
                "goal": "primeira corrida de 10k em 3 meses",
                "pace_5k": "6:30/km",
            },
            "active_markers": [],
        },
    )

    assert_judge_no_regression(
        "test_recommendation_coherence_no_regression",
        verdict["score"],
        model=verdict["model"],
    )
