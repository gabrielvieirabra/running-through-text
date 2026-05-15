"""Pure-function safety triggers and the `evaluate` orchestrator.

Each trigger takes already-loaded structured data and returns a `Marker`
dict or `None`. No DB calls live inside the trigger functions themselves —
DB access is concentrated in `evaluate`, which accepts a `db_module`
parameter so tests can inject a fake without monkeypatching.

A `Marker` is a TypedDict (a plain dict at runtime) with three keys:

    name    : 'high_pain' | 'high_fatigue' | 'volume_jump'
              | 'compensation_attempt' | 'red_medical_symptoms'
    severity: 'soft' | 'red'   ('red' only for red_medical_symptoms)
    detail  : short human-readable string for the LLM, e.g. "pé esquerdo 7/10"

Soft markers drive harm reduction; the lone red marker drives hard refusal.

Per ADR 0003, the keyword lists in `compensation_attempt` and
`red_medical_symptoms` are intentionally crude — substring match is enough
for the MVP. They live here (not in the system prompt) precisely so they
are fail-closed and auditable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, TypedDict

from persistence import db as _default_db


class Marker(TypedDict):
    """Shape of a single risk marker. Returned by triggers, consumed by the agent."""

    name: str
    severity: str
    detail: str


# ---------------------------------------------------------------------------
# Individual triggers
# ---------------------------------------------------------------------------


def high_pain(latest_checkin: dict | None) -> Marker | None:
    """Fire when the latest check-in carries any pain with severity >= 5.

    Returns the worst pain (highest severity) as the marker detail so the
    agent has the single most important data point on hand. If multiple
    pains tie on severity, the first one wins — order is preserved from
    however persistence handed it over.
    """
    if not latest_checkin:
        return None
    pains = latest_checkin.get("pains") or []
    worst: dict | None = None
    for pain in pains:
        try:
            severity = int(pain.get("severity"))
        except (TypeError, ValueError):
            continue
        if severity < 5:
            continue
        if worst is None or severity > int(worst.get("severity", 0)):
            worst = pain
    if worst is None:
        return None
    location = worst.get("location") or "unspecified"
    severity = worst.get("severity")
    return {
        "name": "high_pain",
        "severity": "soft",
        "detail": f"{location} {severity}/10",
    }


def high_fatigue(latest_checkin: dict | None) -> Marker | None:
    """Fire when the latest check-in reports fatigue >= 8."""
    if not latest_checkin:
        return None
    fatigue = latest_checkin.get("fatigue")
    if fatigue is None:
        return None
    try:
        fatigue_int = int(fatigue)
    except (TypeError, ValueError):
        return None
    if fatigue_int < 8:
        return None
    return {
        "name": "high_fatigue",
        "severity": "soft",
        "detail": f"fatigue {fatigue_int}/10",
    }


def _iso_year_week(d: date) -> tuple[int, int]:
    iso = d.isocalendar()
    return (iso.year, iso.week)


def _workout_date(workout: dict) -> date | None:
    """Coerce a workout's `date` field to a `date` regardless of how persistence handed it over."""
    raw = workout.get("date")
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    return None


def volume_jump(recent_workouts: list[dict] | None) -> Marker | None:
    """Fire when this ISO-week's distance is > 15% greater than last week's.

    Comparison is week-over-week with the *latest* ISO week observed in the
    data acting as "this week" — that way the trigger keeps working even
    when called right after `register_workout` (the new row carries today's
    date) and stays stable across days inside the same ISO week.

    Returns None unless both weeks have at least one workout.
    """
    if not recent_workouts:
        return None

    by_week: dict[tuple[int, int], float] = {}
    for w in recent_workouts:
        d = _workout_date(w)
        if d is None:
            continue
        distance = w.get("distance_km")
        if distance is None:
            continue
        try:
            distance_f = float(distance)
        except (TypeError, ValueError):
            continue
        key = _iso_year_week(d)
        by_week[key] = by_week.get(key, 0.0) + distance_f

    if len(by_week) < 2:
        return None

    # Sort by (year, week) descending; the top entry is "this week".
    weeks_sorted = sorted(by_week.keys(), reverse=True)
    this_week = weeks_sorted[0]
    last_week = weeks_sorted[1]
    this_total = by_week[this_week]
    last_total = by_week[last_week]

    if last_total <= 0:
        return None
    if this_total <= last_total * 1.15:
        return None

    pct = (this_total - last_total) / last_total * 100
    return {
        "name": "volume_jump",
        "severity": "soft",
        "detail": (
            f"this week {this_total:.1f}km vs last week {last_total:.1f}km "
            f"(+{pct:.0f}%)"
        ),
    }


# Crude keyword/phrase list for "I missed a workout, want to make up for it"
# detection. Intentionally small: substring match in lowercase is enough
# for the MVP (ADR 0003: don't over-engineer NLP in this slice).
_COMPENSATION_KEYWORDS: tuple[str, ...] = (
    # PT-BR
    "compensar",
    "perdi o treino",
    "perdi treino",
    "faltei",
    "dobrar",
    "dobradinha",
    "recuperar o atraso",
    "recuperar atraso",
    # EN
    "missed",
    "make up for",
    "makeup run",
    "catch up",
    "double up",
)


def compensation_attempt(
    latest_user_message: str | None,
    recent_workouts: list[dict] | None,
    *,
    now: datetime | None = None,
) -> Marker | None:
    """Fire when the runner's message reads as "make up for a missed workout"
    AND there's been no workout logged in the past 2 days.

    Both conditions must hold to suppress false positives — the agent
    shouldn't fire compensation guidance just because the runner used the
    word "dobrar" in a benign context (e.g. asking about double sessions
    when they ran yesterday).
    """
    if not latest_user_message:
        return None
    text = latest_user_message.lower()
    matched = next((kw for kw in _COMPENSATION_KEYWORDS if kw in text), None)
    if matched is None:
        return None

    today = (now or datetime.now(timezone.utc)).date()
    cutoff = today - timedelta(days=2)
    for w in recent_workouts or []:
        d = _workout_date(w)
        if d is not None and d >= cutoff:
            # A workout in the last 2 days — not a "missed days" situation.
            return None

    return {
        "name": "compensation_attempt",
        "severity": "soft",
        "detail": f"matched '{matched}', no workout in last 2 days",
    }


# Red medical-symptom phrases. Substring match, case-insensitive. PT-BR
# and EN side by side. The runner sees the agent's "go to the ER" reply,
# never this list directly — keep crude.
_RED_SYMPTOM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("dor no peito", "chest pain (PT-BR)"),
    ("chest pain", "chest pain"),
    ("aperto no peito", "chest tightness (PT-BR)"),
    ("tontura forte", "severe dizziness (PT-BR)"),
    ("tontura severa", "severe dizziness (PT-BR)"),
    ("severe dizziness", "severe dizziness"),
    ("sangue na urina", "blood in urine (PT-BR)"),
    ("blood in urine", "blood in urine"),
    ("blood in my urine", "blood in urine"),
    ("desmaiei", "fainted (PT-BR)"),
    ("desmaio", "fainted (PT-BR)"),
    ("fainted", "fainted"),
    ("passed out", "fainted"),
)


def red_medical_symptoms(latest_user_message: str | None) -> Marker | None:
    """Fire on any of a short list of red medical phrases.

    This is the ONE trigger that returns `severity: "red"` — the system
    prompt's Safety section instructs the agent to hard-refuse training and
    direct the runner to emergency care.
    """
    if not latest_user_message:
        return None
    text = latest_user_message.lower()
    # Strip basic accents-as-different-case nuance: lowering is enough for
    # PT-BR substring matches because the patterns are stored in lowercase
    # with their natural accents.
    matched = next((label for needle, label in _RED_SYMPTOM_PATTERNS if needle in text), None)
    if matched is None:
        return None
    return {
        "name": "red_medical_symptoms",
        "severity": "red",
        "detail": matched,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def evaluate(
    user_id: str,
    latest_user_message: str | None,
    *,
    db_module: Any = _default_db,
) -> list[Marker]:
    """Run every trigger against the runner's current state and message.

    Loads the latest check-in and the last ~30 workouts (enough to cover
    two ISO weeks) from `db_module`, then runs every trigger and returns
    the non-None markers as a list (order is stable: matches the order of
    the trigger functions below).

    `db_module` is injectable for tests — pass any object exposing
    `load_recent_checkins(user_id, limit=...)` and
    `load_recent_workouts(user_id, limit=...)`.
    """
    try:
        recent_checkins = db_module.load_recent_checkins(user_id, limit=1)
    except Exception:  # noqa: BLE001 — safety is best-effort, never blocks the turn
        recent_checkins = []
    try:
        recent_workouts = db_module.load_recent_workouts(user_id, limit=30)
    except Exception:  # noqa: BLE001
        recent_workouts = []

    latest_checkin = recent_checkins[0] if recent_checkins else None

    markers: list[Marker] = []
    for marker in (
        high_pain(latest_checkin),
        high_fatigue(latest_checkin),
        volume_jump(recent_workouts),
        compensation_attempt(latest_user_message, recent_workouts),
        red_medical_symptoms(latest_user_message),
    ):
        if marker is not None:
            markers.append(marker)
    return markers


