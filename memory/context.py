"""Per-turn memory preamble builder (Slice 6, ADR 0001 M3).

Pulls profile, active injuries, recent check-ins, recent workouts, and the
latest coach note from the structured store, and packs them into a
compact system-message preamble for the main agent. The point of M3 is
the agent doesn't chew through raw rows: the preamble below + the latest
coach note IS the summary.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from persistence import db as _default_db


# Profile keys included in the preamble. We deliberately skip noisy fields
# (height_cm, hr_resting, terrain_access) to keep the preamble compact —
# the LLM can call `update_profile` if it needs to know more, and the
# wizard preamble (built elsewhere) covers profile gaps.
_PROFILE_KEYS_TO_SHOW: tuple[str, ...] = (
    "name",
    "age",
    "experience_level",
    "pace_5k",
    "pace_10k",
    "weekly_days",
    "goal",
)


def _format_profile(profile: dict | None) -> str:
    if not profile:
        return "RUNNER PROFILE: (not set yet)"
    parts: list[str] = []
    for key in _PROFILE_KEYS_TO_SHOW:
        value = profile.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return "RUNNER PROFILE: (no fields filled)"
    return "RUNNER PROFILE: " + ", ".join(parts)


def _format_injuries(injuries: list[dict]) -> str:
    if not injuries:
        return "ACTIVE INJURIES: none acknowledged"
    bits: list[str] = []
    for inj in injuries:
        name = inj.get("name") or "?"
        side = inj.get("side")
        year = inj.get("year")
        suffix = []
        if side:
            suffix.append(str(side))
        if year:
            suffix.append(str(year))
        suffix_str = f" ({', '.join(suffix)})" if suffix else ""
        bits.append(f"{name}{suffix_str}")
    return "ACTIVE INJURIES: " + "; ".join(bits)


def _format_date(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    return str(value) if value is not None else "?"


def _format_checkins(checkins: list[dict]) -> str:
    if not checkins:
        return "RECENT CHECK-INS (last 7 days): none"
    lines = ["RECENT CHECK-INS (last 7 days):"]
    for c in checkins:
        date_str = _format_date(c.get("date"))
        fields: list[str] = []
        if c.get("sleep_quality") is not None:
            fields.append(f"sleep={c['sleep_quality']}")
        if c.get("fatigue") is not None:
            fields.append(f"fatigue={c['fatigue']}")
        if c.get("motivation") is not None:
            fields.append(f"motivation={c['motivation']}")
        pains = c.get("pains") or []
        if pains:
            pains_str = ",".join(
                f"{p.get('location', '?')}:{p.get('severity', '?')}/10" for p in pains
            )
            fields.append(f"pains=[{pains_str}]")
        line = f"- {date_str}"
        if fields:
            line += " " + " ".join(fields)
        lines.append(line)
    return "\n".join(lines)


def _format_workouts(workouts: list[dict]) -> str:
    if not workouts:
        return "RECENT WORKOUTS (last 14 days): none"
    lines = ["RECENT WORKOUTS (last 14 days):"]
    for w in workouts:
        date_str = _format_date(w.get("date"))
        type_str = w.get("type") or "?"
        bits: list[str] = [type_str]
        if w.get("distance_km") is not None:
            bits.append(f"{w['distance_km']}km")
        if w.get("target_pace"):
            bits.append(f"@{w['target_pace']}")
        if w.get("zone"):
            bits.append(w["zone"])
        lines.append(f"- {date_str} " + " ".join(bits))
    return "\n".join(lines)


def _format_coach_note(note: dict | None) -> str:
    if not note:
        return "COACH NOTE: no note yet"
    generated_at = note.get("generated_at")
    when = _format_date(generated_at)
    content = (note.get("content") or "").strip()
    if not content:
        return f"COACH NOTE (generated {when}): (empty)"
    return f"COACH NOTE (generated {when}):\n{content}"


def build_memory_preamble(user_id: str, *, db_module: Any = _default_db) -> str:
    """Return the per-turn memory preamble (profile + recent state + coach note).

    Best-effort: any DB failure on a sub-section collapses to a short
    "(unavailable)" note rather than blocking the turn. Safety/profile
    preambles still get their own pass elsewhere.
    """
    try:
        profile = db_module.load_profile(user_id)
    except Exception:  # noqa: BLE001
        profile = None
    try:
        injuries = db_module.load_active_injuries(user_id)
    except Exception:  # noqa: BLE001
        injuries = []
    try:
        # Checkins are noisier than workouts day-to-day, so we cap them at
        # 7 days (vs 14 for workouts) to keep the preamble tight.
        checkins = db_module.load_checkins_since(user_id, 7)
    except Exception:  # noqa: BLE001
        checkins = []
    try:
        workouts = db_module.load_workouts_since(user_id, 14)
    except Exception:  # noqa: BLE001
        workouts = []
    try:
        note = db_module.load_latest_coach_note(user_id)
    except Exception:  # noqa: BLE001
        note = None

    sections = [
        _format_profile(profile),
        _format_injuries(injuries),
        _format_checkins(checkins),
        _format_workouts(workouts),
        _format_coach_note(note),
    ]
    return "\n\n".join(sections)
