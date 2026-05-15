import json
import os
from contextlib import contextmanager

import psycopg


def _dsn() -> str:
    return os.environ["DATABASE_URL"]


@contextmanager
def connection():
    with psycopg.connect(_dsn()) as conn:
        yield conn


def ensure_user(user_id: str, name: str | None = None) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO users (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (user_id, name),
        )


def load_messages(user_id: str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def save_message(user_id: str, role: str, content: str) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (%s, %s, %s)",
            (user_id, role, content),
        )


def register_checkin(
    user_id: str,
    sleep_quality: int | None = None,
    fatigue: int | None = None,
    motivation: int | None = None,
    pains: list[dict] | None = None,
    notes: str | None = None,
) -> str:
    """Persist a check-in row. Returns the new check-in's UUID as a string.

    `pains` is a list of `{location, severity}` dicts; persisted as JSONB.
    """
    pains_json = json.dumps(pains or [])
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO checkins (user_id, sleep_quality, fatigue, motivation, pains, notes)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            RETURNING id
            """,
            (user_id, sleep_quality, fatigue, motivation, pains_json, notes),
        ).fetchone()
    return str(row[0])


def register_injury(
    user_id: str,
    name: str,
    side: str | None = None,
    year: int | None = None,
    status: str = "active",
    notes: str | None = None,
) -> str:
    """Persist an injury row. Returns the new injury's UUID as a string.

    Also flips `running_profiles.injury_history_acknowledged = TRUE` for the
    user — mentioning a past injury counts as covering the onboarding's
    "injury_history" blocking field. The profile row is upserted so the flag
    can be set even before any other profile field exists.
    """
    with connection() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                INSERT INTO injuries (user_id, name, side, year, status, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, name, side, year, status, notes),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO running_profiles (user_id, injury_history_acknowledged)
                VALUES (%s, TRUE)
                ON CONFLICT (user_id) DO UPDATE
                SET injury_history_acknowledged = TRUE,
                    updated_at = NOW()
                """,
                (user_id,),
            )
    return str(row[0])


# Fields that live on `running_profiles` (everything else routed to `users`).
_PROFILE_FIELDS: tuple[str, ...] = (
    "weight_kg",
    "height_cm",
    "experience_level",
    "pace_5k",
    "pace_10k",
    "longest_run_km",
    "weekly_days",
    "goal",
    "terrain_access",
    "hr_resting",
    "injury_history_acknowledged",
)

# Fields routed to `users` instead of `running_profiles`.
_USER_FIELDS: tuple[str, ...] = ("name", "age")


def upsert_profile(user_id: str, **fields) -> None:
    """Patch-style upsert covering `users` and `running_profiles` in one transaction.

    Only the keys the caller passed are updated. `name` and `age` go to `users`;
    everything else goes to `running_profiles`. Unknown keys raise `ValueError`.
    On `running_profiles` update, `updated_at` is bumped to NOW().
    """
    unknown = set(fields) - set(_PROFILE_FIELDS) - set(_USER_FIELDS)
    if unknown:
        raise ValueError(f"upsert_profile got unknown fields: {sorted(unknown)}")

    user_updates = {k: fields[k] for k in _USER_FIELDS if k in fields}
    profile_updates = {k: fields[k] for k in _PROFILE_FIELDS if k in fields}

    if not user_updates and not profile_updates:
        return

    with connection() as conn:
        with conn.transaction():
            if user_updates:
                set_clause = ", ".join(f"{k} = %s" for k in user_updates)
                values = list(user_updates.values()) + [user_id]
                conn.execute(
                    f"UPDATE users SET {set_clause} WHERE id = %s",
                    values,
                )

            if profile_updates:
                # INSERT with the chosen fields, then on conflict update those
                # same fields plus updated_at. Build columns/placeholders/SET
                # clause dynamically from `profile_updates` keys.
                columns = ["user_id", *profile_updates.keys()]
                placeholders = ", ".join(["%s"] * len(columns))
                set_clause = ", ".join(f"{k} = EXCLUDED.{k}" for k in profile_updates)
                conn.execute(
                    f"""
                    INSERT INTO running_profiles ({", ".join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (user_id) DO UPDATE
                    SET {set_clause}, updated_at = NOW()
                    """,
                    [user_id, *profile_updates.values()],
                )


def load_profile(user_id: str) -> dict | None:
    """Return the merged `users` + `running_profiles` row for the runner, or None."""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT u.name, u.age,
                   p.weight_kg, p.height_cm, p.experience_level,
                   p.pace_5k, p.pace_10k, p.longest_run_km, p.weekly_days,
                   p.goal, p.terrain_access, p.hr_resting,
                   p.injury_history_acknowledged
            FROM users u
            LEFT JOIN running_profiles p ON p.user_id = u.id
            WHERE u.id = %s
            """,
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "name": row[0],
        "age": row[1],
        "weight_kg": row[2],
        "height_cm": row[3],
        "experience_level": row[4],
        "pace_5k": row[5],
        "pace_10k": row[6],
        "longest_run_km": row[7],
        "weekly_days": row[8],
        "goal": row[9],
        "terrain_access": row[10],
        "hr_resting": row[11],
        # `injury_history_acknowledged` is FALSE by default; if the profile row
        # doesn't exist yet the LEFT JOIN gives None — coerce to False.
        "injury_history_acknowledged": bool(row[12]) if row[12] is not None else False,
    }


# Human-readable labels for the 5 blocking onboarding fields.
_BLOCKING_LABELS: dict[str, str] = {
    "pace": "pace (5k or 10k)",
    "weekly_days": "weekly_days",
    "goal": "goal",
    "injury_history": "injury_history",
    "experience_level": "experience_level",
}


def profile_completeness(user_id: str) -> dict:
    """Onboarding gate summary.

    Returns a dict with:
      - `blocking_complete`: bool — all 5 blocking fields filled.
      - `missing_blocking`: list[str] — labels of unfilled blocking fields.
      - `filled_count` / `total_count`: ints for the "N/5" UI.
      - `has_5k_or_10k_pace`: bool.
      - `cooper_needed`: bool — true iff the runner has neither pace AND the
        other blocking fields are filled (so the pace is the only gap left,
        making a Cooper test the right next step).
    """
    profile = load_profile(user_id)
    if profile is None:
        # Runner doesn't exist yet — treat as fully empty.
        return {
            "blocking_complete": False,
            "missing_blocking": list(_BLOCKING_LABELS.values()),
            "filled_count": 0,
            "total_count": len(_BLOCKING_LABELS),
            "has_5k_or_10k_pace": False,
            "cooper_needed": False,
        }

    has_pace = bool(profile.get("pace_5k") or profile.get("pace_10k"))
    # `injury_history` is covered either by an injuries row OR by the explicit
    # "no injuries" acknowledgement — `register_injury` flips the flag too.
    injury_acknowledged = bool(profile.get("injury_history_acknowledged"))
    if not injury_acknowledged:
        with connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM injuries WHERE user_id = %s LIMIT 1",
                (user_id,),
            ).fetchone()
        injury_acknowledged = row is not None

    filled = {
        "pace": has_pace,
        "weekly_days": profile.get("weekly_days") is not None,
        "goal": bool(profile.get("goal")),
        "injury_history": injury_acknowledged,
        "experience_level": bool(profile.get("experience_level")),
    }

    missing_blocking = [_BLOCKING_LABELS[k] for k, ok in filled.items() if not ok]
    blocking_complete = not missing_blocking

    # Cooper is only the right move when the *only* gap is the pace. If other
    # blocking fields are still missing, ask for those first.
    other_blocking_filled = all(ok for k, ok in filled.items() if k != "pace")
    cooper_needed = (not has_pace) and other_blocking_filled

    return {
        "blocking_complete": blocking_complete,
        "missing_blocking": missing_blocking,
        "filled_count": sum(1 for ok in filled.values() if ok),
        "total_count": len(filled),
        "has_5k_or_10k_pace": has_pace,
        "cooper_needed": cooper_needed,
    }


def register_workout(
    user_id: str,
    type: str,
    *,
    target_pace: str | None = None,
    zone: str | None = None,
    distance_km: float | None = None,
    duration_min: float | None = None,
    perceived_effort: int | None = None,
    notes: str | None = None,
    date: str | None = None,
) -> str:
    """Persist a realized workout row. Returns the new workout's UUID as a string.

    `type` must match the canonical enum in schema.sql (PT-BR domain terms).
    When `date` is None, the DB's `CURRENT_DATE` default is used.
    """
    if date is None:
        with connection() as conn:
            row = conn.execute(
                """
                INSERT INTO workouts (
                    user_id, type, target_pace, zone,
                    distance_km, duration_min, perceived_effort, notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    type,
                    target_pace,
                    zone,
                    distance_km,
                    duration_min,
                    perceived_effort,
                    notes,
                ),
            ).fetchone()
    else:
        with connection() as conn:
            row = conn.execute(
                """
                INSERT INTO workouts (
                    user_id, type, target_pace, zone,
                    distance_km, duration_min, perceived_effort, notes, date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    type,
                    target_pace,
                    zone,
                    distance_km,
                    duration_min,
                    perceived_effort,
                    notes,
                    date,
                ),
            ).fetchone()
    return str(row[0])


def load_recent_workouts(user_id: str, limit: int = 10) -> list[dict]:
    """Most recent realized workouts first, capped at `limit`."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, date, type, target_pace, zone,
                   distance_km, duration_min, perceived_effort, notes, created_at
            FROM workouts
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": str(row[0]),
            "date": row[1],
            "type": row[2],
            "target_pace": row[3],
            "zone": row[4],
            "distance_km": row[5],
            "duration_min": row[6],
            "perceived_effort": row[7],
            "notes": row[8],
            "created_at": row[9],
        }
        for row in rows
    ]


def recent_volume(user_id: str, days: int) -> dict:
    """Sum `distance_km` and count workouts over the last `days` days.

    Window is `date >= CURRENT_DATE - (days - 1)` so `days=1` covers today,
    `days=7` covers a rolling week including today. Returns zeros (not
    None) when the runner has no workouts in the window — easier for the
    LLM to reason about.
    """
    with connection() as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(distance_km), 0) AS total_km,
                COUNT(*) AS workout_count
            FROM workouts
            WHERE user_id = %s
              AND date >= CURRENT_DATE - (%s::int - 1)
            """,
            (user_id, days),
        ).fetchone()
    total_km = float(row[0]) if row and row[0] is not None else 0.0
    workout_count = int(row[1]) if row and row[1] is not None else 0
    return {
        "days": int(days),
        "total_km": total_km,
        "workout_count": workout_count,
    }


def load_active_injuries(user_id: str) -> list[dict]:
    """Return active injuries (status='active') for the runner, newest first."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, side, year, status, notes, created_at
            FROM injuries
            WHERE user_id = %s AND status = 'active'
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "id": str(r[0]),
            "name": r[1],
            "side": r[2],
            "year": r[3],
            "status": r[4],
            "notes": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def load_checkins_since(user_id: str, days: int) -> list[dict]:
    """Check-ins within the last `days` days (date >= today - (days-1)), newest first."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, date, sleep_quality, fatigue, motivation, pains, notes, created_at
            FROM checkins
            WHERE user_id = %s
              AND date >= CURRENT_DATE - (%s::int - 1)
            ORDER BY date DESC, created_at DESC
            """,
            (user_id, days),
        ).fetchall()
    return [
        {
            "id": str(r[0]),
            "date": r[1],
            "sleep_quality": r[2],
            "fatigue": r[3],
            "motivation": r[4],
            "pains": r[5],
            "notes": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]


def load_workouts_since(user_id: str, days: int) -> list[dict]:
    """Workouts within the last `days` days (date >= today - (days-1)), newest first."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, date, type, target_pace, zone,
                   distance_km, duration_min, perceived_effort, notes, created_at
            FROM workouts
            WHERE user_id = %s
              AND date >= CURRENT_DATE - (%s::int - 1)
            ORDER BY date DESC, created_at DESC
            """,
            (user_id, days),
        ).fetchall()
    return [
        {
            "id": str(r[0]),
            "date": r[1],
            "type": r[2],
            "target_pace": r[3],
            "zone": r[4],
            "distance_km": r[5],
            "duration_min": r[6],
            "perceived_effort": r[7],
            "notes": r[8],
            "created_at": r[9],
        }
        for r in rows
    ]


def save_coach_note(user_id: str, content: str, trigger: str) -> str:
    """Persist a coach note row. Returns the new note's UUID as a string.

    `trigger` must be one of the five values enforced by the schema CHECK:
    `new_workout`, `risk_flag`, `idle`, `manual`, `bootstrap`. The note's
    `generated_at` is set by the DB default (NOW()).
    """
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO coach_notes (user_id, content, trigger)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (user_id, content, trigger),
        ).fetchone()
    return str(row[0])


def load_latest_coach_note(user_id: str) -> dict | None:
    """Return the most recent coach note row for a runner, or None.

    Result shape: `{id, content, generated_at, trigger}`.
    """
    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, content, generated_at, trigger
            FROM coach_notes
            WHERE user_id = %s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "content": row[1],
        "generated_at": row[2],
        "trigger": row[3],
    }


def days_since_last_coach_note(user_id: str) -> int | None:
    """Number of whole days since the runner's last coach note.

    Returns None when the runner has no coach note yet — the caller should
    interpret that as "bootstrap territory", not "0 days". Computed in SQL so
    the runtime's clock skew vs the DB's clock doesn't matter.
    """
    with connection() as conn:
        row = conn.execute(
            """
            SELECT EXTRACT(EPOCH FROM (NOW() - generated_at)) / 86400.0
            FROM coach_notes
            WHERE user_id = %s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    # Truncate to whole days — "7+ days" reads off `>= 7`, not `>= 7.0`.
    return int(row[0])


def load_recent_checkins(user_id: str, limit: int = 10) -> list[dict]:
    """Most recent check-ins first, capped at `limit`. Pains arrive as a Python list."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, date, sleep_quality, fatigue, motivation, pains, notes, created_at
            FROM checkins
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": str(row[0]),
            "date": row[1],
            "sleep_quality": row[2],
            "fatigue": row[3],
            "motivation": row[4],
            "pains": row[5],
            "notes": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]
