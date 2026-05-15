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
    """Persist an injury row. Returns the new injury's UUID as a string."""
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO injuries (user_id, name, side, year, status, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, name, side, year, status, notes),
        ).fetchone()
    return str(row[0])


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
