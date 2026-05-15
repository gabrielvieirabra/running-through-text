"""Baseline file + delta-gating helpers (ADR 0005).

Each judged test owns a JSON file at `evals/baselines/<test_id>.json`:

    {
        "test_id": "<test_id>",
        "score": <int 0..5>,
        "model": "<judge model id>",
        "recorded_at": "<ISO timestamp>"
    }

`assert_judge_no_regression` is the gate. First time a test runs (no
baseline file yet) it writes the file and passes. Subsequent runs pass
when `score >= baseline.score` and fail otherwise. Setting the env var
`RTT_RECORD_BASELINES=1` switches to record mode: the gate overwrites
the baseline with the new score and passes unconditionally — use this
on an intentional model swap.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# `evals/baselines/` next to this file's package.
_BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"


def _baseline_path(test_id: str) -> Path:
    # Test IDs may contain `::` separators (pytest node-id style) and slashes;
    # flatten so they map cleanly to a single JSON file.
    safe = test_id.replace("/", "__").replace("::", "__").replace(":", "_")
    return _BASELINES_DIR / f"{safe}.json"


def is_recording_baselines() -> bool:
    """True when `RTT_RECORD_BASELINES=1` is set in the environment.

    In record mode `assert_judge_no_regression` overwrites the stored
    baseline with the new score instead of asserting against it.
    """
    return os.environ.get("RTT_RECORD_BASELINES", "").strip() in {"1", "true", "yes"}


def load_baseline(test_id: str) -> int | None:
    """Return the recorded baseline score for this test, or None if absent."""
    path = _baseline_path(test_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    score = data.get("score")
    if isinstance(score, int):
        return score
    return None


def write_baseline(
    test_id: str,
    score: int,
    *,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist the baseline file at evals/baselines/<test_id>.json."""
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "test_id": test_id,
        "score": int(score),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if model:
        payload["model"] = model
    if extra:
        payload.update(extra)
    _baseline_path(test_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def assert_judge_no_regression(
    test_id: str,
    score: int,
    *,
    model: str | None = None,
) -> None:
    """Pass when first run (records baseline) or when `score >= baseline`.

    Fails (AssertionError) when `score < baseline`. Per ADR 0005 the gate
    measures DELTA, not absolute — a bad-but-stable score does not break
    the suite, only a regression does. When `RTT_RECORD_BASELINES=1` is
    set, this function always passes and overwrites the baseline.
    """
    if is_recording_baselines():
        write_baseline(test_id, score, model=model)
        return

    baseline = load_baseline(test_id)
    if baseline is None:
        write_baseline(test_id, score, model=model)
        return

    if score < baseline:
        raise AssertionError(
            f"judge score regressed for {test_id}: "
            f"got {score}, baseline {baseline}. "
            f"If this drop is intentional (e.g. a model swap), re-run with "
            f"RTT_RECORD_BASELINES=1 to overwrite the baseline."
        )
