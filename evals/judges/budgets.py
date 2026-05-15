"""Token-budget helpers for cost-category eval tests (ADR 0005).

Budgets live in `evals/budgets.json`, keyed by a short scenario id:

    {
        "extraction_register_checkin": {
            "prompt_tokens": 5000,
            "completion_tokens": 500
        }
    }

The contract is intentionally tight — when a prompt change blows the
budget the test fails and the dev makes a deliberate choice (raise the
budget in the same commit, or trim the prompt). Same delta posture as
the LLM-judge baselines: numbers move on purpose, not by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BUDGETS_FILE = Path(__file__).resolve().parent.parent / "budgets.json"


def load_budget(scenario_id: str) -> dict[str, int]:
    """Return the budget for `scenario_id`. Missing entry raises KeyError.

    Top-level keys starting with `_` are documentation (e.g. `_schema`)
    and are filtered out.
    """
    if not _BUDGETS_FILE.exists():
        raise KeyError(f"no budgets.json at {_BUDGETS_FILE}")
    data: dict[str, Any] = json.loads(_BUDGETS_FILE.read_text(encoding="utf-8"))
    if scenario_id not in data or scenario_id.startswith("_"):
        raise KeyError(f"no budget recorded for scenario {scenario_id!r}")
    budget = data[scenario_id]
    if not isinstance(budget, dict):
        raise KeyError(f"budget for {scenario_id!r} not an object")
    return {k: int(v) for k, v in budget.items() if isinstance(v, (int, float))}


def assert_within_budget(
    scenario_id: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Fail when measured token counts exceed the recorded budget.

    Raises AssertionError with a message naming which leg blew up, so the
    failure is actionable without scrolling through assertion repr.
    """
    budget = load_budget(scenario_id)
    failures: list[str] = []
    p_cap = budget.get("prompt_tokens")
    c_cap = budget.get("completion_tokens")
    if p_cap is not None and prompt_tokens > p_cap:
        failures.append(f"prompt_tokens={prompt_tokens} > cap={p_cap}")
    if c_cap is not None and completion_tokens > c_cap:
        failures.append(f"completion_tokens={completion_tokens} > cap={c_cap}")
    if failures:
        raise AssertionError(
            f"cost budget exceeded for {scenario_id}: " + "; ".join(failures)
        )
