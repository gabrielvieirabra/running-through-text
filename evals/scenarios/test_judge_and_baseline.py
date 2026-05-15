"""Light unit tests for the LLM-judge / baseline plumbing.

The heavy tests live in their own files and actually call OpenRouter.
Here we mock the judge client end-to-end and verify the contract:

- The judge parses the JSON, clamps to 0-5, raises on garbage.
- `assert_judge_no_regression` records a new baseline on first run.
- Subsequent runs pass when score >= baseline, fail when it drops.
- `RTT_RECORD_BASELINES=1` overwrites the baseline and always passes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evals.judges import (
    JudgeParseError,
    assert_judge_no_regression,
    llm_judge,
    load_baseline,
)
from evals.judges import baseline as baseline_module
from evals.judges import budgets as budgets_module


pytestmark = pytest.mark.light


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_llm_judge_parses_strict_json():
    fake = MagicMock()
    fake.chat.completions.create.return_value = _completion(
        '{"score": 4, "reasoning": "good harm reduction"}'
    )
    out = llm_judge(
        response="ok",
        rubric="dummy",
        client=fake,
        model="anthropic/claude-sonnet-4.6",
    )
    assert out == {
        "score": 4,
        "reasoning": "good harm reduction",
        "model": "anthropic/claude-sonnet-4.6",
    }


def test_llm_judge_extracts_json_when_wrapped_in_prose():
    """Some models prepend a sentence — we still recover the JSON."""
    fake = MagicMock()
    fake.chat.completions.create.return_value = _completion(
        'Sure! Here is the verdict:\n{"score": 3, "reasoning": "borderline"}\n'
    )
    out = llm_judge(response="x", rubric="r", client=fake)
    assert out["score"] == 3


def test_llm_judge_raises_on_unparseable_output():
    fake = MagicMock()
    fake.chat.completions.create.return_value = _completion("totally not json")
    with pytest.raises(JudgeParseError):
        llm_judge(response="x", rubric="r", client=fake)


def test_llm_judge_raises_on_out_of_range_score():
    fake = MagicMock()
    fake.chat.completions.create.return_value = _completion(
        '{"score": 7, "reasoning": "too high"}'
    )
    with pytest.raises(JudgeParseError, match="0..5"):
        llm_judge(response="x", rubric="r", client=fake)


def test_assert_judge_no_regression_records_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_module, "_BASELINES_DIR", tmp_path)
    test_id = "fake_test_record"
    assert load_baseline(test_id) is None
    assert_judge_no_regression(test_id, 3, model="judge-x")
    assert load_baseline(test_id) == 3


def test_assert_judge_no_regression_passes_when_score_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_module, "_BASELINES_DIR", tmp_path)
    test_id = "fake_test_hold"
    assert_judge_no_regression(test_id, 3)
    # Same score — passes.
    assert_judge_no_regression(test_id, 3)
    # Higher score — passes (and does NOT overwrite the baseline; we only
    # overwrite on RTT_RECORD_BASELINES=1).
    assert_judge_no_regression(test_id, 5)
    assert load_baseline(test_id) == 3


def test_assert_judge_no_regression_fails_on_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_module, "_BASELINES_DIR", tmp_path)
    test_id = "fake_test_drop"
    assert_judge_no_regression(test_id, 4)
    with pytest.raises(AssertionError, match="regressed"):
        assert_judge_no_regression(test_id, 3)


def test_record_flag_overwrites_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_module, "_BASELINES_DIR", tmp_path)
    monkeypatch.setenv("RTT_RECORD_BASELINES", "1")
    test_id = "fake_test_overwrite"
    assert_judge_no_regression(test_id, 5)
    # A drop would normally fail, but recording mode replaces it.
    assert_judge_no_regression(test_id, 2)
    assert load_baseline(test_id) == 2


def test_is_recording_baselines_truthy_values(monkeypatch):
    for value in ("1", "true", "yes"):
        monkeypatch.setenv("RTT_RECORD_BASELINES", value)
        assert baseline_module.is_recording_baselines() is True
    monkeypatch.delenv("RTT_RECORD_BASELINES", raising=False)
    assert baseline_module.is_recording_baselines() is False


def test_budgets_load_and_assert(tmp_path, monkeypatch):
    """`load_budget` reads the JSON; `assert_within_budget` fails on overruns."""
    fake_file = tmp_path / "budgets.json"
    fake_file.write_text(
        '{"_schema": "doc", "fake_scenario": {"prompt_tokens": 100, "completion_tokens": 20}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(budgets_module, "_BUDGETS_FILE", fake_file)

    assert budgets_module.load_budget("fake_scenario") == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
    }
    # Within budget — no raise.
    budgets_module.assert_within_budget(
        "fake_scenario", prompt_tokens=80, completion_tokens=10
    )
    # Over budget — raise.
    with pytest.raises(AssertionError, match="prompt_tokens=200"):
        budgets_module.assert_within_budget(
            "fake_scenario", prompt_tokens=200, completion_tokens=10
        )
    with pytest.raises(AssertionError, match="completion_tokens=50"):
        budgets_module.assert_within_budget(
            "fake_scenario", prompt_tokens=10, completion_tokens=50
        )
    with pytest.raises(KeyError):
        budgets_module.load_budget("missing")
