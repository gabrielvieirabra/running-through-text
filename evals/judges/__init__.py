"""LLM-as-judge and delta-baseline helpers (ADR 0005).

The judge measures DELTA, not absolute. A response only fails the PR
gate when its score drops below the recorded baseline — first run for
a given test_id seeds the baseline. Re-record on intentional model
swaps by setting `RTT_RECORD_BASELINES=1`.
"""

from .baseline import (
    assert_judge_no_regression,
    is_recording_baselines,
    load_baseline,
    write_baseline,
)
from .llm_judge import JudgeParseError, llm_judge

__all__ = [
    "JudgeParseError",
    "assert_judge_no_regression",
    "is_recording_baselines",
    "llm_judge",
    "load_baseline",
    "write_baseline",
]
