"""M3 memory package (Slice 6, ADR 0001).

Closes the agent's memory loop without RAG: per turn the agent's context
includes the runner's profile, recent check-ins and workouts, active
injuries, and a narrative `coach_note` that a separate LLM call rewrites
on deterministic triggers (new workout, risk flag, idle, manual,
bootstrap). The note is the "summary" half of "structured + summary"
M3 — preserves long-term coherence without per-turn cost growth.

Public surface:
    - `rewrite_coach_note`: the separate-LLM-call rewriter.
    - `bootstrap_coach_note`: regenerate from scratch (previous note ignored).
    - `build_memory_preamble`: the per-turn system-message preamble.
"""

from memory.coach_note import bootstrap_coach_note, rewrite_coach_note
from memory.context import build_memory_preamble

__all__ = [
    "bootstrap_coach_note",
    "build_memory_preamble",
    "rewrite_coach_note",
]
