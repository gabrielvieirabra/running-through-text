"""Deterministic safety triggers (Slice 5, ADR 0003).

Numeric safety rules — high pain, high fatigue, week-over-week volume jump,
attempt to compensate for missed workouts, and red medical symptoms — are
implemented as pure functions over structured data. The agent receives the
resulting markers in its context preamble, then applies harm reduction (or
hard refusal, only for red medical symptoms) per the system prompt's Safety
section.

This package exposes:
    - `Marker`: the shape of a single risk marker (a TypedDict).
    - `evaluate`: load latest check-in / recent workouts and run all triggers.
    - The individual trigger functions, for unit testing.
"""

from safety.triggers import (
    Marker,
    compensation_attempt,
    evaluate,
    high_fatigue,
    high_pain,
    red_medical_symptoms,
    volume_jump,
)

__all__ = [
    "Marker",
    "compensation_attempt",
    "evaluate",
    "high_fatigue",
    "high_pain",
    "red_medical_symptoms",
    "volume_jump",
]
