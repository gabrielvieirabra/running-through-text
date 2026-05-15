"""LLM-as-judge (ADR 0005).

A stronger model scores the response against a rubric and returns a
strict JSON object `{"score": int 0-5, "reasoning": str}`. The judge
itself is stateless — the delta gating lives in `baseline.py`.

Default judge model is Sonnet (configurable via the
`OPENROUTER_JUDGE_MODEL` env var). Parse failures raise rather than
silently defaulting to 0 or 5; a noisy judge breaks the eval loudly so
prompt/rubric drift is observable.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"


class JudgeParseError(RuntimeError):
    """Raised when the judge's response cannot be parsed as the required JSON shape."""


JUDGE_SYSTEM_PROMPT = """You are an evaluation judge.

Score the supplied RESPONSE against the supplied RUBRIC on a 0-5 integer scale.

5 = response matches the top of the rubric in full.
0 = response matches the bottom of the rubric, or is irrelevant.
Use the whole scale; do not anchor on 3.

Output ONLY a single JSON object with exactly two keys:
{"score": <integer 0..5>, "reasoning": "<one or two sentences>"}

No prose before or after. No markdown fences. No additional keys."""


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def _model_name(model: str | None) -> str:
    return model or os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extract a single JSON object from the judge's reply.

    The system prompt asks for raw JSON; some models still wrap output in
    code fences or add prose. We try a direct parse first, then fall back
    to slicing between the first `{` and the last `}`. If neither yields
    a parseable object, raise `JudgeParseError`.
    """
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JudgeParseError(f"judge returned no JSON object: {text!r}") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeParseError(
                f"judge returned malformed JSON: {text!r}"
            ) from exc
    if not isinstance(parsed, dict):
        raise JudgeParseError(f"judge JSON was not an object: {parsed!r}")
    return parsed


def _coerce_score(value: Any) -> int:
    if isinstance(value, bool):  # bool is a subclass of int — reject explicitly
        raise JudgeParseError(f"judge score was boolean, not integer: {value!r}")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    else:
        raise JudgeParseError(f"judge score not an integer: {value!r}")
    if score < 0 or score > 5:
        raise JudgeParseError(f"judge score out of 0..5 range: {score}")
    return score


def llm_judge(
    *,
    response: str,
    rubric: str,
    context: dict | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Score `response` against `rubric` using a stronger model.

    `context` (profile snapshot, fired markers, etc.) is serialised into
    the user-role payload as a JSON object so the judge has the inputs
    the agent saw. Returns
        {"score": int 0..5, "reasoning": str, "model": str}.

    Raises `JudgeParseError` if the judge replies with anything other
    than the strict `{"score": <int 0..5>, "reasoning": "<str>"}` shape.
    """
    llm = client or _build_client()
    judge_model = _model_name(model)

    user_payload = json.dumps(
        {
            "rubric": rubric,
            "response": response,
            "context": context or {},
        },
        ensure_ascii=False,
        indent=2,
    )

    completion = llm.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.0,
        top_p=1.0,
    )
    raw = completion.choices[0].message.content or ""
    parsed = _extract_json_object(raw)

    if "score" not in parsed:
        raise JudgeParseError(f"judge JSON missing `score` key: {parsed!r}")
    score = _coerce_score(parsed["score"])
    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {"score": score, "reasoning": reasoning, "model": judge_model}
