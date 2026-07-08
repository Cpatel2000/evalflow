"""LLM-as-judge scorer.

Loads and validates the versioned judge YAML, renders it with sample fields
plus the response, calls a Provider, and parses the JSON reply. A malformed
reply is retried exactly once with a "reply with only JSON" nudge appended to
the same prompt; if that also fails to parse, the result is state=judge_error
— never a silent 0 (design doc rule 5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import jinja2
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from gradetrail.errors import JudgeError, ProviderError
from gradetrail.providers.base import Provider
from gradetrail.scorers.base import ScoreResult
from gradetrail.spec import JudgeScorer

_JINJA_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)
_NUDGE = "\n\nReply with only the JSON object and no other text."
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_MAX_RAW_REPLY_CHARS = 200


class JudgeFile(BaseModel):
    """A versioned judge prompt file (docs/design/eval-spec.md: judge prompt files)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    output: Literal["score_0_1", "binary"]
    prompt: str


def load_judge_file(path: Path) -> JudgeFile:
    """Load and validate a versioned judge prompt file. Raises JudgeError on failure."""
    if not path.exists():
        raise JudgeError(f"scorer.judge_prompt: {path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise JudgeError(f"{path}: invalid YAML ({e})") from None
    if not isinstance(raw, dict):
        raise JudgeError(f"{path}: judge file must be a YAML mapping")
    try:
        return JudgeFile.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"]) or "judge file"
        raise JudgeError(f"{path}: {loc}: {first['msg']}") from None


def _truncate(text: str, limit: int = _MAX_RAW_REPLY_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text.strip()


def _parse_reply(text: str, output: Literal["score_0_1", "binary"]) -> tuple[float, str]:
    """Parse a raw judge reply into (score, reason). Raises JudgeError on any failure,
    with a truncated copy of the raw reply in the message — it's the only debugging
    surface left once this bubbles up as a judge_error."""
    try:
        data = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        raise JudgeError(f"reply is not valid JSON ({e}): {_truncate(text)!r}") from None
    if not isinstance(data, dict):
        raise JudgeError(f"reply JSON must be an object: {_truncate(text)!r}")
    if "score" not in data or "reason" not in data:
        raise JudgeError(f"reply JSON missing 'score' or 'reason': {_truncate(text)!r}")
    score = data["score"]
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise JudgeError(f"'score' must be a number, got {score!r}: {_truncate(text)!r}")
    score = float(score)
    if output == "binary" and score not in (0.0, 1.0):
        raise JudgeError(
            f"'score' must be 0 or 1 for output=binary, got {score!r}: {_truncate(text)!r}"
        )
    if output == "score_0_1" and not (0.0 <= score <= 1.0):
        raise JudgeError(f"'score' must be within [0, 1], got {score!r}: {_truncate(text)!r}")
    reason = data["reason"]
    if not isinstance(reason, str):
        raise JudgeError(f"'reason' must be a string, got {reason!r}: {_truncate(text)!r}")
    return score, reason


async def _judge_once(
    prompt: str, provider: Provider, params: object, output: Literal["score_0_1", "binary"]
) -> tuple[float, str]:
    """One judge call, retried exactly once (with a nudge) on a parse failure.

    A ProviderError from provider.complete() is not retried here — base.py's
    own retry loop has already been exhausted, and a text nudge can't fix a
    network/rate-limit failure.
    """
    response = await provider.complete(prompt, params)
    try:
        return _parse_reply(response.text, output)
    except JudgeError:
        nudged = await provider.complete(prompt + _NUDGE, params)
        return _parse_reply(nudged.text, output)  # JudgeError propagates on second failure


async def score_judge(
    sample: dict,
    response_text: str,
    scorer: JudgeScorer,
    judge_file: JudgeFile,
    provider: Provider,
) -> ScoreResult:
    """Render the judge prompt, call provider up to scorer.samples times, average.

    Short-circuits on the first judge_error rather than running all `samples`
    and reporting partial agreement (see NOTES.md for the tradeoff).
    """
    prompt = _JINJA_ENV.from_string(judge_file.prompt).render(**sample, response=response_text)
    scores: list[float] = []
    reasons: list[str] = []
    for _ in range(scorer.samples):
        try:
            score, reason = await _judge_once(
                prompt, provider, scorer.model.params, judge_file.output
            )
        except ProviderError as e:
            detail = str(e)
            if e.__cause__ is not None:
                detail = f"{detail} (caused by: {e.__cause__})"
            return ScoreResult(score=0.0, state="judge_error", detail=detail)
        except JudgeError as e:
            return ScoreResult(score=0.0, state="judge_error", detail=str(e))
        scores.append(score)
        reasons.append(reason)
    return ScoreResult(score=sum(scores) / len(scores), state="scored", detail="; ".join(reasons))
