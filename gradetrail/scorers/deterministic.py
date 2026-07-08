"""Deterministic scorers: exact match and regex.

Can never fail — malformed inputs (bad regex, missing sample fields) are
already rejected at spec load time (gradetrail/spec.py: field_validator on
RegexScorer.pattern, and validate_against_dataset() for both `pattern` and
ExactScorer.target_field against sample 0).
"""

from __future__ import annotations

import re

import jinja2

from gradetrail.scorers.base import ScoreResult
from gradetrail.spec import ExactScorer, RegexScorer

_JINJA_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)

_NORMALIZERS = {
    "strip": str.strip,
    "lower": str.lower,
    "collapse_whitespace": lambda s: re.sub(r"\s+", " ", s),
}

_FLAGS = {"IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE, "DOTALL": re.DOTALL}


def _normalize(text: str, steps: tuple[str, ...]) -> str:
    for step in steps:
        text = _NORMALIZERS[step](text)
    return text


def score_exact(sample: dict, response_text: str, scorer: ExactScorer) -> ScoreResult:
    """Exact-match score: 1.0 if response_text equals sample[target_field], else 0.0."""
    expected = _normalize(str(sample[scorer.target_field]), scorer.normalize)
    actual = _normalize(response_text, scorer.normalize)
    if actual == expected:
        return ScoreResult(score=1.0, state="scored", detail=f"matched {expected!r}")
    return ScoreResult(score=0.0, state="scored", detail=f"expected {expected!r}, got {actual!r}")


def score_regex(sample: dict, response_text: str, scorer: RegexScorer) -> ScoreResult:
    """Regex score: 1.0 if the rendered pattern is found in response_text, else 0.0."""
    flags = 0
    for name in scorer.flags:
        flags |= _FLAGS[name]
    pattern = _JINJA_ENV.from_string(scorer.pattern).render(**sample)
    match = re.search(pattern, response_text, flags)
    if match:
        return ScoreResult(score=1.0, state="scored", detail=f"matched {match.group(0)!r}")
    return ScoreResult(score=0.0, state="scored", detail=f"pattern {pattern!r} not found")
