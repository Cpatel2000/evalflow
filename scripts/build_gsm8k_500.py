"""Build examples/data/gsm8k_500.jsonl from the GSM8K test split.

Requires the `datasets` library (pip install datasets) -- script-only, NOT an
gradetrail dependency, deliberately not added to pyproject.toml. This script is
run once to regenerate the file; gradetrail itself never imports `datasets`.

Each row gets an `answer_pattern` field alongside `answer`: the same number
with optional commas at thousands boundaries (e.g. "114200" -> "114,?200",
matching either "114200" or "114,200"). 52/500 answers in this slice are >=4
digits, where a model could plausibly write either form -- see NOTES.md and
examples/gsm8k_500.yaml, which uses answer_pattern (not answer) in its regex.
"""

from __future__ import annotations

import json
import re

from datasets import load_dataset

# Must stay in sync with examples/gsm8k_500.yaml's scorer.pattern. The trailing
# `\.?` tolerates a sentence-period after the number ("Answer: 25.") without
# permitting a decimal continuation ("Answer: 25.5") -- see _verify_pattern
# below and NOTES.md for why this lives in the spec's literal template rather
# than in answer_pattern (it's a universal formatting concern, not a
# per-sample-value one like the comma grouping is).
_SPEC_PATTERN_TEMPLATE = r"Answer:\s*\$?{answer_pattern}\.?\s*$"


def _comma_tolerant_pattern(answer: str) -> str:
    """`answer` (a plain integer string, optionally negative) as a regex
    fragment that also accepts thousands-comma formatting: "2125" ->
    "2,?125", matching both "2125" and "2,125"."""
    sign = ""
    digits = answer
    if digits.startswith("-"):
        sign, digits = "-", digits[1:]
    grouped: list[str] = []
    for i, ch in enumerate(reversed(digits)):
        if i and i % 3 == 0:
            grouped.append(",?")
        grouped.append(ch)
    return sign + "".join(reversed(grouped))


def _verify_pattern() -> None:
    """Regex-level check of the trailing-period fix against the exact template
    the spec renders. Must hold: a trailing sentence-period matches, but a
    real decimal continuation, or a longer number with the same prefix, does
    not -- run on every invocation of this script, not just once by hand."""
    pattern = _SPEC_PATTERN_TEMPLATE.format(answer_pattern=_comma_tolerant_pattern("25"))
    cases = [
        ("Answer: 25", True),
        ("Answer: 25.", True),
        ("Answer: 25.5", False),
        ("Answer: 250", False),
    ]
    for text, expected in cases:
        matched = re.search(pattern, text) is not None
        assert matched == expected, f"{text!r}: expected match={expected}, got {matched}"


def main() -> None:
    _verify_pattern()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open("examples/data/gsm8k_500.jsonl", "w") as f:
        for i, row in enumerate(ds.select(range(500))):
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            sample = {
                "id": str(i + 1),
                "question": row["question"],
                "answer": answer,
                "answer_pattern": _comma_tolerant_pattern(answer),
            }
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
