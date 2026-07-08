"""Tests for gradetrail.scorers.deterministic: exact and regex scorers.

Deterministic scorers can never fail — malformed inputs (bad regex, wrong
scorer types) are already rejected at spec load time (gradetrail/spec.py).
These tests exercise scoring behavior only, not validation.
"""

from __future__ import annotations

from gradetrail.scorers.deterministic import score_exact, score_regex
from gradetrail.spec import ExactScorer, RegexScorer

# --- exact -------------------------------------------------------------------


def test_score_exact_match_returns_full_score() -> None:
    scorer = ExactScorer(type="exact", target_field="answer")
    result = score_exact({"answer": "42"}, "42", scorer)
    assert result.score == 1.0
    assert result.state == "scored"


def test_score_exact_mismatch_returns_zero() -> None:
    scorer = ExactScorer(type="exact", target_field="answer")
    result = score_exact({"answer": "42"}, "43", scorer)
    assert result.score == 0.0
    assert result.state == "scored"


def test_score_exact_detail_mentions_expected_and_actual_on_mismatch() -> None:
    scorer = ExactScorer(type="exact", target_field="answer")
    result = score_exact({"answer": "42"}, "43", scorer)
    assert "42" in result.detail
    assert "43" in result.detail


def test_score_exact_strip_normalizes_whitespace() -> None:
    scorer = ExactScorer(type="exact", target_field="answer", normalize=("strip",))
    result = score_exact({"answer": "42"}, "  42  ", scorer)
    assert result.score == 1.0


def test_score_exact_lower_normalizes_case() -> None:
    scorer = ExactScorer(type="exact", target_field="answer", normalize=("lower",))
    result = score_exact({"answer": "YES"}, "yes", scorer)
    assert result.score == 1.0


def test_score_exact_collapse_whitespace_normalizes_internal_spacing() -> None:
    scorer = ExactScorer(type="exact", target_field="answer", normalize=("collapse_whitespace",))
    result = score_exact({"answer": "42 is the answer"}, "42   is  the   answer", scorer)
    assert result.score == 1.0


def test_score_exact_normalize_steps_apply_in_order() -> None:
    scorer = ExactScorer(type="exact", target_field="answer", normalize=("strip", "lower"))
    result = score_exact({"answer": "Yes"}, "  YES  ", scorer)
    assert result.score == 1.0


def test_score_exact_without_normalize_is_case_and_whitespace_sensitive() -> None:
    scorer = ExactScorer(type="exact", target_field="answer")
    result = score_exact({"answer": "Yes"}, "yes", scorer)
    assert result.score == 0.0


# --- regex ---------------------------------------------------------------------


def test_score_regex_match_returns_full_score() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}\s*$")
    result = score_regex({"answer": "4"}, "Solve...\nAnswer: 4", scorer)
    assert result.score == 1.0
    assert result.state == "scored"


def test_score_regex_no_match_returns_zero() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}\s*$")
    result = score_regex({"answer": "4"}, "Solve...\nAnswer: 5", scorer)
    assert result.score == 0.0
    assert result.state == "scored"


def test_score_regex_multiline_flag_lets_dollar_anchor_mid_string() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}\s*$", flags=("MULTILINE",))
    result = score_regex({"answer": "4"}, "Answer: 4\nSome trailing line", scorer)
    assert result.score == 1.0


def test_score_regex_without_multiline_flag_dollar_anchors_end_of_string() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}\s*$")
    result = score_regex({"answer": "4"}, "Answer: 4\nSome trailing line", scorer)
    assert result.score == 0.0


def test_score_regex_ignorecase_flag() -> None:
    scorer = RegexScorer(type="regex", pattern=r"answer: {{ answer }}", flags=("IGNORECASE",))
    result = score_regex({"answer": "4"}, "ANSWER: 4", scorer)
    assert result.score == 1.0


def test_score_regex_dotall_flag_lets_dot_match_newlines() -> None:
    scorer = RegexScorer(type="regex", pattern=r"start.*end", flags=("DOTALL",))
    result = score_regex({}, "start\nmiddle\nend", scorer)
    assert result.score == 1.0


def test_score_regex_without_dotall_flag_dot_does_not_match_newlines() -> None:
    scorer = RegexScorer(type="regex", pattern=r"start.*end")
    result = score_regex({}, "start\nmiddle\nend", scorer)
    assert result.score == 0.0


def test_score_regex_renders_sample_fields_into_pattern() -> None:
    scorer = RegexScorer(type="regex", pattern=r"^{{ answer }}$")
    match_result = score_regex({"answer": "17"}, "17", scorer)
    miss_result = score_regex({"answer": "18"}, "17", scorer)
    assert match_result.score == 1.0
    assert miss_result.score == 0.0


def test_score_regex_detail_contains_matched_text_on_match() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}")
    result = score_regex({"answer": "4"}, "Answer: 4", scorer)
    assert "Answer: 4" in result.detail


def test_score_regex_detail_contains_rendered_pattern_on_miss() -> None:
    scorer = RegexScorer(type="regex", pattern=r"Answer:\s*{{ answer }}")
    result = score_regex({"answer": "4"}, "nope", scorer)
    assert "Answer:" in result.detail
    assert "4" in result.detail
