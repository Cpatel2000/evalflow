"""Tests for evalflow.results: SampleResult/RunSummary, summarize(), and JSONL I/O.

Encodes design doc rule 5: every sample terminates in exactly one state
(scored, provider_error, judge_error); summarize() reports counts of each and
computes mean score over scored samples only. Cost math is Decimal-only.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import pytest

from evalflow.errors import ResultsError
from evalflow.results import PRICING, RunSummary, SampleResult, read_jsonl, summarize, write_jsonl
from evalflow.spec import ModelSpec

PRICED_MODEL = ModelSpec(provider="anthropic", name="claude-sonnet-4-6")
UNKNOWN_MODEL = ModelSpec(provider="anthropic", name="claude-nonexistent-model-xyz")


def make_scored(
    sample_id: str,
    score: float,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached: bool = False,
    served_model: str | None = "fake-model",
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        state="scored",
        score=score,
        response_text="some response",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=123.4,
        cached=cached,
        detail="matched 'x'",
        served_model=served_model,
    )


def make_error(
    sample_id: str, state: str, detail: str = "boom", *, cached: bool = False
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        state=state,
        score=None,
        response_text=None if state == "provider_error" else "some response",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        cached=cached,
        detail=detail,
    )


# --- SampleResult / RunSummary shape -------------------------------------------


def test_sample_result_is_frozen_dataclass() -> None:
    result = make_scored("1", 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0  # type: ignore[misc]


def test_run_summary_is_frozen_dataclass() -> None:
    summary = summarize([make_scored("1", 1.0)], PRICED_MODEL, wall_time_s=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.n_samples = 99  # type: ignore[misc]


def test_served_model_defaults_to_none() -> None:
    result = make_error("1", "provider_error")
    assert result.served_model is None


def test_served_model_can_be_set_on_scored_samples() -> None:
    result = make_scored("1", 1.0, served_model="claude-sonnet-4-6-20260115")
    assert result.served_model == "claude-sonnet-4-6-20260115"


def test_run_summary_constructs_directly_with_expected_fields() -> None:
    summary = RunSummary(
        n_samples=2,
        n_scored=1,
        n_provider_error=1,
        n_judge_error=0,
        mean_score=1.0,
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=Decimal("0.01"),
        wall_time_s=5.0,
        cache_hits=0,
    )
    assert summary.n_samples == 2
    assert summary.total_cost_usd == Decimal("0.01")


# --- summarize: closed set of states --------------------------------------------


def test_summarize_unknown_state_raises_instead_of_miscounting() -> None:
    # A typo'd/future state must not be silently dropped from every counter --
    # n_samples would then stop equaling n_scored + n_provider_error + n_judge_error
    # and nobody would notice until the numbers looked wrong in a report.
    results = [make_scored("1", 1.0), make_error("2", "provider-error")]  # hyphen typo
    with pytest.raises(ResultsError, match="2"):
        summarize(results, PRICED_MODEL, wall_time_s=1.0)


def test_summarize_unknown_state_error_names_the_bad_state() -> None:
    results = [make_error("7", "totally-unknown")]
    with pytest.raises(ResultsError, match="totally-unknown"):
        summarize(results, PRICED_MODEL, wall_time_s=1.0)


# --- summarize: counts and states ----------------------------------------------


def test_summarize_counts_per_state() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=10.0)
    assert summary.n_samples == 5
    assert summary.n_scored == 2
    assert summary.n_provider_error == 1
    assert summary.n_judge_error == 2


def test_summarize_wall_time_passthrough() -> None:
    summary = summarize([make_scored("1", 1.0)], PRICED_MODEL, wall_time_s=42.5)
    assert summary.wall_time_s == 42.5


def test_summarize_cache_hits_counts_cached_samples() -> None:
    results = [
        make_scored("1", 1.0, cached=True),
        make_scored("2", 1.0, cached=False),
        make_error("3", "provider_error", cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.cache_hits == 2


# --- summarize: mean score excludes errors -------------------------------------


def test_summarize_mean_score_over_scored_samples_only() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.5),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    # mean of [1.0, 0.5] = 0.75, over n=2 -- NOT diluted to 0.3 by treating the
    # three error samples as zeros, and NOT inflated by excluding them from
    # the denominator while still counting them in the numerator.
    assert summary.mean_score == pytest.approx(0.75)
    assert summary.n_samples == 5
    assert summary.n_scored == 2


def test_summarize_mean_score_exactly_half_with_errors_present() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score == 0.5
    assert summary.n_samples == 5
    assert summary.n_scored == 2


def test_summarize_mean_score_is_none_when_no_scored_samples() -> None:
    results = [make_error("1", "provider_error"), make_error("2", "judge_error")]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score is None


def test_summarize_mean_score_all_scored() -> None:
    results = [make_scored("1", 1.0), make_scored("2", 0.0), make_scored("3", 1.0)]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score == pytest.approx(2 / 3)


# --- summarize: tokens ----------------------------------------------------------


def test_summarize_total_tokens_sum_across_all_samples_including_cached() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=100, output_tokens=50, cached=False),
        make_scored("2", 1.0, input_tokens=200, output_tokens=75, cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 125


def test_summarize_error_samples_contribute_zero_tokens_without_crashing() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=100, output_tokens=50),
        make_error("2", "provider_error"),  # input_tokens/output_tokens are None
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 50


# --- summarize: cost (Decimal only) ----------------------------------------------


def test_summarize_cost_hand_computed() -> None:
    input_price, output_price = PRICING[("anthropic", "claude-sonnet-4-6")]
    assert input_price == Decimal("3.00")
    assert output_price == Decimal("15.00")

    results = [
        make_scored("1", 1.0, input_tokens=200_000, output_tokens=100_000),
        make_scored("2", 1.0, input_tokens=300_000, output_tokens=100_000),
    ]
    # total input = 500_000 -> 0.5M * $3.00  = $1.50
    # total output = 200_000 -> 0.2M * $15.00 = $3.00
    # total                                   = $4.50
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("4.50")
    assert isinstance(summary.total_cost_usd, Decimal)


def test_summarize_cost_is_zero_for_cached_samples_regardless_of_tokens() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000, cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("0")
    # tokens are still counted in the totals even though they cost nothing
    assert summary.total_input_tokens == 1_000_000
    assert summary.total_output_tokens == 1_000_000


def test_summarize_mixed_cached_and_uncached_cost_only_counts_uncached() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=0, cached=True),
        make_scored("2", 1.0, input_tokens=1_000_000, output_tokens=0, cached=False),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("3.00")  # only sample 2's input tokens billed


def test_summarize_unknown_model_gives_none_cost_never_a_crash() -> None:
    results = [make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000)]
    summary = summarize(results, UNKNOWN_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd is None


def test_summarize_empty_results_list() -> None:
    summary = summarize([], PRICED_MODEL, wall_time_s=0.0)
    assert summary.n_samples == 0
    assert summary.mean_score is None
    assert summary.total_cost_usd == Decimal("0")


# --- write_jsonl / read_jsonl ----------------------------------------------------


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0, cached=True),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
    ]
    path = tmp_path / "results.jsonl"
    write_jsonl(results, path)
    round_tripped = read_jsonl(path)
    assert round_tripped == results


def test_round_trip_preserves_none_fields(tmp_path: Path) -> None:
    result = make_error("1", "provider_error")
    assert result.score is None
    assert result.response_text is None
    assert result.input_tokens is None

    path = tmp_path / "results.jsonl"
    write_jsonl([result], path)
    [round_tripped] = read_jsonl(path)
    assert round_tripped.score is None
    assert round_tripped.response_text is None
    assert round_tripped.input_tokens is None
    assert round_tripped.output_tokens is None
    assert round_tripped.latency_ms is None


def test_write_jsonl_produces_one_json_object_per_line(tmp_path: Path) -> None:
    results = [make_scored("1", 1.0), make_scored("2", 0.5), make_scored("3", 0.0)]
    path = tmp_path / "results.jsonl"
    write_jsonl(results, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line parses independently


def test_write_jsonl_field_order_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_jsonl([make_scored("1", 1.0)], path)
    [line] = path.read_text().splitlines()
    keys = list(json.loads(line).keys())
    expected = [f.name for f in dataclasses.fields(SampleResult)]
    assert keys == expected


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    result = make_scored("1", 1.0)
    path.write_text(json.dumps(dataclasses.asdict(result)) + "\n\n")
    assert read_jsonl(path) == [result]
