"""Results model, JSONL writing, and run summary stats.

SampleResult is the per-sample record; every sample terminates in exactly one
state ("scored" | "provider_error" | "judge_error", design doc rule 5).
summarize() reduces a list of them to a RunSummary, including a Decimal cost
estimate looked up from PRICING.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from gradetrail.errors import ResultsError
from gradetrail.spec import ModelSpec

_MILLION = Decimal(1_000_000)

# Prices as of July 2026, USD per million tokens: (input, output).
# Update when adding models.
PRICING: dict[tuple[str, str], tuple[Decimal, Decimal]] = {
    ("anthropic", "claude-opus-4-8"): (Decimal("5.00"), Decimal("25.00")),
    ("anthropic", "claude-sonnet-4-6"): (Decimal("3.00"), Decimal("15.00")),
    ("anthropic", "claude-haiku-4-5"): (Decimal("0.25"), Decimal("1.25")),
    ("openai", "gpt-5.1"): (Decimal("2.50"), Decimal("10.00")),
    ("openai", "gpt-5.1-mini"): (Decimal("0.40"), Decimal("1.60")),
    ("openai", "gpt-5.1-nano"): (Decimal("0.10"), Decimal("0.40")),
}


@dataclass(frozen=True)
class SampleResult:
    """One sample's outcome for the run's manifest/JSONL output."""

    sample_id: str
    state: Literal["scored", "provider_error", "judge_error"]
    score: float | None  # None unless state == "scored"
    response_text: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    cached: bool
    detail: str | None  # error detail or scorer detail
    served_model: str | None = None  # API-reported model; None on error states
    # Only set for a judge scorer (None otherwise). Judge calls are never
    # cached, so these are real, always-billed tokens -- see summarize().
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None


@dataclass(frozen=True)
class RunSummary:
    """Aggregate stats for a run, reduced from its SampleResults."""

    n_samples: int
    n_scored: int
    n_provider_error: int
    n_judge_error: int
    mean_score: float | None  # over scored samples only; None if none scored
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal | None  # None if (provider, model) has no pricing entry
    wall_time_s: float
    cache_hits: int
    total_judge_input_tokens: int = 0
    total_judge_output_tokens: int = 0
    # Human-readable "<role> model <provider>/<name>" entries for every model
    # that made total_cost_usd None. Empty whenever total_cost_usd is a
    # Decimal. Lets a summary say *why* cost is unknown instead of just that
    # it is -- see cli.py's _print_summary.
    cost_unpriced_models: tuple[str, ...] = ()
    # Set by the runner (not computed here) when the run stopped early
    # because the first several samples all failed identically -- the shared
    # failure detail, or None if the run ran to completion normally.
    aborted_reason: str | None = None


def summarize(
    results: list[SampleResult],
    model: ModelSpec,
    wall_time_s: float,
    *,
    judge_model: ModelSpec | None = None,
    aborted_reason: str | None = None,
) -> RunSummary:
    """Reduce results to a RunSummary. Raises ResultsError on an unrecognized state.

    judge_model is the judge scorer's own model (None for a non-judge eval).
    total_cost_usd is strict: if either the primary model or (when present)
    the judge model has no PRICING entry, the whole total is None rather than
    silently summing only the priced side -- same loud-failure-over-silent-
    undercount precedent as ResponseCache.put()'s JSON-serializability check.
    cost_unpriced_models then names exactly which model(s) caused that.

    Judge cost is billed per sample regardless of r.cached: `cached` reflects
    only the primary response's cache status, and judge calls are never
    cached (see NOTES.md), so a fully-cached primary run with a judge scorer
    still has real, nonzero cost from the judge calls alone.

    aborted_reason is a pure pass-through onto RunSummary: the runner (not
    this function) detects an early-abort and already knows the shared
    failure detail, so there is nothing for summarize() itself to compute.
    """
    primary_prices = PRICING.get((model.provider, model.name))
    judge_prices = PRICING.get((judge_model.provider, judge_model.name)) if judge_model else None

    cost_unpriced: list[str] = []
    if primary_prices is None:
        cost_unpriced.append(f"primary model {model.provider}/{model.name}")
    if judge_model is not None and judge_prices is None:
        cost_unpriced.append(f"judge model {judge_model.provider}/{judge_model.name}")

    n_scored = 0
    n_provider_error = 0
    n_judge_error = 0
    scored_scores: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_judge_input_tokens = 0
    total_judge_output_tokens = 0
    cache_hits = 0
    cost: Decimal | None = Decimal(0) if not cost_unpriced else None

    for r in results:
        if r.state == "scored":
            n_scored += 1
            scored_scores.append(r.score)  # type: ignore[arg-type]
        elif r.state == "provider_error":
            n_provider_error += 1
        elif r.state == "judge_error":
            n_judge_error += 1
        else:
            raise ResultsError(
                f"sample {r.sample_id!r}: unknown state {r.state!r} "
                "(expected one of 'scored', 'provider_error', 'judge_error')"
            )

        total_input_tokens += r.input_tokens or 0
        total_output_tokens += r.output_tokens or 0
        total_judge_input_tokens += r.judge_input_tokens or 0
        total_judge_output_tokens += r.judge_output_tokens or 0
        if r.cached:
            cache_hits += 1
        if cost is not None:
            if not r.cached:
                input_price, output_price = primary_prices  # type: ignore[misc]
                cost += (Decimal(r.input_tokens or 0) / _MILLION) * input_price
                cost += (Decimal(r.output_tokens or 0) / _MILLION) * output_price
            if judge_model is not None:
                judge_input_price, judge_output_price = judge_prices  # type: ignore[misc]
                cost += (Decimal(r.judge_input_tokens or 0) / _MILLION) * judge_input_price
                cost += (Decimal(r.judge_output_tokens or 0) / _MILLION) * judge_output_price

    mean_score = sum(scored_scores) / len(scored_scores) if scored_scores else None

    return RunSummary(
        n_samples=len(results),
        n_scored=n_scored,
        n_provider_error=n_provider_error,
        n_judge_error=n_judge_error,
        mean_score=mean_score,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=cost,
        wall_time_s=wall_time_s,
        cache_hits=cache_hits,
        total_judge_input_tokens=total_judge_input_tokens,
        total_judge_output_tokens=total_judge_output_tokens,
        cost_unpriced_models=tuple(cost_unpriced),
        aborted_reason=aborted_reason,
    )


def write_jsonl(results: list[SampleResult], path: str | Path) -> None:
    """Write one JSON object per line, one per SampleResult, in field-declaration order."""
    with Path(path).open("w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def read_jsonl(path: str | Path) -> list[SampleResult]:
    """Read SampleResults written by write_jsonl(). Round-trips exactly."""
    results = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            results.append(SampleResult(**json.loads(line)))
    return results
