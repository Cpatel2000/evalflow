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

from evalflow.errors import ResultsError
from evalflow.spec import ModelSpec

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


def summarize(results: list[SampleResult], model: ModelSpec, wall_time_s: float) -> RunSummary:
    """Reduce results to a RunSummary. Raises ResultsError on an unrecognized state."""
    prices = PRICING.get((model.provider, model.name))

    n_scored = 0
    n_provider_error = 0
    n_judge_error = 0
    scored_scores: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    cache_hits = 0
    cost = Decimal(0) if prices is not None else None

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
        if r.cached:
            cache_hits += 1
        elif cost is not None:
            input_price, output_price = prices  # type: ignore[misc]
            cost += (Decimal(r.input_tokens or 0) / _MILLION) * input_price
            cost += (Decimal(r.output_tokens or 0) / _MILLION) * output_price

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
