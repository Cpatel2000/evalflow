"""Tests for gradetrail.runner.local.LocalRunner: the integration point.

A fake Provider stands in for the SDK (never a real API call). Real
ResponseCache is used against a tmp_path SQLite file -- cache.py already has
its own unit tests, so here it's exercised as a real dependency, only the
Provider boundary is faked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gradetrail.errors import JudgeError, ProviderError
from gradetrail.providers.base import ProviderResponse
from gradetrail.results import RunSummary
from gradetrail.runner.local import _ABORT_THRESHOLD, _SKIPPED_DETAIL, LocalRunner
from gradetrail.spec import (
    DatasetSpec,
    EvalSpec,
    ExactScorer,
    JudgeScorer,
    ModelSpec,
    RegexScorer,
    RunSpec,
)

VALID_JUDGE_YAML = """
version: 1
output: score_0_1
prompt: |
  Question: {{ question }}
  Expected: {{ answer }}
  Response: {{ response }}

  Reply with only a JSON object: {"score": <0 or 1>, "reason": "<one sentence>"}
"""


class FakeProvider:
    """Fake Provider: fixed reply text, optional per-prompt failure predicate.

    Tracks every prompt seen and the max number of concurrent .complete()
    calls in flight, so tests can assert on both call counts and concurrency.
    """

    def __init__(
        self,
        *,
        reply_text: str = "42",
        work_s: float = 0.01,
        fail_when: object = None,  # Callable[[str], bool] | None
    ) -> None:
        self.reply_text = reply_text
        self.work_s = work_s
        self.fail_when = fail_when
        self.calls: list[str] = []
        self._in_flight = 0
        self.max_in_flight = 0

    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        self.calls.append(prompt)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self.work_s)
            if self.fail_when is not None and self.fail_when(prompt):
                raise ProviderError(f"fake provider: simulated failure for prompt {prompt!r}")
            return ProviderResponse(
                text=self.reply_text,
                input_tokens=10,
                output_tokens=5,
                latency_ms=1.0,
                model="fake-model",
            )
        finally:
            self._in_flight -= 1


def make_spec(
    tmp_path: Path,
    *,
    samples: list[dict] | None = None,
    scorer: ExactScorer | RegexScorer | JudgeScorer | None = None,
    model: ModelSpec | None = None,
    concurrency: int = 8,
) -> EvalSpec:
    dataset_path = tmp_path / "data.jsonl"
    rows = samples if samples is not None else [{"id": "1", "question": "2+2?", "answer": "42"}]
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows))
    return EvalSpec(
        name="test-eval",
        dataset=DatasetSpec(path=str(dataset_path)),
        prompt="{{ question }}",
        model=model or ModelSpec(provider="anthropic", name="claude-sonnet-4-6"),
        scorer=scorer or ExactScorer(type="exact", target_field="answer"),
        run=RunSpec(concurrency=concurrency, max_retries=0, timeout_s=5.0),
        base_dir=tmp_path,
    )


# --- happy path ------------------------------------------------------------------


async def test_happy_path_all_samples_scored(tmp_path: Path) -> None:
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(5)]
    spec = make_spec(tmp_path, samples=rows)
    fake = FakeProvider(reply_text="42")
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    assert len(results) == 5
    assert all(r.state == "scored" for r in results)
    assert summary.n_scored == 5
    assert summary.mean_score == 1.0
    assert summary.cache_hits == 0
    assert len(fake.calls) == 5


# --- mixed failures never abort the run -------------------------------------------


async def test_mixed_provider_failures_do_not_abort_the_run(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "question": "ok1", "answer": "42"},
        {"id": "2", "question": "FAIL", "answer": "42"},
        {"id": "3", "question": "ok3", "answer": "42"},
    ]
    spec = make_spec(tmp_path, samples=rows)
    fake = FakeProvider(reply_text="42", fail_when=lambda prompt: "FAIL" in prompt)
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    states = {r.sample_id: r.state for r in results}
    assert len(results) == 3
    assert states == {"1": "scored", "2": "provider_error", "3": "scored"}
    assert summary.n_scored == 2
    assert summary.n_provider_error == 1
    # sibling 2's failure carries the error in detail, not silently dropped
    failed = next(r for r in results if r.sample_id == "2")
    assert failed.detail is not None
    assert "simulated failure" in failed.detail


async def test_unexpected_exception_in_one_sample_does_not_abort_the_run(
    tmp_path: Path,
) -> None:
    # An error that is *not* ProviderError (e.g. a bug) must still be isolated
    # per-sample -- there is no code path where one bad sample crashes the run.
    rows = [
        {"id": "1", "question": "ok1", "answer": "42"},
        {"id": "2", "question": "BOOM", "answer": "42"},
    ]

    class ExplodingProvider:
        async def complete(self, prompt: str, params: object) -> ProviderResponse:
            if "BOOM" in prompt:
                raise RuntimeError("something unrelated broke")
            return ProviderResponse(
                text="42", input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake-model"
            )

    spec = make_spec(tmp_path, samples=rows)
    exploding = ExplodingProvider()
    runner = LocalRunner(
        cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: exploding
    )

    results, summary = await runner.run(spec)

    assert len(results) == 2
    states = {r.sample_id: r.state for r in results}
    assert states["1"] == "scored"
    assert states["2"] == "provider_error"
    assert summary.n_provider_error == 1


# --- cache hit path ----------------------------------------------------------------


async def test_cache_hit_skips_provider_call(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, samples=[{"id": "1", "question": "2+2?", "answer": "42"}])
    cache_path = tmp_path / "cache.sqlite"

    fake1 = FakeProvider(reply_text="42")
    runner1 = LocalRunner(cache_path=cache_path, provider_factory=lambda m, r: fake1)
    results1, _ = await runner1.run(spec)
    assert len(fake1.calls) == 1
    assert results1[0].cached is False

    fake2 = FakeProvider(reply_text="should never be used")
    runner2 = LocalRunner(cache_path=cache_path, provider_factory=lambda m, r: fake2)
    results2, _ = await runner2.run(spec)
    assert len(fake2.calls) == 0  # cache hit -- provider never called
    assert results2[0].cached is True
    assert results2[0].response_text == "42"  # came from cache, not fake2's reply


# --- re-score from cache ------------------------------------------------------------


async def test_rescoring_from_cache_when_regex_scorer_changes(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.sqlite"
    sample = {"id": "1", "question": "2+2?", "answer": "42"}

    spec1 = make_spec(tmp_path, samples=[sample], scorer=RegexScorer(type="regex", pattern=r"42"))
    fake1 = FakeProvider(reply_text="The answer is 42.")
    runner1 = LocalRunner(cache_path=cache_path, provider_factory=lambda m, r: fake1)
    results1, _ = await runner1.run(spec1)
    assert results1[0].state == "scored"
    assert results1[0].score == 1.0
    assert len(fake1.calls) == 1

    # Same prompt (same "question"), different regex -- must rescore the
    # cached response, not re-call the provider.
    spec2 = make_spec(tmp_path, samples=[sample], scorer=RegexScorer(type="regex", pattern=r"99"))
    fake2 = FakeProvider(reply_text="should never be used")
    runner2 = LocalRunner(cache_path=cache_path, provider_factory=lambda m, r: fake2)
    results2, _ = await runner2.run(spec2)
    assert len(fake2.calls) == 0
    assert results2[0].state == "scored"
    assert results2[0].score == 0.0  # rescored with the new pattern
    assert results2[0].response_text == "The answer is 42."  # response came from cache


# --- concurrency ---------------------------------------------------------------------


async def test_concurrency_is_bounded_by_spec_run_concurrency(tmp_path: Path) -> None:
    n_samples = 20
    concurrency = 3
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(n_samples)]
    spec = make_spec(tmp_path, samples=rows, concurrency=concurrency)
    fake = FakeProvider(reply_text="42", work_s=0.02)
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, _ = await runner.run(spec)

    assert len(results) == n_samples
    assert fake.max_in_flight <= concurrency
    assert fake.max_in_flight > 1  # concurrency actually happened, not accidentally serial
    # gather() preserves input order regardless of completion order
    assert [r.sample_id for r in results] == [str(i) for i in range(n_samples)]


# --- judge scorer wiring -------------------------------------------------------------


async def test_judge_scorer_uses_a_separate_provider_for_the_judge_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "judge.yaml").write_text(VALID_JUDGE_YAML)
    sample = {"id": "1", "question": "2+2?", "answer": "4"}
    spec = make_spec(
        tmp_path,
        samples=[sample],
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    main_fake = FakeProvider(reply_text="4")
    judge_fake = FakeProvider(reply_text='{"score": 1, "reason": "correct"}')

    def factory(model: ModelSpec, run: RunSpec) -> FakeProvider:
        return judge_fake if model.name == "judge-model" else main_fake

    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=factory)
    results, _ = await runner.run(spec)

    assert results[0].state == "scored"
    assert results[0].score == 1.0
    assert len(main_fake.calls) == 1
    assert len(judge_fake.calls) == 1


async def test_judge_error_does_not_abort_the_run(tmp_path: Path) -> None:
    (tmp_path / "judge.yaml").write_text(VALID_JUDGE_YAML)
    rows = [
        {"id": "1", "question": "2+2?", "answer": "4"},
        {"id": "2", "question": "3+3?", "answer": "6"},
    ]
    spec = make_spec(
        tmp_path,
        samples=rows,
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    main_fake = FakeProvider(reply_text="4")
    judge_fake = FakeProvider(reply_text="not json, ever")  # always malformed

    def factory(model: ModelSpec, run: RunSpec) -> FakeProvider:
        return judge_fake if model.name == "judge-model" else main_fake

    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=factory)
    results, summary = await runner.run(spec)

    assert len(results) == 2
    assert all(r.state == "judge_error" for r in results)
    assert summary.n_judge_error == 2


async def test_invalid_judge_file_aborts_before_any_sample_runs(tmp_path: Path) -> None:
    (tmp_path / "judge.yaml").write_text("version: not-an-int\noutput: score_0_1\nprompt: hi\n")
    spec = make_spec(
        tmp_path,
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    fake = FakeProvider(reply_text="should never be called")
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    with pytest.raises(JudgeError):
        await runner.run(spec)
    assert len(fake.calls) == 0


# --- judge token accounting -----------------------------------------------------------


async def test_judge_scorer_populates_judge_tokens_on_scored_sample(tmp_path: Path) -> None:
    (tmp_path / "judge.yaml").write_text(VALID_JUDGE_YAML)
    sample = {"id": "1", "question": "2+2?", "answer": "4"}
    spec = make_spec(
        tmp_path,
        samples=[sample],
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    main_fake = FakeProvider(reply_text="4")
    judge_fake = FakeProvider(reply_text='{"score": 1, "reason": "correct"}')

    def factory(model: ModelSpec, run: RunSpec) -> FakeProvider:
        return judge_fake if model.name == "judge-model" else main_fake

    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=factory)
    results, summary = await runner.run(spec)

    assert results[0].judge_input_tokens == 10
    assert results[0].judge_output_tokens == 5
    assert summary.total_judge_input_tokens == 10
    assert summary.total_judge_output_tokens == 5


async def test_non_judge_scorer_leaves_judge_token_fields_none(tmp_path: Path) -> None:
    rows = [{"id": "1", "question": "2+2?", "answer": "42"}]
    spec = make_spec(tmp_path, samples=rows)  # default scorer is ExactScorer
    fake = FakeProvider(reply_text="42")
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    assert results[0].judge_input_tokens is None
    assert results[0].judge_output_tokens is None
    assert summary.total_judge_input_tokens == 0
    assert summary.total_judge_output_tokens == 0


async def test_judge_error_sample_still_reports_judge_tokens_consumed(tmp_path: Path) -> None:
    (tmp_path / "judge.yaml").write_text(VALID_JUDGE_YAML)
    rows = [{"id": "1", "question": "2+2?", "answer": "4"}]
    spec = make_spec(
        tmp_path,
        samples=rows,
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    main_fake = FakeProvider(reply_text="4")
    judge_fake = FakeProvider(reply_text="not json, ever")  # always malformed -> 2 calls/sample

    def factory(model: ModelSpec, run: RunSpec) -> FakeProvider:
        return judge_fake if model.name == "judge-model" else main_fake

    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=factory)
    results, summary = await runner.run(spec)

    assert results[0].state == "judge_error"
    # the malformed first call and its one nudge retry both cost real tokens
    assert results[0].judge_input_tokens == 20
    assert results[0].judge_output_tokens == 10
    assert summary.total_judge_input_tokens == 20
    assert summary.total_judge_output_tokens == 10


# --- fail-fast on uniform fatal errors --------------------------------------------------


class StaggeredProvider:
    """Like FakeProvider, but each call's delay is proportional to a numeric
    suffix parsed out of the prompt ("q7" -> 7 * base_delay_s), so completion
    order is fully deterministic even under real concurrency. FakeProvider's
    single uniform work_s can't guarantee this: two calls with the same delay
    have no guaranteed relative completion order, and these tests need to pin
    exactly which N complete first, which are still genuinely in flight at
    the moment of cancellation, and whether original order survives a
    reversal of completion order.
    """

    def __init__(self, *, base_delay_s: float = 0.02, fail: bool = True) -> None:
        self.base_delay_s = base_delay_s
        self.fail = fail
        self.calls: list[str] = []

    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        self.calls.append(prompt)
        index = int(prompt.removeprefix("q"))
        await asyncio.sleep(self.base_delay_s * index)
        if self.fail:
            # Deliberately NOT prompt-specific: a real uniform fatal error
            # (missing API key, no-credits account) reports the identical
            # message regardless of which sample triggered it.
            raise ProviderError("fake provider: simulated identical fatal failure")
        return ProviderResponse(
            text="42", input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake-model"
        )


def _staggered_rows(n: int) -> list[dict]:
    return [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(1, n + 1)]


async def test_uniform_fatal_errors_abort_the_run_early(tmp_path: Path) -> None:
    n_samples = 20
    spec = make_spec(tmp_path, samples=_staggered_rows(n_samples), concurrency=8)
    fake = StaggeredProvider(fail=True)
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    assert len(results) == n_samples
    assert sorted(r.sample_id for r in results) == sorted(str(i) for i in range(1, n_samples + 1))

    real_failures = [r for r in results if r.detail != _SKIPPED_DETAIL]
    skipped = [r for r in results if r.detail == _SKIPPED_DETAIL]
    assert len(real_failures) == _ABORT_THRESHOLD
    assert len(skipped) == n_samples - _ABORT_THRESHOLD
    assert all(r.state == "provider_error" for r in results)

    assert summary.aborted_reason == real_failures[0].detail

    # The semaphore is a sliding window, not fixed batches of 8: the instant
    # q1 fails and releases its slot, the next queued sample (q9) immediately
    # backfills it -- by the time q5's completion trips the abort, more than
    # 8 samples may already have started. What's guaranteed regardless of
    # that timing detail: at least the 5 that tripped the predicate started
    # (fake.calls isn't empty), and strictly fewer than all 20 did (real
    # requests were actually avoided, not just relabeled after the fact).
    assert _ABORT_THRESHOLD <= len(fake.calls) < n_samples


async def test_cancelled_in_flight_tasks_produce_exactly_one_skipped_result_each(
    tmp_path: Path,
) -> None:
    """The specific risk flagged in review: task.cancel() raises CancelledError
    inside the cancelled task, which run_one_sample's own except clauses must
    NOT catch-and-misreport as an ordinary provider_error (they only catch
    Exception, never BaseException -- CancelledError is deliberately outside
    that net, see providers/base.py's NOTES.md entry on the same point).

    Two separate claims, asserted explicitly rather than one standing in for
    the other: (1) no result's detail mentions CancelledError, and (2)
    runner.run() actually completed -- returned a real, fully-formed summary
    from summarize() -- rather than merely "not raising" for some unrelated
    reason (a hang that timed out at the test-runner level, a swallowed
    exception turning into a degenerate empty return, etc). A future refactor
    could satisfy claim (1) by accident while breaking claim (2); pinning
    both means it can't.
    """
    n_samples = 20
    spec = make_spec(tmp_path, samples=_staggered_rows(n_samples), concurrency=8)
    fake = StaggeredProvider(fail=True)
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await asyncio.wait_for(runner.run(spec), timeout=5.0)

    # claim 2: a genuine, fully-reduced RunSummary came back, not a hang or a
    # degenerate empty/partial return from a swallowed cancellation.
    assert isinstance(summary, RunSummary)
    assert summary.n_samples == n_samples
    assert summary.aborted_reason is not None

    # exactly one SampleResult per original sample: no duplicates (a task
    # that somehow produced both a normal result and a cancellation
    # replacement) and no gaps (a task that vanished).
    ids = sorted(r.sample_id for r in results)
    assert ids == sorted(str(i) for i in range(1, n_samples + 1))
    assert len(ids) == len(set(ids))

    # claim 1: no result carries a leaked CancelledError instead of the
    # intended skipped/real-failure detail.
    for r in results:
        assert r.detail is not None
        assert "CancelledError" not in r.detail


async def test_one_success_before_failures_prevents_abort(tmp_path: Path) -> None:
    n_samples = 8

    class SuccessThenFailProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, prompt: str, params: object) -> ProviderResponse:
            self.calls.append(prompt)
            if prompt == "q1":
                return ProviderResponse(
                    text="42", input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake-model"
                )
            raise ProviderError("fake provider: simulated identical fatal failure")

    spec = make_spec(tmp_path, samples=_staggered_rows(n_samples), concurrency=1)
    fake = SuccessThenFailProvider()
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    # q1 succeeds, q2-q8 fail identically -- but with concurrency=1,
    # completion order == submission order, so q1's success is among the
    # first _ABORT_THRESHOLD completions and must prevent the abort.
    assert summary.aborted_reason is None
    assert not any(r.detail == _SKIPPED_DETAIL for r in results)
    assert len(fake.calls) == n_samples  # every sample actually ran, nothing skipped


async def test_mixed_error_messages_prevent_abort(tmp_path: Path) -> None:
    n_samples = 8

    class DistinctFailureProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, prompt: str, params: object) -> ProviderResponse:
            self.calls.append(prompt)
            raise ProviderError(f"fake provider: distinct failure for {prompt!r}")

    spec = make_spec(tmp_path, samples=_staggered_rows(n_samples), concurrency=1)
    fake = DistinctFailureProvider()
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, summary = await runner.run(spec)

    # every sample fails, but each with a DIFFERENT detail (the prompt is
    # embedded in the message) -- the first _ABORT_THRESHOLD failures don't
    # share one detail string, so the run must not abort.
    assert summary.aborted_reason is None
    assert not any(r.detail == _SKIPPED_DETAIL for r in results)
    assert len(fake.calls) == n_samples


async def test_order_preserved_despite_reversed_completion_order(tmp_path: Path) -> None:
    n_samples = 10

    class ReversedOrderProvider:
        """Later samples finish first (q1 has the longest delay, q10 the
        shortest) -- completion order is the exact reverse of submission
        order, deliberately hostile to any code that conflates the two."""

        def __init__(self, *, base_delay_s: float = 0.02) -> None:
            self.base_delay_s = base_delay_s

        async def complete(self, prompt: str, params: object) -> ProviderResponse:
            index = int(prompt.removeprefix("q"))
            await asyncio.sleep(self.base_delay_s * (n_samples - index))
            return ProviderResponse(
                text="42", input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake-model"
            )

    spec = make_spec(tmp_path, samples=_staggered_rows(n_samples), concurrency=n_samples)
    fake = ReversedOrderProvider()
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, _ = await runner.run(spec)

    assert [r.sample_id for r in results] == [str(i) for i in range(1, n_samples + 1)]


# --- sample_id -------------------------------------------------------------------------


async def test_sample_id_falls_back_to_one_based_position_when_id_field_absent(
    tmp_path: Path,
) -> None:
    rows = [{"question": "q0", "answer": "42"}, {"question": "q1", "answer": "42"}]  # no "id"
    spec = make_spec(tmp_path, samples=rows)
    fake = FakeProvider(reply_text="42")
    runner = LocalRunner(cache_path=tmp_path / "cache.sqlite", provider_factory=lambda m, r: fake)

    results, _ = await runner.run(spec)

    assert [r.sample_id for r in results] == ["1", "2"]
