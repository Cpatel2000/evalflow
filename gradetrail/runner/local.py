"""LocalRunner: asyncio + a semaphore for concurrency, no distributed execution.

Per sample: render the prompt, check the cache, on a miss call the provider
and cache the raw response, then score. A provider failure or a judge error on
one sample becomes a SampleResult in the appropriate error state -- it never
aborts the run or takes down sibling tasks (except a real cancellation, which
must still propagate; see the CancelledError note in providers/base.py).

run_one_sample (and the private helpers it calls) are module-level, not
LocalRunner methods: they take no `self` and close over nothing, so
runner/ray_runner.py can call the exact same per-sample pipeline from inside
a Ray task running in a separate process, instead of duplicating it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import jinja2
import structlog

from gradetrail.cache import ResponseCache
from gradetrail.errors import ProviderError
from gradetrail.providers.anthropic import AnthropicProvider
from gradetrail.providers.base import Provider, ProviderResponse
from gradetrail.providers.openai import OpenAIProvider
from gradetrail.providers.openai_compatible import OpenAICompatibleProvider
from gradetrail.results import RunSummary, SampleResult, summarize
from gradetrail.runner.base import Runner
from gradetrail.scorers.base import ScoreResult
from gradetrail.scorers.deterministic import score_exact, score_regex
from gradetrail.scorers.judge import JudgeFile, load_judge_file, score_judge
from gradetrail.spec import EvalSpec, ExactScorer, JudgeScorer, ModelSpec, RegexScorer, RunSpec

_JINJA_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)
_log = structlog.get_logger(__name__)

ProviderFactory = Callable[[ModelSpec, RunSpec], Provider]


def _default_provider_factory(model: ModelSpec, run: RunSpec) -> Provider:
    if model.provider == "anthropic":
        return AnthropicProvider(
            model=model.name, max_retries=run.max_retries, timeout_s=run.timeout_s
        )
    if model.provider == "openai":
        return OpenAIProvider(
            model=model.name, max_retries=run.max_retries, timeout_s=run.timeout_s
        )
    assert model.provider == "openai_compatible"  # the only remaining case (closed Literal)
    assert model.base_url is not None  # ModelSpec's own validator guarantees this
    return OpenAICompatibleProvider(
        model=model.name,
        base_url=model.base_url,
        max_retries=run.max_retries,
        timeout_s=run.timeout_s,
    )


def _resolve_path(base_dir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base_dir / p


def _error_result(sample_id: str, detail: str) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        state="provider_error",
        score=None,
        response_text=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        cached=False,
        detail=detail,
    )


async def run_one_sample(
    spec: EvalSpec,
    sample: dict,
    provider: Provider,
    judge_provider: Provider | None,
    judge_file: JudgeFile | None,
    cache: ResponseCache,
    semaphore: asyncio.Semaphore,
) -> SampleResult:
    """The per-sample pipeline shared by every runner: render, cache check,
    complete-or-cache-hit, score -- under a concurrency semaphore.

    Never raises for expected failure modes (ProviderError, judge failures);
    those become a SampleResult in the appropriate error state (design doc
    rule 5). A real task cancellation still propagates: CancelledError is
    BaseException, not Exception (see providers/base.py NOTES.md entry), so
    it matches neither except clause below.
    """
    sample_id = str(sample[spec.dataset.id_field])
    async with semaphore:
        start = time.monotonic()
        try:
            result = await _score_one(
                spec, sample, sample_id, provider, judge_provider, judge_file, cache
            )
        except ProviderError as exc:
            result = _error_result(sample_id, str(exc))
        except Exception as exc:  # noqa: BLE001 -- task isolation: one bad
            # sample must never crash the run or its siblings.
            result = _error_result(sample_id, f"unexpected error: {exc!r}")
        elapsed_ms = (time.monotonic() - start) * 1000

    _log.info(
        "sample_completed",
        sample_id=sample_id,
        state=result.state,
        cached=result.cached,
        latency_ms=round(elapsed_ms, 2),
    )
    return result


async def _score_one(
    spec: EvalSpec,
    sample: dict,
    sample_id: str,
    provider: Provider,
    judge_provider: Provider | None,
    judge_file: JudgeFile | None,
    cache: ResponseCache,
) -> SampleResult:
    prompt = _JINJA_ENV.from_string(spec.prompt).render(**sample)
    params_dict = spec.model.params.model_dump()

    cache_entry = await cache.get(
        spec.model.provider, spec.model.name, spec.model.base_url, prompt, params_dict
    )
    cached = cache_entry is not None
    if cached:
        response = ProviderResponse(**cache_entry.response)
    else:
        response = await provider.complete(prompt, spec.model.params)
        await cache.put(
            spec.model.provider,
            spec.model.name,
            spec.model.base_url,
            prompt,
            params_dict,
            dataclasses.asdict(response),
        )
    score_result = await _score(spec, sample, response.text, judge_provider, judge_file)

    return SampleResult(
        sample_id=sample_id,
        state=score_result.state,
        score=score_result.score if score_result.state == "scored" else None,
        response_text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        cached=cached,
        detail=score_result.detail,
        served_model=response.model if score_result.state == "scored" else None,
    )


async def _score(
    spec: EvalSpec,
    sample: dict,
    response_text: str,
    judge_provider: Provider | None,
    judge_file: JudgeFile | None,
) -> ScoreResult:
    scorer = spec.scorer
    if isinstance(scorer, ExactScorer):
        return score_exact(sample, response_text, scorer)
    if isinstance(scorer, RegexScorer):
        return score_regex(sample, response_text, scorer)
    assert judge_provider is not None
    assert judge_file is not None
    return await score_judge(sample, response_text, scorer, judge_file, judge_provider)


class LocalRunner(Runner):
    """Runs an eval spec locally: one ResponseCache and one Provider per model,
    constructed once per run, not once per sample.
    """

    def __init__(
        self,
        *,
        cache_path: str | Path,
        provider_factory: ProviderFactory = _default_provider_factory,
    ) -> None:
        self._cache_path = cache_path
        self._provider_factory = provider_factory

    async def run(self, spec: EvalSpec) -> tuple[list[SampleResult], RunSummary]:
        start = time.monotonic()
        samples = spec.load_samples()

        judge_file: JudgeFile | None = None
        judge_provider: Provider | None = None
        if isinstance(spec.scorer, JudgeScorer):
            judge_path = _resolve_path(spec.base_dir, spec.scorer.judge_prompt)
            judge_file = load_judge_file(judge_path)
            judge_provider = self._provider_factory(spec.scorer.model, spec.run)

        provider = self._provider_factory(spec.model, spec.run)
        semaphore = asyncio.Semaphore(spec.run.concurrency)

        async with ResponseCache(self._cache_path) as cache:
            results = await asyncio.gather(
                *(
                    run_one_sample(
                        spec, sample, provider, judge_provider, judge_file, cache, semaphore
                    )
                    for sample in samples
                )
            )

        wall_time_s = time.monotonic() - start
        summary = summarize(list(results), spec.model, wall_time_s)
        return list(results), summary
