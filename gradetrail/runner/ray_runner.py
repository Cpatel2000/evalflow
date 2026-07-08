"""RayRunner: single-machine Ray backend for LocalRunner-equivalent execution.

Samples are sharded into n_workers contiguous batches; one Ray task per batch
reconstructs its own Provider(s) and its own ResponseCache connection to the
shared cache path (WAL handles the contention -- see cache.py's two-connection
test), then runs the exact same per-sample pipeline as LocalRunner
(runner.local.run_one_sample). A whole-batch failure (the worker died, its
task raised) converts every sample in that batch to a provider_error result
with the failure in detail -- the run itself never crashes, the same
guarantee LocalRunner gives at the per-sample level.

ray is imported lazily, only inside run() -- the base gradetrail install never
requires it; `pip install gradetrail[ray]` is required to actually use this
backend.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from gradetrail.cache import ResponseCache
from gradetrail.errors import GradetrailError
from gradetrail.results import RunSummary, SampleResult, summarize
from gradetrail.runner.base import Runner
from gradetrail.runner.local import (
    ProviderFactory,
    _default_provider_factory,
    _error_result,
    _resolve_path,
    run_one_sample,
)
from gradetrail.scorers.judge import JudgeFile, load_judge_file
from gradetrail.spec import EvalSpec, JudgeScorer


def _require_ray():  # noqa: ANN202 -- return type is the `ray` module, kept untyped to avoid importing it at module level
    try:
        import ray
    except ImportError as exc:
        raise GradetrailError(
            "the ray backend requires the optional ray dependency: install gradetrail[ray]"
        ) from exc
    return ray


def _shard_samples(samples: list[dict], n_workers: int) -> list[list[dict]]:
    """Split samples into up to n_workers contiguous batches, in original order.

    The remainder (when len(samples) doesn't divide evenly) goes to the
    earliest batches, one extra sample each. Never creates more batches than
    samples, and never an empty batch.
    """
    n_workers = max(1, min(n_workers, len(samples)))
    base, remainder = divmod(len(samples), n_workers)
    batches = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < remainder else 0)
        batches.append(samples[start : start + size])
        start += size
    return batches


async def _run_batch(
    spec: EvalSpec,
    batch: list[dict],
    judge_file: JudgeFile | None,
    cache_path: str,
    provider_factory: ProviderFactory,
) -> list[SampleResult]:
    """Ray task body: construct this worker's own Provider(s) and its own
    ResponseCache connection, then run the same per-sample pipeline as
    LocalRunner over this batch.
    """
    provider = provider_factory(spec.model, spec.run)
    judge_provider = (
        provider_factory(spec.scorer.model, spec.run)
        if isinstance(spec.scorer, JudgeScorer)
        else None
    )
    semaphore = asyncio.Semaphore(spec.run.concurrency)

    async with ResponseCache(cache_path) as cache:
        return list(
            await asyncio.gather(
                *(
                    run_one_sample(
                        spec, sample, provider, judge_provider, judge_file, cache, semaphore
                    )
                    for sample in batch
                )
            )
        )


def _run_batch_sync(
    spec: EvalSpec,
    batch: list[dict],
    judge_file: JudgeFile | None,
    cache_path: str,
    provider_factory: ProviderFactory,
) -> list[SampleResult]:
    """Sync wrapper around _run_batch: Ray remote FUNCTIONS (not actors) reject
    `async def` outright ("'async def' should not be used for remote tasks",
    confirmed empirically against ray 2.56) -- they must return a value
    directly, not a coroutine. asyncio.run() is the documented workaround.
    """
    return asyncio.run(_run_batch(spec, batch, judge_file, cache_path, provider_factory))


class RayRunner(Runner):
    """Runs an eval spec on a single-machine Ray cluster: one task per batch of samples."""

    def __init__(
        self,
        *,
        cache_path: str | Path,
        n_workers: int,
        provider_factory: ProviderFactory = _default_provider_factory,
    ) -> None:
        self._cache_path = str(cache_path)
        self._n_workers = n_workers
        self._provider_factory = provider_factory

    async def run(self, spec: EvalSpec) -> tuple[list[SampleResult], RunSummary]:
        ray = _require_ray()
        start = time.monotonic()

        samples = spec.load_samples()
        judge_file: JudgeFile | None = None
        if isinstance(spec.scorer, JudgeScorer):
            judge_file = load_judge_file(_resolve_path(spec.base_dir, spec.scorer.judge_prompt))

        # Pre-warm: open and close the cache once here, before any worker
        # connects. ResponseCache.connect() already retries the WAL-switch
        # race on its own (that's the correctness guarantee), but a
        # pre-warmed file means workers connect to a database already in WAL
        # mode, so that retry path is a safety net that almost never fires
        # rather than a hot path under n_workers-way contention.
        async with ResponseCache(self._cache_path):
            pass

        batches = _shard_samples(samples, self._n_workers)
        remote_run_batch = ray.remote(_run_batch_sync)
        object_refs = [
            remote_run_batch.remote(
                spec, batch, judge_file, self._cache_path, self._provider_factory
            )
            for batch in batches
        ]

        # All batches are already dispatched and running concurrently on the
        # cluster (the .remote() calls above are non-blocking). This loop just
        # collects them in submission order, one at a time -- NOT ray.get(refs)
        # on the whole list, which fails atomically on the first error and
        # would defeat per-batch isolation. A fast batch behind a slow one in
        # this list only waits to be *collected* here, not to *run* -- do not
        # "optimize" this back into a single batched ray.get() call.
        results: list[SampleResult] = []
        for batch, ref in zip(batches, object_refs, strict=True):
            try:
                batch_results = await asyncio.to_thread(ray.get, ref)
            except Exception as exc:  # noqa: BLE001 -- a whole worker/task died;
                # isolate it to this batch's samples, the same guarantee
                # LocalRunner gives per-sample. CancelledError still
                # propagates uncaught (BaseException, not Exception).
                batch_results = [
                    _error_result(str(sample[spec.dataset.id_field]), f"worker failed: {exc!r}")
                    for sample in batch
                ]
            results.extend(batch_results)

        wall_time_s = time.monotonic() - start
        summary = summarize(results, spec.model, wall_time_s)
        return results, summary
