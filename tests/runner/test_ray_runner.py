"""Tests for gradetrail.runner.ray_runner.RayRunner: the Ray backend.

Requires a real (non-local-mode) Ray cluster -- ray.init(local_mode=False,
num_cpus=2) in a session-scoped fixture, torn down at the end. Marked `ray`
so `pytest -m "not ray"` skips this whole module.

Fake providers are constructed via a module-level, picklable factory function
-- never passed as already-constructed instances from the driver/test
process. A lambda or a driver-built FakeProvider instance would make these
tests pass while silently failing on real Ray serialization (Provider
instances hold live SDK clients that generally aren't picklable at all); a
module-level factory function is what Ray actually pickles by reference and
calls fresh inside each worker process, exactly mirroring how real Provider
construction works for the anthropic/openai backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.ray

ray = pytest.importorskip("ray")

from gradetrail.providers.base import ProviderResponse  # noqa: E402
from gradetrail.runner.local import LocalRunner  # noqa: E402
from gradetrail.runner.ray_runner import RayRunner  # noqa: E402
from gradetrail.spec import DatasetSpec, EvalSpec, ExactScorer, ModelSpec, RunSpec  # noqa: E402


@pytest.fixture(scope="session")
def ray_cluster() -> None:
    ray.init(local_mode=False, num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


# --- module-level (picklable-by-reference) fakes, constructed INSIDE workers --------


class FakeProvider:
    """Constructed fresh inside each Ray worker by the factories below -- never
    constructed in the driver/test process and passed across."""

    def __init__(self, *, reply_text: str) -> None:
        self.reply_text = reply_text

    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        return ProviderResponse(
            text=self.reply_text,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            model="fake-model",
        )


class _NeverCallProvider:
    """Fails loudly if a worker actually calls it -- proves a cache hit skipped
    the provider entirely, even across process boundaries."""

    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        raise AssertionError("provider should never be called -- expected a cache hit")


class _UnreachableProvider:
    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        raise AssertionError("unreachable: the factory should have crashed first")


def fake_provider_factory(model: ModelSpec, run: RunSpec) -> FakeProvider:
    return FakeProvider(reply_text="42")


def never_call_provider_factory(model: ModelSpec, run: RunSpec) -> _NeverCallProvider:
    return _NeverCallProvider()


def crashy_provider_factory(model: ModelSpec, run: RunSpec) -> _UnreachableProvider:
    """Raises inside the worker before any sample runs -- simulates the whole
    Ray task (worker) dying, not a single sample's provider call failing."""
    raise RuntimeError("simulated worker crash")


def make_spec(tmp_path: Path, *, samples: list[dict], concurrency: int = 4) -> EvalSpec:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text("\n".join(json.dumps(r) for r in samples))
    return EvalSpec(
        name="ray-test-eval",
        dataset=DatasetSpec(path=str(dataset_path)),
        prompt="{{ question }}",
        model=ModelSpec(provider="anthropic", name="claude-sonnet-4-6"),
        scorer=ExactScorer(type="exact", target_field="answer"),
        run=RunSpec(concurrency=concurrency, max_retries=0, timeout_s=5.0),
        base_dir=tmp_path,
    )


# --- results match LocalRunner --------------------------------------------------------


async def test_results_match_local_runner_on_same_fake_inputs(
    tmp_path: Path, ray_cluster: None
) -> None:
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(6)]
    spec = make_spec(tmp_path, samples=rows)

    local_runner = LocalRunner(
        cache_path=tmp_path / "local_cache.sqlite", provider_factory=fake_provider_factory
    )
    local_results, _ = await local_runner.run(spec)

    ray_runner = RayRunner(
        cache_path=tmp_path / "ray_cache.sqlite",
        n_workers=3,
        provider_factory=fake_provider_factory,
    )
    ray_results, _ = await ray_runner.run(spec)

    def normalize(results: list) -> list:
        # latency_ms/cached legitimately differ by execution path; everything
        # that determines correctness of the eval itself must match exactly.
        return [(r.sample_id, r.state, r.score, r.response_text) for r in results]

    assert normalize(local_results) == normalize(ray_results)


# --- order preservation ---------------------------------------------------------------


async def test_order_is_preserved_across_uneven_worker_batches(
    tmp_path: Path, ray_cluster: None
) -> None:
    n_samples = 11  # deliberately not evenly divisible by n_workers
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(n_samples)]
    spec = make_spec(tmp_path, samples=rows, concurrency=4)
    runner = RayRunner(
        cache_path=tmp_path / "cache.sqlite", n_workers=3, provider_factory=fake_provider_factory
    )

    results, _ = await runner.run(spec)

    assert [r.sample_id for r in results] == [str(i) for i in range(n_samples)]


# --- batch failure isolation -----------------------------------------------------------


async def test_batch_failure_converts_to_provider_error_without_crashing_the_run(
    tmp_path: Path, ray_cluster: None
) -> None:
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(6)]
    spec = make_spec(tmp_path, samples=rows)
    runner = RayRunner(
        cache_path=tmp_path / "cache.sqlite",
        n_workers=2,
        provider_factory=crashy_provider_factory,
    )

    results, summary = await runner.run(spec)  # must not raise

    assert len(results) == 6
    assert all(r.state == "provider_error" for r in results)
    assert all(r.detail is not None and "simulated worker crash" in r.detail for r in results)
    assert summary.n_provider_error == 6
    assert summary.n_scored == 0


# --- cache shared across workers ------------------------------------------------------


async def test_cache_shared_across_workers_and_visible_to_a_subsequent_run(
    tmp_path: Path, ray_cluster: None
) -> None:
    cache_path = tmp_path / "cache.sqlite"
    rows = [{"id": str(i), "question": f"q{i}", "answer": "42"} for i in range(6)]
    spec = make_spec(tmp_path, samples=rows, concurrency=4)

    runner1 = RayRunner(cache_path=cache_path, n_workers=3, provider_factory=fake_provider_factory)
    results1, _ = await runner1.run(spec)
    assert all(r.cached is False for r in results1)
    assert all(r.response_text == "42" for r in results1)

    # A subsequent run must see every entry the (multiple) Ray workers wrote --
    # proves the cache is genuinely shared across workers, not per-worker-local.
    # Using LocalRunner here also proves the cache is shared across runners.
    runner2 = LocalRunner(cache_path=cache_path, provider_factory=never_call_provider_factory)
    results2, _ = await runner2.run(spec)
    assert all(r.cached is True for r in results2)
    assert all(r.response_text == "42" for r in results2)
