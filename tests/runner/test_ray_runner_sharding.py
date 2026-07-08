"""Tests for evalflow.runner.ray_runner's pure helpers: _shard_samples and the
lazy-ray-import guard.

No Ray cluster needed here -- these run fast, always, unmarked (not `ray`),
since ray_runner.py itself has no module-level `import ray`; only run()
imports it lazily.
"""

from __future__ import annotations

import builtins

import pytest

from evalflow.errors import EvalflowError
from evalflow.runner.ray_runner import _require_ray, _shard_samples

# --- _shard_samples ----------------------------------------------------------------


def test_shard_samples_splits_evenly() -> None:
    samples = [{"id": str(i)} for i in range(9)]
    batches = _shard_samples(samples, 3)
    assert [len(b) for b in batches] == [3, 3, 3]


def test_shard_samples_distributes_remainder_to_earlier_batches() -> None:
    samples = [{"id": str(i)} for i in range(11)]
    batches = _shard_samples(samples, 3)
    assert [len(b) for b in batches] == [4, 4, 3]


def test_shard_samples_preserves_order_within_and_across_batches() -> None:
    samples = [{"id": str(i)} for i in range(11)]
    batches = _shard_samples(samples, 3)
    flattened = [s["id"] for batch in batches for s in batch]
    assert flattened == [str(i) for i in range(11)]


def test_shard_samples_caps_worker_count_at_sample_count() -> None:
    samples = [{"id": str(i)} for i in range(2)]
    batches = _shard_samples(samples, 8)
    assert len(batches) == 2  # never more batches than samples
    assert sum(len(b) for b in batches) == 2


def test_shard_samples_handles_single_worker() -> None:
    samples = [{"id": str(i)} for i in range(5)]
    batches = _shard_samples(samples, 1)
    assert len(batches) == 1
    assert len(batches[0]) == 5


# --- _require_ray --------------------------------------------------------------------


def test_require_ray_raises_evalflow_error_when_ray_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ray" or name.startswith("ray."):
            raise ImportError("No module named 'ray'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(EvalflowError, match=r"evalflow\[ray\]"):
        _require_ray()


def test_require_ray_returns_the_module_when_installed() -> None:
    ray = _require_ray()
    assert ray.__name__ == "ray"
