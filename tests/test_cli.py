"""Tests for evalflow.cli: argument wiring, output paths, exit codes.

LocalRunner and RayRunner are stubbed out here -- runner internals
(concurrency, caching, scoring, provider failures, real Ray execution) are
already covered by tests/runner/test_local.py and tests/runner/test_ray_runner.py.
These tests exercise only cli.py's own wiring: spec loading/validation,
--backend/--workers dispatch, output-dir defaulting, file writing, exit
codes, and the summary print. Never a real provider, never a real runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import evalflow.cli as cli_module
from evalflow.results import RunSummary, SampleResult

runner = CliRunner()

VALID_SPEC_YAML = """
name: cli-test-eval
dataset:
  path: {dataset_path}
prompt: |
  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: exact
  target_field: answer
"""

FAKE_RESULTS = [
    SampleResult(
        sample_id="1",
        state="scored",
        score=1.0,
        response_text="42",
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        cached=False,
        detail="matched '42'",
        served_model="claude-sonnet-4-6-20260115",
    )
]
FAKE_SUMMARY = RunSummary(
    n_samples=1,
    n_scored=1,
    n_provider_error=0,
    n_judge_error=0,
    mean_score=1.0,
    total_input_tokens=10,
    total_output_tokens=5,
    total_cost_usd=None,
    wall_time_s=1.5,
    cache_hits=0,
)


class _StubRunner:
    """Stands in for LocalRunner: returns canned results, never calls a provider."""

    def __init__(self, **kwargs: object) -> None:
        pass

    async def run(self, spec: object) -> tuple[list[SampleResult], RunSummary]:
        return FAKE_RESULTS, FAKE_SUMMARY


@pytest.fixture(autouse=True)
def stub_local_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "LocalRunner", _StubRunner)


@pytest.fixture()
def ray_runner_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stub RayRunner that records its constructor kwargs, so --backend/--workers
    wiring can be asserted without ever touching a real Ray cluster."""
    calls: list[dict] = []

    class _StubRayRunner:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        async def run(self, spec: object) -> tuple[list[SampleResult], RunSummary]:
            return FAKE_RESULTS, FAKE_SUMMARY

    monkeypatch.setattr(cli_module, "RayRunner", _StubRayRunner)
    return calls


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"id": "1", "question": "2+2?", "answer": "42"}))
    return path


@pytest.fixture()
def spec_file(tmp_path: Path, dataset: Path) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(VALID_SPEC_YAML.format(dataset_path=dataset))
    return path


# --- green path --------------------------------------------------------------------


def test_run_green_path_produces_results_and_manifest(spec_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "manifest.json").exists()


def test_run_green_path_results_jsonl_matches_returned_results(
    spec_file: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    lines = (output_dir / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sample_id"] == "1"


def test_run_green_path_manifest_reflects_served_models(spec_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["served_models"] == ["claude-sonnet-4-6-20260115"]


def test_run_green_path_prints_summary_table(spec_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert "1" in result.stdout  # n_samples / n_scored show up somewhere
    assert "1.0" in result.stdout or "1.00" in result.stdout  # mean score


def test_run_green_path_exit_code_is_zero(spec_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    assert result.exit_code == 0


def test_run_default_output_dir_uses_name_and_identity_hash(
    spec_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli_module.app, ["run", str(spec_file)])
    assert result.exit_code == 0
    results_dirs = list((tmp_path / "results").iterdir())
    assert len(results_dirs) == 1
    assert results_dirs[0].name.startswith("cli-test-eval-")
    assert (results_dirs[0] / "results.jsonl").exists()
    assert (results_dirs[0] / "manifest.json").exists()


# --- bad spec: exit 1, clean message, no traceback ------------------------------------


def test_run_bad_spec_exits_1_with_clean_message(tmp_path: Path) -> None:
    bad_spec = tmp_path / "bad.yaml"
    bad_spec.write_text(
        "name: Bad Name!\n"
        "dataset:\n  path: x\n"
        "prompt: hi\n"
        "model:\n  provider: anthropic\n  name: x\n"
        "scorer:\n  type: exact\n  target_field: y\n"
    )
    result = runner.invoke(cli_module.app, ["run", str(bad_spec)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "name" in result.output


def test_run_dataset_validation_failure_exits_1(dataset: Path, tmp_path: Path) -> None:
    spec_path = tmp_path / "eval.yaml"
    spec_path.write_text(
        f"""
name: cli-test-eval
dataset:
  path: {dataset}
prompt: |
  {{{{ nonexistent_field }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: exact
  target_field: answer
"""
    )
    result = runner.invoke(cli_module.app, ["run", str(spec_path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "nonexistent_field" in result.output


def test_run_missing_spec_file_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(cli_module.app, ["run", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


# --- --backend / --workers wiring ---------------------------------------------------


def test_run_default_backend_never_constructs_ray_runner(
    spec_file: Path, tmp_path: Path, ray_runner_calls: list[dict]
) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(cli_module.app, ["run", str(spec_file), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert ray_runner_calls == []


def test_run_backend_ray_dispatches_to_ray_runner(
    spec_file: Path, tmp_path: Path, ray_runner_calls: list[dict]
) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(
        cli_module.app,
        ["run", str(spec_file), "--output-dir", str(output_dir), "--backend", "ray"],
    )
    assert result.exit_code == 0
    assert len(ray_runner_calls) == 1
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "manifest.json").exists()


def test_run_backend_ray_with_explicit_workers(
    spec_file: Path, tmp_path: Path, ray_runner_calls: list[dict]
) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(
        cli_module.app,
        [
            "run",
            str(spec_file),
            "--output-dir",
            str(output_dir),
            "--backend",
            "ray",
            "--workers",
            "4",
        ],
    )
    assert result.exit_code == 0
    assert ray_runner_calls[0]["n_workers"] == 4


def test_run_backend_ray_default_workers_is_cpu_count_capped_at_8(
    spec_file: Path,
    tmp_path: Path,
    ray_runner_calls: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.os, "cpu_count", lambda: 32)
    output_dir = tmp_path / "out"
    result = runner.invoke(
        cli_module.app,
        ["run", str(spec_file), "--output-dir", str(output_dir), "--backend", "ray"],
    )
    assert result.exit_code == 0
    assert ray_runner_calls[0]["n_workers"] == 8


def test_run_backend_ray_default_workers_below_cap_uses_cpu_count(
    spec_file: Path,
    tmp_path: Path,
    ray_runner_calls: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.os, "cpu_count", lambda: 4)
    output_dir = tmp_path / "out"
    result = runner.invoke(
        cli_module.app,
        ["run", str(spec_file), "--output-dir", str(output_dir), "--backend", "ray"],
    )
    assert result.exit_code == 0
    assert ray_runner_calls[0]["n_workers"] == 4
