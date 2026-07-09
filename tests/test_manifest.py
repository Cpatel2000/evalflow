"""Tests for gradetrail.manifest.write_manifest: the reproducibility record.

git SHA lookup is mocked at the subprocess boundary (never depends on this
repo's actual git state, hermetic either way).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

import gradetrail.manifest as manifest_module
from gradetrail.manifest import write_manifest
from gradetrail.results import RunSummary
from gradetrail.spec import (
    DatasetSpec,
    EvalSpec,
    ExactScorer,
    JudgeScorer,
    ModelSpec,
    RunSpec,
    compute_identity,
)

JUDGE_YAML = """
version: 3
output: binary
prompt: |
  Question: {{ question }}
  Response: {{ response }}
"""


def make_spec(tmp_path: Path, *, scorer=None) -> EvalSpec:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(json.dumps({"id": "1", "question": "2+2?", "answer": "4"}))
    return EvalSpec(
        name="manifest-test",
        dataset=DatasetSpec(path=str(dataset_path)),
        prompt="{{ question }}",
        model=ModelSpec(provider="anthropic", name="claude-sonnet-4-6"),
        scorer=scorer or ExactScorer(type="exact", target_field="answer"),
        run=RunSpec(),
        base_dir=tmp_path,
    )


SUMMARY = RunSummary(
    n_samples=3,
    n_scored=2,
    n_provider_error=1,
    n_judge_error=0,
    mean_score=0.5,
    total_input_tokens=1000,
    total_output_tokens=500,
    total_cost_usd=Decimal("0.0125"),
    wall_time_s=12.34,
    cache_hits=1,
)


@pytest.fixture(autouse=True)
def no_real_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: pretend there is no git binary at all, unless a test overrides this."""

    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)


# --- basic shape -----------------------------------------------------------------


def test_write_manifest_produces_valid_json(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_write_manifest_identity_hash_matches_compute_identity(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["identity_hash"] == compute_identity(spec)
    assert len(data["identity_hash"]) == 64


def test_write_manifest_includes_spec_name(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["name"] == "manifest-test"


def test_write_manifest_dataset_sha256_matches_file_contents(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    expected = hashlib.sha256(spec.dataset_path().read_bytes()).hexdigest()
    assert data["dataset_sha256"] == expected


# --- judge sha256 ----------------------------------------------------------------


def test_write_manifest_judge_sha256_is_none_for_non_judge_scorer(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)  # default is ExactScorer
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["judge_sha256"] is None


def test_write_manifest_judge_sha256_present_for_judge_scorer(tmp_path: Path) -> None:
    judge_path = tmp_path / "judge.yaml"
    judge_path.write_text(JUDGE_YAML)
    spec = make_spec(
        tmp_path,
        scorer=JudgeScorer(
            type="judge",
            judge_prompt="judge.yaml",
            model=ModelSpec(provider="anthropic", name="judge-model"),
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["judge_sha256"] == hashlib.sha256(judge_path.read_bytes()).hexdigest()


# --- requested vs served model ----------------------------------------------------


def test_write_manifest_requested_model_is_spec_model_name(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["requested_model"] == "claude-sonnet-4-6"


def test_write_manifest_served_models_reflects_input_set(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(
        spec,
        SUMMARY,
        path,
        served_models={"claude-sonnet-4-6-20260115", "claude-sonnet-4-6-20260201"},
    )
    data = json.loads(path.read_text())
    assert data["served_models"] == ["claude-sonnet-4-6-20260115", "claude-sonnet-4-6-20260201"]


def test_write_manifest_served_models_defaults_to_empty_list(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["served_models"] == []


# --- git sha -----------------------------------------------------------------------


def test_git_sha_returned_when_git_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, 0, stdout="abc123def\n", stderr="")

    monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["git_sha"] == "abc123def"


def test_git_sha_is_none_when_not_a_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["git_sha"] is None


def test_git_sha_is_none_when_git_binary_missing(tmp_path: Path) -> None:
    # no_real_git autouse fixture already simulates this
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["git_sha"] is None


# --- timestamps, version, summary fields ----------------------------------------------


def test_write_manifest_created_at_is_recent_utc_timestamp(tmp_path: Path) -> None:
    before = datetime.now(UTC)
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    after = datetime.now(UTC)
    data = json.loads(path.read_text())
    created_at = datetime.fromisoformat(data["created_at"])
    assert before <= created_at <= after


def test_write_manifest_gradetrail_version_matches_installed_package(tmp_path: Path) -> None:
    import gradetrail

    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["gradetrail_version"] == gradetrail.__version__


def test_write_manifest_includes_summary_fields(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["wall_time_s"] == 12.34
    assert data["n_samples"] == 3
    assert data["n_scored"] == 2
    assert data["n_provider_error"] == 1
    assert data["n_judge_error"] == 0
    assert data["mean_score"] == 0.5
    assert data["total_input_tokens"] == 1000
    assert data["total_output_tokens"] == 500
    assert data["cache_hits"] == 1


def test_write_manifest_total_cost_usd_serialized_as_string(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["total_cost_usd"] == "0.0125"
    assert Decimal(data["total_cost_usd"]) == Decimal("0.0125")


# --- dataset path + id field (viewer join, docs/design/viewer.md decision 1) -------


def test_write_manifest_dataset_path_is_resolved_absolute(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["dataset_path"] == str(spec.dataset_path().resolve())
    assert Path(data["dataset_path"]).is_absolute()


def test_write_manifest_dataset_id_field_default(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["dataset_id_field"] == "id"


def test_write_manifest_dataset_id_field_reflects_spec(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(json.dumps({"qid": "1", "question": "2+2?", "answer": "4"}))
    spec = EvalSpec(
        name="manifest-test",
        dataset=DatasetSpec(path=str(dataset_path), id_field="qid"),
        prompt="{{ question }}",
        model=ModelSpec(provider="anthropic", name="claude-sonnet-4-6"),
        scorer=ExactScorer(type="exact", target_field="answer"),
        run=RunSpec(),
        base_dir=tmp_path,
    )
    path = tmp_path / "manifest.json"
    write_manifest(spec, SUMMARY, path)
    data = json.loads(path.read_text())
    assert data["dataset_id_field"] == "qid"


def test_write_manifest_total_cost_usd_is_none_when_unpriced(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    unpriced_summary = RunSummary(
        n_samples=1,
        n_scored=1,
        n_provider_error=0,
        n_judge_error=0,
        mean_score=1.0,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=None,
        wall_time_s=1.0,
        cache_hits=0,
    )
    path = tmp_path / "manifest.json"
    write_manifest(spec, unpriced_summary, path)
    data = json.loads(path.read_text())
    assert data["total_cost_usd"] is None
