"""Tests for gradetrail.spec: loading, validation, and run identity.

These encode the semantics in docs/design/eval-spec.md. If a test here needs
to change, the design doc changes first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gradetrail.errors import DatasetError, SpecError
from gradetrail.spec import compute_identity, load_spec

MINIMAL_YAML = """
name: gsm8k-subset
dataset:
  path: {dataset_path}
prompt: |
  Solve this. End with "Answer: <number>".

  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'Answer:\\s*{{{{ answer }}}}\\s*$'
"""


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "1", "question": "2+2?", "answer": "4"},
        {"id": "2", "question": "3+3?", "answer": "6"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture()
def spec_file(tmp_path: Path, dataset: Path) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(MINIMAL_YAML.format(dataset_path=dataset))
    return path


# --- loading ---------------------------------------------------------------


def test_minimal_spec_loads_with_defaults(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    assert spec.name == "gsm8k-subset"
    assert spec.model.params.max_tokens == 1024
    assert spec.model.params.temperature == 0.0
    assert spec.run.concurrency == 8
    assert spec.run.max_retries == 5


def test_spec_is_immutable(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    with pytest.raises(ValidationError):
        spec.name = "changed"  # type: ignore[misc]


def test_name_must_be_slug(spec_file: Path) -> None:
    text = spec_file.read_text().replace("gsm8k-subset", "Bad Name!")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="name"):
        load_spec(spec_file)


def test_unknown_top_level_field_rejected(spec_file: Path) -> None:
    spec_file.write_text(spec_file.read_text() + "\nsurprise: true\n")
    with pytest.raises(SpecError, match="surprise"):
        load_spec(spec_file)


# --- model validation --------------------------------------------------------


def test_base_url_required_for_openai_compatible(spec_file: Path) -> None:
    text = spec_file.read_text().replace("provider: anthropic", "provider: openai_compatible")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="base_url"):
        load_spec(spec_file)


def test_base_url_forbidden_for_anthropic(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "provider: anthropic", "provider: anthropic\n  base_url: http://localhost:8000"
    )
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="base_url"):
        load_spec(spec_file)


# --- scorer validation -------------------------------------------------------


def test_invalid_regex_fails_at_load(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "pattern: 'Answer:\\s*{{ answer }}\\s*$'", "pattern: '(unbalanced'"
    )
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="regex"):
        load_spec(spec_file)


def test_exact_scorer_parses(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: answer\n  normalize: [strip, lower]",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    assert spec.scorer.type == "exact"


# --- template strictness -----------------------------------------------------


def test_template_syntax_error_fails_at_load(spec_file: Path) -> None:
    text = spec_file.read_text().replace("{{ question }}", "{% broken")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="template"):
        load_spec(spec_file)


def test_missing_sample_field_fails_fast(spec_file: Path, dataset: Path) -> None:
    text = spec_file.read_text().replace("{{ question }}", "{{ nonexistent_field }}")
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    with pytest.raises(DatasetError, match="nonexistent_field"):
        spec.validate_against_dataset()


def test_valid_spec_passes_dataset_validation(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    spec.validate_against_dataset()  # should not raise


def test_exact_scorer_missing_target_field_fails_fast(spec_file: Path, dataset: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: nonexistent_field",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    with pytest.raises(DatasetError, match="nonexistent_field"):
        spec.validate_against_dataset()


def test_exact_scorer_with_valid_target_field_passes_dataset_validation(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: answer",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    spec.validate_against_dataset()  # should not raise


# --- dataset loading: sample ids ----------------------------------------------


def test_missing_id_field_falls_back_to_original_line_number_despite_shuffle(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "no_ids.jsonl"
    rows = [{"question": f"q{i}"} for i in range(5)]  # no "id" field at all
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows))

    spec_path = tmp_path / "eval.yaml"
    spec_path.write_text(
        f"""
name: no-ids
dataset:
  path: {dataset_path}
  shuffle_seed: 42
prompt: |
  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'x'
"""
    )
    spec = load_spec(spec_path)
    samples = spec.load_samples()

    # shuffle_seed=42 must actually reorder the samples relative to the file,
    # otherwise this test would pass trivially without exercising the bug.
    assert [s["question"] for s in samples] != [f"q{i}" for i in range(5)]

    # but each sample's fallback id must still reflect its ORIGINAL file line,
    # not its position after the shuffle.
    by_question = {s["question"]: s["id"] for s in samples}
    assert by_question == {f"q{i}": str(i + 1) for i in range(5)}


def test_present_id_field_is_never_overwritten(tmp_path: Path) -> None:
    dataset_path = tmp_path / "with_ids.jsonl"
    rows = [{"id": "custom-a", "question": "q0"}, {"id": "custom-b", "question": "q1"}]
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows))

    spec_path = tmp_path / "eval.yaml"
    spec_path.write_text(
        f"""
name: with-ids
dataset:
  path: {dataset_path}
prompt: |
  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'x'
"""
    )
    spec = load_spec(spec_path)
    samples = spec.load_samples()
    assert [s["id"] for s in samples] == ["custom-a", "custom-b"]


# --- identity ----------------------------------------------------------------


def test_run_block_does_not_affect_identity(spec_file: Path) -> None:
    id_default = compute_identity(load_spec(spec_file))
    spec_file.write_text(spec_file.read_text() + "\nrun:\n  concurrency: 32\n")
    id_tuned = compute_identity(load_spec(spec_file))
    assert id_default == id_tuned


def test_prompt_change_changes_identity(spec_file: Path) -> None:
    id_before = compute_identity(load_spec(spec_file))
    spec_file.write_text(spec_file.read_text().replace("Solve this.", "Solve it."))
    id_after = compute_identity(load_spec(spec_file))
    assert id_before != id_after


def test_dataset_content_change_changes_identity(spec_file: Path, dataset: Path) -> None:
    id_before = compute_identity(load_spec(spec_file))
    dataset.write_text(dataset.read_text() + '\n{"id": "3", "question": "5+5?", "answer": "10"}')
    id_after = compute_identity(load_spec(spec_file))
    assert id_before != id_after


def test_identity_is_stable_hex_digest(spec_file: Path) -> None:
    a = compute_identity(load_spec(spec_file))
    b = compute_identity(load_spec(spec_file))
    assert a == b
    assert len(a) == 64
    int(a, 16)  # valid hex
