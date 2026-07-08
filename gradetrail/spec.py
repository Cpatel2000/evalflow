"""Eval spec: the public contract of gradetrail.

Loading, validation, and run identity live here. See docs/design/eval-spec.md
for the schema and the semantics the tests enforce. Do not change the schema
without updating that doc first.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

import jinja2
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gradetrail.errors import DatasetError, SpecError

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_JINJA_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSpec(_Frozen):
    path: str
    id_field: str = "id"
    limit: int | None = Field(default=None, gt=0)
    shuffle_seed: int | None = None


class ModelParams(_Frozen):
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    system: str | None = None


class ModelSpec(_Frozen):
    provider: Literal["anthropic", "openai", "openai_compatible"]
    name: str
    base_url: str | None = None
    params: ModelParams = ModelParams()

    @model_validator(mode="after")
    def _check_base_url(self) -> ModelSpec:
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("model.base_url: required when provider is openai_compatible")
        if self.provider != "openai_compatible" and self.base_url:
            raise ValueError(
                f"model.base_url: forbidden for provider {self.provider!r} "
                "(only openai_compatible uses it)"
            )
        return self


_Normalize = Literal["strip", "lower", "collapse_whitespace"]


class ExactScorer(_Frozen):
    type: Literal["exact"]
    target_field: str
    normalize: tuple[_Normalize, ...] = ()


class RegexScorer(_Frozen):
    type: Literal["regex"]
    pattern: str
    flags: tuple[Literal["IGNORECASE", "MULTILINE", "DOTALL"], ...] = ()

    @field_validator("pattern")
    @classmethod
    def _check_regex(cls, v: str) -> str:
        # Render template placeholders with a dummy value first so we validate
        # the regex structure, not the jinja syntax (checked separately).
        probe = re.sub(r"\{\{.*?\}\}", "PROBE", v)
        try:
            re.compile(probe)
        except re.error as e:
            raise ValueError(f"scorer.pattern: invalid regex ({e})") from None
        return v


class JudgeScorer(_Frozen):
    type: Literal["judge"]
    judge_prompt: str
    model: ModelSpec
    samples: int = Field(default=1, gt=0)


ScorerSpec = Annotated[ExactScorer | RegexScorer | JudgeScorer, Field(discriminator="type")]


class RunSpec(_Frozen):
    concurrency: int = Field(default=8, gt=0)
    max_retries: int = Field(default=5, ge=0)
    timeout_s: float = Field(default=120.0, gt=0)


class EvalSpec(_Frozen):
    name: str
    description: str | None = None
    dataset: DatasetSpec
    prompt: str
    model: ModelSpec
    scorer: ScorerSpec
    run: RunSpec = RunSpec()

    # Base dir for resolving relative paths (dataset, judge). Set by load_spec.
    base_dir: Path = Path(".")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(f"name: {v!r} must match [a-z0-9-] (used in run IDs and filenames)")
        return v

    @field_validator("prompt")
    @classmethod
    def _check_template_syntax(cls, v: str) -> str:
        try:
            _JINJA_ENV.parse(v)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"prompt: invalid template (line {e.lineno}: {e.message})") from None
        return v

    def dataset_path(self) -> Path:
        p = Path(self.dataset.path)
        return p if p.is_absolute() else self.base_dir / p

    def load_samples(self) -> list[dict]:
        """Load, optionally shuffle, and limit the dataset per the spec."""
        path = self.dataset_path()
        if not path.exists():
            raise DatasetError(f"dataset.path: {path} does not exist")
        samples: list[dict] = []
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({e.msg})") from None
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: each line must be a JSON object")
            if self.dataset.id_field not in row:
                row[self.dataset.id_field] = str(lineno)
            samples.append(row)
        if not samples:
            raise DatasetError(f"dataset.path: {path} contains no samples")
        if self.dataset.shuffle_seed is not None:
            import random

            random.Random(self.dataset.shuffle_seed).shuffle(samples)
        if self.dataset.limit is not None:
            samples = samples[: self.dataset.limit]
        return samples

    def validate_against_dataset(self) -> None:
        """Fail fast: render templates against sample 0 with strict undefined.

        Raises DatasetError naming the missing field, per design doc rule 3.
        """
        sample = self.load_samples()[0]
        templates = {"prompt": self.prompt}
        if isinstance(self.scorer, RegexScorer):
            templates["scorer.pattern"] = self.scorer.pattern
        for label, source in templates.items():
            try:
                _JINJA_ENV.from_string(source).render(**sample)
            except jinja2.UndefinedError as e:
                raise DatasetError(
                    f"{label}: {e.message} (sample fields: {sorted(sample)})"
                ) from None
        if isinstance(self.scorer, ExactScorer) and self.scorer.target_field not in sample:
            raise DatasetError(
                f"scorer.target_field: {self.scorer.target_field!r} is undefined "
                f"(sample fields: {sorted(sample)})"
            )


def load_spec(path: str | Path) -> EvalSpec:
    """Load and validate an eval spec from a YAML file.

    Raises SpecError with a field-and-fix message on any validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise SpecError(f"spec file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: invalid YAML ({e})") from None
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: spec must be a YAML mapping")
    raw.setdefault("base_dir", path.parent)
    try:
        return EvalSpec.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"]) or "spec"
        raise SpecError(f"{path}: {loc}: {first['msg']}") from None


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_identity(spec: EvalSpec) -> str:
    """The results-relevant identity of a run (design doc rule 1).

    sha256 over canonical JSON of (spec minus run block and base_dir),
    the dataset content hash, and the judge file hash if present.
    The run block never affects identity.
    """
    payload = spec.model_dump(exclude={"run", "base_dir"})
    dataset_hash = hashlib.sha256(spec.dataset_path().read_bytes()).hexdigest()
    parts: dict[str, object] = {"spec": payload, "dataset_sha256": dataset_hash}
    if isinstance(spec.scorer, JudgeScorer):
        judge_path = Path(spec.scorer.judge_prompt)
        if not judge_path.is_absolute():
            judge_path = spec.base_dir / judge_path
        if not judge_path.exists():
            raise SpecError(f"scorer.judge_prompt: {judge_path} does not exist")
        parts["judge_sha256"] = hashlib.sha256(judge_path.read_bytes()).hexdigest()
    return hashlib.sha256(_canonical_json(parts).encode()).hexdigest()
