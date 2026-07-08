"""Run manifest: everything needed to reproduce or audit a run.

Written once per run, after summarize(). See docs/design/eval-spec.md rule 1
(identity) and rule 6 (the determinism caveat this doesn't try to hide).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import gradetrail
from gradetrail.results import RunSummary
from gradetrail.spec import EvalSpec, JudgeScorer, compute_identity


def _dataset_sha256(spec: EvalSpec) -> str:
    return hashlib.sha256(spec.dataset_path().read_bytes()).hexdigest()


def _judge_sha256(spec: EvalSpec) -> str | None:
    if not isinstance(spec.scorer, JudgeScorer):
        return None
    judge_path = Path(spec.scorer.judge_prompt)
    if not judge_path.is_absolute():
        judge_path = spec.base_dir / judge_path
    return hashlib.sha256(judge_path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    """Best-effort: None if git isn't installed or this isn't a repo, never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def write_manifest(
    spec: EvalSpec,
    summary: RunSummary,
    path: str | Path,
    *,
    served_models: set[str] = frozenset(),
) -> None:
    """Write the run's reproducibility manifest as JSON.

    served_models is the set of API-reported model strings seen in responses
    (requested vs served can differ, e.g. an alias resolving to a dated
    snapshot) -- not derivable from spec/summary alone, so the runner passes
    it in explicitly.
    """
    manifest = {
        "identity_hash": compute_identity(spec),
        "name": spec.name,
        "dataset_sha256": _dataset_sha256(spec),
        "judge_sha256": _judge_sha256(spec),
        "requested_model": spec.model.name,
        "served_models": sorted(served_models),
        "gradetrail_version": gradetrail.__version__,
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "wall_time_s": summary.wall_time_s,
        "n_samples": summary.n_samples,
        "n_scored": summary.n_scored,
        "n_provider_error": summary.n_provider_error,
        "n_judge_error": summary.n_judge_error,
        "mean_score": summary.mean_score,
        "total_input_tokens": summary.total_input_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "total_cost_usd": (
            str(summary.total_cost_usd) if isinstance(summary.total_cost_usd, Decimal) else None
        ),
        "cache_hits": summary.cache_hits,
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
