"""Thin CLI: wires spec loading, the runner, and output writing together.

No business logic here (CLAUDE.md) -- everything is delegated to spec.py,
runner/local.py, results.py, and manifest.py.
"""

from __future__ import annotations

import asyncio
import enum
import os
import webbrowser
from decimal import Decimal
from pathlib import Path

import typer

from gradetrail.errors import DatasetError, SpecError
from gradetrail.manifest import write_manifest
from gradetrail.results import RunSummary, write_jsonl
from gradetrail.runner.local import LocalRunner
from gradetrail.runner.ray_runner import RayRunner
from gradetrail.spec import compute_identity, load_spec
from gradetrail.viewer.server import create_server, discover_runs

app = typer.Typer()

_DEFAULT_CACHE_PATH = Path(".gradetrail_cache.sqlite")
_MAX_DEFAULT_WORKERS = 8


class Backend(enum.StrEnum):
    """Execution backend. The spec itself stays execution-agnostic (design doc);
    this choice is CLI-level only."""

    local = "local"
    ray = "ray"


@app.callback()
def _main() -> None:
    """gradetrail: a distributed LLM evaluation harness."""


@app.command()
def run(
    spec_path: Path = typer.Argument(..., help="Path to the eval spec YAML file."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Defaults to ./results/<name>-<identity_hash[:8]>/"
    ),
    backend: Backend = typer.Option(Backend.local, "--backend", help="Execution backend."),
    workers: int | None = typer.Option(
        None,
        "--workers",
        help=("Ray worker count (--backend ray only). Defaults to os.cpu_count() capped at 8."),
    ),
) -> None:
    """Run an eval spec and write results.jsonl + manifest.json."""
    try:
        spec = load_spec(spec_path)
        spec.validate_against_dataset()

        identity_hash = compute_identity(spec)
        resolved_output_dir = output_dir or Path("results") / f"{spec.name}-{identity_hash[:8]}"
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        if backend == Backend.ray:
            n_workers = (
                workers if workers is not None else min(os.cpu_count() or 1, _MAX_DEFAULT_WORKERS)
            )
            runner = RayRunner(cache_path=_DEFAULT_CACHE_PATH, n_workers=n_workers)
        else:
            runner = LocalRunner(cache_path=_DEFAULT_CACHE_PATH)

        results, summary = asyncio.run(runner.run(spec))

        served_models = {r.served_model for r in results if r.served_model}

        write_jsonl(results, resolved_output_dir / "results.jsonl")
        write_manifest(
            spec,
            summary,
            resolved_output_dir / "manifest.json",
            served_models=served_models,
        )
    except (SpecError, DatasetError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    if summary.aborted_reason is not None:
        typer.echo(
            "ABORTED: run stopped early -- multiple samples failed with an "
            f"identical fatal error: {summary.aborted_reason}",
            err=True,
        )

    _print_summary(summary)

    # Exit 1 when nothing scored -- an all-failed run (e.g. an expired API key
    # in CI) must not report success. An aborted run always has n_scored == 0,
    # so this single check subsumes the abort case; the ABORTED line above is
    # what distinguishes the two for a human. Exit 0 iff at least one sample
    # scored (partial success is success).
    if summary.aborted_reason is not None or summary.n_scored == 0:
        raise typer.Exit(code=1)


@app.command()
def view(
    results_root: Path = typer.Argument(
        Path("results"), help="Directory containing run directories."
    ),
    port: int = typer.Option(8600, "--port", help="Port on 127.0.0.1 (0 picks a free one)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser."),
) -> None:
    """Browse run results in a local web viewer (binds 127.0.0.1 only)."""
    runs = discover_runs(results_root)
    if not runs:
        typer.echo(f"No run directories found under {results_root}.", err=True)
        raise typer.Exit(code=1)

    server = create_server(results_root, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    typer.echo(f"Serving {len(runs)} run(s) at {url} (Ctrl+C to stop)")
    if not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _print_summary(summary: RunSummary) -> None:
    typer.echo(
        f"Samples: {summary.n_samples} "
        f"(scored={summary.n_scored}, provider_error={summary.n_provider_error}, "
        f"judge_error={summary.n_judge_error})"
    )
    mean = f"{summary.mean_score:.4f}" if summary.mean_score is not None else "n/a"
    typer.echo(f"Mean score: {mean}")
    typer.echo(f"Tokens: {summary.total_input_tokens} in / {summary.total_output_tokens} out")
    if summary.total_judge_input_tokens or summary.total_judge_output_tokens:
        typer.echo(
            f"Judge tokens: {summary.total_judge_input_tokens} in / "
            f"{summary.total_judge_output_tokens} out"
        )
    if isinstance(summary.total_cost_usd, Decimal):
        cost = f"${summary.total_cost_usd:.4f}"
    elif summary.cost_unpriced_models:
        cost = f"unknown ({'; '.join(summary.cost_unpriced_models)})"
    else:
        cost = "unknown"
    typer.echo(f"Cost: {cost}")
    typer.echo(f"Cache hits: {summary.cache_hits}/{summary.n_samples}")
    typer.echo(f"Wall time: {summary.wall_time_s:.2f}s")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
