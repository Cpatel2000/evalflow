"""Endpoint + helper tests for gradetrail.viewer.server (docs/design/viewer.md).

Fixtures build synthetic run directories (results.jsonl + manifest.json) and a
dataset file, per the design doc's testing section. Requests go through a real
ThreadingHTTPServer bound to 127.0.0.1 port 0, via http.client so traversal
paths reach the server unnormalized.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from gradetrail.viewer.server import create_server, dataset_index, discover_runs

# --- fixture helpers ----------------------------------------------------------------


def make_dataset(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_manifest(
    *,
    name: str,
    dataset_path: Path | None,
    created_at: str = "2026-07-09T12:00:00+00:00",
    n_scored: int = 2,
    n_provider_error: int = 0,
    n_judge_error: int = 0,
    **overrides: object,
) -> dict:
    """A manifest dict shaped like write_manifest's output (0.2, with dataset fields).

    dataset_path=None mimics a pre-0.2 manifest: the dataset_path and
    dataset_id_field keys are absent entirely, not null.
    """
    manifest: dict = {
        "identity_hash": hashlib.sha256(name.encode()).hexdigest(),
        "name": name,
        "dataset_sha256": (
            hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path else "0" * 64
        ),
        "judge_sha256": None,
        "requested_model": "claude-sonnet-4-6",
        "served_models": ["claude-sonnet-4-6"],
        "gradetrail_version": "0.2.0",
        "git_sha": None,
        "created_at": created_at,
        "wall_time_s": 12.5,
        "n_samples": n_scored + n_provider_error + n_judge_error,
        "n_scored": n_scored,
        "n_provider_error": n_provider_error,
        "n_judge_error": n_judge_error,
        "mean_score": 0.5,
        "total_input_tokens": 100,
        "total_output_tokens": 200,
        "total_cost_usd": "0.0125",
        "cache_hits": 0,
    }
    if dataset_path is not None:
        manifest["dataset_path"] = str(dataset_path.resolve())
        manifest["dataset_id_field"] = "id"
    manifest.update(overrides)
    return manifest


def sample_line(sample_id: str, *, score: float = 1.0, state: str = "scored") -> str:
    return json.dumps(
        {
            "sample_id": sample_id,
            "state": state,
            "score": score if state == "scored" else None,
            "response_text": f"Answer: {sample_id}",
            "input_tokens": 10,
            "output_tokens": 20,
            "latency_ms": 100.0,
            "cached": False,
            "detail": None,
            "served_model": "claude-sonnet-4-6",
            "judge_input_tokens": None,
            "judge_output_tokens": None,
        }
    )


def make_run(
    root: Path,
    dirname: str,
    *,
    manifest: dict | str,
    results_lines: list[str] | None = None,
) -> Path:
    """Create root/dirname with manifest.json + results.jsonl.

    manifest may be a raw string to write malformed JSON directly.
    """
    run_dir = root / dirname
    run_dir.mkdir(parents=True)
    if isinstance(manifest, str):
        (run_dir / "manifest.json").write_text(manifest)
    else:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = results_lines if results_lines is not None else [sample_line("1"), sample_line("2")]
    (run_dir / "results.jsonl").write_text("".join(line + "\n" for line in lines))
    return run_dir


@pytest.fixture
def clean_run(tmp_path: Path) -> Path:
    """One well-formed run in an otherwise empty results root; returns the root."""
    dataset = tmp_path / "datasets" / "qa.jsonl"
    make_dataset(
        dataset,
        [
            {"id": "1", "question": "2+2?", "answer": "4"},
            {"id": "2", "question": "3+3?", "answer": "6"},
        ],
    )
    manifest = make_manifest(name="qa", dataset_path=dataset)
    make_run(tmp_path / "results", "qa-abc12345", manifest=manifest)
    return tmp_path / "results"


# --- HTTP plumbing --------------------------------------------------------------------


@pytest.fixture
def serve() -> Iterator:
    """Start create_server(root) on a background thread; yields root -> (host, port)."""
    servers = []

    def _serve(root: Path) -> tuple[str, int]:
        server = create_server(root)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host, port = server.server_address[:2]
        return host, port

    yield _serve
    for server in servers:
        server.shutdown()
        server.server_close()


def get(addr: tuple[str, int], path: str) -> tuple[int, str, dict[str, str]]:
    """GET via http.client (no client-side path normalization, unlike urllib)."""
    conn = http.client.HTTPConnection(*addr)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode(), {k.lower(): v for k, v in resp.getheaders()}
    finally:
        conn.close()


def get_json(addr: tuple[str, int], path: str) -> tuple[int, object]:
    status, body, _ = get(addr, path)
    return status, json.loads(body)


# --- discovery -------------------------------------------------------------------------


def test_discover_runs_finds_dirs_with_both_files(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}])
    make_run(tmp_path / "results", "run-a", manifest=make_manifest(name="a", dataset_path=dataset))
    make_run(tmp_path / "results", "run-b", manifest=make_manifest(name="b", dataset_path=dataset))
    found = discover_runs(tmp_path / "results")
    assert {p.name for p in found} == {"run-a", "run-b"}


def test_discover_runs_ignores_incomplete_dirs_and_files(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "no-manifest").mkdir(parents=True)
    (root / "no-manifest" / "results.jsonl").write_text("")
    (root / "no-results").mkdir()
    (root / "no-results" / "manifest.json").write_text("{}")
    (root / "stray-file.txt").write_text("not a run")
    assert discover_runs(root) == []


def test_discover_runs_empty_or_missing_root(tmp_path: Path) -> None:
    assert discover_runs(tmp_path) == []
    assert discover_runs(tmp_path / "does-not-exist") == []


# --- dataset_index (the join helper) ----------------------------------------------------


def test_dataset_index_keys_rows_by_id_field(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "a", "question": "q1"}, {"id": "b", "question": "q2"}])
    index = dataset_index(make_manifest(name="x", dataset_path=dataset))
    assert index is not None
    assert index["a"] == {"id": "a", "question": "q1"}
    assert index["b"]["question"] == "q2"


def test_dataset_index_stringifies_non_string_ids(tmp_path: Path) -> None:
    # SampleResult.sample_id is always a str; a dataset with integer ids must
    # still join ("7" -> row with id 7).
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": 7, "question": "q"}])
    index = dataset_index(make_manifest(name="x", dataset_path=dataset))
    assert index is not None
    assert index["7"]["question"] == "q"


def test_dataset_index_line_number_fallback_matches_load_samples(tmp_path: Path) -> None:
    # Rows without the id field are keyed by 1-based *file* line number --
    # blank lines are skipped but still counted, exactly like
    # EvalSpec.load_samples() (see NOTES.md 2026-07-06 on shuffle-stable ids).
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"question": "first"})
        + "\n\n"  # blank line at line 2
        + json.dumps({"question": "third"})
        + "\n"
    )
    index = dataset_index(make_manifest(name="x", dataset_path=dataset))
    assert index is not None
    assert index["1"]["question"] == "first"
    assert "2" not in index
    assert index["3"]["question"] == "third"


def test_dataset_index_respects_custom_id_field(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"qid": "q-9", "question": "q"}])
    manifest = make_manifest(name="x", dataset_path=dataset, dataset_id_field="qid")
    index = dataset_index(manifest)
    assert index is not None
    assert index["q-9"]["question"] == "q"


def test_dataset_index_none_for_pre_0_2_manifest(tmp_path: Path) -> None:
    index = dataset_index(make_manifest(name="x", dataset_path=None))
    assert index is None


def test_dataset_index_none_when_dataset_file_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}])
    manifest = make_manifest(name="x", dataset_path=dataset)
    dataset.unlink()
    assert dataset_index(manifest) is None


def test_dataset_index_skips_malformed_dataset_lines(tmp_path: Path) -> None:
    # A bad dataset line degrades to "that row can't join" -- never an exception
    # (server errors degrade, CLAUDE.md viewer conventions).
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"id": "1", "question": "ok"}) + "\n{not json\n")
    index = dataset_index(make_manifest(name="x", dataset_path=dataset))
    assert index is not None
    assert index["1"]["question"] == "ok"
    assert len(index) == 1


# --- GET /api/runs ----------------------------------------------------------------------


def test_api_runs_entry_shape(clean_run: Path, serve) -> None:
    status, runs = get_json(serve(clean_run), "/api/runs")
    assert status == 200
    assert isinstance(runs, list) and len(runs) == 1
    entry = runs[0]
    assert entry["dir"] == "qa-abc12345"
    assert entry["name"] == "qa"
    assert entry["identity_hash"] == hashlib.sha256(b"qa").hexdigest()
    assert entry["created_at"] == "2026-07-09T12:00:00+00:00"
    assert entry["n_samples"] == 2
    assert entry["counts"] == {"scored": 2, "provider_error": 0, "judge_error": 0}
    assert entry["mean_score"] == 0.5
    assert entry["total_cost_usd"] == "0.0125"
    assert entry["wall_time_s"] == 12.5
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["dataset_matches"] is True
    assert entry["parse_errors"] == 0
    assert "error" not in entry


def test_api_runs_sorted_newest_first(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}])
    root = tmp_path / "results"
    make_run(
        root,
        "older",
        manifest=make_manifest(
            name="older", dataset_path=dataset, created_at="2026-07-01T00:00:00+00:00"
        ),
    )
    make_run(
        root,
        "newer",
        manifest=make_manifest(
            name="newer", dataset_path=dataset, created_at="2026-07-08T00:00:00+00:00"
        ),
    )
    _, runs = get_json(serve(root), "/api/runs")
    assert [r["dir"] for r in runs] == ["newer", "older"]


def test_api_runs_dataset_matches_false_when_content_differs(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1", "answer": "4"}])
    root = tmp_path / "results"
    make_run(root, "run-a", manifest=make_manifest(name="a", dataset_path=dataset))
    make_dataset(dataset, [{"id": "1", "answer": "5"}])  # dataset edited after the run
    _, runs = get_json(serve(root), "/api/runs")
    assert runs[0]["dataset_matches"] is False


def test_api_runs_dataset_matches_null_when_file_missing(tmp_path: Path, serve) -> None:
    # Tri-state: a missing file is "cannot verify" (null), not a verified
    # mismatch (false) -- see docs/design/viewer.md decision 1.
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}])
    root = tmp_path / "results"
    make_run(root, "run-a", manifest=make_manifest(name="a", dataset_path=dataset))
    dataset.unlink()
    _, runs = get_json(serve(root), "/api/runs")
    assert runs[0]["dataset_matches"] is None


def test_api_runs_dataset_matches_null_for_pre_0_2_manifest(tmp_path: Path, serve) -> None:
    root = tmp_path / "results"
    make_run(root, "old-run", manifest=make_manifest(name="old", dataset_path=None))
    _, runs = get_json(serve(root), "/api/runs")
    assert runs[0]["dataset_matches"] is None  # unverifiable, distinct from false
    assert "error" not in runs[0]  # degraded, not broken


def test_api_runs_corrupted_results_line_counts_parse_errors(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}, {"id": "2"}])
    root = tmp_path / "results"
    make_run(
        root,
        "run-a",
        manifest=make_manifest(name="a", dataset_path=dataset),
        results_lines=[sample_line("1"), "{corrupted", sample_line("2")],
    )
    _, runs = get_json(serve(root), "/api/runs")
    entry = runs[0]
    assert entry["parse_errors"] == 1
    # The run still loads: manifest-derived fields are intact.
    assert entry["name"] == "a"
    assert entry["counts"] == {"scored": 2, "provider_error": 0, "judge_error": 0}
    assert "error" not in entry


def test_api_runs_malformed_manifest_surfaces_error_not_absence(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}])
    root = tmp_path / "results"
    make_run(root, "good-run", manifest=make_manifest(name="good", dataset_path=dataset))
    make_run(root, "bad-run", manifest="{this is not json")
    status, runs = get_json(serve(root), "/api/runs")
    assert status == 200
    by_dir = {r["dir"]: r for r in runs}
    assert set(by_dir) == {"good-run", "bad-run"}
    bad = by_dir["bad-run"]
    assert isinstance(bad["error"], str) and bad["error"]
    assert "error" not in by_dir["good-run"]


# --- GET /api/runs/{dir} -----------------------------------------------------------------


def test_api_run_detail_shape_and_join(clean_run: Path, serve) -> None:
    status, body = get_json(serve(clean_run), "/api/runs/qa-abc12345")
    assert status == 200
    assert body["dir"] == "qa-abc12345"
    # Manifest verbatim: exactly what's on disk, not a reshaped copy.
    on_disk = json.loads((clean_run / "qa-abc12345" / "manifest.json").read_text())
    assert body["manifest"] == on_disk
    assert body["dataset_matches"] is True
    assert body["parse_errors"] == 0
    assert "error" not in body

    samples = body["samples"]
    assert [s["sample_id"] for s in samples] == ["1", "2"]  # results.jsonl order
    first = samples[0]
    assert first["state"] == "scored"
    assert first["score"] == 1.0
    assert first["response_text"] == "Answer: 1"
    assert first["detail"] is None
    assert first["input_tokens"] == 10
    assert first["output_tokens"] == 20
    assert first["judge_input_tokens"] is None
    assert first["judge_output_tokens"] is None
    assert first["cached"] is False
    # The results.jsonl row passes through verbatim -- fields beyond the doc's
    # example (latency_ms, served_model) are not stripped.
    assert first["latency_ms"] == 100.0
    assert first["served_model"] == "claude-sonnet-4-6"
    assert first["sample"] == {"id": "1", "question": "2+2?", "answer": "4"}
    assert samples[1]["sample"] == {"id": "2", "question": "3+3?", "answer": "6"}


def test_api_run_detail_sample_null_when_id_missing_from_dataset(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1", "question": "q1"}])
    root = tmp_path / "results"
    make_run(
        root,
        "run-a",
        manifest=make_manifest(name="a", dataset_path=dataset),
        results_lines=[sample_line("1"), sample_line("ghost-id")],
    )
    _, body = get_json(serve(root), "/api/runs/run-a")
    by_id = {s["sample_id"]: s for s in body["samples"]}
    assert by_id["1"]["sample"] == {"id": "1", "question": "q1"}
    assert by_id["ghost-id"]["sample"] is None  # degrades per-sample, never 500


def test_api_run_detail_all_samples_null_for_pre_0_2_manifest(tmp_path: Path, serve) -> None:
    root = tmp_path / "results"
    make_run(root, "old-run", manifest=make_manifest(name="old", dataset_path=None))
    _, body = get_json(serve(root), "/api/runs/old-run")
    assert body["dataset_matches"] is None
    assert len(body["samples"]) == 2
    assert all(s["sample"] is None for s in body["samples"])
    assert "error" not in body  # degraded, not broken


def test_api_run_detail_changed_dataset_still_joins_with_warning(tmp_path: Path, serve) -> None:
    # A changed dataset is a warning, not a join failure: the join uses the
    # file's *current* content (decision 1 -- "still renders but shows a
    # visible warning").
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1", "question": "original"}])
    root = tmp_path / "results"
    make_run(
        root,
        "run-a",
        manifest=make_manifest(name="a", dataset_path=dataset),
        results_lines=[sample_line("1")],
    )
    make_dataset(dataset, [{"id": "1", "question": "edited"}])
    _, body = get_json(serve(root), "/api/runs/run-a")
    assert body["dataset_matches"] is False
    assert body["samples"][0]["sample"] == {"id": "1", "question": "edited"}


def test_api_run_detail_corrupted_line_skipped_and_counted(tmp_path: Path, serve) -> None:
    dataset = tmp_path / "data.jsonl"
    make_dataset(dataset, [{"id": "1"}, {"id": "2"}])
    root = tmp_path / "results"
    make_run(
        root,
        "run-a",
        manifest=make_manifest(name="a", dataset_path=dataset),
        results_lines=[sample_line("1"), "{corrupted", "", sample_line("2")],
    )
    _, body = get_json(serve(root), "/api/runs/run-a")
    assert body["parse_errors"] == 1  # the corrupted line; the blank line is not an error
    # No half-parsed ghost samples: exactly the two good rows survive.
    assert [s["sample_id"] for s in body["samples"]] == ["1", "2"]


def test_api_run_detail_malformed_manifest_degrades(tmp_path: Path, serve) -> None:
    root = tmp_path / "results"
    make_run(root, "bad-run", manifest="{this is not json")
    status, body = get_json(serve(root), "/api/runs/bad-run")
    assert status == 200
    assert body["dir"] == "bad-run"
    assert isinstance(body["error"], str) and body["error"]
    assert body["manifest"] is None
    assert body["dataset_matches"] is None
    # Samples still come back (unjoined) -- the run degrades, it doesn't vanish.
    assert [s["sample_id"] for s in body["samples"]] == ["1", "2"]
    assert all(s["sample"] is None for s in body["samples"])


def test_api_run_detail_trailing_path_is_404(clean_run: Path, serve) -> None:
    status, _, _ = get(serve(clean_run), "/api/runs/qa-abc12345/extra")
    assert status == 404


# --- routing, traversal, root page --------------------------------------------------------


def test_api_unknown_run_dir_is_404(clean_run: Path, serve) -> None:
    status, _, _ = get(serve(clean_run), "/api/runs/does-not-exist")
    assert status == 404


def test_api_path_traversal_is_404(clean_run: Path, serve) -> None:
    addr = serve(clean_run)
    for path in (
        "/api/runs/../../etc/passwd",
        "/api/runs/..%2f..%2fetc%2fpasswd",
        "/api/runs/%2e%2e/qa-abc12345",
    ):
        status, body, _ = get(addr, path)
        assert status == 404, path
        assert "passwd" not in body or "root:" not in body


def test_unknown_path_is_404(clean_run: Path, serve) -> None:
    status, _, _ = get(serve(clean_run), "/api/nope")
    assert status == 404


def test_root_serves_index_html(clean_run: Path, serve) -> None:
    status, body, headers = get(serve(clean_run), "/")
    assert status == 200
    assert "text/html" in headers["content-type"]
    assert "/api/runs" in body  # the minimal page fetches the runs listing


def test_responses_send_cache_control_no_store(clean_run: Path, serve) -> None:
    # Stdlib http.server sends no caching headers of its own; without no-store,
    # browsers serve stale HTML across viewer versions (freshness over caching).
    addr = serve(clean_run)
    for path in ("/", "/api/runs", "/api/runs/qa-abc12345"):
        _, _, headers = get(addr, path)
        assert headers["cache-control"] == "no-store", path


def test_create_server_binds_loopback_only(clean_run: Path) -> None:
    server = create_server(clean_run)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
