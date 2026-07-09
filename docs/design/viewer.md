# Design: Results Viewer (v0.2)

Status: locked for viewer v1. Changes require updating this doc first.
Location in repo: `docs/design/viewer.md`
Branch: all viewer work happens on `results-viewer`, merged to main when v1 is complete.

## Problem

The most valuable act in the GSM8K benchmark was the failure audit: hand-inspecting all 14 zero-score samples, which reclassified 6 of them as measurement error rather than model error. That audit was done with a python one-liner squinting at JSON. The viewer makes that workflow a first-class feature: browse runs, read per-sample transcripts, filter to failures, and diff two runs.

## Decisions (made, not open)

1. **The viewer joins results with the dataset; results files are not changed.**
   `results.jsonl` stores outputs (state, score, response_text, tokens, detail) but not inputs (no question text, no expected answer). The run's `manifest.json` records `dataset_path` (resolved to an absolute path at write time) and `dataset_id_field` (both new in 0.2 — pre-0.2 manifests recorded only `dataset_sha256`); the viewer reads those two fields and joins results to dataset rows on `sample_id`. Rows missing the id field are keyed by their 1-based file line number, matching `load_samples()`. If the dataset file's current sha256 does not match the manifest's `dataset_sha256`, the viewer still renders but shows a visible "dataset changed since this run" warning on the run. `dataset_matches` is `false` whenever the live check cannot confirm a match: file changed, file missing, or a pre-0.2 manifest with no `dataset_path`; in the latter two cases the join degrades to `sample: null` per sample. A pre-0.2 run whose spec is unchanged can be regenerated nearly free from cache to pick the new fields up. Making results self-describing (embedding rendered prompt / sample fields in SampleResult) is a separate gradetrail 0.2+ roadmap item, not part of the viewer.

2. **The viewer ships inside gradetrail as `gradetrail view`, not a separate package.**

3. **Zero new dependencies. Stdlib server, single HTML file, no build step.**
   Server: python stdlib (`http.server` / `ThreadingHTTPServer`) on localhost. Frontend: exactly one file, `gradetrail/viewer/static/index.html`, containing all HTML, CSS, and vanilla JS. No npm, no framework, no CDN imports (must work offline). If the UI ever outgrows one file, that is a v0.3 decision made from evidence, not now.

## CLI

```
gradetrail view [RESULTS_ROOT] [--port 8600] [--no-browser]
```

- `RESULTS_ROOT` defaults to `./results`. The command discovers run directories (any subdirectory containing both `results.jsonl` and `manifest.json`).
- Binds to `127.0.0.1` only. Never `0.0.0.0`. No auth because no remote access.
- Opens the browser by default (`webbrowser.open`), suppressed with `--no-browser`.
- Exit: Ctrl+C stops the server cleanly.
- Errors: no run directories found is a clear message and exit 1, not an empty UI.

## HTTP contract

Three endpoints, all GET, all JSON except the page itself:

- `GET /` returns the bundled `index.html`.
- `GET /api/runs` returns the discovered runs:
  ```json
  [
    {
      "dir": "gsm8k-500-abe30645",
      "name": "gsm8k-500",
      "identity_hash": "abe30645...",
      "created_at": "...",
      "n_samples": 500,
      "counts": {"scored": 500, "provider_error": 0, "judge_error": 0},
      "mean_score": 0.978,
      "total_cost_usd": "0.98",
      "wall_time_s": 176.6,
      "model": "claude-sonnet-4-6",
      "dataset_matches": true
    }
  ]
  ```
  Fields come from the manifest; `dataset_matches` is the live sha256 check. Runs sorted newest first.
- `GET /api/runs/{dir}` returns full run data: the manifest verbatim under `manifest`, and `samples`: the results.jsonl rows joined with their dataset rows. Each sample:
  ```json
  {
    "sample_id": "13",
    "state": "scored",
    "score": 0.0,
    "response_text": "...",
    "detail": "...",
    "input_tokens": 92, "output_tokens": 187,
    "judge_input_tokens": null, "judge_output_tokens": null,
    "cached": true,
    "sample": {"question": "...", "answer": "72"}
  }
  ```
  `sample` is the raw dataset row (all fields), or `null` if the join failed for that id (missing/changed dataset). Join failures never 500; they degrade per-sample.
- **Malformed data semantics**: a corrupted results.jsonl line is skipped, counted, and surfaced as `"parse_errors": N` at the run level in both endpoints (0 when clean); the run still loads with the remaining samples. A malformed manifest.json makes that run appear in `/api/runs` with `"error"` set and minimal fields, not disappear silently. No half-parsed ghost samples.
- Path traversal: `{dir}` is validated against the discovered run list; arbitrary paths are rejected with 404.
- **The HTTP API is the compatibility boundary.** The frontend may evolve freely; endpoint semantics change only by updating this doc first.
- **Freshness over caching**: the server reads run files from disk on each API request, no server-side caching (staleness has no invalidation story, and a few MB of JSONL per request is negligible at this scale). The client fetches a run once per view and works from memory; it does not refetch per click.

## UI scope (v1)

Three views in one page. **Requirement: every view and filter state is URL-addressable** (a researcher can send a colleague a link to "run X, failures only"); the mechanism is client-side hash routing, e.g. `#/run/<dir>?state=scored&zero=1`. Target scale: interactive up to roughly 10,000 samples per run, rendered directly without pagination machinery.

1. **Run list** (landing): table of runs from `/api/runs`: name, model, date, samples, state counts, mean score, cost, wall time, and a warning icon when `dataset_matches` is false. Click a row to open the run. Checkboxes to select exactly two runs enable a "Diff" button.

2. **Single run**: summary header (the manifest numbers), then a sample table: id, state, score, cached, tokens. Controls: filter by state (all / scored / provider_error / judge_error), filter "score = 0" (the audit button), sort by id or score, free-text search over response text and dataset fields. Click a row to expand the transcript panel inline: every dataset field, the full response_text, detail, token counts. The audit workflow, "show me every zero-score with its question and response", must be at most two clicks from landing.

3. **Diff**: two runs side by side, joined on sample_id. Header shows both manifests' identity hashes and whether they match. Table of samples where the two runs disagree (score changed, state changed), with both responses viewable in the expand panel. A toggle to show all samples, not just disagreements. Missing-on-one-side samples are shown, not dropped.

Design tone: plain, fast, readable. System font stack, no animation, tables not cards. It should look like a tool, not a product.

## Non-goals for v1 (do not build)

- No editing anything, no launching runs from the UI
- No charts beyond the numeric summaries already in the manifest
- No remote serving, no auth, no HTTPS, no 0.0.0.0
- No persistence, no state beyond the URL hash
- No streaming/live-updating while a run is in progress
- No pagination machinery: render up to a few thousand samples directly; if that is ever slow, that is measured evidence for a v2 decision

## Testing

- Server: endpoint tests via `http.client` or `urllib` against a `ThreadingHTTPServer` started on port 0 in a fixture, using tmp-path fixture run directories (small synthetic results.jsonl + manifest.json + dataset). Cover: discovery, runs listing, run detail with successful join, join with missing dataset (per-sample null + dataset_matches false), path traversal rejected, no-runs-found exit.
- The HTML file is not unit-tested; its data contract is (the endpoints). Manual browser check per session.
- CLI: `gradetrail view` with no results dir exits 1 with a message; `--no-browser` suppresses `webbrowser.open` (patch it).

## Milestones

1. This doc + CLAUDE.md viewer conventions + `gradetrail view` serving the run list end to end.
2. Single-run view with sample table and transcript expand.
3. Failure filtering and search (the audit workflow).
4. Two-run diff.
5. Polish: README section with screenshots, redo the GSM8K failure audit through the viewer as the demo.
