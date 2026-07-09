# gradetrail

A distributed LLM evaluation harness that treats evals like tests: declarative, cached, reproducible.

## What this project is

A Python library + CLI. A user defines an eval in a YAML spec (dataset, prompt template, model, scorer), runs it with `gradetrail run spec.yaml`, and gets scored results plus a reproducibility manifest. Runs are cached, multi-provider, and can execute locally (asyncio) or distributed (Ray) behind the same interface.

Target user: an ML researcher who wants eval results in 30 seconds of setup, not a platform to administer.

## Architecture

Modules and their single responsibility. Do not blur these boundaries.

- `gradetrail/spec.py` - Eval spec dataclasses + YAML loading/validation. The spec is the public contract. NEVER change the spec schema without asking me first.
- `gradetrail/providers/` - Provider abstraction. One base class, one module per provider (`anthropic.py`, `openai.py`, `openai_compatible.py`). ALL model API calls go through this layer. No direct HTTP/SDK calls anywhere else in the codebase.
- `gradetrail/scorers/` - Scoring. `deterministic.py` (exact match, regex, numeric tolerance) and `judge.py` (LLM-as-judge). Judge prompts are versioned artifacts referenced by the spec, never inlined ad hoc.
- `gradetrail/cache.py` - SQLite response cache. Key = sha256 of canonical JSON of (provider, model, resolved prompt, sampling params). Cache logic lives here and only here.
- `gradetrail/runner/` - Execution. `base.py` defines the Runner interface, `local.py` (asyncio + semaphore concurrency), `ray_runner.py` (Ray backend, added in week 3). Runners are swappable via config, callers never know which one they got.
- `gradetrail/manifest.py` - Run manifests: spec hash, model versions, timestamps, seed, git SHA, package version. Every run writes one.
- `gradetrail/results.py` - Results model + JSONL writing + summary stats (scores, cost, tokens, failure rate).
- `gradetrail/cli.py` - Thin CLI (typer). No business logic here, it only wires modules together.

Design docs live in `docs/design/`. Read the relevant one before implementing a module.

## Conventions

- Python 3.11+. Type hints on every function signature. `from __future__ import annotations` at the top of each module.
- Prefer functions and frozen dataclasses over classes with mutable state. Only use a class when there is real state or a real interface (providers, runners).
- Errors: raise specific exceptions from `gradetrail/errors.py`. Never bare `except:`. Never swallow exceptions silently.
- Logging: `structlog`, structured key-value logs. Every provider call logs model, latency_ms, tokens, cached (bool), outcome. No print() outside the CLI.
- Async: provider calls are async. Do not mix sync SDK calls into async paths, use the async clients.
- Dependencies: keep them minimal. Ask before adding any new dependency. Current allowed set: pydantic, typer, structlog, anthropic, openai, pyyaml, aiosqlite, ray (week 3 only).
- Tests: pytest + pytest-asyncio. Tests live in `tests/` mirroring the package layout. Mock at the provider boundary using recorded fixtures, never hit real APIs in tests.
- Style: ruff for lint + format, config in pyproject.toml. Line length 100.

## Workflow rules

- Before writing code, state your plan in a few bullets and wait for my confirmation if the change touches more than one module.
- Write or update tests first for any new behavior. I will review tests before you implement.
- Run `ruff check . && pytest -q` before declaring anything done. If tests fail, fix them, do not weaken assertions to make them pass.
- Small commits with imperative messages ("add sqlite cache keyed on request hash"). Never combine a refactor and a feature in one commit.
- If you make a non-obvious design decision mid-task, append one line explaining it to `NOTES.md`.
- Do not edit `docs/design/*` unless I explicitly ask. Those are my specs, not living docs.
- Do not add a web UI, dashboard, or extra providers. Out of scope for v0.1 no matter how tempting.

## Definition of done for any task

1. `ruff check . && ruff format --check . && pytest -q` all passing, matching CI exactly.
2. Public functions have docstrings (one line + args if non-obvious).
3. NOTES.md updated if a decision was made.
4. No TODOs left in code without an issue-style note in NOTES.md.

## Architecture additions

- `gradetrail/viewer/server.py` - Run discovery, the dataset join, and the three HTTP endpoints (stdlib http.server only). All filesystem reading and join logic lives here, testable without a browser.
- `gradetrail/viewer/static/index.html` - The entire frontend: one file, all HTML/CSS/JS inline. No other frontend files may be created.
- `gradetrail/cli.py` - gains the `view` subcommand, thin wiring only, same rule as `run`.

Design doc: docs/design/viewer.md. Read it before touching any viewer file. The HTTP contract in that doc is the public interface; do not change response shapes without asking me.

## Viewer conventions

- Zero new dependencies. Standard library only for the server. No pip installs, no npm, no build step, no CDN URLs in the HTML (the page must work offline).
- One HTML file, hard rule. If a change seems to require a second frontend file, stop and ask.
- Vanilla JS. No frameworks, no TypeScript, no JSX. Small pure functions, DOM building via createElement or template strings. The frontend communicates only through the documented HTTP API in docs/design/viewer.md; that API is the compatibility boundary and its semantics change only by updating the doc first.
- Every view and filter state must be URL-addressable (deep links are a feature requirement, not a style choice); use hash routing. Fetch a run's data once per view and work from memory; do not refetch per click.
- The server joins results to the dataset by sample_id via the manifest, and exposes dataset-hash mismatches and parse errors as warnings in the API rather than silently joining or crashing; see the design doc for exact semantics.
- Frontend style: system font stack, tables not cards, no animation, no emoji, high information density. It should look like a tool, not a product.
- Escape all user/model-derived text before inserting into the DOM (response_text and dataset fields are untrusted content; use textContent or an escape helper, never innerHTML with raw data).
- The server binds 127.0.0.1 only, never 0.0.0.0. Validate the run-directory path parameter against the discovered list; reject anything else with 404.
- Server errors degrade, never 500 on bad data: an unreadable results.jsonl line, a missing dataset, or a malformed manifest becomes a visible per-run or per-sample warning in the JSON, not an exception.
- Endpoint tests live in tests/viewer/, using tmp-path synthetic run directories. The HTML file itself is not unit-tested; its contract (the endpoints) is.

## Branch workflow (viewer)

- All viewer work happens on the `results-viewer` branch. Never commit viewer work to main.
- Start every session with `git checkout results-viewer` and confirm with `git branch --show-current` before any commit.
- Keep the branch current: if main moves (a gradetrail fix ships), rebase or merge main into results-viewer at session start and tell me it happened.
- Same commit discipline as always: small commits, imperative messages, push to `origin/results-viewer` after each. CI runs on the branch via the pull_request trigger once a PR is open; open a draft PR after milestone 1 so every push gets CI.
- Merge to main only when v1 (all five milestones) is done, via the PR. The PR includes the version bump to 0.2.0 before merge.
