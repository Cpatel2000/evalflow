# evalflow

> Rename "evalflow" everywhere once you pick a final name.

A distributed LLM evaluation harness that treats evals like tests: declarative, cached, reproducible.

## What this project is

A Python library + CLI. A user defines an eval in a YAML spec (dataset, prompt template, model, scorer), runs it with `evalflow run spec.yaml`, and gets scored results plus a reproducibility manifest. Runs are cached, multi-provider, and can execute locally (asyncio) or distributed (Ray) behind the same interface.

Target user: an ML researcher who wants eval results in 30 seconds of setup, not a platform to administer.

## Architecture

Modules and their single responsibility. Do not blur these boundaries.

- `evalflow/spec.py` - Eval spec dataclasses + YAML loading/validation. The spec is the public contract. NEVER change the spec schema without asking me first.
- `evalflow/providers/` - Provider abstraction. One base class, one module per provider (`anthropic.py`, `openai.py`, `openai_compatible.py`). ALL model API calls go through this layer. No direct HTTP/SDK calls anywhere else in the codebase.
- `evalflow/scorers/` - Scoring. `deterministic.py` (exact match, regex, numeric tolerance) and `judge.py` (LLM-as-judge). Judge prompts are versioned artifacts referenced by the spec, never inlined ad hoc.
- `evalflow/cache.py` - SQLite response cache. Key = sha256 of canonical JSON of (provider, model, resolved prompt, sampling params). Cache logic lives here and only here.
- `evalflow/runner/` - Execution. `base.py` defines the Runner interface, `local.py` (asyncio + semaphore concurrency), `ray_runner.py` (Ray backend, added in week 3). Runners are swappable via config, callers never know which one they got.
- `evalflow/manifest.py` - Run manifests: spec hash, model versions, timestamps, seed, git SHA, package version. Every run writes one.
- `evalflow/results.py` - Results model + JSONL writing + summary stats (scores, cost, tokens, failure rate).
- `evalflow/cli.py` - Thin CLI (typer). No business logic here, it only wires modules together.

Design docs live in `docs/design/`. Read the relevant one before implementing a module.

## Conventions

- Python 3.11+. Type hints on every function signature. `from __future__ import annotations` at the top of each module.
- Prefer functions and frozen dataclasses over classes with mutable state. Only use a class when there is real state or a real interface (providers, runners).
- Errors: raise specific exceptions from `evalflow/errors.py`. Never bare `except:`. Never swallow exceptions silently.
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
