"""CLI tests for `gradetrail view` (docs/design/viewer.md, CLI section).

The server loop is neutralized by patching serve_forever at the socketserver
base class, so `view` runs its full wiring (discover -> bind -> maybe open
browser) and returns immediately. webbrowser.open is patched at its module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gradetrail.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_serve_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """serve_forever returns immediately; nothing ever blocks or accepts requests."""
    monkeypatch.setattr("socketserver.BaseServer.serve_forever", lambda self, **kw: None)


@pytest.fixture
def opened_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url, **kw: urls.append(url) or True)
    return urls


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    """A root with one minimal but complete run directory."""
    run_dir = tmp_path / "results" / "qa-abc12345"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"name": "qa"}))
    (run_dir / "results.jsonl").write_text("")
    return tmp_path / "results"


def test_view_no_runs_exits_1_with_message(tmp_path: Path, opened_urls: list[str]) -> None:
    empty_root = tmp_path / "results"
    empty_root.mkdir()
    result = runner.invoke(app, ["view", str(empty_root)])
    assert result.exit_code == 1
    assert "no run directories" in result.output.lower()
    assert opened_urls == []  # exits before serving; never opens a browser


def test_view_missing_root_exits_1_with_message(tmp_path: Path) -> None:
    result = runner.invoke(app, ["view", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "no run directories" in result.output.lower()


def test_view_opens_browser_by_default(results_root: Path, opened_urls: list[str]) -> None:
    result = runner.invoke(app, ["view", str(results_root), "--port", "0"])
    assert result.exit_code == 0
    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("http://127.0.0.1:")


def test_view_no_browser_suppresses_webbrowser_open(
    results_root: Path, opened_urls: list[str]
) -> None:
    result = runner.invoke(app, ["view", str(results_root), "--port", "0", "--no-browser"])
    assert result.exit_code == 0
    assert opened_urls == []


def test_view_prints_the_serving_url(results_root: Path, opened_urls: list[str]) -> None:
    result = runner.invoke(app, ["view", str(results_root), "--port", "0", "--no-browser"])
    assert "http://127.0.0.1:" in result.output
