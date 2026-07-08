"""Reproduction case for the SQLite WAL-mode-switch race documented in
NOTES.md (2026-07-08, evalflow/cache.py).

Multiple connections racing to perform the *first* switch of a brand-new
SQLite file from its default rollback journal to WAL mode can hit
sqlite3.OperationalError("database is locked") even with busy_timeout set
*before* the journal_mode pragma. busy_timeout governs ordinary read/write
lock contention; it does not reliably cover this one-time mode-switch, which
briefly requires an exclusive lock to rewrite the database header. This
script reproduces that directly with plain sqlite3 and multiprocessing --
no Ray, no aiosqlite, no asyncio -- to isolate the mechanism from anything
Ray- or evalflow-specific.

The actual fix in evalflow lives in ResponseCache.connect()'s bounded retry
(tests/test_cache.py mocks that retry path directly, since the race itself
isn't deterministically reproducible on every run -- typically once every
handful of trials on ordinary hardware, as this script demonstrates).

Usage: python scripts/repro_sqlite_wal_race.py [n_workers] [n_trials]
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import sys
import tempfile
from pathlib import Path


def _worker(db_path: str, worker_id: str, n_writes: int, results: dict) -> None:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if mode.lower() != "wal":
            results[worker_id] = f"failed to enable WAL, got {mode!r}"
            return
        conn.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.commit()
        for i in range(n_writes):
            conn.execute(
                "INSERT INTO responses (key, response, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET response = excluded.response",
                (f"{worker_id}-{i}", "{}", "2026-01-01"),
            )
            conn.commit()
        conn.close()
        results[worker_id] = "ok"
    except sqlite3.OperationalError as exc:
        results[worker_id] = f"OperationalError: {exc}"


def run_trial(n_workers: int, n_writes: int) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "race.sqlite")
        with multiprocessing.Manager() as manager:
            results = manager.dict()
            procs = [
                multiprocessing.Process(
                    target=_worker, args=(db_path, chr(65 + i), n_writes, results)
                )
                for i in range(n_workers)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=10)
            return [f"{k}: {v}" for k, v in sorted(results.items())]


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    hits = 0
    for trial in range(1, n_trials + 1):
        outcomes = run_trial(n_workers, n_writes=5)
        failed = [o for o in outcomes if "OperationalError" in o]
        if failed:
            hits += 1
            print(f"trial {trial}: RACE REPRODUCED -- {failed}")
        else:
            print(f"trial {trial}: ok")

    print(f"\n{hits}/{n_trials} trials hit the race ({n_workers} workers per trial)")


if __name__ == "__main__":
    main()
