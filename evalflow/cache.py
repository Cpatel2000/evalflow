"""SQLite response cache, keyed by resolved request identity.

See docs/design/eval-spec.md rule 2: the cache key is per-sample sha256 of
canonical JSON of (provider, model, base_url, resolved prompt, params).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import aiosqlite

from evalflow.errors import CacheError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(provider: str, model: str, base_url: str | None, prompt: str, params: dict) -> str:
    """Per-sample cache key: sha256 of canonical JSON of the resolved request."""
    payload = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "prompt": prompt,
        "params": params,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """A cached provider response and the (Python-generated, UTC) time it was written."""

    response: dict
    created_at: datetime


class ResponseCache:
    """Async SQLite cache of provider responses, keyed by request identity.

    Opens the database in WAL journal mode with a busy timeout so multiple
    connections to the same cache file (e.g. concurrent runner workers) don't
    fail with "database is locked" under normal contention.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        conn = await aiosqlite.connect(self._path)
        async with conn.execute("PRAGMA journal_mode=WAL") as cursor:
            (mode,) = await cursor.fetchone()
        if mode.lower() != "wal":
            await conn.close()
            raise CacheError(f"could not enable WAL journal mode (got {mode!r}) at {self._path}")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute(_SCHEMA)
        await conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> ResponseCache:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise CacheError("ResponseCache used before connect() (or after close())")
        return self._conn

    async def get(
        self, provider: str, model: str, base_url: str | None, prompt: str, params: dict
    ) -> CacheEntry | None:
        """Look up a cached response for this exact request, or None on a miss."""
        conn = self._require_conn()
        key = cache_key(provider, model, base_url, prompt, params)
        async with conn.execute(
            "SELECT response, created_at FROM responses WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        response_json, created_at_text = row
        return CacheEntry(
            response=json.loads(response_json),
            created_at=datetime.fromisoformat(created_at_text),
        )

    async def put(
        self,
        provider: str,
        model: str,
        base_url: str | None,
        prompt: str,
        params: dict,
        response: dict,
    ) -> None:
        """Store a response for this request, overwriting any prior entry for the same key.

        response must be JSON-serializable as-is; the cache's contract is that
        get() returns exactly what was put(), so non-native types (e.g. datetime)
        raise CacheError here rather than being silently coerced to strings.
        """
        conn = self._require_conn()
        key = cache_key(provider, model, base_url, prompt, params)
        created_at = datetime.now(UTC).isoformat()
        try:
            response_json = json.dumps(response)
        except TypeError as e:
            raise CacheError(f"response for key {key} is not JSON-serializable: {e}") from None
        await conn.execute(
            """
            INSERT INTO responses (key, response, created_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                response = excluded.response, created_at = excluded.created_at
            """,
            (key, response_json, created_at),
        )
        await conn.commit()
