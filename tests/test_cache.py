"""Tests for evalflow.cache: SQLite response cache and its key derivation.

Encodes design doc rule 2 (docs/design/eval-spec.md): the cache key is per-sample
sha256 of canonical JSON of (provider, model, base_url, resolved prompt, params).
Changing any one component, and nothing else, must change the key.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import evalflow.cache as cache_module
from evalflow.cache import ResponseCache, cache_key
from evalflow.errors import CacheError

BASE_KWARGS = dict(
    provider="anthropic",
    model="claude-sonnet-4-6",
    base_url=None,
    prompt="Solve this.\n\n2+2?",
    params={"max_tokens": 1024, "temperature": 0.0},
)

RESPONSE = {"text": "Answer: 4", "input_tokens": 12, "output_tokens": 4, "latency_ms": 812.5}


# --- cache_key ---------------------------------------------------------------


def test_cache_key_is_stable_sha256_hex() -> None:
    key = cache_key(**BASE_KWARGS)
    assert len(key) == 64
    int(key, 16)  # valid hex
    assert cache_key(**BASE_KWARGS) == key  # deterministic across calls


def test_cache_key_ignores_params_insertion_order() -> None:
    reordered = dict(BASE_KWARGS, params={"temperature": 0.0, "max_tokens": 1024})
    assert cache_key(**BASE_KWARGS) == cache_key(**reordered)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "openai"},
        {"model": "claude-opus-4-8"},
        {"base_url": "http://localhost:8000"},
        {"prompt": "Solve this.\n\n3+3?"},
        {"params": {"max_tokens": 1024, "temperature": 1.0}},
        {"params": {"max_tokens": 2048, "temperature": 0.0}},
    ],
)
def test_cache_key_changes_when_any_component_changes(overrides: dict) -> None:
    changed = dict(BASE_KWARGS, **overrides)
    assert cache_key(**BASE_KWARGS) != cache_key(**changed)


# --- ResponseCache: get/put ----------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite3"


async def test_miss_on_empty_cache(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        assert await cache.get(**BASE_KWARGS) is None


async def test_put_then_get_round_trips_response(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=RESPONSE)
        entry = await cache.get(**BASE_KWARGS)
        assert entry is not None
        assert entry.response == RESPONSE


async def test_get_misses_when_any_key_component_differs(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=RESPONSE)
        changed = dict(BASE_KWARGS, prompt="Solve this.\n\n3+3?")
        assert await cache.get(**changed) is None


async def test_put_overwrites_existing_entry_for_same_key(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=RESPONSE)
        updated = {**RESPONSE, "text": "Answer: 5"}
        await cache.put(**BASE_KWARGS, response=updated)
        entry = await cache.get(**BASE_KWARGS)
        assert entry.response == updated


async def test_round_trips_nested_and_unicode_payloads(db_path: Path) -> None:
    payload = {"text": "Réponse: café ☕", "meta": {"tags": ["a", "b"], "n": 3}}
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=payload)
        entry = await cache.get(**BASE_KWARGS)
        assert entry.response == payload


async def test_cache_persists_across_connections(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=RESPONSE)

    async with ResponseCache(db_path) as reopened:
        entry = await reopened.get(**BASE_KWARGS)
        assert entry is not None
        assert entry.response == RESPONSE


async def test_connect_creates_schema_on_fresh_file(db_path: Path) -> None:
    assert not db_path.exists()
    async with ResponseCache(db_path) as cache:
        assert await cache.get(**BASE_KWARGS) is None
    assert db_path.exists()


async def test_entry_records_created_at_as_utc_datetime(db_path: Path) -> None:
    before = datetime.now(UTC)
    async with ResponseCache(db_path) as cache:
        await cache.put(**BASE_KWARGS, response=RESPONSE)
        entry = await cache.get(**BASE_KWARGS)
    after = datetime.now(UTC)
    assert entry is not None
    assert before <= entry.created_at <= after


async def test_get_before_connect_raises_cache_error(db_path: Path) -> None:
    cache = ResponseCache(db_path)
    with pytest.raises(CacheError):
        await cache.get(**BASE_KWARGS)


async def test_put_before_connect_raises_cache_error(db_path: Path) -> None:
    cache = ResponseCache(db_path)
    with pytest.raises(CacheError):
        await cache.put(**BASE_KWARGS, response=RESPONSE)


async def test_put_with_non_json_serializable_response_raises_cache_error(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        with pytest.raises(CacheError):
            await cache.put(**BASE_KWARGS, response={"when": datetime.now(UTC)})


async def test_connect_enables_wal_journal_mode(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        conn = cache._conn
        assert conn is not None
        async with conn.execute("PRAGMA journal_mode") as cursor:
            (mode,) = await cursor.fetchone()
        assert mode.lower() == "wal"


# --- connect(): retry on transient lock contention -----------------------------


class _RaiseOnAwait:
    """Awaiting this raises the given exception -- lets aiosqlite.connect(...)
    itself (which returns something awaitable) fail without a real connection."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __await__(self):
        async def _raise() -> None:
            raise self._exc

        return _raise().__await__()


async def test_connect_retries_transient_operational_error_then_succeeds(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(cache_module.asyncio, "sleep", instant_sleep)

    real_connect = cache_module.aiosqlite.connect
    calls = {"n": 0}

    def flaky_connect(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 2:
            return _RaiseOnAwait(sqlite3.OperationalError("database is locked"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(cache_module.aiosqlite, "connect", flaky_connect)

    async with ResponseCache(db_path) as cache:
        # the pragmas ran for real on the successful (3rd) attempt: WAL is
        # enabled and a genuine put/get round-trip works, not just "no exception".
        conn = cache._conn
        assert conn is not None
        async with conn.execute("PRAGMA journal_mode") as cursor:
            (mode,) = await cursor.fetchone()
        assert mode.lower() == "wal"

        await cache.put(**BASE_KWARGS, response=RESPONSE)
        entry = await cache.get(**BASE_KWARGS)
        assert entry is not None
        assert entry.response == RESPONSE

    assert calls["n"] == 3  # 2 simulated failures + 1 real success


async def test_connect_raises_cache_error_after_exhausting_retries(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(cache_module.asyncio, "sleep", instant_sleep)

    calls = {"n": 0}

    def always_flaky_connect(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return _RaiseOnAwait(sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(cache_module.aiosqlite, "connect", always_flaky_connect)

    cache = ResponseCache(db_path)
    with pytest.raises(CacheError) as exc_info:
        await cache.connect()

    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)
    assert calls["n"] == cache_module._CONNECT_MAX_ATTEMPTS  # bounded, not unbounded retry


async def test_two_connections_to_same_file_share_cache_state(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache_a, ResponseCache(db_path) as cache_b:
        kwargs_list = [dict(BASE_KWARGS, prompt=f"Solve this.\n\n{i}+{i}?") for i in range(20)]

        async def put_via(cache: ResponseCache, i: int, kw: dict) -> None:
            await cache.put(**kw, response={**RESPONSE, "text": f"Answer: {2 * i}"})

        # Alternate which connection writes each key so both connections issue
        # concurrent writes to the same file (exercises WAL + busy_timeout).
        await asyncio.gather(
            *(
                put_via(cache_a if i % 2 == 0 else cache_b, i, kw)
                for i, kw in enumerate(kwargs_list)
            )
        )

        results_a = await asyncio.gather(*(cache_a.get(**kw) for kw in kwargs_list))
        results_b = await asyncio.gather(*(cache_b.get(**kw) for kw in kwargs_list))
        for i, (entry_a, entry_b) in enumerate(zip(results_a, results_b, strict=True)):
            assert entry_a is not None and entry_b is not None
            assert entry_a.response["text"] == entry_b.response["text"] == f"Answer: {2 * i}"


async def test_concurrent_get_and_put_do_not_corrupt_cache(db_path: Path) -> None:
    async with ResponseCache(db_path) as cache:
        kwargs_list = [dict(BASE_KWARGS, prompt=f"Solve this.\n\n{i}+{i}?") for i in range(20)]

        async def put_one(i: int, kw: dict) -> None:
            await cache.put(**kw, response={**RESPONSE, "text": f"Answer: {2 * i}"})

        await asyncio.gather(*(put_one(i, kw) for i, kw in enumerate(kwargs_list)))

        results = await asyncio.gather(*(cache.get(**kw) for kw in kwargs_list))
        for i, entry in enumerate(results):
            assert entry is not None
            assert entry.response["text"] == f"Answer: {2 * i}"
