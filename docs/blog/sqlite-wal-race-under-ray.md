**DRAFT. Voice pass pending.**

# A 1-in-5 flake: debugging a SQLite WAL race under Ray

## The symptom

gradetrail's Ray backend test suite boots a real, non-local-mode Ray cluster (`ray.init(local_mode=False, num_cpus=2)`) and runs the runner tests against it. One of those tests checks that multiple Ray workers writing to the same SQLite-backed response cache produce results a later run can read back correctly.

It failed intermittently: not every run, roughly one in four to seven. The failure was always the same shape. A sample that should have scored came back as `provider_error`, with a detail message reading:

```
worker failed: RayTaskError(OperationalError)(OperationalError('database is locked'))
```

This is precisely the failure the cache's WAL (write-ahead log) journal mode and `busy_timeout` pragma exist to prevent. The cache had already been checked for concurrent-connection safety once before, including a same-process two-connection test. This was different: a real Ray cluster with genuinely separate worker processes, contending on one shared SQLite file for the first time.

## First hypothesis: pragma order (wrong)

`ResponseCache.connect()` set `journal_mode=WAL` before `busy_timeout`:

```python
async with conn.execute("PRAGMA journal_mode=WAL") as cursor:
    (mode,) = await cursor.fetchone()
...
await conn.execute("PRAGMA busy_timeout=5000")
```

`busy_timeout` defaults to 0 on a fresh connection. If the WAL pragma runs before `busy_timeout` is set, that one statement executes with a zero-length timeout. The obvious fix was to reorder:

```python
await conn.execute("PRAGMA busy_timeout=5000")
async with conn.execute("PRAGMA journal_mode=WAL") as cursor:
    (mode,) = await cursor.fetchone()
```

This is correct hygiene and worth keeping regardless of whether it fixes anything. It did not fix anything. A 20-iteration loop of the real Ray-marked suite after the reorder still failed 6 times out of 20, statistically indistinguishable from before the change. The hypothesis did not survive contact with a loop.

## Isolating the mechanism outside Ray

Rather than keep debugging inside the combined Ray, pytest, asyncio, and aiosqlite stack, the next step was to strip all of that away and reproduce the underlying SQLite behavior directly, with nothing but the standard library:

```python
def _worker(db_path, worker_id, n_writes, results):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        ...
    except sqlite3.OperationalError as exc:
        results[worker_id] = f"OperationalError: {exc}"
```

Spawn three of these as separate `multiprocessing.Process` workers, all pointed at the same brand-new, empty SQLite file, all racing to run `PRAGMA journal_mode=WAL` at effectively the same instant. `busy_timeout=5000` is set on every connection before the WAL pragma runs: the already-"fixed" order.

Run it 15 times. On one of them:

```
sqlite3.OperationalError: database is locked
```

on the `journal_mode=WAL` line. No Ray, no asyncio, no aiosqlite anywhere in the process. The mechanism was real, and it had nothing to do with Ray or with the pragma order this project controls.

## The actual mechanism

A fresh SQLite file starts in the default rollback-journal mode. Switching a connection to WAL mode is not an ordinary read or write. It briefly requires an exclusive lock to rewrite the database header, because the on-disk format itself changes. `busy_timeout` governs SQLite's normal busy-handler retry loop, the one that applies to ordinary lock contention once a database is already settled into a journal mode. It does not reliably cover this one-time, first-ever mode switch, which sits outside that path.

In practice, when several connections open a brand-new file at nearly the same moment and all attempt the initial WAL switch, more than one can hit `SQLITE_BUSY` in a way `busy_timeout` does not smooth over. Once a file is already in WAL mode, later connections requesting WAL mode again are cheap no-ops and this race no longer applies. That is exactly why the failure only ever showed up against a fresh cache file, and why reproducing it needed genuinely separate processes, not threads or coroutines, racing the very first connection.

## The fix, in two layers

The correctness fix belongs in `ResponseCache.connect()` itself, not in whichever caller happens to connect first. If the fix lived only in the Ray runner, the next concurrent entry point (two `gradetrail run` invocations started at once, a future distributed backend) would silently reintroduce the same race.

```python
async def connect(self) -> None:
    last_exc = None
    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        try:
            self._conn = await self._try_connect()
            return
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if attempt == _CONNECT_MAX_ATTEMPTS:
                break
            await asyncio.sleep(random.uniform(_CONNECT_RETRY_MIN_S, _CONNECT_RETRY_MAX_S))
    raise CacheError(
        f"could not connect to {self._path} after {_CONNECT_MAX_ATTEMPTS} attempts "
        f"(likely concurrent connections racing the initial WAL switch): {last_exc}"
    ) from last_exc
```

Five attempts, a short jittered backoff (50-200ms, since the race window itself is on the order of milliseconds), and a `CacheError` chained from the last underlying exception if every attempt fails. A genuinely non-transient failure to enable WAL, an unsupported filesystem, for example, is not retried: only `sqlite3.OperationalError` triggers a retry, everything else raises immediately.

On top of that, `RayRunner.run()` pre-warms the cache. It opens and closes a connection to the cache path once in the driver, before dispatching any worker, so the file is already in WAL mode by the time workers connect. This is an optimization on top of the fix, not a substitute for it: it turns the retry path into a safety net that almost never fires under real contention, rather than a hot path every worker has to negotiate at startup.

## Verification

The bounded retry is tested by mocking `aiosqlite.connect` to raise `sqlite3.OperationalError` a controlled number of times before succeeding, or always, rather than depending on timing, since the underlying race is not deterministically reproducible in a test suite. One test confirms recovery after two simulated failures, with a real WAL check and a real put/get round-trip afterward proving the pragmas actually ran on the successful attempt. Another confirms exactly `_CONNECT_MAX_ATTEMPTS` calls before a chained `CacheError`.

The empirical confirmation was a loop, not a single run: `pytest -m ray` against the real cluster, 25 consecutive times, zero failures, down from roughly one failure in four to seven before the fix.

## Takeaway

The first, most plausible hypothesis, pragma order, was wrong, and a single passing test run would not have revealed that; only a loop did. The actual mechanism was a real SQLite and WAL edge case, isolated by deliberately removing every layer that was not necessary to reproduce it, Ray, asyncio, aiosqlite, down to a 30-line multiprocessing script. The fix belongs at the layer that owns the invariant, the cache, not the layer that happened to trigger it first, the Ray runner.
