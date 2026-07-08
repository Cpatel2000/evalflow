"""Provider abstraction: the retry/backoff loop shared by every model provider.

Concrete providers (gradetrail/providers/anthropic.py, etc.) only implement one
raw attempt (_complete) and how to classify a raw SDK exception (classify).
Everything about retrying, timing out, backing off, and logging lives here,
once, so no provider module reimplements it.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import structlog

from gradetrail.errors import ProviderError
from gradetrail.spec import ModelParams

_BASE_DELAY_S = 0.5
_MAX_DELAY_S = 20.0
_RETRY_AFTER_CAP_S = 60.0

_log = structlog.get_logger(__name__)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a Retry-After delay (seconds) from an SDK exception's response
    headers, if present and parseable. None otherwise -- caller falls back to
    normal backoff silently.

    Works identically for the openai and anthropic SDKs without importing
    either here: both APIStatusError subclasses carry `.response: httpx.Response`
    (confirmed by reading both SDKs' _exceptions.py), and httpx.Headers is
    case-insensitive, so `.get("retry-after")` matches however the server
    actually cased it. APIConnectionError/APITimeoutError (and a plain
    asyncio TimeoutError) carry no `.response` at all, so getattr falls
    through to None cleanly for those rather than raising.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ProviderResponse:
    """One completed provider call. Tokens only — no dollar cost (that's results.py)."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str  # as reported by the API response, not necessarily the requested name


class Provider(ABC):
    """Base class for model providers: owns retries, timeout, backoff, and logging.

    Subclasses implement _complete() (one raw attempt against the SDK) and
    classify() (retryable vs fatal for a raw SDK exception). They must not
    implement their own retry loop.
    """

    def __init__(self, *, model: str, max_retries: int, timeout_s: float) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    @abstractmethod
    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        """One raw attempt against the provider SDK. No retry/timeout handling here."""

    @abstractmethod
    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        """Classify a raw SDK exception. Never called for a timeout (see complete())."""

    async def complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        """Run _complete() with timeout, retry, and backoff per run.max_retries/timeout_s.

        Raises ProviderError, chained from the last underlying exception, once
        a fatal error is classified or retries are exhausted. A cancellation of
        the enclosing task (asyncio.CancelledError) is never classified or
        retried — it propagates immediately, as required for cooperative
        cancellation under concurrent execution.
        """
        total_attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._complete(prompt, params), timeout=self.timeout_s
                )
            except TimeoutError as exc:
                last_exc = exc
                retryable = True
            except Exception as exc:  # noqa: BLE001 - reclassified immediately below
                last_exc = exc
                retryable = self.classify(exc) == "retryable"
            else:
                self._log_attempt(
                    attempt=attempt,
                    latency_ms=response.latency_ms,
                    response=response,
                    outcome="success",
                )
                return response

            latency_ms = (time.monotonic() - start) * 1000
            if not retryable:
                self._log_attempt(
                    attempt=attempt, latency_ms=latency_ms, response=None, outcome="fatal_error"
                )
                raise ProviderError(
                    f"{self.model}: non-retryable error on attempt "
                    f"{attempt}/{total_attempts}: {last_exc}"
                ) from last_exc
            if attempt == total_attempts:
                self._log_attempt(
                    attempt=attempt,
                    latency_ms=latency_ms,
                    response=None,
                    outcome="retries_exhausted",
                )
                raise ProviderError(
                    f"{self.model}: exhausted {self.max_retries} retries: {last_exc}"
                ) from last_exc
            self._log_attempt(
                attempt=attempt, latency_ms=latency_ms, response=None, outcome="retrying"
            )
            await self._backoff_sleep(attempt, last_exc)

        raise AssertionError("unreachable: loop always returns or raises above")

    async def _backoff_sleep(self, attempt: int, exc: Exception) -> None:
        """Sleep before the next retry attempt.

        A Retry-After header on `exc` (see _retry_after_seconds) takes
        priority over the computed exponential backoff: sleeps for
        max(retry_after, computed_backoff), capped at _RETRY_AFTER_CAP_S.
        Absent or unparseable, falls back to the ordinary jittered
        exponential backoff silently.
        """
        backoff = min(_MAX_DELAY_S, _BASE_DELAY_S * 2 ** (attempt - 1))
        delay = random.uniform(0, backoff)
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(_RETRY_AFTER_CAP_S, max(retry_after, delay))
        await asyncio.sleep(delay)

    def _log_attempt(
        self, *, attempt: int, latency_ms: float, response: ProviderResponse | None, outcome: str
    ) -> None:
        _log.info(
            "provider_attempt",
            model=self.model,
            latency_ms=round(latency_ms, 2),
            tokens=(
                {"input": response.input_tokens, "output": response.output_tokens}
                if response is not None
                else None
            ),
            outcome=outcome,
            attempt=attempt,
        )
