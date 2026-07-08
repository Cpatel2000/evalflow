"""Tests for gradetrail.providers.base: the retry/backoff loop shared by all providers.

Concrete providers only implement _complete() (one raw attempt) and classify()
(retryable vs fatal). This file drives that contract with fakes so the retry
algorithm — timeout handling, backoff, exhaustion, logging — is tested
independently of any real SDK.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
import structlog

import gradetrail.providers.base as base_module
from gradetrail.errors import ProviderError
from gradetrail.providers.base import Provider, ProviderResponse
from gradetrail.spec import ModelParams

MODEL = "claude-sonnet-4-6"
PARAMS = ModelParams(max_tokens=1024, temperature=0.0)

RESPONSE = ProviderResponse(
    text="Answer: 4", input_tokens=10, output_tokens=4, latency_ms=42.0, model=MODEL
)


class FakeError(Exception):
    """Stand-in for an SDK exception, tagged with how it should be classified."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class FakeProvider(Provider):
    """Replays a scripted sequence of outcomes: exceptions, then a final response."""

    def __init__(self, outcomes: list[Exception | ProviderResponse], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._outcomes = outcomes
        self.calls = 0

    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        assert isinstance(exc, FakeError)
        return "retryable" if exc.retryable else "fatal"


class HangingProvider(Provider):
    """Blocks forever on the first `hangs` calls, then responds.

    Uses an Event that is never set (not asyncio.sleep) so this is immune to
    the no_real_sleep autouse fixture below and only ever ends via wait_for's
    own timeout cancellation in complete().
    """

    def __init__(self, *, response: ProviderResponse, hangs: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._response = response
        self._hangs_remaining = hangs
        self.calls = 0

    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        self.calls += 1
        if self._hangs_remaining > 0:
            self._hangs_remaining -= 1
            await asyncio.Event().wait()
        return self._response

    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        return "fatal"  # never consulted: timeouts are handled before classify()


class CancellableProvider(Provider):
    """Blocks forever so a test can cancel the enclosing task mid-attempt."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.calls = 0
        self.started = asyncio.Event()

    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable: should have been cancelled first")

    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        raise AssertionError("classify() must not be called for cancellation")


def make_provider(cls: type[Provider], *, max_retries: int = 5, timeout_s: float = 5.0, **kwargs):
    return cls(model=MODEL, max_retries=max_retries, timeout_s=timeout_s, **kwargs)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff sleeps are real seconds; replace with a no-op so tests run instantly."""

    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(base_module.asyncio, "sleep", instant_sleep)


# --- happy path ----------------------------------------------------------------


async def test_returns_response_on_first_success() -> None:
    provider = make_provider(FakeProvider, outcomes=[RESPONSE])
    result = await provider.complete("2+2?", PARAMS)
    assert result == RESPONSE
    assert provider.calls == 1


# --- retry behavior --------------------------------------------------------------


async def test_retries_retryable_error_then_succeeds() -> None:
    provider = make_provider(
        FakeProvider,
        outcomes=[FakeError("rate limited", retryable=True), RESPONSE],
        max_retries=3,
    )
    result = await provider.complete("2+2?", PARAMS)
    assert result == RESPONSE
    assert provider.calls == 2


async def test_fatal_error_raises_immediately_without_retry() -> None:
    provider = make_provider(
        FakeProvider,
        outcomes=[FakeError("bad request", retryable=False), RESPONSE],
        max_retries=5,
    )
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete("2+2?", PARAMS)
    assert provider.calls == 1  # no retry attempted
    assert isinstance(exc_info.value.__cause__, FakeError)


async def test_exhausts_retries_and_raises_provider_error() -> None:
    errors = [FakeError(f"attempt {i}", retryable=True) for i in range(4)]
    provider = make_provider(FakeProvider, outcomes=errors, max_retries=3)
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete("2+2?", PARAMS)
    assert provider.calls == 4  # 1 initial + 3 retries
    assert exc_info.value.__cause__ is errors[-1]


async def test_max_retries_zero_means_single_attempt() -> None:
    provider = make_provider(
        FakeProvider,
        outcomes=[FakeError("rate limited", retryable=True)],
        max_retries=0,
    )
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", PARAMS)
    assert provider.calls == 1


@pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
async def test_total_attempts_equals_max_retries_plus_one(max_retries: int) -> None:
    errors = [FakeError(f"attempt {i}", retryable=True) for i in range(max_retries + 1)]
    provider = make_provider(FakeProvider, outcomes=errors, max_retries=max_retries)
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", PARAMS)
    assert provider.calls == max_retries + 1


# --- timeout handling --------------------------------------------------------------


async def test_timeout_is_treated_as_retryable_regardless_of_classify() -> None:
    provider = make_provider(
        HangingProvider, response=RESPONSE, hangs=1, max_retries=2, timeout_s=0.02
    )
    result = await provider.complete("2+2?", PARAMS)
    assert result == RESPONSE
    assert provider.calls == 2


async def test_timeout_exhausting_retries_raises_provider_error() -> None:
    provider = make_provider(
        HangingProvider, response=RESPONSE, hangs=99, max_retries=1, timeout_s=0.02
    )
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", PARAMS)
    assert provider.calls == 2  # 1 initial + 1 retry, both timed out


async def test_cancellation_propagates_without_retry() -> None:
    provider = make_provider(CancellableProvider, max_retries=3, timeout_s=5.0)
    task = asyncio.create_task(provider.complete("2+2?", PARAMS))
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == 1  # cancellation is not classified and burns no retry


# --- backoff -------------------------------------------------------------------------


async def test_backoff_grows_exponentially_and_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def capturing_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(base_module.asyncio, "sleep", capturing_sleep)
    base_module.random.seed(1234)  # seeded: bounds below are reproducible, not just probable

    num_retries = 8  # 0.5 * 2**7 = 64, well past the 20s cap, so the cap actually gets exercised
    errors = [FakeError(f"attempt {i}", retryable=True) for i in range(num_retries)]
    provider = make_provider(FakeProvider, outcomes=[*errors, RESPONSE], max_retries=num_retries)

    await provider.complete("2+2?", PARAMS)

    assert len(delays) == num_retries
    caps = [
        min(base_module._MAX_DELAY_S, base_module._BASE_DELAY_S * 2**i) for i in range(num_retries)
    ]
    for delay, cap in zip(delays, caps, strict=True):
        assert 0 <= delay <= cap
    assert caps[-1] == base_module._MAX_DELAY_S  # backoff caps out rather than growing forever


# --- structured logging ------------------------------------------------------------------


async def test_logs_structured_fields_on_retry_and_success() -> None:
    provider = make_provider(
        FakeProvider,
        outcomes=[FakeError("rate limited", retryable=True), RESPONSE],
        max_retries=3,
    )
    with structlog.testing.capture_logs() as logs:
        await provider.complete("2+2?", PARAMS)

    assert len(logs) == 2
    retry_log, success_log = logs
    assert retry_log["outcome"] == "retrying"
    assert retry_log["attempt"] == 1
    assert retry_log["model"] == MODEL
    assert retry_log["tokens"] is None
    assert isinstance(retry_log["latency_ms"], float)

    assert success_log["outcome"] == "success"
    assert success_log["attempt"] == 2
    assert success_log["tokens"] == {"input": 10, "output": 4}


async def test_logs_fatal_error_outcome() -> None:
    provider = make_provider(FakeProvider, outcomes=[FakeError("bad request", retryable=False)])
    with structlog.testing.capture_logs() as logs, pytest.raises(ProviderError):
        await provider.complete("2+2?", PARAMS)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "fatal_error"
    assert logs[0]["attempt"] == 1


async def test_logs_retries_exhausted_outcome() -> None:
    errors = [FakeError(f"attempt {i}", retryable=True) for i in range(2)]
    provider = make_provider(FakeProvider, outcomes=errors, max_retries=1)
    with structlog.testing.capture_logs() as logs, pytest.raises(ProviderError):
        await provider.complete("2+2?", PARAMS)
    assert [log["outcome"] for log in logs] == ["retrying", "retries_exhausted"]
