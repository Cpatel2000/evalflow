"""Tests for gradetrail.providers.anthropic: the async Anthropic SDK binding.

Mocks only at the SDK boundary (client.messages.create), using hand-built
anthropic.types.Message/Usage objects and real anthropic exception instances
(built from minimal httpx.Request/Response objects). Never hits a real API.
Retry/backoff behavior itself is covered by test_base.py; these tests only
check request/response mapping and classify().
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import Message, TextBlock, Usage

from gradetrail.errors import ProviderError
from gradetrail.providers.anthropic import AnthropicProvider
from gradetrail.spec import ModelParams

MODEL = "claude-sonnet-4-6"
_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST)


def make_message(
    text: str = "Answer: 4",
    *,
    input_tokens: int = 12,
    output_tokens: int = 4,
    model: str = MODEL,
) -> Message:
    return Message(
        id="msg_01",
        content=[TextBlock(text=text, type="text")],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    """Stand-in for client.messages: replays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Exception | Message]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> Message:
        self.calls.append(kwargs)
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    """Stand-in for anthropic.AsyncAnthropic: only .messages is ever touched."""

    def __init__(self, outcomes: list[Exception | Message]) -> None:
        self.messages = FakeMessages(outcomes)


def make_provider(
    outcomes: list[Exception | Message], *, max_retries: int = 3, timeout_s: float = 5.0
) -> tuple[AnthropicProvider, FakeClient]:
    client = FakeClient(outcomes)
    provider = AnthropicProvider(
        model=MODEL, max_retries=max_retries, timeout_s=timeout_s, client=client
    )
    return provider, client


# --- request/response mapping -----------------------------------------------------


async def test_complete_maps_message_to_provider_response() -> None:
    provider, _client = make_provider([make_message()])
    result = await provider.complete("2+2?", ModelParams(max_tokens=256, temperature=0.0))
    assert result.text == "Answer: 4"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert result.model == MODEL
    assert result.latency_ms >= 0


async def test_complete_sends_resolved_prompt_and_params() -> None:
    provider, client = make_provider([make_message()])
    await provider.complete("2+2?", ModelParams(max_tokens=256, temperature=0.7, system="be terse"))
    [call] = client.messages.calls
    assert call["model"] == MODEL
    assert call["max_tokens"] == 256
    assert call["temperature"] == 0.7
    assert call["system"] == "be terse"
    assert call["messages"] == [{"role": "user", "content": "2+2?"}]


async def test_complete_omits_system_when_not_set() -> None:
    provider, client = make_provider([make_message()])
    await provider.complete("2+2?", ModelParams(max_tokens=256, temperature=0.0))
    assert "system" not in client.messages.calls[0]


async def test_concatenates_multiple_text_blocks() -> None:
    msg = Message(
        id="msg_01",
        content=[TextBlock(text="Answer: ", type="text"), TextBlock(text="4", type="text")],
        model=MODEL,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=12, output_tokens=4),
    )
    provider, _client = make_provider([msg])
    result = await provider.complete("2+2?", ModelParams())
    assert result.text == "Answer: 4"


async def test_model_on_response_is_as_reported_by_api() -> None:
    provider, _client = make_provider([make_message(model="claude-sonnet-4-6-20260115")])
    result = await provider.complete("2+2?", ModelParams())
    assert result.model == "claude-sonnet-4-6-20260115"


# --- classify: retryable vs fatal ----------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RateLimitError("rate limited", response=_status_response(429), body=None),
        InternalServerError("server error", response=_status_response(500), body=None),
        APIConnectionError(request=_REQUEST),
        APITimeoutError(_REQUEST),
    ],
    ids=["rate_limit_429", "server_error_500", "connection_error", "timeout_error"],
)
async def test_retryable_sdk_errors_are_retried(exc: Exception) -> None:
    provider, client = make_provider([exc, make_message()], max_retries=1)
    result = await provider.complete("2+2?", ModelParams())
    assert result.text == "Answer: 4"
    assert len(client.messages.calls) == 2


async def test_fatal_sdk_error_is_not_retried() -> None:
    exc = AuthenticationError("bad key", response=_status_response(401), body=None)
    provider, client = make_provider([exc, make_message()], max_retries=3)
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", ModelParams())
    assert len(client.messages.calls) == 1
