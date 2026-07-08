"""Tests for gradetrail.providers.openai: the async OpenAI SDK binding.

Mocks only at the SDK boundary (client.chat.completions.create), using
hand-built openai.types.chat.ChatCompletion objects and real openai exception
instances (built from minimal httpx.Request/Response objects, same shape as
anthropic's exception hierarchy -- both SDKs are Stainless-generated). Never
hits a real API. Retry/backoff behavior itself is covered by test_base.py;
these tests only check request/response mapping and classify().
"""

from __future__ import annotations

import httpx
import pytest
import structlog
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from gradetrail.errors import ProviderError
from gradetrail.providers.openai import OpenAIProvider
from gradetrail.spec import ModelParams

MODEL = "gpt-5.1"
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST)


def make_message(
    text: str | None = "Answer: 4",
    *,
    input_tokens: int = 12,
    output_tokens: int = 4,
    model: str = MODEL,
    include_usage: bool = True,
) -> ChatCompletion:
    kwargs: dict = dict(
        id="chatcmpl-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        created=1234567890,
        model=model,
        object="chat.completion",
    )
    if include_usage:
        kwargs["usage"] = CompletionUsage(
            completion_tokens=output_tokens,
            prompt_tokens=input_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return ChatCompletion(**kwargs)


class FakeCompletions:
    """Stand-in for client.chat.completions: replays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Exception | ChatCompletion]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> ChatCompletion:
        self.calls.append(kwargs)
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    """Stand-in for client.chat: only .completions is ever touched."""

    def __init__(self, outcomes: list[Exception | ChatCompletion]) -> None:
        self.completions = FakeCompletions(outcomes)


class FakeClient:
    """Stand-in for openai.AsyncOpenAI: only .chat is ever touched."""

    def __init__(self, outcomes: list[Exception | ChatCompletion]) -> None:
        self.chat = FakeChat(outcomes)


def make_provider(
    outcomes: list[Exception | ChatCompletion], *, max_retries: int = 3, timeout_s: float = 5.0
) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(outcomes)
    provider = OpenAIProvider(
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


async def test_complete_sends_resolved_prompt_and_params_with_system_message() -> None:
    provider, client = make_provider([make_message()])
    await provider.complete("2+2?", ModelParams(max_tokens=256, temperature=0.7, system="be terse"))
    [call] = client.chat.completions.calls
    assert call["model"] == MODEL
    assert call["max_tokens"] == 256
    assert call["temperature"] == 0.7
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "2+2?"},
    ]


async def test_complete_omits_system_message_when_not_set() -> None:
    provider, client = make_provider([make_message()])
    await provider.complete("2+2?", ModelParams(max_tokens=256, temperature=0.0))
    assert client.chat.completions.calls[0]["messages"] == [{"role": "user", "content": "2+2?"}]


async def test_model_on_response_is_as_reported_by_api() -> None:
    provider, _client = make_provider([make_message(model="gpt-5.1-2026-05-01")])
    result = await provider.complete("2+2?", ModelParams())
    assert result.model == "gpt-5.1-2026-05-01"


async def test_complete_treats_null_content_as_empty_string() -> None:
    # ChatCompletionMessage.content is nullable per the SDK's own type (e.g. a
    # refusal or tool-call-only reply) -- ProviderResponse.text is not optional.
    provider, _client = make_provider([make_message(text=None)])
    result = await provider.complete("2+2?", ModelParams())
    assert result.text == ""


async def test_complete_treats_missing_usage_as_zero_tokens_without_raising() -> None:
    # usage is Optional per the SDK's own type; some OpenAI-compatible servers
    # omit it on non-streaming calls. The response text is still perfectly
    # scoreable, so this must not surface as a crash (see openai_compatible).
    provider, _client = make_provider([make_message(include_usage=False)])
    result = await provider.complete("2+2?", ModelParams())
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.text == "Answer: 4"


async def test_complete_logs_a_warning_when_usage_is_missing() -> None:
    provider, _client = make_provider([make_message(include_usage=False)])
    with structlog.testing.capture_logs() as logs:
        await provider.complete("2+2?", ModelParams())
    assert any(log["event"] == "usage_missing" and log.get("model") == MODEL for log in logs)


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
    assert len(client.chat.completions.calls) == 2


async def test_fatal_sdk_error_is_not_retried() -> None:
    exc = AuthenticationError("bad key", response=_status_response(401), body=None)
    provider, client = make_provider([exc, make_message()], max_retries=3)
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", ModelParams())
    assert len(client.chat.completions.calls) == 1
