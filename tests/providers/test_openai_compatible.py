"""Tests for gradetrail.providers.openai_compatible: the OpenAI client pointed at
a custom base_url (vLLM, Together, local inference servers, etc.).

_complete/classify are inherited unchanged from OpenAIProvider and are fully
covered by test_openai.py -- these tests focus on what's actually different
for this provider: that base_url reaches the real client constructor, plus
one smoke test confirming the inherited response mapping still works through
the subclass.

Patches gradetrail.providers.openai.openai.AsyncOpenAI (not
.openai_compatible.openai.AsyncOpenAI) deliberately: OpenAICompatibleProvider
calls the inherited OpenAIProvider.__init__, whose body resolves the `openai`
name from gradetrail.providers.openai's own module globals, not this module's.
"""

from __future__ import annotations

import httpx
import pytest
from openai import AuthenticationError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from gradetrail.errors import ProviderError
from gradetrail.providers.openai_compatible import OpenAICompatibleProvider
from gradetrail.spec import ModelParams

BASE_URL = "http://localhost:8000/v1"
_REQUEST = httpx.Request("POST", BASE_URL + "/chat/completions")


def make_message(
    text: str = "Answer: 4",
    *,
    input_tokens: int = 12,
    output_tokens: int = 4,
    model: str = "local-llama",
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
    def __init__(self, outcomes: list[Exception | ChatCompletion]) -> None:
        self.completions = FakeCompletions(outcomes)


class FakeClient:
    def __init__(self, outcomes: list[Exception | ChatCompletion]) -> None:
        self.chat = FakeChat(outcomes)


# --- base_url wiring ---------------------------------------------------------------


async def test_base_url_reaches_the_real_client_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class RecordingAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            self.chat = FakeChat([make_message()])

    import gradetrail.providers.openai as openai_provider_module

    monkeypatch.setattr(openai_provider_module.openai, "AsyncOpenAI", RecordingAsyncOpenAI)

    OpenAICompatibleProvider(model="local-llama", base_url=BASE_URL, max_retries=3, timeout_s=5.0)

    assert captured_kwargs["base_url"] == BASE_URL
    assert captured_kwargs["max_retries"] == 0  # the SDK's own retries stay disabled


async def test_explicit_client_bypasses_base_url_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If a client is injected (as in every other test here), the real
    # AsyncOpenAI constructor -- and base_url -- must never be touched at all.
    import gradetrail.providers.openai as openai_provider_module

    def boom(**kwargs: object) -> None:
        raise AssertionError("real AsyncOpenAI constructor should not be called")

    monkeypatch.setattr(openai_provider_module.openai, "AsyncOpenAI", boom)

    client = FakeClient([make_message()])
    provider = OpenAICompatibleProvider(
        model="local-llama", base_url=BASE_URL, max_retries=3, timeout_s=5.0, client=client
    )
    result = await provider.complete("2+2?", ModelParams())
    assert result.text == "Answer: 4"


# --- inherited behavior still works through the subclass ---------------------------


async def test_complete_maps_response_through_the_subclass() -> None:
    client = FakeClient([make_message()])
    provider = OpenAICompatibleProvider(
        model="local-llama", base_url=BASE_URL, max_retries=3, timeout_s=5.0, client=client
    )
    result = await provider.complete("2+2?", ModelParams())
    assert result.text == "Answer: 4"
    assert result.model == "local-llama"
    [call] = client.chat.completions.calls
    assert call["model"] == "local-llama"


async def test_classify_is_inherited_from_openai_provider() -> None:
    exc = AuthenticationError("bad key", response=httpx.Response(401, request=_REQUEST), body=None)
    client = FakeClient([exc, make_message()])
    provider = OpenAICompatibleProvider(
        model="local-llama", base_url=BASE_URL, max_retries=3, timeout_s=5.0, client=client
    )
    with pytest.raises(ProviderError):
        await provider.complete("2+2?", ModelParams())
    assert len(client.chat.completions.calls) == 1  # fatal -- no retry


async def test_missing_usage_is_zero_tokens_not_a_crash() -> None:
    # The actual motivating scenario for this provider: some vLLM/proxy
    # configurations omit usage on non-streaming calls entirely.
    client = FakeClient([make_message(include_usage=False)])
    provider = OpenAICompatibleProvider(
        model="local-llama", base_url=BASE_URL, max_retries=3, timeout_s=5.0, client=client
    )
    result = await provider.complete("2+2?", ModelParams())
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.text == "Answer: 4"
