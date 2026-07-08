"""Anthropic provider: binds Provider._complete/classify to the async Anthropic SDK."""

from __future__ import annotations

import time
from typing import Literal

import anthropic

from gradetrail.providers.base import Provider, ProviderResponse
from gradetrail.spec import ModelParams


class AnthropicProvider(Provider):
    """Provider backed by anthropic.AsyncAnthropic.

    The real client is constructed with max_retries=0: the SDK has its own
    retry policy, and Provider.complete() must be the only thing that ever
    retries a request, or the two retry loops compound (max_retries=5 here
    becomes up to 5x the SDK's own retries under the hood).
    """

    def __init__(
        self,
        *,
        model: str,
        max_retries: int,
        timeout_s: float,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries, timeout_s=timeout_s)
        self._client = client if client is not None else anthropic.AsyncAnthropic(max_retries=0)

    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if params.system is not None:
            kwargs["system"] = params.system

        start = time.monotonic()
        message = await self._client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        text = "".join(block.text for block in message.content if block.type == "text")
        return ProviderResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
            model=message.model,
        )

    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
            return "retryable"
        if isinstance(exc, anthropic.APIStatusError):
            return "retryable" if exc.status_code == 429 or exc.status_code >= 500 else "fatal"
        return "fatal"
