"""OpenAI provider: binds Provider._complete/classify to the async OpenAI SDK."""

from __future__ import annotations

import time
from typing import Literal

import openai
import structlog

from gradetrail.providers.base import Provider, ProviderResponse
from gradetrail.spec import ModelParams

_log = structlog.get_logger(__name__)


class OpenAIProvider(Provider):
    """Provider backed by openai.AsyncOpenAI.

    The real client is constructed with max_retries=0: the SDK has its own
    retry policy, and Provider.complete() must be the only thing that ever
    retries a request, or the two retry loops compound (max_retries=5 here
    becomes up to 5x the SDK's own retries under the hood).

    base_url is optional here (None uses OpenAI's default endpoint); see
    openai_compatible.py for the subclass that makes it mandatory.
    """

    def __init__(
        self,
        *,
        model: str,
        max_retries: int,
        timeout_s: float,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries, timeout_s=timeout_s)
        if client is not None:
            self._client = client
        elif base_url is not None:
            self._client = openai.AsyncOpenAI(base_url=base_url, max_retries=0)
        else:
            self._client = openai.AsyncOpenAI(max_retries=0)

    async def _complete(self, prompt: str, params: ModelParams) -> ProviderResponse:
        messages: list[dict] = []
        if params.system is not None:
            messages.append({"role": "system", "content": params.system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        # max_tokens (not max_completion_tokens): works on current mainstream
        # OpenAI chat models and on OpenAI-compatible servers (vLLM et al.),
        # which mostly don't support the newer param yet. OpenAI's reasoning
        # models reject max_tokens -- deferred rename, see NOTES.md.
        completion = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=messages,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = completion.usage
        if usage is None:
            # Nullable per the SDK's own type; some OpenAI-compatible servers
            # omit it on non-streaming calls. The response text is still
            # perfectly scoreable, so zero tokens beats crashing the sample.
            _log.warning("usage_missing", model=self.model)
            input_tokens, output_tokens = 0, 0
        else:
            input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens

        return ProviderResponse(
            text=completion.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=completion.model,
        )

    def classify(self, exc: Exception) -> Literal["retryable", "fatal"]:
        if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
            return "retryable"
        if isinstance(exc, openai.APIStatusError):
            return "retryable" if exc.status_code == 429 or exc.status_code >= 500 else "fatal"
        return "fatal"
