"""OpenAI-compatible provider: the OpenAI client pointed at a custom base_url.

For any OpenAI-API-compatible server (vLLM, Together, local inference, etc.)
that isn't OpenAI itself. Request/response mapping and error classification
are identical to OpenAIProvider -- only the client's base_url differs -- so
this subclasses it rather than duplicating _complete/classify.
"""

from __future__ import annotations

import openai

from gradetrail.providers.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAIProvider with a mandatory base_url (ModelSpec requires it for this provider)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        max_retries: int,
        timeout_s: float,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(
            model=model,
            max_retries=max_retries,
            timeout_s=timeout_s,
            base_url=base_url,
            client=client,
        )
