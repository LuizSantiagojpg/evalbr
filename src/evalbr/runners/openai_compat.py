"""OpenAI-compatible runner (supports OpenAI API and local LM Studio servers)."""

from __future__ import annotations

import os
from typing import Any

from evalbr.cache import ResponseCache
from evalbr.runners.base import BaseRunner, ProviderResponse
from evalbr.schema import ReviewSample, SystemSpec


class OpenAICompatRunner(BaseRunner):
    """Runner targeting any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        spec: SystemSpec,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
        api_key: str | None = None,
    ) -> None:
        super().__init__(spec=spec, cache=cache, max_retries=max_retries)
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        self._client = OpenAI(base_url=spec.base_url, api_key=resolved_key)

    def _call_provider(
        self,
        messages: list[dict[str, str]],
        sample: ReviewSample,
        attempt: int,
    ) -> ProviderResponse:
        del sample, attempt
        kwargs: dict[str, Any] = {
            "model": self.spec.model_name,
            "messages": messages,
            "temperature": self.spec.temperature,
        }
        # Only pass OpenAI's response_format when targeting official OpenAI endpoints;
        # local OpenAI-compatible servers (e.g., LM Studio) often reject `json_object`.
        if self.spec.strategy == "structured" and self.spec.base_url is None:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        return ProviderResponse(
            text=content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
