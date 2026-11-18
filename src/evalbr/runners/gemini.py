"""Google Gemini runner using the official google-genai SDK."""

from __future__ import annotations

import os

from evalbr.cache import ResponseCache
from evalbr.runners.base import BaseRunner, ProviderResponse
from evalbr.schema import ReviewLabel, ReviewSample, SystemSpec


class GeminiRunner(BaseRunner):
    """Runner targeting Google Gemini models via google-genai."""

    def __init__(
        self,
        spec: SystemSpec,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
        api_key: str | None = None,
    ) -> None:
        super().__init__(spec=spec, cache=cache, max_retries=max_retries)
        from google import genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=resolved_key)

    def _call_provider(
        self,
        messages: list[dict[str, str]],
        sample: ReviewSample,
        attempt: int,
    ) -> ProviderResponse:
        del sample, attempt
        from google.genai import types

        system_instruction = "\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        contents = "\n".join(
            m["content"] for m in messages if m["role"] != "system"
        )
        config_kwargs: dict[str, object] = {
            "system_instruction": system_instruction,
            "temperature": self.spec.temperature,
        }
        if self.spec.strategy == "structured":
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = ReviewLabel

        response = self._client.models.generate_content(
            model=self.spec.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=response.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )
