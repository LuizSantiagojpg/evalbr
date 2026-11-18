"""Abstract runner with caching, JSON extraction, Pydantic validation, and retries."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from evalbr.cache import ResponseCache
from evalbr.prompts import PROMPT_VERSION, build_messages, hash_payload
from evalbr.schema import EvalRecord, ReviewLabel, ReviewSample, SystemSpec

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class ProviderResponse:
    """Raw response returned by an underlying LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int


def extract_json_object(raw_text: str) -> str:
    """Extract a JSON object string, stripping markdown code fences if present."""
    stripped = raw_text.strip()
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class BaseRunner(ABC):
    """Base class for all LLM runners with deterministic caching and Pydantic retries."""

    def __init__(
        self,
        spec: SystemSpec,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
    ) -> None:
        self.spec = spec
        self.cache = cache
        self.max_retries = max_retries

    @abstractmethod
    def _call_provider(
        self,
        messages: list[dict[str, str]],
        sample: ReviewSample,
        attempt: int,
    ) -> ProviderResponse:
        """Execute a single provider call and return text plus token usage."""

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.spec.input_cost_per_1m_usd
            + (output_tokens / 1_000_000.0) * self.spec.output_cost_per_1m_usd
        )

    def run_sample(
        self,
        sample: ReviewSample,
        variant: Literal["clean", "noisy"] = "clean",
    ) -> EvalRecord:
        """Run inference on a single review sample with caching and schema retries."""
        messages = build_messages(sample.text, self.spec.strategy)
        prompt_hash = hash_payload({"version": PROMPT_VERSION, "messages": messages})
        input_hash = hash_payload({"review_id": sample.review_id, "text": sample.text})
        cache_key = ResponseCache.make_key(self.spec.system_id, prompt_hash, input_hash, variant)

        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        total_in = 0
        total_out = 0
        retries_used = 0
        last_raw = ""
        last_error: str | None = None
        parsed: ReviewLabel | None = None

        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._call_provider(messages, sample, attempt)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                last_raw = resp.text

                candidate_json = extract_json_object(resp.text)
                parsed = ReviewLabel.model_validate_json(candidate_json)
                last_error = None
                break
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    retries_used += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": last_raw},
                        {
                            "role": "user",
                            "content": (
                                "Sua resposta anterior falhou na validação do schema JSON: "
                                f"{last_error}. Retorne APENAS um JSON válido."
                            ),
                        },
                    ]
            except Exception as exc:  # noqa: BLE001 - surface provider/network errors cleanly
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    retries_used += 1
                    time.sleep(0.25 * (2**attempt))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        record = EvalRecord(
            system_id=self.spec.system_id,
            review_id=sample.review_id,
            variant=variant,
            raw_output=last_raw,
            parsed_label=parsed,
            json_valid=parsed is not None,
            retries=retries_used,
            latency_ms=round(latency_ms, 2),
            input_tokens=total_in,
            output_tokens=total_out,
            cost_usd=round(self._compute_cost(total_in, total_out), 8),
            from_cache=False,
            error=last_error,
        )
        if self.cache is not None and record.json_valid:
            self.cache.put(cache_key, record)
        return record
