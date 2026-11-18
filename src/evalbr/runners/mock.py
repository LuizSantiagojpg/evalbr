"""Deterministic mock runner for offline CI, unit tests, and reproducible demos."""

from __future__ import annotations

import hashlib
import json
import random

from evalbr.cache import ResponseCache
from evalbr.runners.base import BaseRunner, ProviderResponse
from evalbr.schema import CATEGORIES, Category, LabeledReview, ReviewSample, SystemSpec

_PROFILE_ACCURACY: dict[str, float] = {
    "zero_shot": 0.72,
    "few_shot": 0.84,
    "structured": 0.90,
}

_MODEL_BONUS: dict[str, float] = {
    "qwen-local-4b": -0.04,
    "gemini-2.5-flash": 0.02,
    "gemini-2.5-pro": 0.05,
}


class MockRunner(BaseRunner):
    """Simulates realistic model/strategy behavior using ground truth when available."""

    def __init__(
        self,
        spec: SystemSpec,
        ground_truth: dict[str, LabeledReview] | None = None,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(spec=spec, cache=cache, max_retries=max_retries)
        self.ground_truth = ground_truth or {}

    def _call_provider(
        self,
        messages: list[dict[str, str]],
        sample: ReviewSample,
        attempt: int,
    ) -> ProviderResponse:
        seed_hex = hashlib.sha256(
            f"{self.spec.system_id}:{sample.review_id}:{attempt}".encode()
        ).hexdigest()
        rng = random.Random(int(seed_hex[:12], 16))

        gt = self.ground_truth.get(sample.review_id)
        base_acc = _PROFILE_ACCURACY[self.spec.strategy] + _MODEL_BONUS.get(
            self.spec.model_name, 0.0
        )
        # Noisy inputs degrade zero_shot slightly more than structured few-shot.
        if "  " not in sample.text and not any(ch in sample.text for ch in "áéíóúãõç"):
            base_acc -= 0.05 if self.spec.strategy == "zero_shot" else 0.02
        target_acc = max(0.3, min(0.97, base_acc))

        # Simulate occasional invalid JSON on zero_shot first attempt.
        if self.spec.strategy == "zero_shot" and attempt == 0 and rng.random() < 0.08:
            return ProviderResponse(
                text='{"categoria": "atraso_entrega", INVALID_JSON',
                input_tokens=220,
                output_tokens=25,
            )

        if gt is not None:
            true_label = gt.label
            if rng.random() <= target_acc:
                cat: Category = true_label.categoria
            else:
                alts = [c for c in CATEGORIES if c != true_label.categoria]
                cat = rng.choice(alts)

            # Simulate occasional hallucinated evidence (lower on structured).
            hallucination_rate = 0.09 if self.spec.strategy == "zero_shot" else 0.02
            if rng.random() < hallucination_rate:
                evidence = "trecho inexistente inventado pelo modelo"
            else:
                words = sample.text.split()
                span = " ".join(words[: min(5, len(words))])
                evidence = (
                    true_label.evidencia
                    if true_label.evidencia in sample.text
                    else span
                )

            payload = {
                "categoria": cat,
                "sentimento": true_label.sentimento,
                "menciona_prazo": true_label.menciona_prazo,
                "acao_desejada": true_label.acao_desejada,
                "evidencia": evidence,
            }
        else:
            words = sample.text.split()
            payload = {
                "categoria": "atraso_entrega",
                "sentimento": "negativo" if sample.score <= 2 else "positivo",
                "menciona_prazo": "prazo" in sample.text.lower(),
                "acao_desejada": "nao_informado",
                "evidencia": " ".join(words[: min(4, len(words))]),
            }

        in_tokens = len(" ".join(m["content"] for m in messages).split()) + 40
        out_tokens = 55
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
