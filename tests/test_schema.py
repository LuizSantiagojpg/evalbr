"""Tests for Pydantic domain schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalbr.schema import ReviewLabel, SystemSpec


def test_review_label_valid() -> None:
    label = ReviewLabel(
        categoria="atraso_entrega",
        sentimento="negativo",
        menciona_prazo=True,
        acao_desejada="entrega",
        evidencia="passou do prazo",
    )
    assert label.categoria == "atraso_entrega"


def test_review_label_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewLabel(
            categoria="elogio",
            sentimento="positivo",
            menciona_prazo=False,
            acao_desejada="nenhuma",
            evidencia="muito bom",
            extra_field="forbidden",  # type: ignore[call-arg]
        )


def test_system_spec_rejects_spaces_in_id() -> None:
    with pytest.raises(ValidationError):
        SystemSpec(
            system_id="bad id",
            provider="mock",
            model_name="qwen-local-4b",
            strategy="zero_shot",
        )
