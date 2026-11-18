"""Tests for deterministic evaluation metrics and hallucination detection."""

from __future__ import annotations

from evalbr.metrics import compute_system_metrics, is_evidence_grounded
from evalbr.schema import EvalRecord, LabeledReview, ReviewLabel


def test_evidence_grounding_normalizes_whitespace() -> None:
    assert is_evidence_grounded("chegou   atrasado", "O pedido chegou atrasado hoje.")
    assert not is_evidence_grounded("texto inventado", "O pedido chegou atrasado hoje.")


def test_compute_system_metrics_accounts_for_invalid_json_and_hallucination() -> None:
    gt = {
        "r1": LabeledReview(
            review_id="r1",
            score=1,
            text="Produto veio quebrado na caixa.",
            label=ReviewLabel(
                categoria="defeito_ou_qualidade",
                sentimento="negativo",
                menciona_prazo=False,
                acao_desejada="troca",
                evidencia="veio quebrado",
            ),
        ),
        "r2": LabeledReview(
            review_id="r2",
            score=5,
            text="Chegou rápido e bem embalado.",
            label=ReviewLabel(
                categoria="elogio",
                sentimento="positivo",
                menciona_prazo=True,
                acao_desejada="nenhuma",
                evidencia="Chegou rápido",
            ),
        ),
    }

    records = [
        EvalRecord(
            system_id="s1",
            review_id="r1",
            raw_output="{}",
            parsed_label=gt["r1"].label,
            json_valid=True,
            retries=0,
            latency_ms=100.0,
            cost_usd=0.001,
        ),
        EvalRecord(
            system_id="s1",
            review_id="r2",
            raw_output="broken",
            parsed_label=None,
            json_valid=False,
            retries=2,
            latency_ms=300.0,
            cost_usd=0.003,
        ),
    ]

    m = compute_system_metrics(records, gt)
    assert m.accuracy == 0.5
    assert m.json_valid_rate == 0.5
    assert m.retry_rate == 0.5
    assert m.hallucination_rate == 0.5
    assert m.cost_per_1k_usd == 2.0
