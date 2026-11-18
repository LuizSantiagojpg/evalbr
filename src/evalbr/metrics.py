"""Deterministic quality, reliability, hallucination, latency, and cost metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from evalbr.schema import CATEGORIES, EvalRecord, LabeledReview

_WS_RE = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def is_evidence_grounded(evidence: str, source_text: str) -> bool:
    """Return True iff `evidence` appears literally in `source_text` (case/ws normalized)."""
    norm_ev = _norm_ws(evidence)
    if not norm_ev:
        return False
    return norm_ev in _norm_ws(source_text)


@dataclass(frozen=True)
class SystemMetrics:
    """Aggregated evaluation metrics for a single system on a dataset."""

    system_id: str
    n_samples: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    json_valid_rate: float
    retry_rate: float
    hallucination_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_usd: float


def compute_system_metrics(
    records: list[EvalRecord],
    ground_truth: dict[str, LabeledReview],
) -> SystemMetrics:
    """Compute quality, reliability, hallucination, latency, and cost metrics."""
    if not records:
        raise ValueError("Cannot compute metrics on an empty record list.")

    system_id = records[0].system_id
    y_true: list[str] = []
    y_pred: list[str] = []
    hallucinations = 0
    valid_count = 0
    retried_count = 0

    for rec in records:
        gt = ground_truth[rec.review_id]
        y_true.append(gt.label.categoria)
        if rec.retries > 0:
            retried_count += 1

        if rec.parsed_label is not None and rec.json_valid:
            valid_count += 1
            y_pred.append(rec.parsed_label.categoria)
            if not is_evidence_grounded(rec.parsed_label.evidencia, gt.text):
                hallucinations += 1
        else:
            # Invalid JSON cannot be trusted in production; count as wrong + hallucinated.
            y_pred.append("__invalid__")
            hallucinations += 1

    acc = float(accuracy_score(y_true, y_pred))
    labels = list(CATEGORIES)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    per_class_arr = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        label: round(float(score), 4) for label, score in zip(labels, per_class_arr, strict=True)
    }

    latencies = np.array([r.latency_ms for r in records], dtype=float)
    total_cost = sum(r.cost_usd for r in records)
    n = len(records)

    return SystemMetrics(
        system_id=system_id,
        n_samples=n,
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        per_class_f1=per_class,
        json_valid_rate=round(valid_count / n, 4),
        retry_rate=round(retried_count / n, 4),
        hallucination_rate=round(hallucinations / n, 4),
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2),
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2),
        cost_per_1k_usd=round((total_cost / n) * 1000.0, 4),
    )


def build_confusion_matrix(
    records: list[EvalRecord],
    ground_truth: dict[str, LabeledReview],
) -> tuple[list[str], np.ndarray]:
    """Return class labels and confusion matrix array for error analysis."""
    labels = list(CATEGORIES)
    y_true = [ground_truth[r.review_id].label.categoria for r in records]
    y_pred = [
        r.parsed_label.categoria if r.parsed_label is not None else "outro"
        for r in records
    ]
    return labels, confusion_matrix(y_true, y_pred, labels=labels)
