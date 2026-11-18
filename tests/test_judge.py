"""Tests for LLM-as-a-Judge calibration."""

from __future__ import annotations

from evalbr.data import generate_synthetic_ptbr_dataset
from evalbr.judge import calibrate_judge_on_action
from evalbr.schema import SystemSpec


def test_judge_calibration_perfect_agreement_in_mock() -> None:
    dataset = generate_synthetic_ptbr_dataset(n_samples=20, seed=9)
    gt_map = {item.review_id: item for item in dataset}
    preds = {item.review_id: item.label.acao_desejada for item in dataset}
    # Flip 4 predictions so both "sim" and "nao" classes exist for kappa.
    for item in dataset[:4]:
        preds[item.review_id] = (
            "nao_informado" if item.label.acao_desejada != "nao_informado" else "troca"
        )

    judge_spec = SystemSpec(
        system_id="judge_mock",
        provider="mock",
        model_name="gemini-2.5-flash",
        strategy="structured",
    )
    res = calibrate_judge_on_action(judge_spec, "target_sys", preds, gt_map)
    assert res.n_evaluated == 20
    assert res.kappa_vs_human == 1.0
    assert res.agreement_rate == 1.0
