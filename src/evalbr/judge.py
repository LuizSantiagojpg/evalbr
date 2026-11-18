"""LLM-as-a-Judge calibration module for evaluating `acao_desejada` alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evalbr.cache import ResponseCache
from evalbr.runners.base import extract_json_object
from evalbr.schema import LabeledReview, SystemSpec
from evalbr.stats import cohens_kappa


class JudgeVerdict(BaseModel):
    """Structured verdict emitted by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    agrees_with_desired_action: Literal["sim", "nao"]
    rationale: str


@dataclass(frozen=True)
class JudgeCalibrationResult:
    """Calibration agreement between the LLM judge and human ground truth."""

    judge_system_id: str
    target_system_id: str
    n_evaluated: int
    kappa_vs_human: float
    agreement_rate: float


def calibrate_judge_on_action(
    judge_spec: SystemSpec,
    target_system_id: str,
    predictions: dict[str, str],
    ground_truth: dict[str, LabeledReview],
    cache: ResponseCache | None = None,
) -> JudgeCalibrationResult:
    """Measure agreement (Cohen's kappa) between an LLM judge and human labels."""
    del cache  # Reserved for persistent judge verdict caching.
    human_match: list[str] = []
    judge_match: list[str] = []

    if judge_spec.provider == "mock":
        for rid, pred_action in predictions.items():
            gt_action = ground_truth[rid].label.acao_desejada
            human_verdict = "sim" if pred_action == gt_action else "nao"
            human_match.append(human_verdict)
            # Deterministic high-agreement judge simulation.
            judge_match.append(human_verdict)
    elif judge_spec.provider == "openai_compat":
        import os

        from openai import OpenAI

        client = OpenAI(
            base_url=judge_spec.base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "lm-studio",
        )
        for rid, pred_action in predictions.items():
            gt = ground_truth[rid]
            human_verdict = "sim" if pred_action == gt.label.acao_desejada else "nao"
            prompt = (
                "Avalie se a ação desejada predita corresponde à intenção do cliente.\n"
                f'Comentário: "{gt.text}"\n'
                f"Ação predita: {pred_action}\n"
                f"Ação esperada (referência): {gt.label.acao_desejada}\n"
                'Responda APENAS JSON: '
                '{"agrees_with_desired_action": "sim"|"nao", "rationale": "..."}'
            )
            resp = client.chat.completions.create(
                model=judge_spec.model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_json_object(resp.choices[0].message.content or "")
            verdict = JudgeVerdict.model_validate_json(raw)
            human_match.append(human_verdict)
            judge_match.append(verdict.agrees_with_desired_action)
    else:
        raise ValueError(f"Judge provider `{judge_spec.provider}` not supported yet.")

    if not human_match:
        raise ValueError("No predictions supplied for judge calibration.")

    kappa = cohens_kappa(human_match, judge_match)
    agree = sum(1 for h, j in zip(human_match, judge_match, strict=True) if h == j) / len(
        human_match
    )
    return JudgeCalibrationResult(
        judge_system_id=judge_spec.system_id,
        target_system_id=target_system_id,
        n_evaluated=len(human_match),
        kappa_vs_human=kappa,
        agreement_rate=round(agree, 4),
    )
