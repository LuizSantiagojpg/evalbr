"""End-to-end experiment pipeline: execution, stats, charts, and decision memo."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evalbr.cache import ResponseCache
from evalbr.judge import calibrate_judge_on_action
from evalbr.metrics import (
    SystemMetrics,
    build_confusion_matrix,
    compute_system_metrics,
    is_evidence_grounded,
)
from evalbr.prompts import inject_ptbr_noise
from evalbr.runners import BaseRunner, GeminiRunner, MockRunner, OpenAICompatRunner
from evalbr.schema import EvalRecord, LabeledReview, ReviewSample, SystemSpec
from evalbr.stats import (
    BootstrapCI,
    PairedTestResult,
    accuracy_metric,
    bootstrap_ci,
    cohens_kappa,
    holm_bonferroni,
    macro_f1_metric,
    mcnemar_test,
    paired_bootstrap_diff,
    pareto_frontier,
    sample_size_paired_mcnemar,
)


def build_runner(
    spec: SystemSpec,
    cache: ResponseCache | None = None,
    ground_truth: dict[str, LabeledReview] | None = None,
) -> BaseRunner:
    """Instantiate the concrete provider runner for a SystemSpec."""
    if spec.provider == "mock":
        return MockRunner(spec=spec, ground_truth=ground_truth, cache=cache)
    if spec.provider == "openai_compat":
        return OpenAICompatRunner(spec=spec, cache=cache)
    if spec.provider == "gemini":
        return GeminiRunner(spec=spec, cache=cache)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def specs_from_json(path: Path | str) -> list[SystemSpec]:
    """Load a list of SystemSpec definitions from a JSON config file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SystemSpec.model_validate(item) for item in raw]


def default_experiment_matrix() -> list[SystemSpec]:
    """9-cell benchmark matrix across 3 models x 3 prompting strategies."""
    models = [
        ("qwen-local-4b", 0.00, 0.00),
        ("gemini-2.5-flash", 0.15, 0.60),
        ("gemini-2.5-pro", 1.25, 10.00),
    ]
    strategies = ("zero_shot", "few_shot", "structured")
    specs: list[SystemSpec] = []
    for model_name, in_cost, out_cost in models:
        for strategy in strategies:
            specs.append(
                SystemSpec(
                    system_id=f"{model_name}__{strategy}",
                    provider="mock",
                    model_name=model_name,
                    strategy=strategy,
                    input_cost_per_1m_usd=in_cost,
                    output_cost_per_1m_usd=out_cost,
                )
            )
    return specs


def _extract_predictions(
    records: list[EvalRecord],
) -> list[str]:
    return [
        r.parsed_label.categoria if (r.parsed_label is not None and r.json_valid) else "__invalid__"
        for r in records
    ]


def _plot_pareto(
    metrics_list: list[SystemMetrics],
    pareto_ids: set[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for m in metrics_list:
        is_pareto = m.system_id in pareto_ids
        ax.scatter(
            m.cost_per_1k_usd,
            m.macro_f1,
            s=110 if is_pareto else 65,
            color="#0f766e" if is_pareto else "#94a3b8",
            edgecolors="#0f172a" if is_pareto else "none",
            zorder=3 if is_pareto else 2,
        )
        ax.annotate(
            m.system_id.replace("__", "\n"),
            (m.cost_per_1k_usd, m.macro_f1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.5,
            color="#0f172a" if is_pareto else "#475569",
        )

    pareto_pts = sorted(
        (m for m in metrics_list if m.system_id in pareto_ids),
        key=lambda x: x.cost_per_1k_usd,
    )
    if len(pareto_pts) >= 2:
        ax.plot(
            [p.cost_per_1k_usd for p in pareto_pts],
            [p.macro_f1 for p in pareto_pts],
            linestyle="--",
            color="#0f766e",
            linewidth=1.5,
            label="Fronteira de Pareto",
        )

    ax.set_title("EvalBR — Fronteira de Pareto: Custo por 1k reqs vs. Macro-F1", fontweight="bold")
    ax.set_xlabel("Custo estimado por 1.000 requisições (USD)")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(pareto_pts) >= 2:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion_matrix(
    labels: list[str],
    cm: np.ndarray,
    system_id: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real (Ground Truth)")
    ax.set_title(f"Matriz de Confusão — {system_id}", fontweight="bold")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "#0f172a",
                fontsize=8.5,
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ci_forest(
    metrics_list: list[SystemMetrics],
    f1_cis: dict[str, BootstrapCI],
    out_path: Path,
) -> None:
    ordered = sorted(metrics_list, key=lambda m: f1_cis[m.system_id].estimate)
    labels = [m.system_id for m in ordered]
    ests = [f1_cis[sid].estimate for sid in labels]
    low_err = [f1_cis[sid].estimate - f1_cis[sid].ci_low for sid in labels]
    high_err = [f1_cis[sid].ci_high - f1_cis[sid].estimate for sid in labels]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    y_pos = list(range(len(labels)))
    ax.errorbar(
        ests,
        y_pos,
        xerr=[low_err, high_err],
        fmt="o",
        color="#1d4ed8",
        ecolor="#64748b",
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Macro-F1 (IC 95% via Bootstrap, B=2.000)")
    ax.set_title("Comparação de Macro-F1 com Intervalos de Confiança", fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion_matrix(
    labels: list[str],
    cm: np.ndarray,
    system_id: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real (Ground Truth)")
    ax.set_title(f"Matriz de Confusão — {system_id}", fontweight="bold")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "#0f172a",
                fontsize=8.5,
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_benchmark(
    dataset: list[LabeledReview],
    specs: list[SystemSpec],
    output_dir: Path | str,
    cache_path: Path | str | None = None,
    noisy_subset_size: int = 100,
    max_workers: int = 1,
    judge_spec: SystemSpec | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Execute the full EvalBR benchmark, statistical analysis, and report export."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_map = {item.review_id: item for item in dataset}
    y_true = [item.label.categoria for item in dataset]

    cache = ResponseCache(cache_path) if cache_path is not None else None
    try:
        all_records: dict[str, list[EvalRecord]] = {}
        metrics_by_id: dict[str, SystemMetrics] = {}
        acc_cis: dict[str, BootstrapCI] = {}
        f1_cis: dict[str, BootstrapCI] = {}

        for spec in specs:
            runner = build_runner(spec, cache=cache, ground_truth=gt_map)
            samples = [
                ReviewSample(review_id=item.review_id, score=item.score, text=item.text)
                for item in dataset
            ]
            if max_workers > 1 and spec.provider != "mock":
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    recs = list(pool.map(runner.run_sample, samples))
            else:
                recs = [runner.run_sample(s, variant="clean") for s in samples]
            all_records[spec.system_id] = recs
            metrics_by_id[spec.system_id] = compute_system_metrics(recs, gt_map)
            preds = _extract_predictions(recs)
            acc_cis[spec.system_id] = bootstrap_ci(y_true, preds, accuracy_metric, seed=seed)
            f1_cis[spec.system_id] = bootstrap_ci(y_true, preds, macro_f1_metric, seed=seed)

        # Reference system = highest Macro-F1 overall.
        reference_id = max(metrics_by_id, key=lambda sid: metrics_by_id[sid].macro_f1)
        ref_preds = _extract_predictions(all_records[reference_id])

        raw_mcnemar_ps: list[float] = []
        raw_f1_ps: list[float] = []
        temp_pairs: list[tuple[str, float, float, int, int, float, float, float, float]] = []

        other_ids = [sid for sid in metrics_by_id if sid != reference_id]
        for sid in other_ids:
            cand_preds = _extract_predictions(all_records[sid])
            m_stat, m_p, a_only, b_only = mcnemar_test(y_true, cand_preds, ref_preds)
            d_f1, f1_lo, f1_hi, f1_p = paired_bootstrap_diff(
                y_true, cand_preds, ref_preds, macro_f1_metric, seed=seed
            )
            d_acc = round(metrics_by_id[sid].accuracy - metrics_by_id[reference_id].accuracy, 4)
            raw_mcnemar_ps.append(m_p)
            raw_f1_ps.append(f1_p)
            temp_pairs.append((sid, d_acc, m_stat, a_only, b_only, d_f1, f1_lo, f1_hi, m_p, f1_p))

        holm_mcnemar = holm_bonferroni(raw_mcnemar_ps)
        holm_f1 = holm_bonferroni(raw_f1_ps)

        paired_results: list[PairedTestResult] = []
        for idx, (sid, d_acc, m_stat, a_only, b_only, d_f1, f1_lo, f1_hi, m_p, f1_p) in enumerate(
            temp_pairs
        ):
            paired_results.append(
                PairedTestResult(
                    system_a=sid,
                    system_b=reference_id,
                    delta_acc=d_acc,
                    mcnemar_stat=m_stat,
                    mcnemar_p_value=round(m_p, 6),
                    mcnemar_p_holm=holm_mcnemar[idx],
                    delta_macro_f1=d_f1,
                    f1_ci_low=f1_lo,
                    f1_ci_high=f1_hi,
                    f1_p_value=round(f1_p, 6),
                    f1_p_holm=holm_f1[idx],
                    discordant_a_only=a_only,
                    discordant_b_only=b_only,
                )
            )

        pareto_ids = pareto_frontier(
            [(m.system_id, m.cost_per_1k_usd, m.macro_f1) for m in metrics_by_id.values()]
        )

        # Robustness test on noisy PT-BR subset for Pareto-optimal systems.
        noisy_subset = dataset[: min(noisy_subset_size, len(dataset))]
        noisy_gt_map = {item.review_id: item for item in noisy_subset}
        noisy_y_true = [item.label.categoria for item in noisy_subset]
        robustness: dict[str, dict[str, float]] = {}
        for sid in pareto_ids:
            spec = next(s for s in specs if s.system_id == sid)
            runner = build_runner(spec, cache=cache, ground_truth=noisy_gt_map)
            clean_recs = all_records[sid][: len(noisy_subset)]
            noisy_recs = [
                runner.run_sample(
                    ReviewSample(
                        review_id=item.review_id,
                        score=item.score,
                        text=inject_ptbr_noise(item.text, seed=seed + idx),
                    ),
                    variant="noisy",
                )
                for idx, item in enumerate(noisy_subset)
            ]
            d_f1, lo, hi, p_val = paired_bootstrap_diff(
                noisy_y_true,
                _extract_predictions(noisy_recs),
                _extract_predictions(clean_recs),
                macro_f1_metric,
                seed=seed,
            )
            robustness[sid] = {
                "delta_macro_f1_noisy_vs_clean": d_f1,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p_val,
            }

        metrics_list = list(metrics_by_id.values())
        _plot_pareto(metrics_list, set(pareto_ids), out_dir / "pareto_f1_cost.png")
        _plot_ci_forest(metrics_list, f1_cis, out_dir / "macro_f1_ci.png")

        # Recommend cheapest Pareto system with non-significant F1 loss (p_holm >= 0.05).
        non_inferior = [
            r.system_a
            for r in paired_results
            if r.system_a in pareto_ids and r.f1_p_holm >= 0.05
        ]
        recommended_id = (
            min(non_inferior, key=lambda sid: metrics_by_id[sid].cost_per_1k_usd)
            if non_inferior
            else reference_id
        )

        power_n = sample_size_paired_mcnemar(
            discordant_rate=0.15, delta=0.05, alpha=0.05, power=0.80
        )

        cm_labels, cm_arr = build_confusion_matrix(all_records[recommended_id], gt_map)
        _plot_confusion_matrix(cm_labels, cm_arr, recommended_id, out_dir / "confusion_matrix.png")

        judge_cal = None
        if judge_spec is not None:
            action_preds = {
                r.review_id: (
                    r.parsed_label.acao_desejada
                    if (r.parsed_label is not None and r.json_valid)
                    else "nao_informado"
                )
                for r in all_records[recommended_id]
            }
            judge_cal = asdict(
                calibrate_judge_on_action(
                    judge_spec=judge_spec,
                    target_system_id=recommended_id,
                    predictions=action_preds,
                    ground_truth=gt_map,
                    cache=cache,
                )
            )

        summary = {
            "n_samples": len(dataset),
            "power_analysis_min_n": power_n,
            "reference_system": reference_id,
            "recommended_system": recommended_id,
            "pareto_frontier": pareto_ids,
            "judge_calibration": judge_cal,
            "metrics": [asdict(m) for m in metrics_list],
            "accuracy_cis": {k: asdict(v) for k, v in acc_cis.items()},
            "macro_f1_cis": {k: asdict(v) for k, v in f1_cis.items()},
            "paired_vs_reference": [asdict(r) for r in paired_results],
            "robustness_noisy_ptbr": robustness,
        }

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_decision_memo(out_dir / "decision_memo.md", summary, metrics_by_id, f1_cis)
        return summary
    finally:
        if cache is not None:
            cache.close()


def _write_decision_memo(
    path: Path,
    summary: dict[str, object],
    metrics_by_id: dict[str, SystemMetrics],
    f1_cis: dict[str, BootstrapCI],
) -> None:
    ref_id = str(summary["reference_system"])
    rec_id = str(summary["recommended_system"])
    ref_m = metrics_by_id[ref_id]
    rec_m = metrics_by_id[rec_id]
    rec_ci = f1_cis[rec_id]
    ref_ci = f1_cis[ref_id]

    cost_note = (
        f"custo estimado de **US$ {rec_m.cost_per_1k_usd:.4f} / 1k reqs** "
        f"(vs. US$ {ref_m.cost_per_1k_usd:.4f} na referência)"
    )

    pareto_str = ", ".join(str(x) for x in summary["pareto_frontier"])  # type: ignore[attr-defined]
    lines = [
        "# Memo de Decisão — Triagem de Reclamações PT-BR (EvalBR)",
        "",
        f"**Amostra avaliada:** `n = {summary['n_samples']}` "
        f"(poder para Δ=5 p.p., α=0,05, 80% exige `n ≥ {summary['power_analysis_min_n']}`).",
        "",
        "## Recomendação de Produção",
        f"- **Sistema recomendado:** `{rec_id}`",
        f"- **Macro-F1:** `{rec_m.macro_f1:.4f}` "
        f"(IC 95%: `[{rec_ci.ci_low:.4f}, {rec_ci.ci_high:.4f}]`) "
        f"vs. referência `{ref_id}` com `{ref_m.macro_f1:.4f}` "
        f"(IC 95%: `[{ref_ci.ci_low:.4f}, {ref_ci.ci_high:.4f}]`).",
        f"- **Economia & Operação:** {cost_note}, JSON válido `{rec_m.json_valid_rate:.1%}` "
        f"e alucinação `{rec_m.hallucination_rate:.1%}`.",
        f"- **Fronteira de Pareto:** `{pareto_str}`",
        "",
        "## Tabela Consolidada",
        "| Sistema | Macro-F1 (IC 95%) | Acurácia | JSON Válido | Alucinação | Custo / 1k (USD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in sorted(metrics_by_id.values(), key=lambda x: x.macro_f1, reverse=True):
        ci = f1_cis[m.system_id]
        lines.append(
            f"| `{m.system_id}` | {m.macro_f1:.3f} [{ci.ci_low:.3f}, {ci.ci_high:.3f}] | "
            f"{m.accuracy:.3f} | {m.json_valid_rate:.1%} | {m.hallucination_rate:.1%} | "
            f"${m.cost_per_1k_usd:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_olist_audit(
    samples: list[ReviewSample],
    specs: list[SystemSpec],
    output_dir: Path | str,
    cache_path: Path | str | None = None,
) -> dict[str, object]:
    """Run real inference on unlabeled Olist reviews and measure reliability & agreement."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_map = {s.review_id: s for s in samples}
    cache = ResponseCache(cache_path) if cache_path is not None else None
    try:
        records_by_id: dict[str, list[EvalRecord]] = {}
        rows: list[dict[str, object]] = []
        for spec in specs:
            runner = build_runner(spec, cache=cache)
            recs = [runner.run_sample(s, variant="clean") for s in samples]
            records_by_id[spec.system_id] = recs
            n = len(recs)
            valid = sum(1 for r in recs if r.json_valid and r.parsed_label is not None)
            retried = sum(1 for r in recs if r.retries > 0)
            hallucinated = sum(
                1
                for r in recs
                if not (
                    r.json_valid
                    and r.parsed_label is not None
                    and is_evidence_grounded(
                        r.parsed_label.evidencia, sample_map[r.review_id].text
                    )
                )
            )
            latencies = np.array([r.latency_ms for r in recs], dtype=float)
            rows.append(
                {
                    "system_id": spec.system_id,
                    "n_samples": n,
                    "json_valid_rate": round(valid / n, 4),
                    "retry_rate": round(retried / n, 4),
                    "hallucination_rate": round(hallucinated / n, 4),
                    "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                }
            )

        pairwise_kappa: list[dict[str, object]] = []
        spec_ids = [s.system_id for s in specs]
        for i, id_a in enumerate(spec_ids):
            for id_b in spec_ids[i + 1 :]:
                preds_a = [
                    r.parsed_label.categoria if r.parsed_label else "__invalid__"
                    for r in records_by_id[id_a]
                ]
                preds_b = [
                    r.parsed_label.categoria if r.parsed_label else "__invalid__"
                    for r in records_by_id[id_b]
                ]
                pairwise_kappa.append(
                    {
                        "system_a": id_a,
                        "system_b": id_b,
                        "cohens_kappa": cohens_kappa(preds_a, preds_b),
                    }
                )

        # Save predictions for human annotation/inspection.
        pred_lines: list[str] = []
        for idx, sample in enumerate(samples):
            preds = {
                sid: (
                    records_by_id[sid][idx].parsed_label.model_dump()
                    if records_by_id[sid][idx].parsed_label is not None
                    else None
                )
                for sid in spec_ids
            }
            pred_lines.append(
                json.dumps(
                    {
                        "review_id": sample.review_id,
                        "score": sample.score,
                        "text": sample.text,
                        "predictions": preds,
                    },
                    ensure_ascii=False,
                )
            )
        (out_dir / "olist_predictions.jsonl").write_text(
            "\n".join(pred_lines) + "\n", encoding="utf-8"
        )

        audit = {
            "n_samples": len(samples),
            "systems": rows,
            "inter_model_kappa": pairwise_kappa,
        }
        (out_dir / "olist_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return audit
    finally:
        if cache is not None:
            cache.close()
