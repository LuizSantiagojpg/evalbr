"""Command-line interface for EvalBR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalbr.data import (
    download_olist_reviews,
    generate_synthetic_ptbr_dataset,
    load_labeled_jsonl,
    load_olist_reviews_csv,
    save_labeled_jsonl,
    save_samples_jsonl,
)
from evalbr.pipeline import (
    default_experiment_matrix,
    run_benchmark,
    run_olist_audit,
    specs_from_json,
)
from evalbr.stats import sample_size_paired_mcnemar, sample_size_two_proportions


def _cmd_power(args: argparse.Namespace) -> None:
    n_paired = sample_size_paired_mcnemar(
        discordant_rate=args.discordant_rate,
        delta=args.delta,
        alpha=args.alpha,
        power=args.power,
    )
    n_indep = sample_size_two_proportions(
        p1=args.baseline_acc,
        delta=args.delta,
        alpha=args.alpha,
        power=args.power,
    )
    print(
        json.dumps(
            {
                "paired_mcnemar_n": n_paired,
                "independent_per_arm_n": n_indep,
                "delta": args.delta,
                "alpha": args.alpha,
                "power": args.power,
            },
            indent=2,
        )
    )


def _cmd_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    dataset_path = out_dir / "demo_dataset.jsonl"
    dataset = generate_synthetic_ptbr_dataset(n_samples=args.samples, seed=args.seed)
    save_labeled_jsonl(dataset, dataset_path)

    summary = run_benchmark(
        dataset=dataset,
        specs=default_experiment_matrix(),
        output_dir=out_dir,
        cache_path=out_dir / "cache.sqlite",
        seed=args.seed,
    )
    print(f"Benchmark concluído (n={summary['n_samples']}).")
    print(f"Sistema referência: {summary['reference_system']}")
    print(f"Sistema recomendado: {summary['recommended_system']}")
    print(f"Fronteira de Pareto: {summary['pareto_frontier']}")
    print(f"Relatórios salvos em: {out_dir.resolve()}")


def _cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    if args.dataset:
        dataset = load_labeled_jsonl(args.dataset)
    else:
        dataset = generate_synthetic_ptbr_dataset(n_samples=args.samples, seed=args.seed)
        save_labeled_jsonl(dataset, out_dir / "dataset.jsonl")

    specs = specs_from_json(args.config)
    summary = run_benchmark(
        dataset=dataset,
        specs=specs,
        output_dir=out_dir,
        cache_path=out_dir / "cache.sqlite",
        noisy_subset_size=args.noisy_samples,
        max_workers=args.workers,
        seed=args.seed,
    )
    print(f"Benchmark real concluído (n={summary['n_samples']}).")
    print(f"Sistema referência: {summary['reference_system']}")
    print(f"Sistema recomendado: {summary['recommended_system']}")
    print(f"Fronteira de Pareto: {summary['pareto_frontier']}")
    print(f"Relatórios salvos em: {out_dir.resolve()}")


def _cmd_olist(args: argparse.Namespace) -> None:
    csv_path = (
        Path(args.csv)
        if args.csv
        else download_olist_reviews(output_dir=args.data_dir)
    )
    samples = load_olist_reviews_csv(csv_path, n_samples=args.samples, seed=args.seed)
    out_dir = Path(args.output_dir)
    save_samples_jsonl(samples, out_dir / "olist_sample.jsonl")
    specs = specs_from_json(args.config)
    audit = run_olist_audit(
        samples=samples,
        specs=specs,
        output_dir=out_dir,
        cache_path=out_dir / "cache.sqlite",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalbr", description="EvalBR LLM Evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_power = sub.add_parser("power", help="Compute required sample size via power analysis.")
    p_power.add_argument("--baseline-acc", type=float, default=0.80)
    p_power.add_argument("--discordant-rate", type=float, default=0.15)
    p_power.add_argument("--delta", type=float, default=0.05)
    p_power.add_argument("--alpha", type=float, default=0.05)
    p_power.add_argument("--power", type=float, default=0.80)
    p_power.set_defaults(func=_cmd_power)

    p_demo = sub.add_parser("demo", help="Run full offline benchmark demo with stats & charts.")
    p_demo.add_argument("--samples", type=int, default=420)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--output-dir", type=str, default="reports")
    p_demo.set_defaults(func=_cmd_demo)

    p_run = sub.add_parser("run", help="Run benchmark from a JSON specs config (LM Studio / APIs).")
    p_run.add_argument("--config", type=str, required=True)
    p_run.add_argument("--dataset", type=str, default=None)
    p_run.add_argument("--samples", type=int, default=60)
    p_run.add_argument("--noisy-samples", type=int, default=20)
    p_run.add_argument("--workers", type=int, default=2)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--output-dir", type=str, default="reports_real")
    p_run.set_defaults(func=_cmd_run)

    p_olist = sub.add_parser(
        "olist",
        help="Download/load real Kaggle Olist reviews and run reliability & kappa audit.",
    )
    p_olist.add_argument("--config", type=str, required=True)
    p_olist.add_argument("--csv", type=str, default=None)
    p_olist.add_argument("--data-dir", type=str, default="data")
    p_olist.add_argument("--samples", type=int, default=30)
    p_olist.add_argument("--seed", type=int, default=42)
    p_olist.add_argument("--output-dir", type=str, default="reports_olist")
    p_olist.set_defaults(func=_cmd_olist)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
