"""Integration test for the full end-to-end EvalBR pipeline."""

from __future__ import annotations

from pathlib import Path

from evalbr.data import generate_synthetic_ptbr_dataset
from evalbr.pipeline import default_experiment_matrix, run_benchmark


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    dataset = generate_synthetic_ptbr_dataset(n_samples=60, seed=123)
    specs = default_experiment_matrix()[:4]
    out_dir = tmp_path / "reports"
    cache_path = tmp_path / "cache.sqlite"

    summary = run_benchmark(
        dataset=dataset,
        specs=specs,
        output_dir=out_dir,
        cache_path=cache_path,
        noisy_subset_size=20,
        seed=123,
    )

    assert summary["n_samples"] == 60
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "decision_memo.md").exists()
    assert (out_dir / "pareto_f1_cost.png").exists()
    assert (out_dir / "macro_f1_ci.png").exists()
