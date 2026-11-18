"""Tests for statistical layer: power, bootstrap, McNemar, Holm, kappa, Pareto."""

from __future__ import annotations

from evalbr.stats import (
    accuracy_metric,
    bootstrap_ci,
    cohens_kappa,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_diff,
    pareto_frontier,
    sample_size_paired_mcnemar,
    sample_size_two_proportions,
)


def test_power_analysis_reasonable_ranges() -> None:
    n_paired = sample_size_paired_mcnemar(discordant_rate=0.15, delta=0.05)
    n_indep = sample_size_two_proportions(p1=0.80, delta=0.05)
    assert 400 <= n_paired <= 550
    assert n_indep > n_paired


def test_bootstrap_ci_contains_estimate() -> None:
    y_true = ["elogio"] * 30 + ["atraso_entrega"] * 30
    y_pred = ["elogio"] * 27 + ["atraso_entrega"] * 3 + ["atraso_entrega"] * 27 + ["elogio"] * 3
    ci = bootstrap_ci(y_true, y_pred, accuracy_metric, n_resamples=500, seed=7)
    assert ci.ci_low <= ci.estimate <= ci.ci_high
    assert abs(ci.estimate - 0.9) < 1e-6


def test_mcnemar_exact_and_chi2_branches() -> None:
    y_true = ["a"] * 40
    pred_a = ["a"] * 36 + ["b"] * 4
    pred_b = ["a"] * 20 + ["b"] * 20
    stat, p_val, a_only, b_only = mcnemar_test(y_true, pred_a, pred_b)
    assert a_only == 16
    assert b_only == 0
    assert p_val < 0.001
    assert stat > 0


def test_paired_bootstrap_diff_detects_clear_winner() -> None:
    y_true = ["a"] * 50
    pred_a = ["a"] * 48 + ["b"] * 2
    pred_b = ["a"] * 25 + ["b"] * 25
    delta, lo, hi, p_val = paired_bootstrap_diff(
        y_true, pred_a, pred_b, accuracy_metric, n_resamples=500, seed=1
    )
    assert delta > 0.4
    assert lo > 0.2
    assert hi <= 1.0
    assert p_val < 0.01


def test_holm_bonferroni_monotonicity() -> None:
    adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.20])
    assert adjusted == [0.04, 0.09, 0.09, 0.2]


def test_cohens_kappa_and_pareto() -> None:
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    front = pareto_frontier(
        [("cheap_good", 0.1, 0.88), ("expensive_worse", 2.0, 0.85), ("sota", 3.0, 0.93)]
    )
    assert set(front) == {"cheap_good", "sota"}
