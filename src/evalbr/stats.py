"""Statistical rigor layer: power analysis, bootstrap CIs, McNemar, Holm, kappa, Pareto."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, chi2, norm
from sklearn.metrics import cohen_kappa_score, f1_score

from evalbr.schema import CATEGORIES


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate and percentile confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95


@dataclass(frozen=True)
class PairedTestResult:
    """Paired statistical comparison between system A (candidate) and system B (baseline)."""

    system_a: str
    system_b: str
    delta_acc: float
    mcnemar_stat: float
    mcnemar_p_value: float
    mcnemar_p_holm: float
    delta_macro_f1: float
    f1_ci_low: float
    f1_ci_high: float
    f1_p_value: float
    f1_p_holm: float
    discordant_a_only: int
    discordant_b_only: int


def sample_size_two_proportions(
    p1: float,
    delta: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute per-arm sample size for detecting `delta` accuracy gain (two-sided)."""
    if not (0.0 < p1 < 1.0) or not (0.0 < p1 + delta < 1.0):
        raise ValueError("p1 and p1 + delta must lie strictly inside (0, 1).")
    p2 = p1 + delta
    p_bar = 0.5 * (p1 + p2)
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return math.ceil(numerator / (delta**2))


def sample_size_paired_mcnemar(
    discordant_rate: float = 0.15,
    delta: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute paired sample size N for McNemar's test given discordant proportion."""
    if not (0.0 < abs(delta) < discordant_rate <= 1.0):
        raise ValueError("Must satisfy 0 < |delta| < discordant_rate <= 1.")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    num = (
        z_alpha * math.sqrt(discordant_rate)
        + z_beta * math.sqrt(discordant_rate - delta**2)
    ) ** 2
    return math.ceil(num / (delta**2))


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Non-parametric percentile bootstrap CI over paired items."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length.")

    n = len(y_true)
    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(y_true, y_pred))
    samples = np.empty(n_resamples, dtype=float)

    true_arr = np.array(y_true, dtype=object)
    pred_arr = np.array(y_pred, dtype=object)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(true_arr[idx].tolist(), pred_arr[idx].tolist())

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(
        estimate=round(estimate, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        confidence=confidence,
    )


def macro_f1_metric(y_true: list[str], y_pred: list[str]) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(CATEGORIES), average="macro", zero_division=0)
    )


def accuracy_metric(y_true: list[str], y_pred: list[str]) -> float:
    matches = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return matches / len(y_true)


def mcnemar_test(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
) -> tuple[float, float, int, int]:
    """Paired McNemar test on item-level correctness.

    Uses exact two-sided binomial test when discordant count (b + c) < 25,
    and Edwards continuity-corrected chi-squared statistic otherwise.
    Returns (statistic, p_value, a_only_correct, b_only_correct).
    """
    a_only = 0
    b_only = 0
    for yt, pa, pb in zip(y_true, pred_a, pred_b, strict=True):
        a_ok = pa == yt
        b_ok = pb == yt
        if a_ok and not b_ok:
            a_only += 1
        elif b_ok and not a_ok:
            b_only += 1

    discordant = a_only + b_only
    if discordant == 0:
        return 0.0, 1.0, 0, 0
    if discordant < 25:
        res = binomtest(min(a_only, b_only), n=discordant, p=0.5, alternative="two-sided")
        return float(abs(a_only - b_only)), float(res.pvalue), a_only, b_only

    stat = ((abs(a_only - b_only) - 1.0) ** 2) / discordant
    p_val = float(chi2.sf(stat, df=1))
    return round(float(stat), 4), p_val, a_only, b_only


def paired_bootstrap_diff(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired bootstrap for metric(A) - metric(B), returning (delta, low, high, p_value)."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    delta_obs = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))

    true_arr = np.array(y_true, dtype=object)
    a_arr = np.array(pred_a, dtype=object)
    b_arr = np.array(pred_b, dtype=object)
    diffs = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        t_s = true_arr[idx].tolist()
        diffs[i] = metric_fn(t_s, a_arr[idx].tolist()) - metric_fn(t_s, b_arr[idx].tolist())

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))

    # Centered bootstrap distribution for two-sided p-value under H0: delta = 0.
    centered = diffs - float(np.mean(diffs))
    p_val = float(np.mean(np.abs(centered) >= abs(delta_obs)))
    p_val = max(p_val, 1.0 / (n_resamples + 1))
    return round(delta_obs, 4), round(ci_low, 4), round(ci_high, 4), round(p_val, 6)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni correction preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = (m - rank) * p_val
        running_max = min(1.0, max(running_max, scaled))
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for inter-annotator or judge-vs-human agreement."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Label lists must be non-empty and equal length.")
    return round(float(cohen_kappa_score(labels_a, labels_b)), 4)


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Identify non-dominated systems given (system_id, cost_per_1k, macro_f1).

    A system is dominated if another achieves <= cost and >= macro_f1 with
    at least one strict inequality.
    """
    non_dominated: list[str] = []
    for sid, cost_i, f1_i in points:
        dominated = any(
            (cost_j <= cost_i and f1_j >= f1_i) and (cost_j < cost_i or f1_j > f1_i)
            for other_id, cost_j, f1_j in points
            if other_id != sid
        )
        if not dominated:
            non_dominated.append(sid)
    return non_dominated
