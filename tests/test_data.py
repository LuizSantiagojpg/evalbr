"""Tests for Olist CSV stratified loader and JSONL persistence."""

from __future__ import annotations

from pathlib import Path

from evalbr.data import (
    load_olist_reviews_csv,
    load_samples_jsonl,
    save_samples_jsonl,
)


def test_load_olist_reviews_csv_stratified(tmp_path: Path) -> None:
    csv_path = tmp_path / "olist_reviews.csv"
    rows = ["review_id,review_score,review_comment_message"]
    for i in range(15):
        score = (i % 5) + 1
        rows.append(f'rev_{i},{score},"Comentário real de teste número {i} com mais de 10 chars"')
    # Add short/empty comments that must be filtered out.
    rows.append('rev_empty,1,""')
    rows.append('rev_short,5,"bom"')
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    samples = load_olist_reviews_csv(csv_path, n_samples=10, seed=42)
    assert len(samples) == 10
    assert {s.score for s in samples} == {1, 2, 3, 4, 5}

    jsonl_path = tmp_path / "sample.jsonl"
    save_samples_jsonl(samples, jsonl_path)
    reloaded = load_samples_jsonl(jsonl_path)
    assert reloaded == samples
