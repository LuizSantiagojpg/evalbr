"""Tests for SQLite response cache."""

from __future__ import annotations

from pathlib import Path

from evalbr.cache import ResponseCache
from evalbr.schema import EvalRecord, ReviewLabel


def test_cache_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    rec = EvalRecord(
        system_id="sys_a",
        review_id="rev_1",
        raw_output="{}",
        parsed_label=ReviewLabel(
            categoria="elogio",
            sentimento="positivo",
            menciona_prazo=False,
            acao_desejada="nenhuma",
            evidencia="ótimo produto",
        ),
        json_valid=True,
        retries=0,
        latency_ms=12.5,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.0001,
    )

    with ResponseCache(db_path) as cache:
        key = ResponseCache.make_key("sys_a", "p1", "i1", "clean")
        assert cache.get(key) is None
        cache.put(key, rec)
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded.from_cache is True
        assert loaded.parsed_label == rec.parsed_label
