"""SQLite-backed deterministic response cache for LLM inference calls."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from evalbr.schema import EvalRecord


class ResponseCache:
    """Thread-safe persistent cache keyed by (system_id, prompt_hash, input_hash, variant)."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(system_id: str, prompt_hash: str, input_hash: str, variant: str) -> str:
        return f"{system_id}:{variant}:{prompt_hash}:{input_hash}"

    def get(self, key: str) -> EvalRecord | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT payload_json FROM eval_cache WHERE cache_key = ?", (key,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["from_cache"] = True
        return EvalRecord.model_validate(data)

    def put(self, key: str, record: EvalRecord) -> None:
        payload = record.model_copy(update={"from_cache": False}).model_dump_json()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_cache (cache_key, payload_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (key, payload),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
