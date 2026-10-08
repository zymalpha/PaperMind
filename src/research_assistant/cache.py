from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np


class SemanticAnswerCache:
    """Local SQLite cache keyed by question vectors and knowledge-base fingerprint."""

    def __init__(self, path: Path, threshold: float = 0.96, ttl_seconds: int = 86400, max_entries: int = 500) -> None:
        self.path, self.threshold, self.ttl_seconds, self.max_entries = path, threshold, ttl_seconds, max_entries
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, vector TEXT, payload TEXT, created REAL)")

    def get(self, vector: list[float], namespace: str) -> dict | None:
        now, target = time.time(), np.asarray(vector, dtype=np.float32)
        best: tuple[float, str] | None = None
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT key, vector, payload, created FROM answers WHERE key LIKE ?", (f"{namespace}:%",)).fetchall()
            for key, raw_vector, payload, created in rows:
                if now - created > self.ttl_seconds:
                    continue
                candidate = np.asarray(json.loads(raw_vector), dtype=np.float32)
                denominator = float(np.linalg.norm(target) * np.linalg.norm(candidate))
                similarity = float(np.dot(target, candidate) / denominator) if denominator else 0.0
                if similarity >= self.threshold and (best is None or similarity > best[0]):
                    best = (similarity, payload)
        return json.loads(best[1]) if best else None

    def put(self, vector: list[float], namespace: str, payload: dict) -> None:
        key = f"{namespace}:{time.time_ns()}"
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO answers VALUES(?,?,?,?)", (key, json.dumps(vector), json.dumps(payload, ensure_ascii=False), time.time()))
            db.execute("DELETE FROM answers WHERE key IN (SELECT key FROM answers ORDER BY created DESC LIMIT -1 OFFSET ?)", (self.max_entries,))
