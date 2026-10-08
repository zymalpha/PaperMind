from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class ConversationStore:
    """SQLite store for UI conversations and rendered evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    tools TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )"""
            )

    def create(self, title: str = "新对话") -> dict[str, Any]:
        now = time.time()
        session_id = str(uuid.uuid4())
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?)", (session_id, title[:120], now, now))
        return {"session_id": session_id, "title": title[:120], "created_at": now, "updated_at": now}

    def ensure(self, session_id: str | None) -> dict[str, Any]:
        return self.get(session_id) if session_id and self.get(session_id) else self.create()

    def list(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT session_id,title,created_at,updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
        return [self._session_row(row) for row in rows]

    def get(self, session_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT session_id,title,created_at,updated_at FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return self._session_row(row) if row else None

    def rename(self, session_id: str, title: str) -> dict[str, Any] | None:
        if not self.get(session_id):
            return None
        now = time.time()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE sessions SET title=?,updated_at=? WHERE session_id=?", (title[:120], now, session_id))
        return self.get(session_id)

    def delete(self, session_id: str) -> bool:
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA foreign_keys=ON")
            cursor = db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        now = time.time()
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO messages(session_id,role,content,citations,tools,created_at) VALUES(?,?,?,?,?,?)",
                (session_id, role, content, json.dumps(citations or [], ensure_ascii=False), json.dumps(tools or [], ensure_ascii=False), now),
            )
            db.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT role,content,citations,tools,created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        result = []
        for role, content, citations, tools, created_at in rows:
            result.append({"role": role, "content": content, "citations": self._json_list(citations), "tools": self._json_list(tools), "created_at": created_at})
        return result

    @staticmethod
    def _session_row(row: tuple[Any, ...]) -> dict[str, Any]:
        return {"session_id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    @staticmethod
    def _json_list(value: str) -> list[Any]:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, json.JSONDecodeError):
            return []
