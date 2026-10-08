from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from server import main  # noqa: E402
from server.storage import ConversationStore  # noqa: E402


def test_health_settings_and_session_lifecycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "web.sqlite3"))
    client = TestClient(main.app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model"] == "deepseek-flash"
    updated_settings = client.patch("/api/settings", json={"retrieval_mode": "vector", "top_k": 3})
    assert updated_settings.status_code == 200
    assert updated_settings.json()["retrieval_mode"] == "vector"
    # Restore global settings for any test process that reuses this module.
    client.patch("/api/settings", json={"retrieval_mode": "hybrid_rerank", "top_k": 5})
    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "web.sqlite3"))
    created = client.post("/api/sessions", json={"title": "检索实验"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    assert client.get(f"/api/sessions/{session_id}").json()["messages"] == []
    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "新标题"})
    assert renamed.json()["title"] == "新标题"
    assert client.delete(f"/api/sessions/{session_id}").json()["deleted"] is True


def test_sse_chat_persists_citations_without_model_call(tmp_path: Path, monkeypatch) -> None:
    class FakeCitation:
        index, source_name, page, chunk_id, score, excerpt = 1, "paper.pdf", 3, "chunk-1", .91, "真实证据"

    class FakeEngine:
        retrieval_mode = "vector"
        def stream_answer(self, question, document_id=None):
            assert question == "Transformer 是什么？"
            return iter(["这是", "一段回答。"]), [FakeCitation()]

    class FakeService:
        def create_rag_engine(self): return FakeEngine()

    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "web.sqlite3"))
    monkeypatch.setattr(main.state, "_service", FakeService())
    client = TestClient(main.app)
    response = client.post("/api/chat/stream", json={"question": "Transformer 是什么？"})
    assert response.status_code == 200
    assert "event: token" in response.text and "event: citations" in response.text
    sessions = client.get("/api/sessions").json()["sessions"]
    messages = client.get(f"/api/sessions/{sessions[0]['session_id']}").json()["messages"]
    assert messages[-1]["content"] == "这是一段回答。"
    assert messages[-1]["citations"][0]["page"] == 3
