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
    client.patch("/api/settings", json={"retrieval_mode": "hybrid_rerank", "top_k": 5})
    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "web.sqlite3"))
    created = client.post("/api/sessions", json={"title": "检索实验"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    assert client.get(f"/api/sessions/{session_id}").json()["messages"] == []
    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "新标题"})
    assert renamed.json()["title"] == "新标题"
    main.state.conversations.add_message(session_id, "user", "删除后应清理")
    assert client.delete(f"/api/sessions/{session_id}").json()["deleted"] is True
    assert main.state.conversations.messages(session_id) == []
    invalid = client.patch("/api/settings", json={"top_k": 0})
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["error"] == "请求参数无效"


def test_health_reports_degraded_vector_store_instead_of_500(monkeypatch) -> None:
    class BrokenVectorStore:
        @property
        def count(self):
            raise RuntimeError("hnsw index is unreadable")

    class FakeService:
        vector_store = BrokenVectorStore()
        _reranker = None

    monkeypatch.setattr(main.state, "_service", FakeService())
    response = TestClient(main.app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["vector_store"] == {
        "status": "error",
        "count": None,
        "error": "RuntimeError",
    }


def test_upload_filename_is_sanitized_cross_platform() -> None:
    assert main._safe_upload_name(r"..\..\paper.pdf") == "paper.pdf"
    try:
        main._safe_upload_name("malware.exe")
    except Exception as error:
        assert "PDF" in str(error)
    else:
        raise AssertionError("unknown extension should be rejected")


def test_attachment_ids_are_session_scoped(tmp_path: Path, monkeypatch) -> None:
    class FakeManifest:
        def list(self): return []
        def get(self, _document_id): return None

    class FakeAttachmentService:
        manifest = FakeManifest()
        def list_documents(self): return []

    store = ConversationStore(tmp_path / "attachments.sqlite3")
    monkeypatch.setattr(main.state, "conversations", store)
    monkeypatch.setattr(main.state, "attachment_service", lambda _session_id: FakeAttachmentService())
    client = TestClient(main.app)
    session = client.post("/api/sessions", json={"title": "附件隔离"}).json()
    listed = client.get(f"/api/sessions/{session['session_id']}/attachments")
    assert listed.status_code == 200 and listed.json() == {"attachments": [], "count": 0}
    assert client.get("/api/sessions/missing/attachments").status_code == 404


def test_sse_chat_persists_citations_without_model_call(tmp_path: Path, monkeypatch) -> None:
    class FakeCitation:
        index, source_name, page, chunk_id, score, excerpt = 1, "paper.pdf", 3, "chunk-1", .91, "真实证据"

    received_history = []

    class FakeEngine:
        retrieval_mode = "vector"
        def stream_answer(self, question, document_id=None, conversation_history=None, cancel_event=None):
            assert question == "Transformer 是什么？"
            received_history.extend(conversation_history or [])
            return iter(["这是", "一段回答。"]), [FakeCitation()]

    class FakeService:
        def create_rag_engine(self): return FakeEngine()

    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "web.sqlite3"))
    monkeypatch.setattr(main.state, "_service", FakeService())
    client = TestClient(main.app)
    session = client.post("/api/sessions", json={"title": "多轮测试"}).json()
    main.state.conversations.add_message(session["session_id"], "user", "上一轮问题")
    main.state.conversations.add_message(session["session_id"], "assistant", "上一轮回答")
    response = client.post("/api/chat/stream", json={"question": "Transformer 是什么？", "session_id": session["session_id"]})
    assert response.status_code == 200
    assert "event: token" in response.text and "event: citations" in response.text
    messages = client.get(f"/api/sessions/{session['session_id']}").json()["messages"]
    assert messages[-1]["content"] == "这是一段回答。"
    assert messages[-1]["citations"][0]["page"] == 3
    assert received_history == [{"role": "user", "content": "上一轮问题"}, {"role": "assistant", "content": "上一轮回答"}]
    invalid_session = client.post("/api/chat/stream", json={"question": "不应自动创建新会话", "session_id": "missing-session"})
    assert invalid_session.status_code == 404


def test_agent_tool_trace_is_returned_as_source_citation(tmp_path: Path, monkeypatch) -> None:
    class FakeAgent:
        def run(self, question, session_id=None, cancel_event=None):
            return {"answer": "根据检索片段回答。", "usage": {}, "tool_trace": [{
                "tool": "knowledge_search", "status": "success", "ok": True, "elapsed_ms": 4,
                "result": {"results": [{"source": "paper.pdf", "page": 8, "score": .88,
                                         "chunk_id": "evidence-1", "text": "检索到的原始段落"}]},
            }]}

    class FakeService:
        def create_agent(self): return FakeAgent()

    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "agent.sqlite3"))
    monkeypatch.setattr(main.state, "_service", FakeService())
    response = TestClient(main.app).post("/api/chat/stream", json={"question": "检索证据", "use_agent": True})
    assert response.status_code == 200
    assert '"source_name": "paper.pdf"' in response.text
    sessions = main.state.conversations.list()
    messages = main.state.conversations.messages(sessions[0]["session_id"])
    assert messages[1]["citations"][0]["page"] == 8
    assert messages[1]["tools"][0]["tool"] == "knowledge_search"


def test_streaming_generation_failure_is_an_error_event_not_a_successful_answer(tmp_path: Path, monkeypatch) -> None:
    class BrokenEngine:
        retrieval_mode = "vector"
        def stream_answer(self, *_args, **_kwargs):
            def broken():
                yield "部分内容"
                raise RuntimeError("provider failure")
            return broken(), []

    class FakeService:
        def create_rag_engine(self): return BrokenEngine()

    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "broken.sqlite3"))
    monkeypatch.setattr(main.state, "_service", FakeService())
    client = TestClient(main.app)
    session = client.post("/api/sessions", json={"title": "流失败"}).json()
    response = client.post("/api/chat/stream", json={"question": "请用中文检查流错误", "session_id": session["session_id"]})
    assert response.status_code == 200
    assert 'event: token\ndata: "部分内容"' in response.text
    assert "event: error" in response.text
    assert "event: final" not in response.text
    stored = client.get(f"/api/sessions/{session['session_id']}").json()["messages"]
    assert stored[-1]["content"].startswith("请求失败")


def test_agent_api_preserves_chinese_question_utf8(tmp_path: Path, monkeypatch) -> None:
    question = "请计算 123 乘以 456。"

    class FakeAgent:
        def run(self, received, session_id=None, cancel_event=None):
            assert received == question
            return {"answer": "56088", "tool_trace": [], "usage": {}}

    class FakeService:
        def create_agent(self): return FakeAgent()

    monkeypatch.setattr(main.state, "conversations", ConversationStore(tmp_path / "utf8.sqlite3"))
    monkeypatch.setattr(main.state, "_service", FakeService())
    response = TestClient(main.app).post("/api/chat/stream", json={"question": question, "use_agent": True})
    assert response.status_code == 200
    assert '"answer": "56088"' in response.text
