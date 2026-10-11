from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_assistant.config import Settings, load_settings
from research_assistant.document_store import DocumentManifest
from research_assistant.service import ResearchAssistantService
from research_assistant.llm import LLMError

from .storage import ConversationStore

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    session_id: str | None = None
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    direction: str | None = Field(default=None, max_length=80)
    include_attachments: bool = True
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)
    retrieval_mode: Literal["vector", "hybrid", "hybrid_rerank"] | None = None
    use_agent: bool = False


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=120)


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class SettingsPatch(BaseModel):
    retrieval_mode: Literal["vector", "hybrid", "hybrid_rerank"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class AppState:
    def __init__(self) -> None:
        self.settings: Settings = load_settings()
        self._service: ResearchAssistantService | None = None
        self._lock = threading.RLock()
        self.conversations = ConversationStore(self.settings.index_dir / "web_conversations.sqlite3")
        self._attachment_services: dict[str, ResearchAssistantService] = {}

    def service(self) -> ResearchAssistantService:
        with self._lock:
            if self._service is None:
                self._service = ResearchAssistantService(self.settings)
            return self._service

    def update_settings(self, patch: SettingsPatch) -> Settings:
        updates = patch.model_dump(exclude_none=True)
        with self._lock:
            self.settings = replace(self.settings, **updates)
            self._service = None
            self.conversations = ConversationStore(self.settings.index_dir / "web_conversations.sqlite3")
            self._attachment_services.clear()
            return self.settings

    def attachment_service(self, session_id: str) -> ResearchAssistantService:
        """Return a session-local index sharing the loaded embedding model."""
        with self._lock:
            service = self._attachment_services.get(session_id)
            if service is not None:
                return service
            root = (self.settings.data_dir / "attachments" / session_id).resolve()
            scoped = replace(self.settings, data_dir=root, upload_dir=root / "uploads",
                             index_dir=root / "index", log_dir=root / "logs",
                             collection_name=f"attachments_{session_id.replace('-', '')[:24]}")
            scoped.ensure_directories()
            service = ResearchAssistantService(scoped, embedding=self.service().embedding)
            self._attachment_services[session_id] = service
            return service


state = AppState()
app = FastAPI(title="PaperMind API", version="0.3.0")
origins = os.getenv("PAPERMIND_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> Any:
    fields = [{"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": {"error": "请求参数无效", "fields": fields}})


def _error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": message})


def _citation_dict(citation: Any) -> dict[str, Any]:
    return {
        "index": citation.index,
        "source_name": citation.source_name,
        "page": citation.page,
        "chunk_id": citation.chunk_id,
        "score": citation.score,
        "excerpt": citation.excerpt,
    }


def _agent_citations(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in tool_trace:
        if trace.get("tool") != "knowledge_search" or not trace.get("ok"):
            continue
        result = trace.get("result") or {}
        items = result.get("results", []) if isinstance(result, dict) else []
        for item in items:
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            citations.append({"index": len(citations) + 1, "source_name": item.get("source", ""),
                              "page": item.get("page", 1), "chunk_id": chunk_id,
                              "score": item.get("score", 0), "excerpt": item.get("text", "")[:240]})
    return citations


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings, service = state.settings, state._service
    reranker = getattr(service, "_reranker", None) if service else None
    reranker_loaded = getattr(reranker, "_instance", None) is not None
    vector_status: dict[str, Any] = {
        "status": "not_initialized",
        "count": None,
    }
    if service is not None:
        try:
            vector_status = {"status": "ready", "count": service.vector_store.count}
        except Exception as exc:
            # A damaged persisted HNSW segment must not turn the health route
            # into an unstructured 500. Keep the failure visible to operators
            # while allowing the UI to report a degraded local index.
            vector_status = {
                "status": "error",
                "count": None,
                "error": type(exc).__name__,
            }
    return {
        "status": "ok", "app": settings.app_name, "model": settings.llm_model,
        "llm_configured": settings.llm_configured, "service_initialized": service is not None,
        "vector_store": vector_status,
        "embedding": {"model": settings.embedding_model, "status": "loaded" if service else "configured_not_loaded"},
        "reranker": {"model": settings.reranker_model,
                     "status": "loaded" if reranker_loaded else "enabled_not_loaded" if settings.reranker_enabled else "disabled"},
    }


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    settings = state.settings
    return {"model": settings.llm_model, "base_url": settings.llm_base_url, "api_configured": settings.llm_configured,
            "retrieval_mode": settings.retrieval_mode, "top_k": settings.top_k, "candidate_k": settings.candidate_k,
            "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model,
            "reranker_enabled": settings.reranker_enabled, "web_search_enabled": settings.web_search_enabled}


@app.patch("/api/settings")
def patch_settings(patch: SettingsPatch) -> dict[str, Any]:
    state.update_settings(patch)
    return get_settings()


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    return {"sessions": state.conversations.list()}


@app.post("/api/sessions")
def create_session(payload: SessionCreate) -> dict[str, Any]:
    return state.conversations.create(payload.title)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = state.conversations.get(session_id)
    if not session:
        raise _error(404, "会话不存在")
    return {**session, "messages": state.conversations.messages(session_id)}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, payload: SessionRename) -> dict[str, Any]:
    result = state.conversations.rename(session_id, payload.title)
    if not result:
        raise _error(404, "会话不存在")
    return result


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    if not state.conversations.delete(session_id):
        raise _error(404, "会话不存在")
    return {"deleted": True, "session_id": session_id}


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    manifest = DocumentManifest(state.settings.index_dir / "documents.json")
    docs = sorted(manifest.list(), key=lambda item: item.indexed_at, reverse=True)
    return {"documents": [{"document_id": doc.document_id, "source_name": doc.source_name, "chunk_count": doc.chunk_count,
                            "indexed_at": doc.indexed_at, "chunk_strategy": doc.chunk_strategy, "status": "indexed",
                            **(doc.metadata or {})} for doc in docs], "count": len(docs)}


def _document_dict(doc: Any) -> dict[str, Any]:
    return {"document_id": doc.document_id, "source_name": doc.source_name, "chunk_count": doc.chunk_count,
            "indexed_at": doc.indexed_at, "chunk_strategy": doc.chunk_strategy, "status": "indexed",
            **(doc.metadata or {})}


def _session_or_404(session_id: str) -> None:
    if not state.conversations.get(session_id):
        raise _error(404, "会话不存在")


@app.get("/api/sessions/{session_id}/attachments")
def list_attachments(session_id: str) -> dict[str, Any]:
    _session_or_404(session_id)
    service = state.attachment_service(session_id)
    docs = service.list_documents()
    return {"attachments": [_document_dict(doc) for doc in docs], "count": len(docs)}


@app.post("/api/sessions/{session_id}/attachments/upload")
async def upload_attachments(session_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    _session_or_404(session_id)
    if len(files) > 10:
        raise _error(413, "单次最多上传 10 个临时附件")
    service = state.attachment_service(session_id)
    results: list[dict[str, Any]] = []
    for upload in files:
        name, target = _safe_upload_name(upload.filename), None
        started = time.perf_counter()
        try:
            target = service._unique_upload_path(name)
            total = 0
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise _error(413, f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制")
                    handle.write(chunk)
            document, created = await run_in_threadpool(
                service.index_file, target, False, None, False,
                {"scope": "session_attachment", "session_id": session_id},
            )
            results.append({"ok": True, "created": created, "elapsed_ms": round((time.perf_counter() - started) * 1000),
                            "attachment": _document_dict(document)})
        except HTTPException:
            if target and target.exists():
                target.unlink()
            raise
        except Exception as exc:
            if target and target.exists():
                target.unlink()
            results.append({"ok": False, "filename": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "elapsed_ms": round((time.perf_counter() - started) * 1000)})
        finally:
            await upload.close()
    return {"results": results}


@app.delete("/api/sessions/{session_id}/attachments/{document_id}")
def delete_attachment(session_id: str, document_id: str) -> dict[str, Any]:
    _session_or_404(session_id)
    if not state.attachment_service(session_id).delete_document(document_id):
        raise _error(404, "临时附件不存在")
    return {"deleted": True, "document_id": document_id}


@app.post("/api/sessions/{session_id}/attachments/{document_id}/save")
def save_attachment(session_id: str, document_id: str) -> dict[str, Any]:
    _session_or_404(session_id)
    temporary = state.attachment_service(session_id).manifest.get(document_id)
    if temporary is None or not Path(temporary.source_path).is_file():
        raise _error(404, "临时附件不存在")
    document, _ = state.service().index_file(temporary.source_path, copy_to_uploads=True,
                                              metadata={k: v for k, v in (temporary.metadata or {}).items()
                                                        if k not in {"scope", "session_id"}})
    return {"saved": True, "document": _document_dict(document)}


def _safe_upload_name(filename: str | None) -> str:
    name = Path((filename or "upload").replace("\\", "/")).name
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS or not name:
        raise _error(400, "仅支持 PDF、DOCX、TXT 和 Markdown 文件")
    return name


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if len(files) > 20:
        raise _error(413, "单次最多上传 20 个文件")
    service = state.service()
    results: list[dict[str, Any]] = []
    for upload in files:
        name, target = _safe_upload_name(upload.filename), None
        try:
            target = service._unique_upload_path(name)
            total = 0
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise _error(413, f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制")
                    handle.write(chunk)
            document, created = await run_in_threadpool(service.index_file, target, False)
            if not created:
                existing = service.manifest.get(document.document_id)
                if existing and Path(existing.source_path).resolve() != target.resolve() and target.exists():
                    target.unlink()
            results.append({"ok": True, "created": created, "document": {"document_id": document.document_id,
                             "source_name": document.source_name, "chunk_count": document.chunk_count,
                             "indexed_at": document.indexed_at, "chunk_strategy": document.chunk_strategy, "status": "indexed"}})
        except HTTPException:
            if target and target.exists(): target.unlink()
            raise
        except Exception as exc:
            if target and target.exists(): target.unlink()
            results.append({"ok": False, "filename": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        finally:
            await upload.close()
    return {"results": results}


@app.post("/api/documents/{document_id}/reindex")
async def reindex_document(document_id: str) -> dict[str, Any]:
    service = state.service()
    document = service.manifest.get(document_id)
    if not document:
        raise _error(404, "文档不存在")
    path = Path(document.source_path).resolve()
    if not path.is_file() or not path.is_relative_to(service.settings.upload_dir.resolve()):
        raise _error(404, "文档源文件不可用")
    updated, _ = await run_in_threadpool(service.index_file, path, False, document.chunk_strategy, True)
    return {"ok": True, "document": {"document_id": updated.document_id, "source_name": updated.source_name,
                                      "chunk_count": updated.chunk_count, "indexed_at": updated.indexed_at,
                                      "chunk_strategy": updated.chunk_strategy, "status": "indexed"}}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    if not state.service().delete_document(document_id):
        raise _error(404, "文档不存在")
    return {"deleted": True, "document_id": document_id}


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    try:
        session = state.conversations.ensure(payload.session_id)
    except KeyError:
        raise _error(404, "会话不存在") from None
    session_id = session["session_id"]
    prior_messages = state.conversations.messages(session_id)
    if not prior_messages and session["title"] == "新对话":
        state.conversations.rename(session_id, payload.question.strip()[:36])
    state.conversations.add_message(session_id, "user", payload.question)
    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()
    cancel_event = threading.Event()
    started = time.perf_counter()

    def worker() -> None:
        try:
            service = state.service()
            events.put(("session", {"session_id": session_id}))
            if payload.use_agent:
                if payload.attachment_ids:
                    raise ValueError("Agent 当前不支持临时附件，请先保存到长期知识库后再使用")
                result = service.create_agent().run(payload.question, session_id=session_id, cancel_event=cancel_event)
                citations = _agent_citations(result.get("tool_trace", []))
                for trace in result.get("tool_trace", []):
                    events.put(("tool", trace))
                state.conversations.add_message(session_id, "assistant", result["answer"], citations=citations, tools=result.get("tool_trace", []))
                events.put(("final", {"answer": result["answer"], "citations": citations, "tools": result.get("tool_trace", []),
                                      "usage": result.get("usage", {}), "elapsed_ms": round((time.perf_counter() - started) * 1000)}))
            else:
                engine = service.create_rag_engine()
                if payload.retrieval_mode:
                    engine.retrieval_mode = payload.retrieval_mode
                history = [{"role": item["role"], "content": item["content"]} for item in prior_messages]
                requested_ids = set(payload.document_ids)
                if payload.document_id:
                    requested_ids.add(payload.document_id)
                if payload.direction:
                    direction_ids = {doc.document_id for doc in service.manifest.list()
                                     if (doc.metadata or {}).get("direction") == payload.direction}
                    if not direction_ids:
                        raise ValueError("该研究方向当前没有已索引论文")
                    requested_ids.update(direction_ids)
                if requested_ids:
                    valid_ids = {doc.document_id for doc in service.manifest.list()}
                    unknown_ids = requested_ids - valid_ids
                    if unknown_ids:
                        raise ValueError("检索范围中包含不存在的论文")
                scoped_ids = sorted(requested_ids) if requested_ids else None
                if payload.attachment_ids:
                    attachment_service = state.attachment_service(session_id)
                    attachment_ids = set(payload.attachment_ids)
                    known = {doc.document_id for doc in attachment_service.manifest.list()}
                    if not attachment_ids.issubset(known):
                        raise ValueError("临时附件不存在或不属于当前会话")
                    engine = attachment_service.create_rag_engine()
                    if payload.retrieval_mode:
                        engine.retrieval_mode = payload.retrieval_mode
                    stream, citations = engine.stream_answer(payload.question, None, history, cancel_event, document_ids=sorted(attachment_ids))
                else:
                    if scoped_ids is None:
                        stream, citations = engine.stream_answer(payload.question, payload.document_id, history, cancel_event)
                    else:
                        stream, citations = engine.stream_answer(payload.question, None, history, cancel_event, document_ids=scoped_ids)
                citation_data = [_citation_dict(item) for item in citations]
                events.put(("citations", citation_data))
                pieces: list[str] = []
                for piece in stream:
                    pieces.append(piece)
                    events.put(("token", piece))
                answer = "".join(pieces)
                if cancel_event.is_set():
                    answer = (answer + "\n\n（已停止生成，以上为部分回答。）").strip()
                state.conversations.add_message(session_id, "assistant", answer, citations=citation_data)
                if not cancel_event.is_set():
                    events.put(("final", {"answer": answer, "citations": citation_data, "tools": [],
                                          "elapsed_ms": round((time.perf_counter() - started) * 1000)}))
        except Exception as exc:
            message = str(exc) if isinstance(exc, LLMError) else "服务处理失败，请检查本地日志后重试。"
            if not cancel_event.is_set():
                try:
                    state.conversations.add_message(session_id, "assistant", f"请求失败：{message}")
                except Exception:
                    pass
                events.put(("error", {"message": message}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        completed = False
        try:
            yield _sse("session", {"session_id": session_id})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.to_thread(events.get, True, 0.25)
                except queue.Empty:
                    continue
                if item is None:
                    completed = True
                    yield _sse("done", {"session_id": session_id})
                    break
                event, data = item
                yield _sse(event, data)
        finally:
            if not completed:
                cancel_event.set()

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> dict[str, Any]:
    if not state.conversations.get(session_id):
        raise _error(404, "会话不存在")
    return {"messages": state.conversations.messages(session_id)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="127.0.0.1", port=int(os.getenv("PAPERMIND_API_PORT", "8000")), reload=False)
