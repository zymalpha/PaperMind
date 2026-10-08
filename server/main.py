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

from .storage import ConversationStore

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    session_id: str | None = None
    document_id: str | None = None
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
            return self.settings


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
    return {
        "status": "ok", "app": settings.app_name, "model": settings.llm_model,
        "llm_configured": settings.llm_configured, "service_initialized": service is not None,
        "vector_store": {"status": "ready" if service else "not_initialized", "count": service.vector_store.count if service else None},
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
                            "indexed_at": doc.indexed_at, "chunk_strategy": doc.chunk_strategy, "status": "indexed"} for doc in docs], "count": len(docs)}


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
                result = service.create_agent().run(payload.question, session_id=session_id)
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
                stream, citations = engine.stream_answer(payload.question, payload.document_id, history, cancel_event)
                citation_data = [_citation_dict(item) for item in citations]
                events.put(("citations", citation_data))
                pieces: list[str] = []
                for piece in stream:
                    pieces.append(piece)
                    events.put(("token", piece))
                answer = "".join(pieces)
                state.conversations.add_message(session_id, "assistant", answer, citations=citation_data)
                events.put(("final", {"answer": answer, "citations": citation_data, "tools": [],
                                      "elapsed_ms": round((time.perf_counter() - started) * 1000)}))
        except Exception as exc:
            events.put(("error", {"message": f"{type(exc).__name__}: {str(exc)[:500]}"}))
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
