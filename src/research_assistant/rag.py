from __future__ import annotations

import time
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from .cache import SemanticAnswerCache
from .embeddings import EmbeddingProvider
from .llm import DeepSeekClient
from .models import Citation, RAGResponse, SearchResult
from .prompts import build_rag_messages
from .retrieval import HybridRetriever
from .vector_store import ChromaVectorStore


class RAGEngine:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm: DeepSeekClient,
        top_k: int = 5,
        min_relevance: float = 0.12,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        retriever: HybridRetriever | None = None,
        retrieval_mode: str = "vector",
        max_context_chars: int = 14000,
        semantic_cache_enabled: bool = True,
        semantic_cache_threshold: float = 0.96,
        embedding: EmbeddingProvider | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.max_context_chars = max_context_chars
        self.embedding = embedding or vector_store.embedding
        self.cache = SemanticAnswerCache(log_path.parent / "semantic_cache.sqlite3", semantic_cache_threshold) if semantic_cache_enabled and log_path else None
        self.log_path = log_path

    def retrieve(self, question: str, document_id: str | None = None) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("问题不能为空")
        if self.retriever:
            return self.retriever.search(question.strip(), self.top_k, self.retrieval_mode, document_id)
        return self.vector_store.query(question.strip(), self.top_k, document_id=document_id)

    def answer(self, question: str, document_id: str | None = None) -> RAGResponse:
        started = time.perf_counter()
        results = self.retrieve(question, document_id=document_id)
        usable = [result for result in results if result.score >= self.min_relevance]
        usable = self._bounded_results(usable, self.max_context_chars)
        if not usable:
            response = RAGResponse(
                answer="当前知识库中未找到与该问题足够相关的内容。请尝试换一种表述，或先导入相关论文。",
                citations=self._citations(results),
                model=self.llm.model,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                warning="low_relevance" if results else "empty_knowledge_base",
            )
            self._log(question, results, response, response.warning or "empty")
            return response
        namespace = self._namespace(usable)
        query_vector = self.embedding.embed_query(question)
        cached = self.cache.get(query_vector, namespace) if self.cache else None
        if cached:
            response = RAGResponse(cached["answer"], self._citations(usable), self.llm.model,
                                   round((time.perf_counter() - started) * 1000), warning="semantic_cache_hit")
            self._log(question, usable, response, "cache_hit")
            return response
        try:
            contexts = self._contexts(usable, self.max_context_chars)
            result = self.llm.chat(build_rag_messages(question, contexts), temperature=self.temperature, max_tokens=self.max_tokens)
            response = RAGResponse(result.content, self._citations(usable), result.model,
                                   round((time.perf_counter() - started) * 1000), result.usage,
                                   warning=self.retriever.last_warning if self.retriever else None)
            if self.cache:
                self.cache.put(query_vector, namespace, {"answer": response.answer})
        except Exception as exc:
            excerpt = "\n\n".join(f"[{i}] {item.chunk.text[:500]}" for i, item in enumerate(usable, 1))
            response = RAGResponse("模型暂时不可用。以下为本地检索到的原文片段，请依据引用自行核对；稍后可重试生成。\n\n" + excerpt,
                                   self._citations(usable), self.llm.model,
                                   round((time.perf_counter() - started) * 1000),
                                   warning=f"generation_failed:{type(exc).__name__}")
        self._log(question, usable, response, "fallback" if response.warning else "ok")
        return response

    def stream_answer(self, question: str, document_id: str | None = None) -> tuple[Iterator[str], list[Citation]]:
        started = time.perf_counter()
        results = self.retrieve(question, document_id=document_id)
        usable = [result for result in results if result.score >= self.min_relevance]
        usable = self._bounded_results(usable, self.max_context_chars)
        citations = self._citations(usable or results)
        if not usable:
            message = "当前知识库中未找到与该问题足够相关的内容。请尝试换一种表述，或先导入相关论文。"
            return iter([message]), citations
        namespace = self._namespace(usable)
        query_vector = self.embedding.embed_query(question)
        cached = self.cache.get(query_vector, namespace) if self.cache else None
        if cached:
            response = RAGResponse(cached["answer"], citations, self.llm.model,
                                   round((time.perf_counter() - started) * 1000), warning="semantic_cache_hit")
            self._log(question, usable, response, "stream_cache_hit")
            return iter([cached["answer"]]), citations

        def generate():
            pieces: list[str] = []
            try:
                for part in self.llm.stream_chat(
                    build_rag_messages(question, self._contexts(usable, self.max_context_chars)),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    pieces.append(part)
                    yield part
                answer = "".join(pieces)
                if self.cache and answer:
                    self.cache.put(query_vector, namespace, {"answer": answer})
                response = RAGResponse(answer, citations, self.llm.model, round((time.perf_counter() - started) * 1000))
                self._log(question, usable, response, "stream_ok")
            except Exception as exc:
                if not pieces:
                    yield "模型暂时不可用。以下为检索到的原文片段，请依据引用核对：\n\n" + usable[0].chunk.text[:700]
                response = RAGResponse("".join(pieces), citations, self.llm.model,
                                       round((time.perf_counter() - started) * 1000),
                                       warning=f"stream_failed:{type(exc).__name__}")
                self._log(question, usable, response, "stream_fallback")

        return generate(), citations

    @staticmethod
    def _contexts(results: list[SearchResult], max_chars: int = 14000) -> list[str]:
        contexts, used = [], 0
        for index, item in enumerate(results, 1):
            content = f"[{index}] 文档：{item.chunk.source_name}；页码：{item.chunk.page}\n{item.chunk.text}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            contexts.append(content[:remaining])
            used += min(len(content), remaining)
        return contexts

    @staticmethod
    def _bounded_results(results: list[SearchResult], max_chars: int) -> list[SearchResult]:
        selected, used = [], 0
        for item in results:
            header = f"[{len(selected) + 1}] 文档：{item.chunk.source_name}；页码：{item.chunk.page}\n"
            remaining = max_chars - used
            if remaining <= 0:
                break
            selected.append(item)
            used += min(len(header) + len(item.chunk.text), remaining)
        return selected

    def _namespace(self, results: list[SearchResult]) -> str:
        # Cache answers only while the exact ordered evidence set remains the same.
        # This also keeps [1], [2] citations aligned with the current prompt context.
        identity = ":".join(item.chunk.id for item in results)
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    def _log(self, question: str, results: list[SearchResult], response: RAGResponse, status: str) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": time.time(), "query_sha256": hashlib.sha256(question.encode()).hexdigest(),
                 "chunk_ids": [item.chunk.id for item in results], "scores": [round(item.score, 4) for item in results],
                 "elapsed_ms": response.elapsed_ms, "usage": response.usage, "status": status, "warning": response.warning}
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _citations(results: list[SearchResult]) -> list[Citation]:
        return [
            Citation(
                index=index,
                source_name=item.chunk.source_name,
                page=item.chunk.page,
                chunk_id=item.chunk.id,
                score=round(item.score, 4),
                excerpt=item.chunk.text[:240],
            )
            for index, item in enumerate(results, start=1)
        ]
