from __future__ import annotations

import time
from collections.abc import Iterator

from .llm import DeepSeekClient
from .models import Citation, RAGResponse, SearchResult
from .prompts import build_rag_messages
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
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.temperature = temperature
        self.max_tokens = max_tokens

    def retrieve(self, question: str, document_id: str | None = None) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("问题不能为空")
        return self.vector_store.query(question.strip(), self.top_k, document_id=document_id)

    def answer(self, question: str, document_id: str | None = None) -> RAGResponse:
        started = time.perf_counter()
        results = self.retrieve(question, document_id=document_id)
        usable = [result for result in results if result.score >= self.min_relevance]
        if not usable:
            return RAGResponse(
                answer="当前知识库中未找到与该问题足够相关的内容。请尝试换一种表述，或先导入相关论文。",
                citations=self._citations(results),
                model=self.llm.model,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                warning="low_relevance" if results else "empty_knowledge_base",
            )
        result = self.llm.chat(
            build_rag_messages(question, self._contexts(usable)),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return RAGResponse(
            answer=result.content,
            citations=self._citations(usable),
            model=result.model,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            usage=result.usage,
        )

    def stream_answer(self, question: str, document_id: str | None = None) -> tuple[Iterator[str], list[Citation]]:
        results = self.retrieve(question, document_id=document_id)
        usable = [result for result in results if result.score >= self.min_relevance]
        citations = self._citations(usable or results)
        if not usable:
            message = "当前知识库中未找到与该问题足够相关的内容。请尝试换一种表述，或先导入相关论文。"
            return iter([message]), citations
        stream = self.llm.stream_chat(
            build_rag_messages(question, self._contexts(usable)),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return stream, citations

    @staticmethod
    def _contexts(results: list[SearchResult]) -> list[str]:
        return [
            f"[{index}] 文档：{item.chunk.source_name}；页码：{item.chunk.page}\n{item.chunk.text}"
            for index, item in enumerate(results, start=1)
        ]

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
