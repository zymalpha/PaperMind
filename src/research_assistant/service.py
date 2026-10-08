from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .chunking import create_chunker
from .config import Settings
from .document_store import DocumentManifest
from .embeddings import EmbeddingProvider, create_embedding_provider
from .llm import DeepSeekClient
from .loaders import LoaderRegistry
from .models import IndexedDocument
from .rag import RAGEngine
from .retrieval import BGEReranker, HybridRetriever
from .vector_store import ChromaVectorStore


class ResearchAssistantService:
    def __init__(self, settings: Settings, embedding: EmbeddingProvider | None = None) -> None:
        self.settings = settings
        self.embedding = embedding or create_embedding_provider(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
        self.vector_store = ChromaVectorStore(
            settings.index_dir / "chroma",
            settings.collection_name,
            self.embedding,
        )
        self.manifest = DocumentManifest(settings.index_dir / "documents.json")
        self.loader = LoaderRegistry()
        self.chunker = create_chunker(settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap)
        self._reranker = _LazyReranker(settings.reranker_model, settings.reranker_device) if settings.reranker_enabled else None
        self.retriever = HybridRetriever(
            self.vector_store, self._reranker, settings.rrf_k, settings.candidate_k
        )

    def index_file(
        self, source_path: str | Path, copy_to_uploads: bool = True, strategy: str | None = None, force: bool = False
    ) -> tuple[IndexedDocument, bool]:
        source = Path(source_path).resolve()
        document_id, pages = self.loader.load(source)
        existing = self.manifest.get(document_id)
        if existing and self.vector_store.existing_chunk_ids(document_id) and not force:
            return existing, False

        target = source
        if copy_to_uploads and source.parent != self.settings.upload_dir.resolve():
            prior = next((doc for doc in self.manifest.list() if doc.source_name.casefold() == source.name.casefold()
                          and Path(doc.source_path).resolve().is_relative_to(self.settings.upload_dir.resolve())), None)
            target = Path(prior.source_path) if prior else self._unique_upload_path(source.name)
            # Index the source before replacing a managed upload. A failed
            # embedding/index operation must not destroy the last working copy.
            pages = [replace(page, source_path=str(target), source_name=target.name) for page in pages]

        selected_strategy = strategy or self.settings.chunk_strategy
        chunker = create_chunker(selected_strategy, self.settings.chunk_size, self.settings.chunk_overlap)
        chunks = chunker.split_pages(pages)
        previous_chunk_ids = self.vector_store.existing_chunk_ids(document_id)
        try:
            self.vector_store.upsert(chunks)
        except Exception:
            # Chroma upsert is batched, not transactional. Remove only chunks
            # created by this failed attempt and leave the prior index intact.
            current_ids = self.vector_store.existing_chunk_ids(document_id)
            created_ids = current_ids - previous_chunk_ids
            if created_ids:
                self.vector_store.delete_chunks(created_ids)
            raise
        if copy_to_uploads and source.parent != self.settings.upload_dir.resolve():
            try:
                shutil.copy2(source, target)
            except Exception:
                # Roll back this attempted version; if it already had an
                # index, restore its prior chunks from the source file.
                if existing:
                    self.vector_store.delete_document(document_id)
                    old_source = Path(existing.source_path)
                    if old_source.is_file():
                        _, old_pages = self.loader.load(old_source)
                        old_chunks = create_chunker(existing.chunk_strategy, self.settings.chunk_size,
                                                    self.settings.chunk_overlap).split_pages(old_pages)
                        self.vector_store.upsert(old_chunks)
                else:
                    self.vector_store.delete_document(document_id)
                raise

        # Retire old same-name versions only after the new index is available.
        # A strategy-only reindex keeps the same document ID but uses new chunk IDs.
        keep_ids = {chunk.id for chunk in chunks}
        self.vector_store.delete_stale_chunks(document_id, keep_ids)
        for previous in self.manifest.list():
            same_name = previous.source_name.casefold() == target.name.casefold()
            if same_name and previous.document_id != document_id:
                self.vector_store.delete_document(previous.document_id)
                self.manifest.remove(previous.document_id)
                old_path = Path(previous.source_path).resolve()
                if old_path != target.resolve() and old_path.is_relative_to(self.settings.upload_dir.resolve()) and old_path.is_file():
                    old_path.unlink()
        document = IndexedDocument(
            document_id=document_id,
            source_name=target.name,
            source_path=str(target),
            chunk_count=len(chunks),
            indexed_at=datetime.now(timezone.utc).isoformat(),
            chunk_strategy=selected_strategy,
        )
        self.manifest.put(document)
        return document, True

    def list_documents(self) -> list[IndexedDocument]:
        return sorted(self.manifest.list(), key=lambda item: item.indexed_at, reverse=True)

    def delete_document(self, document_id: str) -> bool:
        document = self.manifest.get(document_id)
        if document is None:
            return False
        self.vector_store.delete_document(document_id)
        self.manifest.remove(document_id)
        # Only delete files that are inside the managed upload directory.
        upload_root = self.settings.upload_dir.resolve()
        path = Path(document.source_path).resolve()
        if path.is_relative_to(upload_root) and path.is_file():
            path.unlink()
        return True

    def create_rag_engine(self) -> RAGEngine:
        llm = DeepSeekClient(
            self.settings.deepseek_api_key,
            self.settings.llm_base_url,
            self.settings.llm_model,
            self.settings.llm_timeout_seconds,
            self.settings.llm_max_retries,
        )
        return RAGEngine(
            self.vector_store, llm, self.settings.top_k, self.settings.min_relevance,
            self.settings.llm_temperature, self.settings.llm_max_tokens,
            retriever=self.retriever, retrieval_mode=self.settings.retrieval_mode,
            max_context_chars=self.settings.max_context_chars,
            semantic_cache_enabled=self.settings.semantic_cache_enabled,
            semantic_cache_threshold=self.settings.semantic_cache_threshold,
            embedding=self.embedding, log_path=self.settings.log_dir / "rag_requests.jsonl",
        )

    def create_agent(self):
        from .agent import ResearchAgent
        from .tools import create_tool_registry

        llm = DeepSeekClient(
            self.settings.deepseek_api_key, self.settings.llm_base_url, self.settings.llm_model,
            self.settings.llm_timeout_seconds, self.settings.llm_max_retries,
        )
        rag = self.create_rag_engine()
        tools = create_tool_registry(self, rag, llm, self.settings.web_search_enabled)
        return ResearchAgent(
            llm, tools, self.settings.index_dir / "agent_memory.sqlite3",
            self.settings.log_dir / "agent_tools.jsonl", self.settings.agent_max_iterations,
            self.settings.memory_max_chars, self.settings.memory_recent_turns,
            self.settings.tool_timeout_seconds, self.settings.max_parallel_tools,
        )

    def _unique_upload_path(self, name: str) -> Path:
        candidate = self.settings.upload_dir / Path(name).name
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        counter = 2
        while candidate.exists():
            candidate = self.settings.upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate


class _LazyReranker:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name, self.device, self._instance = model_name, device, None

    def rerank(self, query, results, top_k):
        if self._instance is None:
            self._instance = BGEReranker(self.model_name, self.device)
        return self._instance.rerank(query, results, top_k)
