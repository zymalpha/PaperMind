from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .chunking import RecursiveTextChunker
from .config import Settings
from .document_store import DocumentManifest
from .embeddings import EmbeddingProvider, create_embedding_provider
from .llm import DeepSeekClient
from .loaders import LoaderRegistry
from .models import IndexedDocument
from .rag import RAGEngine
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
        self.chunker = RecursiveTextChunker(settings.chunk_size, settings.chunk_overlap)

    def index_file(self, source_path: str | Path, copy_to_uploads: bool = True) -> tuple[IndexedDocument, bool]:
        source = Path(source_path).resolve()
        document_id, pages = self.loader.load(source)
        existing = self.manifest.get(document_id)
        if existing and self.vector_store.existing_chunk_ids(document_id):
            return existing, False

        target = source
        if copy_to_uploads and source.parent != self.settings.upload_dir.resolve():
            target = self._unique_upload_path(source.name)
            shutil.copy2(source, target)
            document_id, pages = self.loader.load(target)

        chunks = self.chunker.split_pages(pages)
        self.vector_store.upsert(chunks)
        document = IndexedDocument(
            document_id=document_id,
            source_name=target.name,
            source_path=str(target),
            chunk_count=len(chunks),
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.manifest.put(document)
        return document, True

    def list_documents(self) -> list[IndexedDocument]:
        return sorted(self.manifest.list(), key=lambda item: item.indexed_at, reverse=True)

    def create_rag_engine(self) -> RAGEngine:
        llm = DeepSeekClient(
            self.settings.deepseek_api_key,
            self.settings.llm_base_url,
            self.settings.llm_model,
            self.settings.llm_timeout_seconds,
            self.settings.llm_max_retries,
        )
        return RAGEngine(
            self.vector_store,
            llm,
            self.settings.top_k,
            self.settings.min_relevance,
            self.settings.llm_temperature,
            self.settings.llm_max_tokens,
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
