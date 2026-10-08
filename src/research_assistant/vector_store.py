from __future__ import annotations

from pathlib import Path

from .embeddings import EmbeddingProvider
from .models import Chunk, SearchResult


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding: EmbeddingProvider,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("未安装 chromadb，请先执行 pip install -r requirements.txt") from exc
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": getattr(embedding, "model_name", "custom")},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        if not chunks:
            return 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            vectors = self.embedding.embed_documents(texts)
            self._collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=texts,
                metadatas=[chunk.chroma_metadata() for chunk in batch],
                embeddings=vectors,
            )
        return len(chunks)

    def query(self, text: str, top_k: int = 5, document_id: str | None = None) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.count == 0:
            return []
        where = {"document_id": document_id} if document_id else None
        result = self._collection.query(
            query_embeddings=[self.embedding.embed_query(text)],
            n_results=min(top_k, self.count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        search_results: list[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
            meta = metadata or {}
            known_keys = {"document_id", "source_path", "source_name", "page", "chunk_index"}
            chunk = Chunk(
                id=str(chunk_id),
                text=str(content or ""),
                document_id=str(meta.get("document_id", "")),
                source_path=str(meta.get("source_path", "")),
                source_name=str(meta.get("source_name", "")),
                page=int(meta.get("page", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata={key: value for key, value in meta.items() if key not in known_keys},
            )
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            search_results.append(SearchResult(chunk=chunk, score=score))
        return search_results

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def existing_chunk_ids(self, document_id: str) -> set[str]:
        result = self._collection.get(where={"document_id": document_id}, include=[])
        return {str(value) for value in result.get("ids", [])}

    def delete_stale_chunks(self, document_id: str, keep_ids: set[str]) -> None:
        stale = self.existing_chunk_ids(document_id) - keep_ids
        if stale:
            self._collection.delete(ids=sorted(stale))

    def all_chunks(self, document_id: str | None = None) -> list[Chunk]:
        """Read persisted corpus metadata for local lexical retrieval."""
        result = self._collection.get(
            where={"document_id": document_id} if document_id else None,
            include=["documents", "metadatas"],
        )
        chunks: list[Chunk] = []
        for chunk_id, content, metadata in zip(
            result.get("ids", []), result.get("documents", []), result.get("metadatas", []), strict=True
        ):
            meta = metadata or {}
            known = {"document_id", "source_path", "source_name", "page", "chunk_index"}
            chunks.append(Chunk(
                id=str(chunk_id), text=str(content or ""), document_id=str(meta.get("document_id", "")),
                source_path=str(meta.get("source_path", "")), source_name=str(meta.get("source_name", "")),
                page=int(meta.get("page", 1)), chunk_index=int(meta.get("chunk_index", 0)),
                metadata={key: value for key, value in meta.items() if key not in known},
            ))
        return chunks
