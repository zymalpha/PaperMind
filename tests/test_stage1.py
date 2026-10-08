from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.chunking import RecursiveTextChunker  # noqa: E402
from research_assistant.loaders import LoaderRegistry  # noqa: E402
from research_assistant.models import DocumentPage  # noqa: E402
from research_assistant.vector_store import ChromaVectorStore  # noqa: E402


class DeterministicEmbedding:
    model_name = "test-deterministic"

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts):
        values = []
        for text in texts:
            lowered = text.lower()
            vector = [float("rag" in lowered), float("agent" in lowered), float("retrieval" in lowered)]
            norm = np.linalg.norm(vector) or 1.0
            values.append((np.asarray(vector) / norm).tolist())
        return values

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_loader_and_chunk_metadata(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("RAG retrieval improves evidence grounding.\n\nAgent routing selects tools.", encoding="utf-8")
    document_id, pages = LoaderRegistry().load(source)
    chunks = RecursiveTextChunker(chunk_size=80, chunk_overlap=10).split_pages(pages)
    assert document_id
    assert chunks
    assert chunks[0].source_name == "paper.txt"
    assert chunks[0].page == 1


def test_chunk_size_and_overlap_are_bounded() -> None:
    text = "中文段落。" * 1000
    chunks = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)._split(text)
    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chroma_persistence_and_top_k(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "index", "test_collection", DeterministicEmbedding())
    chunks = RecursiveTextChunker(chunk_size=100, chunk_overlap=10).split_pages(
        [
            DocumentPage("RAG retrieval grounding", "paper.txt", "paper.txt", 1, "doc-1"),
            DocumentPage("Agent tool routing", "paper.txt", "paper.txt", 1, "doc-1"),
        ]
    )
    store.upsert(chunks)
    results = store.query("retrieval", top_k=1)
    assert len(results) == 1
    assert "retrieval" in results[0].chunk.text
    assert store.count == 2
