from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentPage:
    text: str
    source_path: str
    source_name: str
    page: int
    document_id: str


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    document_id: str
    source_path: str
    source_name: str
    page: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, str | int | float | bool]:
        base: dict[str, str | int | float | bool] = {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }
        base.update(
            {key: value for key, value in self.metadata.items() if isinstance(value, (str, int, float, bool))}
        )
        return base


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class Citation:
    index: int
    source_name: str
    page: int
    chunk_id: str
    score: float
    excerpt: str


@dataclass(slots=True)
class RAGResponse:
    answer: str
    citations: list[Citation]
    model: str
    elapsed_ms: int
    usage: dict[str, int] = field(default_factory=dict)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexedDocument:
    document_id: str
    source_name: str
    source_path: str
    chunk_count: int
    indexed_at: str

    @property
    def path(self) -> Path:
        return Path(self.source_path)

