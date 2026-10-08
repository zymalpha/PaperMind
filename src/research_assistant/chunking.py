from __future__ import annotations

import hashlib
import re
from typing import Protocol

from .models import Chunk, DocumentPage


class TextChunker(Protocol):
    strategy: str

    def split_pages(self, pages: list[DocumentPage]) -> list[Chunk]: ...


class BaseTextChunker:
    strategy = "base"

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: list[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in pages:
            for text in self._split(page.text):
                clean = re.sub(r"[ \t]+", " ", text).strip()
                if not clean:
                    continue
                raw_id = f"{page.document_id}:{self.strategy}:{page.page}:{chunk_index}:{clean}"
                chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:32]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=clean,
                        document_id=page.document_id,
                        source_path=page.source_path,
                        source_name=page.source_name,
                        page=page.page,
                        chunk_index=chunk_index,
                        metadata={"chunk_strategy": self.strategy},
                    )
                )
                chunk_index += 1
        return chunks

    def _split(self, text: str) -> list[str]:
        raise NotImplementedError


class FixedSizeChunker(BaseTextChunker):
    """Overlapping fixed-character windows."""

    strategy = "fixed"

    def _split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        return [text[start : start + self.chunk_size] for start in range(0, len(text), step)]


class RecursiveTextChunker(BaseTextChunker):
    """Page-aware recursive character splitter with deterministic chunk IDs."""

    strategy = "recursive"

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120) -> None:
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ("\n\n", "\n", "。", "！", "？", "; ", " ")

    def _split(self, text: str) -> list[str]:
        text = text.strip()
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                minimum_boundary = start + max(1, self.chunk_size // 2)
                # Prefer paragraph/sentence boundaries while guaranteeing a hard size cap.
                candidates = [
                    (text.rfind(separator, minimum_boundary, end), separator)
                    for separator in self.separators
                ]
                candidates = [(position, sep) for position, sep in candidates if position >= minimum_boundary]
                if candidates:
                    position, separator = max(candidates, key=lambda item: item[0])
                    end = position + len(separator)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            next_start = end - self.chunk_overlap
            start = max(start + 1, next_start)
        return chunks


class SemanticChunker(BaseTextChunker):
    """Sentence/paragraph boundary splitter tailored to natural-language papers."""

    strategy = "semantic"
    _boundary = re.compile(r"(?<=[。！？.!?])\s+|\n{2,}")

    def _split(self, text: str) -> list[str]:
        segments = [segment.strip() for segment in self._boundary.split(text.strip()) if segment.strip()]
        output: list[str] = []
        current = ""
        for segment in segments:
            if len(segment) > self.chunk_size:
                if current:
                    output.append(current)
                    current = ""
                output.extend(RecursiveTextChunker(self.chunk_size, self.chunk_overlap)._split(segment))
                continue
            candidate = f"{current} {segment}".strip() if current else segment
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                output.append(current)
            prefix = current[-self.chunk_overlap :] if current and self.chunk_overlap else ""
            current = f"{prefix} {segment}".strip() if prefix else segment
        if current:
            output.append(current)
        return output


def create_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> TextChunker:
    chunkers: dict[str, type[BaseTextChunker]] = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveTextChunker,
        "semantic": SemanticChunker,
    }
    try:
        return chunkers[strategy.lower()](chunk_size, chunk_overlap)
    except KeyError as exc:
        raise ValueError(f"未知分块策略 {strategy}；可选: {', '.join(chunkers)}") from exc
