from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage


class RecursiveTextChunker:
    """Page-aware recursive character splitter with deterministic chunk IDs."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ("\n\n", "\n", "。", "！", "？", "; ", " ")

    def split_pages(self, pages: list[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in pages:
            for text in self._split(page.text):
                clean = re.sub(r"[ \t]+", " ", text).strip()
                if not clean:
                    continue
                raw_id = f"{page.document_id}:{page.page}:{chunk_index}:{clean}"
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
                    )
                )
                chunk_index += 1
        return chunks

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
