from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import IndexedDocument


class DocumentManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[IndexedDocument]:
        data = self._read()
        return [IndexedDocument(**item) for item in data.values()]

    def get(self, document_id: str) -> IndexedDocument | None:
        item = self._read().get(document_id)
        return IndexedDocument(**item) if item else None

    def put(self, document: IndexedDocument) -> None:
        data = self._read()
        data[document.document_id] = asdict(document)
        self._write(data)

    def remove(self, document_id: str) -> None:
        data = self._read()
        if data.pop(document_id, None) is not None:
            self._write(data)

    def _read(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        try:
            content = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"文档清单损坏或不可读: {self.path}") from exc
        if not isinstance(content, dict):
            raise RuntimeError(f"文档清单格式无效: {self.path}")
        return content

    def _write(self, data: dict[str, dict[str, object]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
