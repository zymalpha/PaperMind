from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index a document into the local knowledge base.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    service = ResearchAssistantService(load_settings())
    document, created = service.index_file(args.path)
    state = "indexed" if created else "already indexed"
    print(f"{state}: {document.source_name}, chunks={document.chunk_count}, id={document.document_id[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

