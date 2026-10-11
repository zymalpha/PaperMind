from __future__ import annotations

import argparse
import json
import sqlite3
import time
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the downloaded research paper dataset in small batches.")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "papers")
    parser.add_argument("--limit", type=int, default=0, help="Index at most N PDFs; 0 means all.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N PDFs (for resumable batches).")
    parser.add_argument("--isolated", action="store_true", help="Use data/papers_test_index instead of the existing index.")
    parser.add_argument("--data-dir", type=Path, help="Use an explicit independent data directory (recommended for the formal library).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-path", type=Path, default=ROOT / "data" / "logs" / "index_research_papers.jsonl")
    parser.add_argument("--persistence-wait-seconds", type=float, default=15,
                        help="Wait at most this long for Chroma's async queue; use 0 to skip waiting.")
    parser.add_argument("--offline", action="store_true", help="Disable model downloads and fail fast if the local embedding model is absent.")
    parser.add_argument("--download-timeout-seconds", type=int, default=30,
                        help="Hugging Face metadata/download timeout when online (minimum 5 seconds).")
    args = parser.parse_args()

    import os
    timeout = str(max(5, args.download_timeout_seconds))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", timeout)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", timeout)
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run_started = time.perf_counter()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(event: str, **fields: object) -> None:
        record = {"timestamp": time.time(), "event": event, **fields}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        with args.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    settings_started = time.perf_counter()
    settings = load_settings()
    log("settings_loaded", elapsed_ms=round((time.perf_counter() - settings_started) * 1000, 1), model=settings.embedding_model)
    if args.isolated:
        isolated_root = ROOT / "data" / "papers_test_index"
        settings = replace(
            settings,
            data_dir=isolated_root,
            upload_dir=isolated_root / "uploads",
            index_dir=isolated_root / "index",
            log_dir=isolated_root / "logs",
            collection_name="research_papers_test",
        )
        settings.ensure_directories()
    elif args.data_dir:
        target_root = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
        settings = replace(settings, data_dir=target_root.resolve(), upload_dir=target_root.resolve() / "uploads",
                           index_dir=target_root.resolve() / "index", log_dir=target_root.resolve() / "logs",
                           collection_name="research_papers")
        settings.ensure_directories()

    files = sorted(args.source.resolve().rglob("*.pdf"))
    if args.offset:
        files = files[args.offset:]
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"No PDFs found under {args.source}")
        return 1
    log("files_selected", source=str(args.source.resolve()), available=len(sorted(args.source.resolve().rglob("*.pdf"))),
        offset=args.offset, limit=args.limit, selected=len(files))

    init_started = time.perf_counter()
    service = ResearchAssistantService(settings)
    log("service_initialized", elapsed_ms=round((time.perf_counter() - init_started) * 1000, 1))
    metadata_rows = json.loads((args.source / "metadata.json").read_text(encoding="utf-8")) if (args.source / "metadata.json").exists() else []
    by_path = {str(Path(row.get("local_pdf_path", "")).resolve()): row for row in metadata_rows if row.get("local_pdf_path")}
    created = skipped = failed = 0
    for index, path in enumerate(files, 1):
        item_started = time.perf_counter()
        try:
            row = by_path.get(str(path.resolve()), {})
            doc_meta = {key: row[key] for key in ("direction", "topic", "title", "title_zh", "authors", "first_author", "year", "arxiv_id", "doi", "venue_or_status") if row.get(key) not in (None, "")}
            if isinstance(doc_meta.get("authors"), list):
                doc_meta["authors"] = "; ".join(doc_meta["authors"])
            document, was_created = service.index_file(path, copy_to_uploads=False, force=args.force, metadata=doc_meta)
            created += int(was_created)
            skipped += int(not was_created)
            state = "indexed" if was_created else "already indexed"
            timings = service.last_index_timings
            log("paper_finished", index=index, total=len(files), source_name=document.source_name,
                state=state, chunks=document.chunk_count,
                elapsed_ms=round((time.perf_counter() - item_started) * 1000, 1), timings=timings)
        except Exception as exc:
            failed += 1
            log("paper_failed", index=index, total=len(files), source_name=path.name,
                elapsed_ms=round((time.perf_counter() - item_started) * 1000, 1), error=f"{type(exc).__name__}: {exc}")
    log("batch_finished", created=created, skipped=skipped, failed=failed,
        elapsed_ms=round((time.perf_counter() - run_started) * 1000, 1))
    # Chroma 1.x persists the HNSW graph asynchronously. Wait for its queue to
    # drain before process exit; otherwise a large batch can reopen with a
    # stale graph even though the SQLite embedding rows are present.
    chroma_db = settings.index_dir / "chroma" / "chroma.sqlite3"
    if failed == 0 and args.persistence_wait_seconds > 0 and chroma_db.exists():
        deadline = time.time() + args.persistence_wait_seconds
        while time.time() < deadline:
            try:
                with sqlite3.connect(chroma_db) as conn:
                    pending = int(conn.execute("select count(*) from embeddings_queue").fetchone()[0])
                log("chroma_persistence", pending=pending)
                if pending == 0:
                    break
            except sqlite3.Error:
                pass
            time.sleep(1)
    log("run_finished", elapsed_ms=round((time.perf_counter() - run_started) * 1000, 1))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
