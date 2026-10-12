"""One bounded, isolated real DeepSeek SSE smoke test for the VLA corpus."""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


def main() -> int:
    output_path = ROOT / "完整提交材料" / "评测数据" / "deepseek_stream_answer.json"
    if not output_path.parent.is_dir():
        raise FileNotFoundError("评测输出目录不存在。")
    settings = load_settings()
    if not settings.llm_configured:
        raise RuntimeError("DeepSeek 未配置；本测试不会输出任何凭据。")

    data_dir = ROOT / "data" / f"papermind_eval_stream_{uuid.uuid4().hex[:8]}"
    settings = replace(
        settings,
        data_dir=data_dir,
        upload_dir=data_dir / "uploads",
        index_dir=data_dir / "index",
        log_dir=data_dir / "logs",
        collection_name=f"stream_eval_{uuid.uuid4().hex[:8]}",
        retrieval_mode="hybrid",
        reranker_enabled=False,
        semantic_cache_enabled=False,
        top_k=5,
        candidate_k=20,
    )
    settings.ensure_directories()
    service = ResearchAssistantService(settings)

    metadata = json.loads((ROOT / "data" / "papers" / "metadata.json").read_text(encoding="utf-8"))
    paper = next(row for row in metadata if row["arxiv_id"] == "2406.09246")
    source = (ROOT / paper["local_pdf_path"]).resolve()
    doc_metadata = {key: paper[key] for key in (
        "direction", "topic", "title", "title_zh", "first_author", "year", "arxiv_id", "doi", "venue_or_status"
    ) if paper.get(key) not in (None, "")}
    document, _created = service.index_file(source, copy_to_uploads=False, metadata=doc_metadata)
    rag = service.create_rag_engine()
    question = "根據 OpenVLA 論文，模型参数规模和训练数据规模分别是什么？请只依据论文回答并标注来源页码。"
    started = time.perf_counter()
    first_chunk_ms = None
    chunks: list[str] = []
    error_type = None
    citations = []
    try:
        stream, citation_rows = rag.stream_answer(question)
        citations = [{
            "source_name": item.source_name,
            "page": item.page,
            "chunk_id": item.chunk_id,
            "score": item.score,
            "excerpt": item.excerpt,
        } for item in citation_rows]
        for piece in stream:
            if first_chunk_ms is None:
                first_chunk_ms = round((time.perf_counter() - started) * 1000, 1)
            chunks.append(piece)
    except Exception as exc:
        error_type = type(exc).__name__

    result = {
        "question": question,
        "model": settings.llm_model,
        "status": "success" if chunks and error_type is None else "failed_or_empty",
        "answer": "".join(chunks),
        "chunk_count": len(chunks),
        "time_to_first_chunk_ms": first_chunk_ms,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "token_usage": None,
        "token_usage_note": "Current stream_chat wrapper does not expose final provider usage fields.",
        "error_type": error_type,
        "citations": citations,
        "indexed_paper": {
            "arxiv_id": paper["arxiv_id"], "source_name": document.source_name,
            "chunks": document.chunk_count,
        },
        "isolated_data_dir": str(data_dir),
        "api_key_logged": False,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"answer", "citations"}},
                     ensure_ascii=True, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
