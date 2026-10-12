"""Run a small, isolated Agent decision check without touching the main index.

This evaluation deliberately uses three representative tools. It makes billable
DeepSeek calls, so it is opt-in and bounded; it stops after a model-call failure.
The local PDF and Chroma index live under data/papermind_eval_agent_* only.
"""
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
    data_dir = ROOT / "data" / "papermind_eval_agent_20261012"
    output_path = ROOT / "完整提交材料" / "评测数据" / "agent_tool_decisions.jsonl"
    if not output_path.parent.is_dir():
        raise FileNotFoundError("评测输出目录不存在；请先准备本地交付目录。")

    settings = load_settings()
    if not settings.llm_configured:
        print(json.dumps({"status": "not_configured", "agent_cases": 0}, ensure_ascii=True))
        return 2

    settings = replace(
        settings,
        data_dir=data_dir,
        upload_dir=data_dir / "uploads",
        index_dir=data_dir / "index",
        log_dir=data_dir / "logs",
        collection_name="agent_evaluation_20261012",
        retrieval_mode="hybrid",
        reranker_enabled=False,
        semantic_cache_enabled=False,
        agent_max_iterations=min(settings.agent_max_iterations, 4),
    )
    settings.ensure_directories()
    service = ResearchAssistantService(settings)

    metadata_rows = json.loads((ROOT / "data" / "papers" / "metadata.json").read_text(encoding="utf-8"))
    openvla = next(row for row in metadata_rows if row["arxiv_id"] == "2406.09246")
    source = (ROOT / openvla["local_pdf_path"]).resolve()
    doc_metadata = {key: openvla[key] for key in (
        "direction", "topic", "title", "title_zh", "first_author", "year", "arxiv_id", "doi", "venue_or_status"
    ) if openvla.get(key) not in (None, "")}
    indexed_doc, created = service.index_file(source, copy_to_uploads=False, metadata=doc_metadata)
    agent = service.create_agent()

    cases = [
        {
            "id": "AG01", "expected_tool": "calculator",
            "question": "请计算 123 乘以 456。",
        },
        {
            "id": "AG02", "expected_tool": "knowledge_search",
            "question": "从知识库检索 OpenVLA 训练数据集和规模，并附论文名及页码。",
        },
        {
            "id": "AG03", "expected_tool": "paper_metadata",
            "question": f"提取论文 {indexed_doc.source_name} 的标题、作者和年份。",
        },
    ]
    records = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = agent.run(case["question"], session_id=f"eval-{case['id']}-{uuid.uuid4().hex[:8]}")
        except Exception as exc:  # Do not persist exception text or request details.
            records.append({
                **case, "selected_expected_tool": False, "tool_names": [], "iterations": 0,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "usage": {}, "answer": "", "tool_trace": [], "error_type": type(exc).__name__,
            })
            break

        tool_names = [entry.get("tool") for entry in result.get("tool_trace", [])]
        record = {
            **case,
            "selected_expected_tool": case["expected_tool"] in tool_names,
            "tool_names": tool_names,
            "iterations": result.get("iterations", 0),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": result.get("usage", {}),
            "answer": result.get("answer", ""),
            "tool_trace": result.get("tool_trace", []),
            "cancelled": result.get("cancelled", False),
        }
        records.append(record)
        if "模型调用失败" in record["answer"]:
            break

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful = [record for record in records if "error_type" not in record]
    summary = {
        "status": "complete_with_cases" if successful else "api_failed_before_result",
        "indexed_papers": 1,
        "index_created_this_run": created,
        "chunks": indexed_doc.chunk_count,
        "agent_case_count": len(successful),
        "expected_tool_selected": sum(record["selected_expected_tool"] for record in successful),
        "tool_selection_accuracy": round(
            sum(record["selected_expected_tool"] for record in successful) / len(successful), 4
        ) if successful else None,
        "mean_iterations": round(sum(record["iterations"] for record in successful) / len(successful), 3)
        if successful else None,
        "mean_elapsed_ms": round(sum(record["elapsed_ms"] for record in successful) / len(successful), 1)
        if successful else None,
        "token_usage": {
            key: sum(int(record.get("usage", {}).get(key, 0) or 0) for record in successful)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "result_file": str(output_path),
        "isolated_data_dir": str(data_dir),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
