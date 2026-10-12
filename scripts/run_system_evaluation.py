"""Run a reproducible, isolated RAG/Agent evaluation on the VLA paper subset.

The script never clears or changes the application's primary data/index directory.
The LLM portion is opt-in because it makes billable DeepSeek requests.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


DEFAULT_DATASET = ROOT / "evaluation" / "vla_qa_benchmark.jsonl"
DEFAULT_DATA_DIR = ROOT / "data" / "papermind_eval_vla_20261012"
DEFAULT_RESULTS = ROOT / "完整提交材料" / "评测数据"
TOP_K = 5
BASE_CANDIDATE_K = 20
OPTIMIZED_CANDIDATE_K = 40


def load_jsonl(path: Path) -> list[dict]:
    # Only LF delimits JSONL records. splitlines() also treats U+2028/U+2029 in
    # extracted PDF text as separators and can corrupt otherwise valid JSON.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(rows) < 50:
        raise ValueError(f"评测集至少需要 50 条，实际 {len(rows)} 条")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("评测集 id 重复")
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_evidence(rows: list[dict], metadata: dict[str, dict]) -> list[dict]:
    """Verify each cited page and needle against the original local PDF."""
    verified: list[dict] = []
    opened: dict[str, fitz.Document] = {}
    for row in rows:
        evidence_rows = []
        for evidence in row.get("evidence", []):
            paper = metadata.get(evidence["arxiv_id"])
            if not paper:
                raise ValueError(f"{row['id']}: 缺少论文元数据 {evidence['arxiv_id']}")
            pdf_path = (ROOT / paper["local_pdf_path"]).resolve()
            if not pdf_path.is_file():
                raise FileNotFoundError(f"评测来源 PDF 不存在: {pdf_path}")
            if str(pdf_path) not in opened:
                opened[str(pdf_path)] = fitz.open(pdf_path)
            pdf = opened[str(pdf_path)]
            page_number = int(evidence["page"])
            if page_number < 1 or page_number > len(pdf):
                raise ValueError(f"{row['id']}: 页码越界 {paper['title']} p.{page_number}")
            page_text = normalize(pdf[page_number - 1].get_text("text", sort=True))
            needle = normalize(evidence["needle"])
            position = page_text.casefold().find(needle.casefold())
            if position < 0:
                raise ValueError(f"{row['id']}: 来源页未找到证据词 {needle!r} ({paper['title']} p.{page_number})")
            start, end = max(0, position - 150), min(len(page_text), position + len(needle) + 180)
            evidence_rows.append({
                "arxiv_id": evidence["arxiv_id"],
                "title": paper.get("title", ""),
                "file": Path(paper["local_pdf_path"]).name,
                "page": page_number,
                "evidence_excerpt": page_text[start:end],
            })
        if not evidence_rows:
            raise ValueError(f"{row['id']}: 缺少来源页证据")
        verified.append({**row, "verified_evidence": evidence_rows})
    for pdf in opened.values():
        pdf.close()
    return verified


def metric_rows(results: list[dict], config: str) -> dict:
    selected = [item for item in results if item["config"] == config]
    total = max(1, len(selected))
    hit = mrr = recall = 0.0
    for item in selected:
        gold = set(item["gold_arxiv_ids"])
        retrieved = [record["arxiv_id"] for record in item["results"]]
        first_rank = next((rank for rank, paper_id in enumerate(retrieved, 1) if paper_id in gold), None)
        hit += float(first_rank is not None)
        mrr += (1.0 / first_rank) if first_rank is not None else 0.0
        recall += len(gold.intersection(retrieved)) / max(1, len(gold))
    latencies = [item["elapsed_ms"] for item in selected]
    return {
        "config": config,
        "n": len(selected),
        "hit_at_5": round(hit / total, 4),
        "mrr": round(mrr / total, 4),
        "recall_at_5": round(recall / total, 4),
        "mean_retrieval_ms": round(mean(latencies), 1) if latencies else None,
        "p50_retrieval_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
        "reranker_warning_count": sum(bool(item.get("warning")) for item in selected),
    }


def write_metrics(rows: list[dict], path: Path) -> list[dict]:
    configs = list(dict.fromkeys(item["config"] for item in rows))
    summary = [metric_rows(rows, config) for config in configs]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]) if summary else [])
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated retrieval benchmark; --llm enables billable DeepSeek tests.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--llm", action="store_true", help="Run bounded real DeepSeek RAG and Agent tests.")
    parser.add_argument("--resume", action="store_true", help="Resume missing config/question pairs from retrieval_results.jsonl.")
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    data_dir, results_dir = data_dir.resolve(), results_dir.resolve()
    app_data = (ROOT / "data").resolve()
    if data_dir == app_data:
        raise ValueError("评测数据目录不能覆盖 data/ 根目录")
    protected = [app_data / name for name in ("index", "uploads", "attachments", "research_library", "logs")]
    if any(data_dir == path or data_dir in path.parents or path in data_dir.parents for path in protected):
        raise ValueError("评测目录与主知识库、上传、附件或日志目录冲突；请使用独立的 data/papermind_eval_* 路径")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    qa_rows = load_jsonl(dataset_path)
    metadata_rows = json.loads((ROOT / "data" / "papers" / "metadata.json").read_text(encoding="utf-8"))
    metadata = {row["arxiv_id"]: row for row in metadata_rows}
    verified_rows = check_evidence(qa_rows, metadata)
    paper_ids = sorted({paper_id for row in qa_rows for paper_id in row["gold_arxiv_ids"]})
    if len(paper_ids) != 10:
        raise ValueError(f"评測 corpus 应恰好覆盖选定的 10 篇论文，实际 {len(paper_ids)}")
    checked_dataset = results_dir / "具身智能问答评测集_含页码证据.jsonl"
    with checked_dataset.open("w", encoding="utf-8") as file:
        for row in verified_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    settings = load_settings()
    settings = replace(
        settings,
        data_dir=data_dir,
        upload_dir=data_dir / "uploads",
        index_dir=data_dir / "index",
        log_dir=data_dir / "logs",
        collection_name="vla_evaluation_20261012",
        chunk_size=700,
        chunk_overlap=120,
        top_k=TOP_K,
        candidate_k=BASE_CANDIDATE_K,
        retrieval_mode="hybrid_rerank",
        reranker_enabled=True,
        semantic_cache_enabled=False,
        llm_max_tokens=min(settings.llm_max_tokens, 1400),
    )
    settings.ensure_directories()
    service = ResearchAssistantService(settings)
    indexed_papers: list[dict] = []
    indexing_started = time.perf_counter()
    for paper_id in paper_ids:
        paper = metadata[paper_id]
        source = (ROOT / paper["local_pdf_path"]).resolve()
        doc_metadata = {key: paper[key] for key in (
            "direction", "topic", "title", "title_zh", "first_author", "year", "arxiv_id", "doi", "venue_or_status"
        ) if paper.get(key) not in (None, "")}
        doc, created = service.index_file(source, copy_to_uploads=False, metadata=doc_metadata)
        indexed_papers.append({
            "arxiv_id": paper_id,
            "title": paper["title"],
            "source_name": doc.source_name,
            "document_id_sha256": doc.document_id,
            "created_in_this_run": created,
            "chunks": doc.chunk_count,
            "index_timings": service.last_index_timings,
        })
        print(json.dumps({"event": "paper_indexed", "arxiv_id": paper_id, "file": doc.source_name,
                          "chunks": doc.chunk_count, "created": created, "timings": service.last_index_timings},
                         ensure_ascii=False), flush=True)
    indexing_elapsed_ms = round((time.perf_counter() - indexing_started) * 1000, 1)

    # Warm the local reranker once, and report cold model load separately from request latency.
    sample_query = "OpenVLA 970k robot episodes dataset"
    reranker_warmup_started = time.perf_counter()
    warmup_results = service.retriever.search(sample_query, top_k=TOP_K, mode="hybrid_rerank")
    reranker_load_elapsed_ms = round((time.perf_counter() - reranker_warmup_started) * 1000, 1)
    reranker_loaded = getattr(service._reranker, "_instance", None) is not None

    indexed_by_arxiv = {item["arxiv_id"]: item for item in indexed_papers}
    retrieval_records: list[dict] = []
    configs = [("vector_k5", "vector", BASE_CANDIDATE_K),
               ("hybrid_candidate20", "hybrid", BASE_CANDIDATE_K),
               ("hybrid_rerank_candidate20", "hybrid_rerank", BASE_CANDIDATE_K),
               ("optimization_hybrid_rerank_candidate40", "hybrid_rerank", OPTIMIZED_CANDIDATE_K)]
    optimized_question_ids = set()
    for question_type in sorted({row["type"] for row in verified_rows}):
        optimized_question_ids.update(row["id"] for row in [q for q in verified_rows if q["type"] == question_type][:5])
    detail_path = results_dir / "retrieval_results.jsonl"
    retrieval_records: list[dict] = []
    if args.resume and detail_path.is_file():
        valid_configs = {name for name, _, _ in configs}
        valid_ids = {row["id"] for row in verified_rows}
        for line in detail_path.read_text(encoding="utf-8").split("\n"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("config") in valid_configs and record.get("id") in valid_ids:
                retrieval_records.append(record)
    completed_pairs = {(item["config"], item["id"]) for item in retrieval_records}
    output_mode = "a" if args.resume and detail_path.is_file() else "w"
    with detail_path.open(output_mode, encoding="utf-8") as detail_file:
        for config_name, mode, candidate_k in configs:
            service.retriever.candidate_k = candidate_k
            config_questions = verified_rows
            if config_name in {"hybrid_rerank_candidate20", "optimization_hybrid_rerank_candidate40"}:
                config_questions = [row for row in verified_rows if row["id"] in optimized_question_ids]
            pending = [row for row in config_questions if (config_name, row["id"]) not in completed_pairs]
            if not pending:
                print(json.dumps({"event": "config_skipped_complete", "config": config_name,
                                  "questions": len(config_questions)}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({"event": "config_started", "config": config_name,
                              "questions": len(pending), "total": len(config_questions), "mode": mode, "candidate_k": candidate_k},
                             ensure_ascii=False), flush=True)
            for row_index, row in enumerate(pending, 1):
                started = time.perf_counter()
                service.retriever.last_warning = None
                found = service.retriever.search(row["question"], top_k=TOP_K, mode=mode)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                output = {
                    "id": row["id"], "type": row["type"], "question": row["question"],
                    "gold_arxiv_ids": row["gold_arxiv_ids"], "config": config_name,
                    "mode": mode, "candidate_k": candidate_k, "elapsed_ms": elapsed_ms,
                    "warning": service.retriever.last_warning,
                    "results": [{
                        "rank": rank, "arxiv_id": result.chunk.metadata.get("arxiv_id"),
                        "source_name": result.chunk.source_name, "page": result.chunk.page,
                        "chunk_id": result.chunk.id, "score": round(result.score, 6),
                        "excerpt": result.chunk.text[:220],
                    } for rank, result in enumerate(found[:TOP_K], 1)],
                }
                retrieval_records.append(output)
                completed_pairs.add((config_name, row["id"]))
                detail_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                detail_file.flush()
                if row_index % 10 == 0 or row_index == len(verified_rows):
                    print(json.dumps({"event": "query_progress", "config": config_name,
                                      "completed": row_index, "total": len(verified_rows),
                                      "last_id": row["id"], "elapsed_ms": elapsed_ms,
                                      "reranker_warning": output["warning"]}, ensure_ascii=False), flush=True)

    metrics = write_metrics(retrieval_records, results_dir / "retrieval_metrics.csv")
    matched_records = [item for item in retrieval_records
                       if item["id"] in optimized_question_ids
                       and item["config"] in {"hybrid_rerank_candidate20", "optimization_hybrid_rerank_candidate40"}]
    comparison = [metric_rows(matched_records, config) for config in (
        "hybrid_rerank_candidate20", "optimization_hybrid_rerank_candidate40")]
    with (results_dir / "candidate_k_optimization_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]) if comparison else [])
        writer.writeheader()
        writer.writerows(comparison)
    type_metrics = []
    for config_name, _, _ in configs:
        for question_type in sorted({row["type"] for row in verified_rows}):
            subset = [item for item in retrieval_records if item["config"] == config_name and item["type"] == question_type]
            type_metrics.append({"config": config_name, "type": question_type,
                                 "n": len(subset), "hit_at_5": metric_rows(subset, config_name)["hit_at_5"],
                                 "mrr": metric_rows(subset, config_name)["mrr"],
                                 "recall_at_5": metric_rows(subset, config_name)["recall_at_5"],
                                 "mean_retrieval_ms": metric_rows(subset, config_name)["mean_retrieval_ms"]})
    with (results_dir / "retrieval_metrics_by_type.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(type_metrics[0]) if type_metrics else [])
        writer.writeheader()
        writer.writerows(type_metrics)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "direction": "具身智能：通用机器人策略与视觉-语言-动作模型",
        "dataset_size": len(verified_rows),
        "question_type_counts": {t: sum(row["type"] == t for row in verified_rows) for t in sorted({r["type"] for r in verified_rows})},
        "paper_count": len(indexed_papers), "papers": indexed_papers,
        "chunk_count": sum(item["chunks"] for item in indexed_papers),
        "index_elapsed_ms": indexing_elapsed_ms,
        "reranker_loaded": reranker_loaded,
        "reranker_model": settings.reranker_model,
        "reranker_device": settings.reranker_device,
        "reranker_warmup_elapsed_ms": reranker_load_elapsed_ms,
        "reranker_warmup_results": [{"source_name": r.chunk.source_name, "page": r.chunk.page,
                                     "chunk_id": r.chunk.id, "score": round(r.score, 6)} for r in warmup_results],
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": TOP_K,
        "evaluated_configs": [{"name": n, "mode": m, "candidate_k": k} for n, m, k in configs],
        "reranker_sample_method": "按四类问题分层抽取，每类取前 5 条；vector 与 hybrid 覆盖全 55 条。",
        "metrics": metrics,
        "optimization_matched_sample": comparison,
        "data_dir_isolated": str(data_dir),
        "deepseek_configured": settings.llm_configured,
    }

    if args.llm:
        if not settings.llm_configured:
            manifest["llm_evaluation_error"] = {"error_type": "NotConfigured",
                                                 "message": "未配置 DeepSeek；LLM 评测中止。密钥只从本地环境读取，不打印。"}
        else:
            try:
                manifest["llm_evaluation"] = run_llm_evaluation(service, settings, verified_rows, results_dir)
            except Exception as exc:
                # Retrieval evidence and partial LLM results remain reportable even
                # when the billable API portion fails unexpectedly.
                manifest["llm_evaluation_error"] = {"error_type": type(exc).__name__,
                                                     "message": "LLM 评测阶段发生异常；请查看本机隔离日志，不记录密钥或请求头。"}

    (results_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "evaluation_complete", "dataset": len(verified_rows), "chunks": manifest["chunk_count"],
                      "index_elapsed_ms": indexing_elapsed_ms, "reranker_loaded": reranker_loaded,
                      "metrics": metrics, "llm": manifest.get("llm_evaluation", {"run": False})},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


def run_llm_evaluation(service, settings, qa_rows: list[dict], results_dir: Path) -> dict:
    """Bounded real API validation; prompts are preselected and results are persisted without secrets."""
    from research_assistant.agent import ResearchAgent

    llm_started = time.perf_counter()
    rag = service.create_rag_engine()
    samples = [
        {"id": "RAG01", "question": "OpenVLA 使用多少机器人示范回合、模型有多少参数？请只按论文回答并引用页码。"},
        {"id": "RAG02", "question": "比较 Diffusion Policy 与 Streaming Flow Policy 的动作生成思路和推理延迟来源，分别引用论文页码。"},
        {"id": "RAG03", "question": "FAST 为什么对连续动作使用 DCT 与 BPE？请基于证据说明。"},
        {"id": "RAG04", "question": "V-JEPA 2 如何把视频预训练表征用于机器人规划？回答数据量、动作条件模型规模并标注引用。"},
        {"id": "RAG05", "question": "RDT-1B 针对双臂操作的困难是什么，其数据与统一动作空间设计如何回应？请提供来源。"},
    ]
    rag_records = []
    ratings_template = []
    for sample in samples:
        start = time.perf_counter()
        response = rag.answer(sample["question"])
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        citations = [{"source_name": item.source_name, "page": item.page, "chunk_id": item.chunk_id,
                      "score": item.score, "excerpt": item.excerpt} for item in response.citations]
        rag_records.append({"id": sample["id"], "question": sample["question"], "answer": response.answer,
                            "model": response.model, "usage": response.usage, "elapsed_ms": elapsed_ms,
                            "warning": response.warning, "citations": citations})
        ratings_template.append({"id": sample["id"], "correctness_1_to_5": "", "completeness_1_to_5": "",
                                 "citation_accuracy_1_to_5": "", "reviewer_note": "待人工复核"})
        if response.warning and response.warning.startswith("generation_failed:"):
            break

    # Verify the actual streaming path once, apart from the five answer() calls above.
    stream_question = "OpenVLA 的主要结构和训练数据是什么？只根据本地论文回答并给出引用。"
    stream_started = time.perf_counter()
    stream_citations = []
    streamed_parts = []
    stream_error = None
    # Streaming is a separate acceptance path. A failed non-streaming answer
    # must not suppress it; retain only a safe error class in the artifact.
    try:
        iterator, stream_citations = rag.stream_answer(stream_question)
        streamed_parts = list(iterator)
    except Exception as exc:
        stream_error = type(exc).__name__
    stream_elapsed_ms = round((time.perf_counter() - stream_started) * 1000, 1)
    stream_record = {
        "id": "RAG_STREAM_01", "question": stream_question,
        "answer": "".join(streamed_parts), "stream_chunk_count": len(streamed_parts),
        "elapsed_ms": stream_elapsed_ms, "error": stream_error,
        "citations": [{"source_name": item.source_name, "page": item.page, "chunk_id": item.chunk_id,
                       "score": item.score, "excerpt": item.excerpt} for item in stream_citations],
    }

    agent = service.create_agent()
    document_by_arxiv = {str(doc.metadata.get("arxiv_id")): doc for doc in service.list_documents()}
    openvla = document_by_arxiv.get("2406.09246")
    pi0 = document_by_arxiv.get("2410.24164")
    if not openvla or not pi0:
        raise RuntimeError("评测 Agent 缺少 OpenVLA 或 π₀ 文档")
    agent_cases = [
        {"id": "A01", "expected_tool": "knowledge_search", "question": "从知识库检索 OpenVLA 使用的数据集规模，并返回来源。"},
        {"id": "A02", "expected_tool": "paper_metadata", "question": f"提取论文 {openvla.source_name} 的标题、作者和年份。"},
        {"id": "A03", "expected_tool": "compare_papers", "question": f"对比 {openvla.source_name} 与 {pi0.source_name} 的方法。"},
        {"id": "A04", "expected_tool": "extract_keywords", "question": "为这句话提取关键词：vision-language-action model, robot demonstrations, action chunking, generalization。"},
        {"id": "A05", "expected_tool": "summarize_paper", "question": f"总结 {openvla.source_name} 的核心方法、实验结果和局限。"},
        {"id": "A06", "expected_tool": "current_time", "question": "现在的本地日期和时间是什么？"},
        {"id": "A07", "expected_tool": "calculator", "question": "请计算 123 乘以 456。"},
        {"id": "A08", "expected_tool": "web_search", "question": "请使用联网搜索查询 VLA 最近的公开论文；若系统设置禁用了联网搜索，请明确报告不可用。"},
    ]
    agent_records = []
    # RAG generation and Agent tool selection are independent evaluation paths.
    # A failed RAG completion must not silently suppress Agent measurements.
    for case in agent_cases:
        start = time.perf_counter()
        try:
            result = agent.run(case["question"], session_id=f"eval-{case['id']}")
        except Exception as exc:
            agent_records.append({**case, "selected_expected_tool": False, "tool_names": [],
                                  "iterations": 0, "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                                  "usage": {}, "answer": "", "tool_trace": [], "cancelled": False,
                                  "error_type": type(exc).__name__})
            break
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        tool_names = [entry.get("tool") for entry in result["tool_trace"]]
        agent_records.append({
            **case,
            "selected_expected_tool": case["expected_tool"] in tool_names,
            "tool_names": tool_names,
            "iterations": result.get("iterations", 0),
            "elapsed_ms": elapsed_ms,
            "usage": result.get("usage", {}),
            "answer": result.get("answer", ""),
            "tool_trace": result.get("tool_trace", []),
            "cancelled": result.get("cancelled", False),
        })
        if not result.get("tool_trace") and "模型调用失败" in result.get("answer", ""):
            break

    def usage_sum(records: list[dict]) -> dict:
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        return {key: sum(int(record.get("usage", {}).get(key, 0) or 0) for record in records) for key in keys}

    (results_dir / "deepseek_rag_answers.json").write_text(json.dumps({"one_shot": rag_records, "stream": stream_record}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (results_dir / "agent_tool_decisions.jsonl").open("w", encoding="utf-8") as file:
        for record in agent_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    ratings_path = results_dir / "manual_answer_ratings.csv"
    preserved_ratings = {}
    if ratings_path.is_file():
        with ratings_path.open("r", encoding="utf-8-sig", newline="") as previous_file:
            for previous in csv.DictReader(previous_file):
                dimensions = ("correctness_1_to_5", "completeness_1_to_5", "citation_accuracy_1_to_5")
                if all(previous.get(key, "").strip() for key in dimensions):
                    preserved_ratings[previous.get("id", "")] = previous
    for rating in ratings_template:
        previous = preserved_ratings.get(rating["id"])
        if previous:
            rating.update({key: previous.get(key, "") for key in (
                "correctness_1_to_5", "completeness_1_to_5", "citation_accuracy_1_to_5", "reviewer_note"
            )})
    with ratings_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ratings_template[0]))
        writer.writeheader()
        writer.writerows(ratings_template)

    total_usage = usage_sum(rag_records)  # Streaming usage is not exposed by the current stream client.
    total_agent_usage = usage_sum(agent_records)
    correct_tools = sum(record["selected_expected_tool"] for record in agent_records)
    decision_elapsed = [record["elapsed_ms"] for record in agent_records]
    return {
        "model": settings.llm_model,
        "rag_one_shot_count": len(rag_records),
        "rag_stream_count": 1,
        "rag_stream_chunk_count": len(streamed_parts),
        "rag_stream_elapsed_ms": stream_elapsed_ms,
        "rag_elapsed_ms_mean": round(mean(record["elapsed_ms"] for record in rag_records), 1) if rag_records else None,
        "rag_token_usage": total_usage,
        "agent_case_count": len(agent_records),
        "agent_expected_tool_selected": correct_tools,
        "agent_tool_accuracy": round(correct_tools / len(agent_records), 4) if agent_records else None,
        "agent_mean_tool_rounds": round(mean(record["iterations"] for record in agent_records), 2) if agent_records else None,
        "agent_elapsed_ms_mean": round(mean(decision_elapsed), 1) if decision_elapsed else None,
        "agent_token_usage": total_agent_usage,
        "elapsed_ms_total": round((time.perf_counter() - llm_started) * 1000, 1),
        "api_key_printed_or_saved": False,
        "note": "DeepSeek Token 统计来自非流式 completion usage；底层 SDK 当前不回传流式 usage。web_search 配置关闭时仅验证禁用回执，不执行外网搜索。人工质量评分模板需完成者填写。",
    }


if __name__ == "__main__":
    raise SystemExit(main())
