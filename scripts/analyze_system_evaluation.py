"""Analyze saved retrieval/e2e evaluation artifacts and generate reproducible tables/plots."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "完整提交材料" / "评测数据"
DEFAULT_RUNTIME = ROOT / "data" / "papermind_eval_vla_20261012_run2"
CONFIGS = ["vector_k5", "hybrid_candidate20", "hybrid_rerank_candidate20",
           "optimization_hybrid_rerank_candidate40"]
LABELS = {
    "vector_k5": "Vector top-5",
    "hybrid_candidate20": "Hybrid RRF k=20",
    "hybrid_rerank_candidate20": "Rerank k=20",
    "optimization_hybrid_rerank_candidate40": "Rerank k=40",
}
QUESTION_TYPES = ["事实性", "对比性", "归纳性", "推理性"]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "hit_at_5": None, "mrr": None, "recall_at_5": None,
                "mean_retrieval_ms": None, "p50_retrieval_ms": None, "reranker_warning_count": 0}
    hits, reciprocal, recalls = [], [], []
    durations = []
    warning_count = 0
    for item in records:
        gold = set(item["gold_arxiv_ids"])
        retrieved = [row.get("arxiv_id") for row in item["results"]]
        first = next((rank for rank, arxiv_id in enumerate(retrieved, 1) if arxiv_id in gold), None)
        hits.append(first is not None)
        reciprocal.append(1.0 / first if first else 0.0)
        recalls.append(len(gold.intersection(retrieved)) / max(1, len(gold)))
        durations.append(float(item["elapsed_ms"]))
        warning_count += int(bool(item.get("warning")))
    ordered = sorted(durations)
    return {
        "n": len(records),
        "hit_at_5": round(sum(hits) / len(records), 4),
        "mrr": round(statistics.mean(reciprocal), 4),
        "recall_at_5": round(statistics.mean(recalls), 4),
        "mean_retrieval_ms": round(statistics.mean(durations), 1),
        "p50_retrieval_ms": round(ordered[len(ordered) // 2], 1),
        "reranker_warning_count": warning_count,
    }


def load_jsonl(path: Path) -> list[dict]:
    # str.splitlines() also splits on U+2028/U+2029. Those are valid characters
    # in JSON strings and occur in extracted paper text, so only split on the
    # JSONL record delimiter (LF) to avoid corrupting otherwise valid records.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_charts(matched: dict[str, list[dict]], rows_by_config: dict[str, list[dict]], timing_rows: list[dict], output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = [name for name in CONFIGS if matched.get(name)]
    summary = {name: metrics(matched[name]) for name in names}
    labels = [LABELS[name] for name in names]
    x = list(range(len(names)))
    width = 0.23
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.8, 1]})
    for offset, metric, title, color in [
        (-width, "hit_at_5", "Hit@5", "#3568A8"),
        (0, "mrr", "MRR", "#50A387"),
        (width, "recall_at_5", "Recall@5", "#D69A3B"),
    ]:
        axes[0].bar([index + offset for index in x], [summary[name][metric] for name in names],
                    width, label=title, color=color)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("分数（0–1）")
    axes[0].set_title("同一 20 条分层样本上的检索质量")
    axes[0].legend(frameon=False, ncols=3, loc="lower right")
    axes[0].grid(axis="y", alpha=0.22)
    latencies = [summary[name]["mean_retrieval_ms"] / 1000 for name in names]
    bars = axes[1].bar(range(len(names)), latencies, color=["#3568A8", "#50A387", "#6B7DAA", "#C46A54"][:len(names)])
    axes[1].set_xticks(range(len(names)), labels, rotation=15, ha="right")
    axes[1].set_ylabel("平均检索时延（秒）")
    axes[1].set_title("CPU 检索/重排时延")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, latencies):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    figure.suptitle("PaperMind VLA 检索参数对照（同源论文、同一分层问答子集）", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "检索质量与时延对比.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    parse = [row["parse_ms"] / 1000 for row in timing_rows]
    index = [row["index_ms"] / 1000 for row in timing_rows]
    x = list(range(len(timing_rows)))
    axis.bar(x, parse, label="PDF 解析", color="#3568A8")
    axis.bar(x, index, bottom=parse, label="Embedding + Chroma 写入", color="#50A387")
    axis.set_xticks(x, [row["source_name"] for row in timing_rows], rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("秒 / 文档")
    axis.set_title("10 篇论文入库阶段耗时（分块耗时低于 0.03s，图中未单列）")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(output / "论文索引分阶段耗时.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    results = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    runtime = args.runtime_dir if args.runtime_dir.is_absolute() else ROOT / args.runtime_dir
    results, runtime = results.resolve(), runtime.resolve()
    retrieval = load_jsonl(results / "retrieval_results.jsonl")
    qa_rows = load_jsonl(ROOT / "evaluation" / "vla_qa_benchmark.jsonl")
    runtime_stdout = runtime / "logs" / "evaluation.stdout.log"
    index_events = [json.loads(line) for line in runtime_stdout.read_text(encoding="utf-8").split("\n")
                    if line.strip().startswith("{")]
    indexed = [item for item in index_events if item.get("event") == "paper_indexed"]
    if len(indexed) != 10:
        raise RuntimeError(f"期望从阶段日志读到 10 篇已索引论文，实际 {len(indexed)}")
    timing_rows = []
    for event in indexed:
        timings = event["timings"]
        timing_rows.append({"arxiv_id": event["arxiv_id"], "source_name": event["file"],
                            "chunks": event["chunks"], "pages": timings["pages"],
                            "parse_ms": timings["parse_ms"], "chunk_ms": timings["chunk_ms"],
                            "index_ms": timings["index_ms"], "total_ms": timings["total_ms"],
                            "embedding_batch_count": len(timings.get("embedding_batches", []))})
    write_csv(results / "indexing_timing_by_paper.csv", timing_rows)

    by_config: dict[str, list[dict]] = defaultdict(list)
    for item in retrieval:
        by_config[item["config"]].append(item)
    qa_type = {item["id"]: item["type"] for item in qa_rows}
    matched_ids = {item["id"] for qtype in QUESTION_TYPES
                   for item in [q for q in qa_rows if q["type"] == qtype][:5]}
    matched = {name: [item for item in by_config.get(name, []) if item["id"] in matched_ids] for name in CONFIGS}
    metric_rows = [{"config": name, "question_type": "matched_20", **metrics(matched[name])} for name in CONFIGS]
    write_csv(results / "retrieval_metrics_matched20.csv", metric_rows)

    grouped = []
    for name in CONFIGS:
        for qtype in QUESTION_TYPES:
            subset = [item for item in by_config.get(name, []) if item["type"] == qtype]
            if subset:
                grouped.append({"config": name, "question_type": qtype, **metrics(subset)})
    write_csv(results / "retrieval_metrics_by_type_actual_n.csv", grouped)

    cases = []
    for config in CONFIGS:
        for item in by_config.get(config, []):
            gold = set(item["gold_arxiv_ids"])
            retrieved = [result.get("arxiv_id") for result in item["results"]]
            missed = sorted(gold - set(retrieved))
            if missed:
                cases.append({"id": item["id"], "question_type": item["type"], "config": config,
                              "question": item["question"], "gold_arxiv_ids": ";".join(item["gold_arxiv_ids"]),
                              "missed_arxiv_ids": ";".join(missed),
                              "top5_arxiv_ids": ";".join(str(x) for x in retrieved),
                              "elapsed_ms": item["elapsed_ms"], "warning": item.get("warning") or ""})
    hit5_failure_count = sum(
        not bool(set(item["gold_arxiv_ids"]) & {result.get("arxiv_id") for result in item["results"]})
        for rows in by_config.values() for item in rows
    )
    unique_questions_with_missing_gold = len({item["id"] for item in cases})
    write_csv(results / "retrieval_bad_cases.csv", cases)

    rank_changes = []
    hybrid = {item["id"]: item for item in by_config.get("hybrid_candidate20", [])}
    reranked = {item["id"]: item for item in by_config.get("hybrid_rerank_candidate20", [])}
    opt = {item["id"]: item for item in by_config.get("optimization_hybrid_rerank_candidate40", [])}
    for question_id in sorted(set(hybrid) & set(reranked)):
        base, after = hybrid[question_id], reranked[question_id]
        before_ids = [item["chunk_id"] for item in base["results"]]
        after_ids = [item["chunk_id"] for item in after["results"]]
        changed = before_ids != after_ids
        candidate_after = opt.get(question_id)
        rank_changes.append({
            "id": question_id, "question_type": base["type"],
            "hybrid_top1": base["results"][0]["arxiv_id"] if base["results"] else "",
            "rerank20_top1": after["results"][0]["arxiv_id"] if after["results"] else "",
            "hybrid_top5": ";".join(str(row["arxiv_id"]) for row in base["results"]),
            "rerank20_top5": ";".join(str(row["arxiv_id"]) for row in after["results"]),
            "rerank20_order_changed": changed,
            "rerank20_elapsed_ms": after["elapsed_ms"],
            "candidate40_top1": candidate_after["results"][0]["arxiv_id"] if candidate_after and candidate_after["results"] else "",
            "candidate40_elapsed_ms": candidate_after["elapsed_ms"] if candidate_after else "",
        })
    write_csv(results / "reranker_rank_changes.csv", rank_changes)

    comparisons = {}
    for name in ["hybrid_rerank_candidate20", "optimization_hybrid_rerank_candidate40"]:
        comparisons[name] = metrics(matched[name])
    baseline = comparisons["hybrid_rerank_candidate20"]
    optimized = comparisons["optimization_hybrid_rerank_candidate40"]
    paired = []
    for question_id in sorted(set(reranked) & set(opt)):
        b, o = reranked[question_id], opt[question_id]
        def rr(record):
            gold = set(record["gold_arxiv_ids"])
            rank = next((i for i, result in enumerate(record["results"], 1) if result["arxiv_id"] in gold), None)
            return 1 / rank if rank else 0.0
        paired.append({"id": question_id, "mrr_k20": rr(b), "mrr_k40": rr(o),
                       "hit_k20": int(rr(b) > 0), "hit_k40": int(rr(o) > 0),
                       "recall_k20": len(set(b["gold_arxiv_ids"]) & {r["arxiv_id"] for r in b["results"]}) / len(b["gold_arxiv_ids"]),
                       "recall_k40": len(set(o["gold_arxiv_ids"]) & {r["arxiv_id"] for r in o["results"]}) / len(o["gold_arxiv_ids"]),
                       "elapsed_k20_ms": b["elapsed_ms"], "elapsed_k40_ms": o["elapsed_ms"]})
    write_csv(results / "candidate_k_paired_questions.csv", paired)
    paired_summary = {
        "n": len(paired),
        "mrr_k20": baseline["mrr"], "mrr_k40": optimized["mrr"],
        "mrr_absolute_delta": round(optimized["mrr"] - baseline["mrr"], 4),
        "hit_at_5_k20": baseline["hit_at_5"], "hit_at_5_k40": optimized["hit_at_5"],
        "recall_at_5_k20": baseline["recall_at_5"], "recall_at_5_k40": optimized["recall_at_5"],
        "mean_latency_k20_ms": baseline["mean_retrieval_ms"], "mean_latency_k40_ms": optimized["mean_retrieval_ms"],
        "latency_ratio_k40_over_k20": round(optimized["mean_retrieval_ms"] / baseline["mean_retrieval_ms"], 3),
        "questions_mrr_up": sum(row["mrr_k40"] > row["mrr_k20"] for row in paired),
        "questions_mrr_tied": sum(row["mrr_k40"] == row["mrr_k20"] for row in paired),
        "questions_mrr_down": sum(row["mrr_k40"] < row["mrr_k20"] for row in paired),
    }

    write_charts(matched, by_config, timing_rows, results)
    try:
        deepseek = json.loads((results / "deepseek_rag_answers.json").read_text(encoding="utf-8"))
        one_shot = deepseek.get("one_shot", [])
        rag_summary = {
            "attempted_one_shot": len(one_shot),
            "successful_one_shot": sum(not row.get("warning") for row in one_shot),
            "failed_or_fallback_one_shot": sum(bool(row.get("warning")) for row in one_shot),
            "successful_elapsed_ms": [row["elapsed_ms"] for row in one_shot if not row.get("warning")],
            "token_usage_successful": {key: sum(int(row.get("usage", {}).get(key, 0)) for row in one_shot if not row.get("warning"))
                                       for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "stream": deepseek.get("stream", {}),
        }
    except Exception:
        rag_summary = {"status": "not_run_or_missing"}
    separate_stream_path = results / "deepseek_stream_answer.json"
    if separate_stream_path.is_file():
        separate_stream = json.loads(separate_stream_path.read_text(encoding="utf-8"))
        rag_summary["stream"] = {
            "status": separate_stream.get("status"),
            "chunk_count": separate_stream.get("chunk_count"),
            "time_to_first_chunk_ms": separate_stream.get("time_to_first_chunk_ms"),
            "elapsed_ms": separate_stream.get("elapsed_ms"),
            "error_type": separate_stream.get("error_type"),
            "citation_count": len(separate_stream.get("citations", [])),
            "source_names": sorted({row.get("source_name") for row in separate_stream.get("citations", [])}),
            "token_usage": separate_stream.get("token_usage"),
            "result_file": str(separate_stream_path),
        }

    agent_path = results / "agent_tool_decisions.jsonl"
    agent_records = load_jsonl(agent_path) if agent_path.is_file() else []
    completed_agent_records = [row for row in agent_records if not row.get("error_type")]
    agent_expected_tool_hits = sum(bool(row.get("selected_expected_tool")) for row in completed_agent_records)
    agent_tool_calls = [len(row.get("tool_trace", [])) for row in completed_agent_records]
    agent_usage = {
        key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in completed_agent_records)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    agent_summary = {
        "attempted_cases": len(agent_records),
        "completed_cases": len(completed_agent_records),
        "failed_cases": len(agent_records) - len(completed_agent_records),
        "expected_tool_selected": agent_expected_tool_hits,
        "tool_call_accuracy": round(agent_expected_tool_hits / len(completed_agent_records), 4)
        if completed_agent_records else None,
        "mean_react_rounds": round(statistics.mean(row.get("iterations", 0) for row in completed_agent_records), 3)
        if completed_agent_records else None,
        "mean_tool_calls_per_case": round(statistics.mean(agent_tool_calls), 3) if agent_tool_calls else None,
        "mean_elapsed_ms": round(statistics.mean(row["elapsed_ms"] for row in completed_agent_records), 1)
        if completed_agent_records else None,
        "token_usage": agent_usage,
        "cases": [{
            "id": row.get("id"), "expected_tool": row.get("expected_tool"),
            "selected_expected_tool": row.get("selected_expected_tool"),
            "tool_names": row.get("tool_names", []), "iterations": row.get("iterations", 0),
            "elapsed_ms": row.get("elapsed_ms"), "error_type": row.get("error_type"),
            "tool_statuses": [event.get("status") for event in row.get("tool_trace", [])],
        } for row in agent_records],
        "source": str(agent_path) if agent_path.is_file() else None,
        "scope_note": "本轮独立隔离的 3 例烟雾评测，不能外推为 8 个工具的总体准确率。",
    }
    ratings_path = results / "manual_answer_ratings.csv"
    rating_rows = []
    if ratings_path.is_file():
        with ratings_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rating_rows = list(csv.DictReader(handle))
    rating_fields = ("correctness_1_to_5", "completeness_1_to_5", "citation_accuracy_1_to_5")
    completed_ratings = [row for row in rating_rows if all(row.get(key, "").strip() for key in rating_fields)]
    answer_quality_review = {
        "rated_answer_count": len(completed_ratings),
        "means": {
            key.removesuffix("_1_to_5"): round(statistics.mean(float(row[key]) for row in completed_ratings), 2)
            for key in rating_fields
        } if completed_ratings else {},
        "reviewer_note": "单人证据核查，非独立盲评；不代表稳定总体质量分布。" if completed_ratings else "尚无完成评分。",
        "source": str(ratings_path) if ratings_path.is_file() else None,
    }

    documents_manifest = json.loads((runtime / "index" / "documents.json").read_text(encoding="utf-8"))
    paper_rows = []
    for event in indexed:
        paper_rows.append({"arxiv_id": event["arxiv_id"], "file": event["file"], "pages": event["timings"]["pages"],
                           "chunks": event["chunks"], "parse_ms": event["timings"]["parse_ms"],
                           "chunk_ms": event["timings"]["chunk_ms"], "index_ms": event["timings"]["index_ms"],
                           "total_ms": event["timings"]["total_ms"]})
    total_chunks = sum(item["chunks"] for item in paper_rows)
    rerank_rows = by_config.get("hybrid_rerank_candidate20", [])
    manifest = {
        "evaluation_date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "direction": "具身智能：通用机器人策略与视觉-语言-动作模型",
        "source_paper_count": 10,
        "question_count": len(qa_rows),
        "question_type_counts": dict(Counter(row["type"] for row in qa_rows)),
        "evidence_page_verified_count": sum(len(row.get("verified_evidence", [])) for row in load_jsonl(results / "具身智能问答评测集_含页码证据.jsonl")),
        "indexed_document_count": len(documents_manifest),
        "chunk_count": total_chunks,
        "paper_indexing": paper_rows,
        "index_total_document_ms": round(sum(item["total_ms"] for item in paper_rows), 1),
        "retrieval_configs": {name: metrics(by_config.get(name, [])) for name in CONFIGS},
        "matched_20_metrics": {name: metrics(matched[name]) for name in CONFIGS},
        "candidate_k_matched_comparison": paired_summary,
        "reranker_warning_count_on_sample": sum(bool(item.get("warning")) for item in rerank_rows),
        "reranker_sample_n": len(rerank_rows),
        "reranker_rank_changed_vs_hybrid_count": sum(row["rerank20_order_changed"] for row in rank_changes),
        "actual_deepseek_evaluation": rag_summary,
        "agent_decision_eval": agent_summary,
        "answer_quality_manual_review": answer_quality_review,
        "chroma_restart_note": "Index was built and queried in a single process. A prior interrupted run demonstrated Windows HNSW index load failure after process restart; this is a persistence limitation, not hidden or repaired by deleting the isolated data.",
        "data_dir": str(runtime),
        "results_dir": str(results),
        "manifest_collection_entry_count": len(documents_manifest),
        "privacy": "No API key, original PDF, model weights, or production index is copied into the delivery results.",
    }
    (results / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"matched_metrics": manifest["matched_20_metrics"],
                      "candidate_comparison": paired_summary,
                      "reranker_order_changed": manifest["reranker_rank_changed_vs_hybrid_count"],
                      "missed_gold_document_occurrences": len(cases),
                      "hit_at_5_failure_records": hit5_failure_count,
                      "unique_questions_with_any_missing_gold": unique_questions_with_missing_gold,
                      "index_documents": len(paper_rows), "chunks": total_chunks,
                      "rag_summary": rag_summary, "agent_summary": agent_summary,
                      "manifest": str(results / "evaluation_manifest.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
