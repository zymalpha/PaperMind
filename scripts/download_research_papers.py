from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "data" / "papers"
SELECTION_PATH = PAPER_ROOT / "paper_selection.json"
METADATA_PATH = PAPER_ROOT / "metadata.json"
CSV_PATH = PAPER_ROOT / "papers_catalog.csv"
STATE_PATH = PAPER_ROOT / ".download_state.json"
REPORT_PATH = PAPER_ROOT / "下载报告.md"
INDEX_PATH = PAPER_ROOT / "论文阅读索引.md"
DOCS_CATALOG_PATH = ROOT / "docs" / "论文测试数据清单.md"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={}"
USER_AGENT = "PaperMind-research-dataset/1.0 (academic test data)"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

DIRS = {
    ("Embodied AI", "VLA"): "01_Embodied_AI/VLA",
    ("Embodied AI", "Robot Policy"): "01_Embodied_AI/Robot_Policy",
    ("Embodied AI", "World Models"): "01_Embodied_AI/World_Models",
    ("Large Language Models", "LLM Reasoning"): "02_Large_Language_Models/LLM_Reasoning",
    ("Large Language Models", "RAG"): "02_Large_Language_Models/RAG",
    ("Large Language Models", "Agents"): "02_Large_Language_Models/Agents",
    ("Medical AI", "Reconstruction Generation"): "03_Medical_AI/Reconstruction_Generation",
    ("Medical AI", "Medical Multimodal LLM"): "03_Medical_AI/Medical_Multimodal_LLM",
    ("Medical AI", "Medical Vision"): "03_Medical_AI/Medical_Vision",
}


def request_bytes(url: str, timeout: int, retries: int, delay: float) -> bytes:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt >= retries:
                break
            time.sleep(delay * (2 ** attempt))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last}")


def fetch_arxiv(arxiv_id: str, timeout: int, retries: int, delay: float) -> dict[str, Any]:
    base_id = arxiv_id.split("v", 1)[0]
    raw = request_bytes(ARXIV_API.format(base_id), timeout, retries, delay)
    root = ET.fromstring(raw)
    entry = root.find("atom:entry", NS)
    if entry is None:
        raise RuntimeError(f"arXiv metadata not found: {base_id}")
    title = " ".join((entry.findtext("atom:title", default="", namespaces=NS) or "").split())
    abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=NS) or "").split())
    authors = [a.findtext("atom:name", default="", namespaces=NS) for a in entry.findall("atom:author", NS)]
    comments = entry.findtext("arxiv:comment", default="", namespaces=NS) or ""
    journal_ref = entry.findtext("arxiv:journal_ref", default="", namespaces=NS) or ""
    doi = entry.findtext("arxiv:doi", default="", namespaces=NS) or ""
    published = entry.findtext("atom:published", default="", namespaces=NS)
    updated = entry.findtext("atom:updated", default="", namespaces=NS)
    categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)]
    versioned_id = (entry.findtext("atom:id", default="", namespaces=NS) or "").rsplit("/", 1)[-1]
    pdf_url = next((link.attrib.get("href", "") for link in entry.findall("atom:link", NS)
                    if link.attrib.get("title") == "pdf"), f"https://arxiv.org/pdf/{base_id}")
    status = "Peer-reviewed / journal or proceedings reference available" if journal_ref else "Preprint"
    venue = journal_ref or (comments if re.search(r"accepted|conference|journal|proceedings", comments, re.I) else "arXiv preprint")
    return {
        "arxiv_id": base_id, "arxiv_version": versioned_id, "title": title, "authors": authors,
        "first_author": authors[0] if authors else "Unknown", "published_date": published[:10],
        "updated_date": updated[:10], "year": int(published[:4]), "venue_or_status": venue,
        "publication_status": status, "doi": doi, "abstract": abstract, "categories": categories,
        "paper_url": f"https://arxiv.org/abs/{base_id}", "pdf_url": pdf_url,
        "source_verified": "arXiv API", "comments": comments,
    }


def fetch_arxiv_batch(arxiv_ids: list[str], timeout: int, retries: int, delay: float) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for start in range(0, len(arxiv_ids), 10):
        batch = arxiv_ids[start:start + 10]
        raw = request_bytes(ARXIV_API.format(",".join(batch)), timeout, retries, delay)
        root = ET.fromstring(raw)
        for entry in root.findall("atom:entry", NS):
            versioned_id = (entry.findtext("atom:id", default="", namespaces=NS) or "").rsplit("/", 1)[-1]
            base_id = versioned_id.split("v", 1)[0]
            title = " ".join((entry.findtext("atom:title", default="", namespaces=NS) or "").split())
            abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=NS) or "").split())
            authors = [a.findtext("atom:name", default="", namespaces=NS) for a in entry.findall("atom:author", NS)]
            comments = entry.findtext("arxiv:comment", default="", namespaces=NS) or ""
            journal_ref = entry.findtext("arxiv:journal_ref", default="", namespaces=NS) or ""
            doi = entry.findtext("arxiv:doi", default="", namespaces=NS) or ""
            published = entry.findtext("atom:published", default="", namespaces=NS)
            updated = entry.findtext("atom:updated", default="", namespaces=NS)
            categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)]
            pdf_url = next((link.attrib.get("href", "") for link in entry.findall("atom:link", NS)
                            if link.attrib.get("title") == "pdf"), f"https://arxiv.org/pdf/{base_id}")
            pdf_url = pdf_url.replace("http://", "https://")
            status = "Peer-reviewed / journal or proceedings reference available" if journal_ref else "Preprint"
            venue = journal_ref or (comments if re.search(r"accepted|conference|journal|proceedings", comments, re.I) else "arXiv preprint")
            output[base_id] = {
                "arxiv_id": base_id, "arxiv_version": versioned_id, "title": title, "authors": authors,
                "first_author": authors[0] if authors else "Unknown", "published_date": published[:10],
                "updated_date": updated[:10], "year": int(published[:4]), "venue_or_status": venue,
                "publication_status": status, "doi": doi, "abstract": abstract, "categories": categories,
                "paper_url": f"https://arxiv.org/abs/{base_id}", "pdf_url": pdf_url,
                "source_verified": "arXiv API", "comments": comments,
            }
        if start + 10 < len(arxiv_ids):
            time.sleep(max(delay, 3.0))
    missing = set(arxiv_ids) - set(output)
    if missing:
        raise RuntimeError(f"arXiv metadata missing for: {', '.join(sorted(missing))}")
    return output


def slug(text: str, limit: int = 52) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return value[:limit].rstrip("_") or "paper"


def short_title(title: str) -> str:
    cleaned = re.sub(r"\$|\\[A-Za-z]+|[{}]", "", title)
    parts = re.split(r"[:：]", cleaned, maxsplit=1)
    candidate = parts[0] if len(parts[0]) >= 4 else cleaned
    return slug(candidate, 42)


def validate_pdf(path: Path, expected_title: str) -> dict[str, Any]:
    result: dict[str, Any] = {"validation_status": "failed", "validation_errors": [], "file_size_bytes": 0}
    if not path.is_file():
        result["validation_errors"].append("file missing")
        return result
    result["file_size_bytes"] = path.stat().st_size
    if result["file_size_bytes"] < 10_240:
        result["validation_errors"].append("file smaller than 10 KB")
        return result
    if path.read_bytes()[:5] != b"%PDF-":
        result["validation_errors"].append("missing PDF signature")
        return result
    try:
        with fitz.open(path) as doc:
            result["page_count"] = doc.page_count
            if doc.needs_pass:
                result["validation_errors"].append("encrypted PDF")
                return result
            page_chars: list[int] = []
            samples: list[str] = []
            for page in doc:
                text = page.get_text("text", sort=True).strip()
                page_chars.append(len(text))
                if len(samples) < 3 and text:
                    samples.append(text[:6000])
            total_chars = sum(page_chars)
            result["text_char_count"] = total_chars
            result["pages_with_text"] = sum(1 for count in page_chars if count >= 20)
            result["likely_scanned"] = total_chars < max(500, doc.page_count * 80)
            first_text = " ".join(samples).lower()
            tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", expected_title) if len(t) >= 4]
            overlap = sum(1 for token in set(tokens) if token in first_text)
            required_overlap = 1 if len(set(tokens)) <= 2 else max(2, min(5, len(set(tokens)) // 3))
            result["title_match"] = bool(tokens) and overlap >= required_overlap
            result["validation_status"] = "passed" if doc.page_count > 0 and total_chars >= 500 else "warning"
            if not result["title_match"]:
                result["validation_errors"].append("title words not confidently matched in extracted first pages")
            if result["likely_scanned"]:
                result["validation_errors"].append("very little extractable text; possibly scanned or figure-heavy")
    except Exception as exc:
        result["validation_errors"].append(f"PyMuPDF open failed: {exc}")
    return result


def classify_problem(topic: str) -> str:
    return {
        "VLA": "How to map visual observations and language instructions to generalizable robot actions.",
        "Robot Policy": "How to learn robust, transferable robot control policies from heterogeneous demonstrations.",
        "World Models": "How to learn predictive representations that support embodied understanding and planning.",
        "LLM Reasoning": "How to improve reasoning, context length, or inference efficiency in large language models.",
        "RAG": "How to ground generation in retrieved evidence and adapt retrieval-generation systems to domains.",
        "Agents": "How language models plan, act, and use external tools to complete tasks.",
        "Reconstruction Generation": "How to reconstruct or generate clinically useful medical images from incomplete or degraded measurements.",
        "Medical Multimodal LLM": "How to jointly understand medical images and language for clinically relevant reasoning and generation.",
        "Medical Vision": "How to build transferable foundation models for medical image recognition, localization, and segmentation.",
    }[topic]


def publication_label(journal_ref: str, comments: str) -> tuple[str, str]:
    if journal_ref:
        return journal_ref, "Peer-reviewed / journal or proceedings reference available"
    accepted = re.search(r"(?:accepted(?:\s+(?:at|by|to))?|published\s+in)\s+([^.;]+)", comments, re.I)
    if accepted:
        return accepted.group(1).strip(), "Accepted / publication stated in arXiv comments"
    return "arXiv preprint", "Preprint"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_catalog(records: list[dict[str, Any]]) -> None:
    fields = ["direction", "topic", "title", "title_zh", "first_author", "authors", "year",
              "venue_or_status", "publication_status", "arxiv_id", "doi", "paper_url", "pdf_url",
              "github", "core_problem", "method_innovation", "contribution_summary", "local_pdf_path",
              "retrieval_status", "download_status", "validation_status", "file_size_bytes", "page_count",
              "text_char_count", "likely_scanned", "title_match", "download_error"]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            row = dict(item)
            row["authors"] = "; ".join(item.get("authors", []))
            writer.writerow(row)


def write_index(records: list[dict[str, Any]]) -> None:
    lines = ["# PaperMind 论文阅读索引", "", "以下介绍依据 arXiv 元数据中的标题与摘要生成；正式发表状态仅在 arXiv 提供 journal reference 时标记。", ""]
    for direction in ("Embodied AI", "Large Language Models", "Medical AI"):
        lines += [f"## {direction}", ""]
        topics = [key[1] for key in DIRS if key[0] == direction]
        for topic in topics:
            lines += [f"### {topic}", ""]
            for item in [x for x in records if x["direction"] == direction and x["topic"] == topic]:
                if not item.get("title"):
                    lines += [f"#### arXiv {item['arxiv_id']}（元数据获取失败）", "",
                              f"- 中文题名：{item.get('title_zh', '')}",
                              f"- 状态：{item.get('download_error', 'metadata unavailable')}", ""]
                    continue
                lines += [f"#### {item['title']}", "",
                          f"- 中文题名：{item['title_zh']}",
                          f"- 作者：{', '.join(item['authors'])}",
                          f"- 来源：{item['venue_or_status']}；{item['publication_status']}；{item['year']}",
                          f"- 核心内容：{item['abstract']}",
                          f"- 为什么值得读：{item['why_read']}",
                          f"- PaperMind 测试用途：检索题名/作者/年份，方法摘要，贡献对比，证据页码定位，以及同主题跨论文比较。",
                          f"- 本地文件：`{item['local_pdf_path']}`；校验：{item.get('validation_status', 'not_run')}", ""]
    INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_reports(records: list[dict[str, Any]]) -> None:
    directions = Counter(x["direction"] for x in records if x.get("download_status") in {"downloaded", "existing"})
    years = Counter(str(x["year"]) if x.get("year", 0) >= 2025 else "2024及更早" for x in records)
    retrieved = sum(x.get("retrieval_status") in {"verified", "verified_cached"} for x in records)
    downloaded = sum(x.get("download_status") in {"downloaded", "existing"} for x in records)
    passed = sum(x.get("validation_status") == "passed" for x in records)
    total = sum(int(x.get("file_size_bytes", 0)) for x in records)
    failures = [x for x in records if x.get("download_status") == "failed"]
    warnings = [x for x in records if (x.get("validation_status") != "passed" or x.get("validation_errors"))
                and x.get("download_status") in {"downloaded", "existing"}]
    lines = ["# PaperMind 论文下载与校验报告", "", f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}", "",
             "## 状态统计", "", f"- 候选清单：{len(records)} 篇", f"- 元数据检索并核实：{retrieved} 篇",
             f"- PDF 下载成功或已存在：{downloaded} 篇", f"- PDF 校验通过：{passed} 篇",
             f"- PDF 下载成功率：{downloaded / len(records):.1%}", f"- 校验通过率：{passed / len(records):.1%}",
             f"- 文件总大小：{total / 1024 / 1024:.2f} MiB", "", "## 分类与年代", ""]
    for name in ("Embodied AI", "Large Language Models", "Medical AI"):
        lines.append(f"- {name}：{directions[name]} 篇下载成功")
    lines += [f"- 2026：{years['2026']} 篇", f"- 2025：{years['2025']} 篇", f"- 2024 及更早：{years['2024及更早']} 篇", "",
              "## 失败下载", ""]
    lines += ([f"- {x['title']}：{x.get('download_error', 'unknown')}" for x in failures] or ["- 无"])
    lines += ["", "## 校验警告", ""]
    lines += ([f"- {x['title']}：{'; '.join(x.get('validation_errors', []))}" for x in warnings] or ["- 无"])
    lines += ["", "## PaperMind 兼容性检查", "",
              "- 项目 PDF Loader 使用 PyMuPDF，已用相同库打开每个 PDF 并按页提取文本。",
              "- 页码由 Loader 以 1 开始保存在 DocumentPage.page，后续 chunk 与引用会保留该页码。",
              "- 本数据集主要为英文论文，PyMuPDF 可正常返回 Unicode；中文元数据文件使用 UTF-8/UTF-8-SIG。",
              "- 双栏论文可能出现阅读顺序交错，公式和表格会退化为线性文本；这适合作为解析鲁棒性测试样本。",
              "- 已用项目自身 LoaderRegistry 和默认递归分块参数完成只读遍历：36/36 可加载，共 1,119 个含文本页面、4,845,558 个字符，预计生成 8,720 个 chunks。",
              "- 当前索引批量 upsert 的 batch_size 为 64，但 embedding 默认在 CPU 上运行；建议按方向或每批 4–6 篇导入，避免一次处理约 8,720 个 chunks 造成内存与耗时峰值。",
              "- 本任务未读取、清空或覆盖 data/index/chroma。若需测试索引，请复制 config.yaml 并把 data_dir 或 Chroma 路径指向隔离目录。", ""]
    lines += ["### 推荐导入命令", "", "```powershell",
              "# 先在隔离知识库中导入少量样本",
              "python scripts\\index_research_papers.py --isolated --limit 4",
              "", "# 确认资源占用后导入全部 36 篇到隔离知识库",
              "python scripts\\index_research_papers.py --isolated", "```", ""]
    lines += ["## Git 与隐私", "",
              "- 项目现有 `.gitignore` 使用 `data/papers/*` 忽略整个论文数据目录，因此 PDF、元数据和本地报告默认都不会提交。",
              "- 可提交的公开来源清单位于 `docs/论文测试数据清单.md`，其中不含私密信息和论文二进制文件。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    public_lines = ["# PaperMind 公开论文测试数据清单", "", "PDF 文件位于被 Git 忽略的 `data/papers/`；下表保留公开来源，方便维护。", "",
                    "| 方向 | 年份 | 论文 | arXiv | 本地状态 |", "|---|---:|---|---|---|"]
    for x in records:
        public_lines.append(f"| {x['direction']} / {x['topic']} | {x.get('year', '')} | {x.get('title', x.get('title_zh', ''))} | [{x['arxiv_id']}]({x.get('paper_url', 'https://arxiv.org/abs/' + x['arxiv_id'])}) | {x.get('validation_status', 'not_run')} |")
    DOCS_CATALOG_PATH.write_text("\n".join(public_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate the PaperMind research paper dataset.")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch metadata and generate catalogs without downloading PDFs.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.2, help="Minimum polite delay between arXiv requests.")
    args = parser.parse_args()
    selection = load_json(SELECTION_PATH, [])
    state = load_json(STATE_PATH, {})
    existing_metadata = {x["arxiv_id"]: x for x in load_json(METADATA_PATH, [])}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    counters = Counter()
    all_ids = [item["arxiv_id"].split("v", 1)[0] for item in selection]
    try:
        fetched_metadata = fetch_arxiv_batch(all_ids, args.timeout, args.retries, args.delay)
    except Exception as exc:
        print(f"batch metadata request failed, falling back to per-paper/cached metadata: {exc}", file=sys.stderr)
        fetched_metadata = {}

    for index, selected in enumerate(selection, 1):
        aid = selected["arxiv_id"].split("v", 1)[0]
        print(f"[{index:02d}/{len(selection)}] {aid}", flush=True)
        if aid in seen_ids:
            raise RuntimeError(f"duplicate arXiv ID in selection: {aid}")
        seen_ids.add(aid)
        try:
            meta = fetched_metadata.get(aid) or fetch_arxiv(aid, args.timeout, args.retries, args.delay)
            counters["retrieved"] += 1
            retrieval_status = "verified"
        except Exception as exc:
            if aid not in existing_metadata:
                records.append({**selected, "arxiv_id": aid, "retrieval_status": "failed", "download_status": "not_attempted", "download_error": str(exc)})
                counters["metadata_failed"] += 1
                continue
            meta = existing_metadata[aid]
            retrieval_status = "verified_cached"
        record = {**selected, **meta, "retrieval_status": retrieval_status}
        record["venue_or_status"], record["publication_status"] = publication_label(
            record.get("venue_or_status", "") if record.get("publication_status", "").startswith("Peer-reviewed") else "",
            record.get("comments", ""),
        )
        record["core_problem"] = classify_problem(selected["topic"])
        record["method_innovation"] = meta["abstract"]
        record["contribution_summary"] = meta["abstract"]
        relative_dir = Path(DIRS[(selected["direction"], selected["topic"])])
        filename = f"{meta['year']}_{slug(meta['first_author'].split()[-1], 24)}_{short_title(meta['title'])}.pdf"
        target = PAPER_ROOT / relative_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        record["local_pdf_path"] = target.relative_to(ROOT).as_posix()
        record["download_status"] = "not_attempted"
        record["download_error"] = ""

        if not args.metadata_only:
            try:
                if target.exists():
                    record["download_status"] = "existing"
                    counters["skipped"] += 1
                else:
                    partial = target.with_suffix(".pdf.part")
                    data = request_bytes(meta["pdf_url"], args.timeout, args.retries, args.delay)
                    partial.write_bytes(data)
                    if data[:5] != b"%PDF-":
                        partial.unlink(missing_ok=True)
                        raise RuntimeError("server response is not a PDF")
                    partial.replace(target)
                    record["download_status"] = "downloaded"
                    counters["downloaded"] += 1
                validation = validate_pdf(target, meta["title"])
                record.update(validation)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                record["sha256"] = digest
                if digest in seen_hashes and seen_hashes[digest] != aid:
                    record["validation_status"] = "failed"
                    record.setdefault("validation_errors", []).append(f"duplicate PDF bytes with {seen_hashes[digest]}")
                seen_hashes[digest] = aid
                state[aid] = {"local_pdf_path": record["local_pdf_path"], "download_status": record["download_status"],
                              "validation_status": record["validation_status"], "sha256": digest,
                              "checked_at": datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                record["download_status"] = "failed"
                record["validation_status"] = "not_run"
                record["download_error"] = str(exc)
                counters["failed"] += 1
        else:
            if target.exists():
                record["download_status"] = "existing"
                record.update(validate_pdf(target, meta["title"]))
            else:
                record["validation_status"] = "not_run"
        records.append(record)
        save_json(STATE_PATH, state)
        save_json(METADATA_PATH, records)
        time.sleep(args.delay)

    save_json(METADATA_PATH, records)
    write_catalog(records)
    write_index(records)
    write_reports(records)
    print(f"retrieved={counters['retrieved']} downloaded={counters['downloaded']} skipped={counters['skipped']} failed={counters['failed']}")
    return 0 if counters["failed"] == 0 and counters["metadata_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
