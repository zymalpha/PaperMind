from __future__ import annotations

import ast
import operator
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jsonschema import validate

from .llm import DeepSeekClient
from .rag import RAGEngine
from .retrieval import tokenize


class ToolError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

    def register(self, name: str, description: str, schema: dict[str, Any], handler: Callable[..., Any]) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = ({"type": "function", "function": {"name": name, "description": description, "parameters": schema}}, handler)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [item[0] for item in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        definition, handler = self._tools[name]
        validate(instance=arguments, schema=definition["function"]["parameters"])
        return handler(**arguments)


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


def create_tool_registry(service, rag: RAGEngine, llm: DeepSeekClient, web_enabled: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    papers = lambda: service.list_documents()

    def retrieve(query: str) -> dict[str, Any]:
        results = rag.retrieve(query)
        return {"query": query, "results": [{"source": r.chunk.source_name, "page": r.chunk.page, "score": round(r.score, 4), "chunk_id": r.chunk.id, "text": r.chunk.text[:1200]} for r in results]}

    def paper_info(paper_id: str) -> dict[str, Any]:
        matches = [p for p in papers() if p.document_id.startswith(paper_id) or p.source_name == paper_id]
        if not matches:
            raise ToolError("未找到论文；paper_id 可使用文件名或文档 ID 前缀。")
        paper = matches[0]
        text = "\n".join(c.text for c in service.vector_store.all_chunks(paper.document_id))
        title = next((line.strip() for line in text.splitlines() if len(line.strip()) > 8), paper.source_name)
        doi = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
        year = re.search(r"\b(?:19|20)\d{2}\b", text[:5000])
        authors = next((line.strip() for line in text.splitlines()[1:8] if re.search(r"[A-Z][a-z]+\s+[A-Z]", line)), "未能从文档文本中可靠提取")
        abstract = re.search(r"(?:abstract|摘要)\s*[:：]?\s*(.{80,1200}?)(?:\n\s*\n|\n(?:keywords|关键词|1\s+introduction)|$)", text, re.I | re.S)
        return {"document_id": paper.document_id, "file": paper.source_name, "title": title[:300], "authors": authors[:300], "year": year.group(0) if year else None, "doi": doi.group(0).rstrip(".,") if doi else None, "abstract": abstract.group(1).strip() if abstract else "未能可靠提取摘要"}

    def compare_papers(paper_a: str, paper_b: str) -> dict[str, Any]:
        left, right = paper_info(paper_a), paper_info(paper_b)
        chunks_a = service.vector_store.all_chunks(left["document_id"])
        chunks_b = service.vector_store.all_chunks(right["document_id"])
        return {"paper_a": left, "paper_b": right,
                "evidence_a": [c.text[:700] for c in chunks_a[:4]], "evidence_b": [c.text[:700] for c in chunks_b[:4]],
                "instruction": "只可基于所给证据比较；这是一组原文证据，不是未经验证的结论。"}

    def extract_keywords(text: str, limit: int = 12) -> dict[str, Any]:
        terms = tokenize(text)
        stop = {"研究", "本文", "结果", "方法", "进行", "使用", "through", "using", "with", "from", "that", "this"}
        counts: dict[str, int] = {}
        for term in terms:
            if len(term) > 1 and term not in stop:
                counts[term] = counts.get(term, 0) + 1
        return {"keywords": [word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]}

    def summarize_paper(paper_id: str) -> dict[str, Any]:
        info = paper_info(paper_id)
        chunks = service.vector_store.all_chunks(info["document_id"])
        text = "\n".join(chunk.text for chunk in chunks)[:10000]
        response = llm.chat([
            {"role": "system", "content": "你是科研论文助理。只根据提供的原文，按背景、方法、结果、结论四部分做简洁摘要；未知内容明确写未找到。"},
            {"role": "user", "content": text},
        ], max_tokens=3072)
        return {"paper": info, "structured_summary": response.content, "usage": response.usage}

    def now() -> dict[str, str]:
        current = datetime.now(timezone.utc)
        return {"utc": current.isoformat(), "local": datetime.now().astimezone().isoformat()}

    def calculate(expression: str) -> dict[str, float]:
        operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                     ast.USub: operator.neg, ast.UAdd: operator.pos}
        def evaluate(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("Exponent is limited to 8")
                result = operators[type(node.op)](left, right)
                if abs(result) > 1e100:
                    raise ValueError("Result out of safe range")
                return result
            if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                return operators[type(node.op)](evaluate(node.operand))
            raise ValueError("Only arithmetic numbers and + - * / ** % parentheses are allowed")
        return {"expression": expression, "result": float(evaluate(ast.parse(expression, mode="eval").body))}

    def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        if not web_enabled:
            return {"available": False, "status": "disabled", "results": [], "message": "联网搜索未启用；未执行网络检索，也未构造搜索结果。"}
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=max_results)
            return {"available": True, "status": "ok", "results": list(results)}
        except Exception as exc:
            return {"available": False, "status": "error", "results": [], "message": f"真实联网搜索不可用: {type(exc).__name__}"}

    registry.register("knowledge_search", "Search indexed scientific papers with citations.", _object({"query": {"type": "string", "minLength": 1}}, ["query"]), retrieve)
    registry.register("paper_metadata", "Extract available metadata from an indexed paper.", _object({"paper_id": {"type": "string", "minLength": 1}}, ["paper_id"]), paper_info)
    registry.register("compare_papers", "Compare two indexed papers using their actual indexed text evidence.", _object({"paper_a": {"type": "string"}, "paper_b": {"type": "string"}}, ["paper_a", "paper_b"]), compare_papers)
    registry.register("extract_keywords", "Extract frequent domain terms from supplied text.", _object({"text": {"type": "string", "minLength": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}} , ["text"]), extract_keywords)
    registry.register("summarize_paper", "Generate a four-part summary from an indexed paper with the LLM.", _object({"paper_id": {"type": "string", "minLength": 1}}, ["paper_id"]), summarize_paper)
    registry.register("current_time", "Return current local and UTC time.", _object({}), lambda: now())
    registry.register("calculator", "Safely evaluate basic arithmetic.", _object({"expression": {"type": "string", "minLength": 1, "maxLength": 200}}, ["expression"]), calculate)
    registry.register("web_search", "Search the public web when explicitly enabled in configuration.", _object({"query": {"type": "string", "minLength": 1}, "max_results": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]), web_search)
    return registry
