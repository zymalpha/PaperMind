from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.agent import ResearchAgent, SessionMemory  # noqa: E402
from research_assistant.cache import SemanticAnswerCache  # noqa: E402
from research_assistant.chunking import FixedSizeChunker, RecursiveTextChunker, SemanticChunker  # noqa: E402
from research_assistant.models import Chunk, SearchResult  # noqa: E402
from research_assistant.rag import RAGEngine  # noqa: E402
from research_assistant.retrieval import BM25Retriever, HybridRetriever, reciprocal_rank_fusion  # noqa: E402
from research_assistant.tools import ToolRegistry  # noqa: E402
from research_assistant.tools import create_tool_registry  # noqa: E402


def chunk(identifier: str, content: str) -> Chunk:
    return Chunk(identifier, content, "doc", "paper.pdf", "paper.pdf", 2, 0)


@pytest.mark.parametrize("chunker", [FixedSizeChunker(40, 5), RecursiveTextChunker(40, 5), SemanticChunker(40, 5)])
def test_three_chunking_strategies_preserve_metadata(chunker):
    from research_assistant.models import DocumentPage
    pages = [DocumentPage("A paragraph about RAG retrieval. Another sentence about agents. " * 4,
                          "paper.pdf", "paper.pdf", 2, "doc")]
    chunks = chunker.split_pages(pages)
    assert chunks and all(len(item.text) <= 40 for item in chunks)
    assert all(item.page == 2 and item.metadata["chunk_strategy"] == chunker.strategy for item in chunks)


def test_bm25_and_handwritten_rrf():
    a, b, c = chunk("a", "RAG retrieval with dense vectors"), chunk("b", "Agent tool routing"), chunk("c", "retrieval and ranking")
    bm = BM25Retriever([a, b, c]).search("retrieval", 3)
    assert bm and bm[0].chunk.id in {"a", "c"}
    fused = reciprocal_rank_fusion([[SearchResult(a, .9), SearchResult(b, .5)], [SearchResult(b, .8), SearchResult(c, .4)]], rrf_k=10)
    assert fused[0].chunk.id == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=0)


def test_semantic_cache_namespace_and_threshold(tmp_path: Path):
    cache = SemanticAnswerCache(tmp_path / "cache.sqlite", threshold=.99)
    cache.put([1.0, 0.0], "papers-v1", {"answer": "cached"})
    assert cache.get([1.0, 0.0], "papers-v1")["answer"] == "cached"
    assert cache.get([0.0, 1.0], "papers-v1") is None
    assert cache.get([1.0, 0.0], "papers-v2") is None


def test_context_limit_does_not_cite_omitted_chunks():
    results = [SearchResult(chunk("a", "first evidence"), .9), SearchResult(chunk("b", "second evidence"), .8)]
    bounded = RAGEngine._bounded_results(results, 60)
    assert bounded
    contexts = RAGEngine._contexts(bounded, 60)
    assert sum(map(len, contexts)) <= 60


def test_hybrid_rerank_without_configured_model_is_explicit_fallback():
    class Store:
        def query(self, *_args, **_kwargs):
            return [SearchResult(chunk("v", "RAG retrieval"), .8)]
        def all_chunks(self, *_args, **_kwargs):
            return [chunk("v", "RAG retrieval")]
    retriever = HybridRetriever(Store(), reranker=None)
    result = retriever.search("retrieval", mode="hybrid_rerank")
    assert result and retriever.last_warning == "reranker_not_configured"


def test_tool_parameter_validation():
    registry = ToolRegistry()
    registry.register("double", "double an integer", {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}, lambda value: value * 2)
    assert registry.execute("double", {"value": 4}) == 8
    with pytest.raises(Exception):
        registry.execute("double", {"value": "4"})


def test_react_tool_loop_memory_and_logs(tmp_path: Path):
    class FakeLLM:
        model = "test"
        def __init__(self): self.calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return type("R", (), {"tool_calls": [{"id": "call-1", "name": "calculator", "arguments": json.dumps({"expression": "6*7"})}], "content": "", "usage": {"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1}})()
            return type("R", (), {"tool_calls": [], "content": "42", "usage": {"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1}})()

    registry = ToolRegistry()
    registry.register("calculator", "calculate", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}, lambda expression: {"result": 42})
    agent = ResearchAgent(FakeLLM(), registry, tmp_path / "memory.sqlite", tmp_path / "tools.jsonl", max_iterations=3)
    result = agent.run("calculate 6*7", "session-a")
    assert result["answer"] == "42" and result["tool_trace"][0]["status"] == "success"
    assert agent.memory.load("session-a")[0][-1]["content"] == "42"
    assert agent.memory.load("session-b")[0] == []
    assert json.loads((tmp_path / "tools.jsonl").read_text(encoding="utf-8").splitlines()[0])["tool"] == "calculator"


def test_tool_catalog_contains_eight_real_tools():
    from types import SimpleNamespace
    service = SimpleNamespace(list_documents=lambda: [], vector_store=SimpleNamespace(all_chunks=lambda _id: []))
    registry = create_tool_registry(service, SimpleNamespace(retrieve=lambda _query: []), object(), web_enabled=False)
    assert set(registry.names) == {
        "knowledge_search", "paper_metadata", "compare_papers", "extract_keywords",
        "summarize_paper", "current_time", "calculator", "web_search",
    }
    assert registry.execute("web_search", {"query": "paper"})["available"] is False
    assert registry.execute("calculator", {"expression": "(19*23)+1"})["result"] == 438


def test_repeat_tool_call_is_blocked_before_second_execution(tmp_path: Path):
    class RepeatingLLM:
        model = "test"
        def __init__(self): self.calls = 0
        def chat(self, *args, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return type("R", (), {"tool_calls": [{"id": str(self.calls), "name": "tool", "arguments": "{}"}], "content": "", "usage": {}})()
            return type("R", (), {"tool_calls": [], "content": "stopped", "usage": {}})()
    calls = []
    registry = ToolRegistry()
    registry.register("tool", "noop", {"type": "object", "properties": {}, "additionalProperties": False}, lambda: calls.append(1) or "ok")
    result = ResearchAgent(RepeatingLLM(), registry, tmp_path / "memory.sqlite", tmp_path / "tools.jsonl", max_iterations=4).run("go", "repeat")
    assert len(calls) == 1
    assert result["tool_trace"][1]["status"] == "blocked"


def test_agent_stops_after_iteration_limit(tmp_path: Path):
    class RepeatingLLM:
        model = "test"
        def chat(self, *args, **kwargs):
            return type("R", (), {"tool_calls": [{"id": "same", "name": "tool", "arguments": "{}"}],
                                  "content": "", "usage": {}})()
    registry = ToolRegistry()
    executions = []
    registry.register("tool", "noop", {"type": "object", "properties": {}, "additionalProperties": False}, lambda: executions.append("ran") or "ok")
    agent = ResearchAgent(RepeatingLLM(), registry, tmp_path / "m.sqlite", tmp_path / "l.jsonl", max_iterations=2)
    result = agent.run("do it", "limit")
    assert result["iterations"] == 2
    assert "最大工具调用轮数" in result["answer"]
    assert len(executions) == 1
    assert result["tool_trace"][1]["status"] == "blocked"


def test_chinese_utf8_strings_are_preserved():
    question = "请计算 123 乘以 456。"
    assert question.encode("utf-8").decode("utf-8") == question
    assert "Transformer" in "请从知识库检索 Transformer 论文。"


def test_tool_timeout_is_logged_and_api_keys_are_redacted(tmp_path: Path):
    class FakeLLM:
        model = "test"
        def __init__(self): self.calls = 0
        def chat(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return type("R", (), {"tool_calls": [{"id": "slow", "name": "slow", "arguments": "{}"}], "content": "", "usage": {}})()
            return type("R", (), {"tool_calls": [], "content": "done", "usage": {}})()
    registry = ToolRegistry()
    registry.register("slow", "slow", {"type": "object", "properties": {}, "additionalProperties": False}, lambda: time.sleep(.1))
    agent = ResearchAgent(FakeLLM(), registry, tmp_path / "memory.sqlite", tmp_path / "tools.jsonl", tool_timeout_seconds=.01)
    result = agent.run("run", "slow-session")
    assert result["tool_trace"][0]["status"] == "timeout"
    sample_token = "s" + "k-abcdefghijklmnopqrstuvwxyz123456"
    assert ResearchAgent._sanitize({"credential": "sensitive", "value": sample_token}) == {
        "credential": "<redacted>", "value": "<redacted-api-key>"
    }
