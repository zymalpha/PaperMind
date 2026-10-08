from __future__ import annotations

import sys
import uuid
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["calculator", "summarize", "compare", "knowledge_search", "metadata_and_keywords"])
    selected = parser.parse_args().scenario
    settings = load_settings()
    service = ResearchAssistantService(settings)
    agent = service.create_agent()
    paper_a = "2005.11401_RAG.pdf"
    paper_b = "1706.03762_Attention_Is_All_You_Need.pdf"
    scenarios = [
        ("calculator", "请计算 123 乘以 456。"),
        ("summarize", f"请总结知识库中 {paper_a} 这篇论文的核心创新点。"),
        ("compare", f"请对比 {paper_a} 和 {paper_b} 两篇论文的主要方法。"),
        ("knowledge_search", "请从知识库中检索有关 Transformer 的论文，并提供来源和页码。"),
        ("metadata_and_keywords", f"请提取 {paper_b} 这篇论文的作者信息，并结合其论文内容提取关键词。"),
    ]
    if selected:
        scenarios = [item for item in scenarios if item[0] == selected]
    failed = False
    for label, prompt in scenarios:
        result = agent.run(prompt, session_id=f"stage2-acceptance-{uuid.uuid4().hex[:8]}")
        tools = [item["tool"] for item in result["tool_trace"]]
        print(f"[{label}] input={prompt}")
        print(f"tools={tools} iterations={result['iterations']} tokens={result['usage']['total_tokens']}")
        print(f"answer={result['answer']}")
        print(f"trace_statuses={[item['status'] for item in result['tool_trace']]}")
        print()
        expected = {
            "calculator": {"calculator"},
            "summarize": {"summarize_paper"},
            "compare": {"compare_papers"},
            "knowledge_search": {"knowledge_search"},
            "metadata_and_keywords": {"paper_metadata", "extract_keywords", "knowledge_search"},
        }[label]
        if not expected.intersection(tools) or any(item["status"] not in {"success", "timeout", "error"} for item in result["tool_trace"]):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
