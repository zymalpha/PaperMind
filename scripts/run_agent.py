from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.service import ResearchAssistantService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ReAct research agent")
    parser.add_argument("question")
    parser.add_argument("--session-id", help="Reuse a conversation session")
    args = parser.parse_args()
    result = ResearchAssistantService(load_settings()).create_agent().run(args.question, args.session_id)
    print(result["answer"])
    print(f"session={result['session_id']} iterations={result['iterations']} tokens={result['usage']['total_tokens']}")
    for record in result["tool_trace"]:
        print(f"{record['step']}. {record['tool']} [{record['status']}] {record['elapsed_ms']}ms")
    print("Full redacted tool trace: data/logs/agent_tools.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
