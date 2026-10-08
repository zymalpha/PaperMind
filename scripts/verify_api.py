from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings  # noqa: E402
from research_assistant.llm import DeepSeekClient, LLMError  # noqa: E402


def main() -> int:
    settings = load_settings()
    try:
        result = DeepSeekClient(
            settings.deepseek_api_key,
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_timeout_seconds,
            settings.llm_max_retries,
        ).chat([{"role": "user", "content": "Reply with exactly: OK"}], temperature=0, max_tokens=1024)
    except (ValueError, LLMError) as exc:
        print(f"API verification failed: {exc}")
        return 1
    print(f"API verification succeeded: model={result.model}, response_received={bool(result.content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
