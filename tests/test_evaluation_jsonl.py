from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_system_evaluation import load_jsonl as load_analysis_jsonl
from scripts.run_system_evaluation import load_jsonl as load_runner_jsonl


def test_evaluation_jsonl_readers_preserve_unicode_line_separator(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    expected = {"id": "Q1", "excerpt": "first section\u2028second section"}
    rows = [expected] + [{"id": f"Q{index}", "excerpt": "ordinary text"} for index in range(2, 51)]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    assert load_runner_jsonl(path) == rows
    assert load_analysis_jsonl(path) == rows
