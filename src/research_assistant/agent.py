from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from .llm import DeepSeekClient, LLMError
from .tools import ToolRegistry


AGENT_SYSTEM_PROMPT = """你是科研助理 Agent，使用 ReAct 工具循环完成任务。
根据用户意图选择工具；知识库事实优先使用 knowledge_search，论文比较用 compare_papers，算术用 calculator，当前时间用 current_time。联网工具若禁用会明确返回不可用。
可同时调用互相独立的工具。工具结果是证据，不得编造。工具调用后根据 observation 决定继续调用或给用户最终答复。
不要输出私有的逐 token 思维过程，只输出简短任务计划/结果摘要；引用必须保留知识库返回的来源和页码。
"""


class SessionMemory:
    def __init__(self, path: Path, max_chars: int = 12000, recent_turns: int = 8) -> None:
        self.path, self.max_chars, self.recent_turns = path, max_chars, recent_turns
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, history TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', updated REAL NOT NULL)")

    def load(self, session_id: str) -> tuple[list[dict[str, str]], str]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT history, summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return (json.loads(row[0]), row[1]) if row else ([], "")

    def save(self, session_id: str, history: list[dict[str, str]], summary: str = "") -> None:
        encoded = json.dumps(history, ensure_ascii=False)
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET history=excluded.history,summary=excluded.summary,updated=excluded.updated",
                       (session_id, encoded, summary, time.time()))

    def bounded(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        # Keep complete recent user/assistant pairs. Never persist internal tool protocol as chat memory.
        visible = [item for item in history if item.get("role") in {"user", "assistant"}]
        visible = visible[-self.recent_turns * 2 :]
        budget, output = self.max_chars, []
        for item in reversed(visible):
            size = len(item.get("content", ""))
            if size > budget:
                continue
            output.append(item)
            budget -= size
        return list(reversed(output))


class ResearchAgent:
    def __init__(self, llm: DeepSeekClient, tools: ToolRegistry, memory_path: Path, log_path: Path,
                 max_iterations: int = 6, memory_max_chars: int = 12000, recent_turns: int = 8,
                 tool_timeout_seconds: float = 12, max_parallel_tools: int = 3) -> None:
        self.llm, self.tools = llm, tools
        self.memory = SessionMemory(memory_path, memory_max_chars, recent_turns)
        self.log_path = log_path
        self.max_iterations, self.tool_timeout_seconds, self.max_parallel_tools = max_iterations, tool_timeout_seconds, max_parallel_tools

    def run(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("问题不能为空")
        session_id = session_id or str(uuid.uuid4())
        history, summary = self.memory.load(session_id)
        bounded = self.memory.bounded(history)
        history = bounded
        messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT + (f"\n此前对话摘要：{summary}" if summary else "")}, *history, {"role": "user", "content": question.strip()}]
        trace: list[dict[str, Any]] = []
        seen: set[str] = set()
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer = ""
        for step in range(1, self.max_iterations + 1):
            try:
                result = self.llm.chat(messages, tools=self.tools.definitions, tool_choice="auto")
                self._add_usage(total_usage, result.usage)
            except Exception as exc:
                answer = f"Agent 模型调用失败，可稍后重试。({type(exc).__name__})"
                break
            if not result.tool_calls:
                answer = result.content
                break
            # Every call returned by the model must receive a tool response to
            # preserve the OpenAI tool-call protocol, even if executed in batches.
            calls = result.tool_calls
            assistant_call = {"role": "assistant", "content": result.content or None,
                              "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}} for call in calls]}
            messages.append(assistant_call)
            outcomes = self._execute_calls(calls)
            for call, outcome in zip(calls, outcomes, strict=True):
                signature = hashlib.sha256((call["name"] + call["arguments"]).encode()).hexdigest()
                repeated = signature in seen
                seen.add(signature)
                if repeated:
                    outcome = {"ok": False, "error": "Repeated identical tool call blocked to prevent a loop."}
                entry = {"step": step, "tool": call["name"], "arguments": self._redact_args(call["arguments"]), **outcome}
                trace.append(entry)
                self._write_log(session_id, entry)
                observation = json.dumps(outcome, ensure_ascii=False, default=str)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation[:10000]})
        else:
            answer = "已达到 Agent 最大工具调用轮数。当前执行记录已保留，请缩小任务范围后重试。"
        if not answer:
            answer = "Agent 未能生成最终答复；请检查工具执行记录后重试。"
        history.extend([{"role": "user", "content": question.strip()}, {"role": "assistant", "content": answer}])
        bounded = self.memory.bounded(history)
        if len(bounded) < len(history):
            retained = {id(item) for item in bounded}
            removed = [item for item in history if id(item) not in retained]
            summary = self._compress(summary, removed)
        self.memory.save(session_id, bounded, summary)
        return {"session_id": session_id, "answer": answer, "tool_trace": trace, "usage": total_usage,
                "iterations": max((entry["step"] for entry in trace), default=0)}

    def _compress(self, previous: str, removed: list[dict[str, str]]) -> str:
        transcript = "\n".join(f"{item['role']}: {item.get('content', '')}" for item in removed)
        prompt = f"已有摘要：{previous or '无'}\n\n需要压缩的旧对话：\n{transcript[:8000]}\n\n请用不超过 600 字中文保留用户目标、已确认事实、论文 ID 和未完成事项。"
        try:
            result = self.llm.chat([{"role": "system", "content": "你负责压缩会话记忆，不添加新事实。"}, {"role": "user", "content": prompt}], max_tokens=1200)
            return result.content[:2000]
        except Exception:
            return (previous + "\n" + transcript[-1500:])[-2000:]

    def _execute_calls(self, calls: list[dict[str, str]]) -> list[dict[str, Any]]:
        def execute(call):
            started = time.perf_counter()
            try:
                arguments = json.loads(call["arguments"] or "{}")
                result = self.tools.execute(call["name"], arguments)
                return {"status": "success", "ok": True, "result": result,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000)}
            except Exception as exc:
                return {"status": "error", "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000)}
        if len(calls) == 1:
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(execute, calls[0])
            try:
                return [future.result(timeout=self.tool_timeout_seconds)]
            except FutureTimeout:
                future.cancel()
                return [{"status": "timeout", "ok": False, "error": "Tool execution timed out.", "elapsed_ms": round(self.tool_timeout_seconds * 1000)}]
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        outcomes = []
        for start in range(0, len(calls), self.max_parallel_tools):
            batch = calls[start : start + self.max_parallel_tools]
            pool = ThreadPoolExecutor(max_workers=len(batch))
            futures = [pool.submit(execute, call) for call in batch]
            try:
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=self.tool_timeout_seconds))
                    except FutureTimeout:
                        future.cancel()
                        outcomes.append({"status": "timeout", "ok": False, "error": "Tool execution timed out.", "elapsed_ms": round(self.tool_timeout_seconds * 1000)})
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        return outcomes

    def _write_log(self, session_id: str, entry: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = self._sanitize({"timestamp": time.time(), "session_id": session_id, **entry})
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _redact_args(raw: str) -> dict[str, Any]:
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"invalid_json": True}
        return ResearchAgent._sanitize(args)

    @staticmethod
    def _sanitize(value: Any, key: str = "") -> Any:
        if any(secret in key.lower() for secret in ("key", "token", "secret", "password", "credential")):
            return "<redacted>"
        if isinstance(value, dict):
            return {str(child_key): ResearchAgent._sanitize(child, str(child_key)) for child_key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [ResearchAgent._sanitize(item) for item in value]
        if isinstance(value, str):
            clipped = value[:500] + "…" if len(value) > 500 else value
            return re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "<redacted-api-key>", clipped)
        return value

    @staticmethod
    def _add_usage(total: dict[str, int], current: dict[str, int]) -> None:
        for key in total:
            total[key] += int(current.get(key, 0))
