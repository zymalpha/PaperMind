from __future__ import annotations

import time
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


class LLMError(RuntimeError):
    """User-safe error raised when the model service cannot complete a request."""


@dataclass(slots=True)
class LLMResult:
    content: str
    model: str
    usage: dict[str, int]
    tool_calls: list[dict[str, str]]


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        timeout_seconds: float = 45,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY，请在 .env 中设置。")
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    request["tools"] = list(tools)
                    request["tool_choice"] = tool_choice or "auto"
                response = self._client.chat.completions.create(
                    **request,
                )
                content = (response.choices[0].message.content or "").strip()
                raw_tool_calls = response.choices[0].message.tool_calls or []
                parsed_tool_calls = [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in raw_tool_calls
                ]
                if not content and not parsed_tool_calls:
                    raise LLMError("模型返回了空内容")
                usage = response.usage
                return LLMResult(
                    content=content,
                    model=response.model or self.model,
                    usage={
                        "prompt_tokens": int(usage.prompt_tokens) if usage else 0,
                        "completion_tokens": int(usage.completion_tokens) if usage else 0,
                        "total_tokens": int(usage.total_tokens) if usage else 0,
                    },
                    tool_calls=parsed_tool_calls,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, LLMError) as exc:
                last_error = exc
                retryable = not isinstance(exc, APIStatusError) or exc.status_code == 429 or exc.status_code >= 500
                if attempt < self.max_retries and retryable:
                    time.sleep(min(2**attempt, 4))
                elif not retryable:
                    break
        raise self._safe_error(last_error)

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            emitted = False
            stream = None
            watcher_stop = threading.Event()
            try:
                if cancel_event and cancel_event.is_set():
                    return
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                def close_on_cancel() -> None:
                    while not watcher_stop.wait(0.05):
                        if cancel_event and cancel_event.is_set():
                            try:
                                stream.close()
                            except Exception:
                                pass
                            return

                if cancel_event:
                    threading.Thread(target=close_on_cancel, daemon=True).start()
                for event in stream:
                    if cancel_event and cancel_event.is_set():
                        return
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta:
                        emitted = True
                        yield delta
                if not emitted:
                    if cancel_event and cancel_event.is_set():
                        return
                    raise LLMError("模型返回了空内容")
                return
            except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, LLMError) as exc:
                if cancel_event and cancel_event.is_set():
                    return
                last_error = exc
                # Once partial text is visible, automatic retries would duplicate the answer.
                retryable = not isinstance(exc, APIStatusError) or exc.status_code == 429 or exc.status_code >= 500
                if emitted or attempt >= self.max_retries or not retryable:
                    break
                time.sleep(min(2**attempt, 4))
            finally:
                watcher_stop.set()
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        raise self._safe_error(last_error)

    @staticmethod
    def _safe_error(error: Exception | None) -> LLMError:
        if isinstance(error, APITimeoutError):
            return LLMError("DeepSeek 请求超时，请稍后重试。")
        if isinstance(error, RateLimitError):
            return LLMError("DeepSeek 请求过于频繁或额度受限，请稍后重试。")
        if isinstance(error, APIStatusError):
            return LLMError(f"DeepSeek 服务返回 HTTP {error.status_code}，请检查模型与账户配置。")
        if isinstance(error, APIConnectionError):
            return LLMError("无法连接 DeepSeek 服务，请检查网络。")
        return LLMError(str(error or "DeepSeek 服务调用失败。"))
