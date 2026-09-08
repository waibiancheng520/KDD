from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI, RateLimitError

_RATE_LIMIT_MAX_ATTEMPTS = 3


def _looks_like_tools_unsupported(exc: Exception) -> bool:
    """True when the endpoint rejected the request because it has no tool support."""
    text = str(exc).lower()
    return "tool" in text and any(
        marker in text
        for marker in ("not support", "unsupported", "unknown parameter", "unrecognized", "invalid parameter")
    )


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Honour the server's retryDelay when present, else exponential backoff."""
    match = re.search(r"'retryDelay': '(\d+)s'", str(exc))
    if match is not None:
        return min(float(match.group(1)) + 1.0, 60.0)
    return min(2.0 ** attempt, 60.0)


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage], tools: list[dict[str, Any]] | None = None) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        native_tools: bool = False,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.native_tools = native_tools
        # Build the client once and reuse it. Creating a new OpenAI client on every
        # call leaks its connection pool, which piles up CLOSE-WAIT sockets behind a
        # local proxy and eventually makes every request hang.
        self._client = (
            OpenAI(
                api_key=api_key,
                base_url=self.api_base,
                timeout=60.0,
                max_retries=2,
            )
            if api_key
            else None
        )
        # Flipped on if the endpoint rejects the `tools` parameter, so we stop
        # re-sending it on every later step.
        self._tools_unsupported = False

    @property
    def supports_native_tools(self) -> bool:
        return self.native_tools and not self._tools_unsupported

    def complete(self, messages: list[ModelMessage], tools: list[dict[str, Any]] | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = self._client
        assert client is not None

        payload = [{"role": message.role, "content": message.content} for message in messages]
        # Native function calling removes a whole class of failures: the model emits a
        # structured tool call instead of prose-wrapped JSON we have to re-parse, and
        # arguments no longer need hand-escaped quotes/newlines. If the endpoint does
        # not support tools we fall back to text mode for the rest of the run.
        use_tools = self.native_tools and bool(tools) and not self._tools_unsupported
        extra: dict[str, Any] = {"tools": tools, "tool_choice": "required"} if use_tools else {}

        # Rate limits (429) are transient: wait for the delay the server asks for and
        # retry, instead of killing the whole task on the first throttled request.
        last_error: Exception | None = None
        for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
            _t0 = time.time()
            print(f"    [api] 请求中 (第{attempt+1}次尝试, 输入{sum(len(m['content']) for m in payload)}字符)...",
                  file=sys.stderr, flush=True)
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=self.temperature,
                    **extra,
                )
                print(f"    [api] 完成 {time.time()-_t0:.1f}秒", file=sys.stderr, flush=True)
                break
            except RateLimitError as exc:
                last_error = exc
                if attempt == _RATE_LIMIT_MAX_ATTEMPTS - 1:
                    raise RuntimeError(f"Model request failed after retries: {exc}") from exc
                _d = _retry_delay_seconds(exc, attempt)
                print(f"    [api] 限流429，等待{_d:.0f}秒后重试", file=sys.stderr, flush=True)
                time.sleep(_d)
            except APIError as exc:
                if use_tools and _looks_like_tools_unsupported(exc):
                    print("    [api] 端点不支持 tools，降级为文本模式", file=sys.stderr, flush=True)
                    self._tools_unsupported = True
                    use_tools = False
                    extra = {}
                    continue
                print(f"    [api] 出错 {time.time()-_t0:.1f}秒: {type(exc).__name__}", file=sys.stderr, flush=True)
                raise RuntimeError(f"Model request failed: {exc}") from exc
        else:
            raise RuntimeError(f"Model request failed after retries: {last_error}")

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        message = choices[0].message
        content = message.content

        # A native tool call is re-emitted as the canonical
        # {"thought","action","action_input"} envelope, so the ReAct loop keeps a
        # single parsing path (and the text fallback keeps working unchanged).
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            raw_arguments = getattr(call.function, "arguments", "") or "{}"
            try:
                action_input = json.loads(raw_arguments)
            except json.JSONDecodeError:
                action_input = {}
            if not isinstance(action_input, dict):
                action_input = {}
            # `thought` is a schema-level argument, not a tool input: lift it out so
            # handlers receive only their own parameters.
            thought = action_input.pop("thought", None)
            if not isinstance(thought, str) or not thought.strip():
                thought = content if isinstance(content, str) else ""
            return json.dumps(
                {
                    "thought": thought,
                    "action": call.function.name,
                    "action_input": action_input,
                },
                ensure_ascii=False,
            )

        if not isinstance(content, str) or not content.strip():
            # An empty completion (filtered / truncated) must not kill the whole task.
            # Return a parseable no-op so the ReAct loop records a failed step and retries.
            finish_reason = getattr(choices[0], "finish_reason", None)
            return json.dumps(
                {
                    "thought": f"Empty model response (finish_reason={finish_reason}). Retrying.",
                    "action": "__empty_response__",
                    "action_input": {},
                }
            )
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage], tools: list[dict[str, Any]] | None = None) -> str:
        del messages, tools
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
