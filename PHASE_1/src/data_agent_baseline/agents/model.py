from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI, RateLimitError

_RATE_LIMIT_MAX_ATTEMPTS = 3


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
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
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

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = self._client
        assert client is not None

        payload = [{"role": message.role, "content": message.content} for message in messages]

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
                print(f"    [api] 出错 {time.time()-_t0:.1f}秒: {type(exc).__name__}", file=sys.stderr, flush=True)
                raise RuntimeError(f"Model request failed: {exc}") from exc
        else:
            raise RuntimeError(f"Model request failed after retries: {last_error}")

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
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

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
