"""LLM clients.

`LLMClient` is a one-method protocol so swapping providers never touches the
agent loop. Two implementations ship in-tree:

* `OpenAICompatClient` — any `/chat/completions` endpoint (OpenAI, DeepSeek,
  Moonshot, DashScope, aiping, vLLM, Ollama, a local gateway...).
* `MockLLMClient` — scripted or callback-driven, used by tests and by the
  demo so the framework is provably runnable with zero API keys.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import requests

from .message import Message, ToolCall


def _sleep(seconds: float) -> None:
    """Indirection so tests can make retry backoff instant."""
    time.sleep(seconds)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """Minimal protocol: turn messages (+tool schemas) into an LLMResponse."""

    name: str = "llm"

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    """Any OpenAI-compatible chat completions endpoint."""

    #: Sent when the caller does not pass `max_tokens` of its own.
    #:
    #: Not cosmetic: aiping.cn returns 503 for a completion body with no
    #: `max_tokens` while the identical body carrying a modest cap returns 200
    #: (12 requests, cleanly separated -- see probes/probe_gateway_max_tokens.py).
    #: Omitting the field is therefore a provider-specific landmine, and the
    #: safe place to defuse it is here, once, rather than at every call site.
    #: Pass `default_max_tokens=None` for providers that reject the field.
    DEFAULT_MAX_TOKENS: int | None = 2048

    #: Statuses worth sending again: the request was well-formed and the answer
    #: is "later", not "no". Any other 4xx means the payload or the credentials
    #: are wrong, and retrying it only delays the real message by seconds.
    #:
    #: Not hypothetical: a forge round against aiping.cn died on a 503, and
    #: because a round is the unit of budget, one transient blip cost the whole
    #: tool. Retrying belongs here, one layer under the round, where a hiccup is
    #: invisible instead of expensive.
    RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        proxies: dict[str, str] | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = DEFAULT_MAX_TOKENS,
        extra_headers: dict[str, str] | None = None,
        max_attempts: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.proxies = proxies
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_headers = extra_headers or {}
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff = retry_backoff
        self.name = f"openai-compat:{model}"

    def _wait_before_retry(self, attempt: int, resp: Any) -> None:
        """Exponential backoff, but never earlier than the server asked for."""
        delay = self.retry_backoff ** attempt
        if resp is not None:
            asked = (getattr(resp, "headers", None) or {}).get("Retry-After")
            if asked:
                try:
                    delay = max(delay, float(asked))
                except ValueError:
                    pass
        _sleep(delay)

    def _to_response(self, data: dict[str, Any]) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=[ToolCall.from_api(tc) for tc in (msg.get("tool_calls") or [])],
            raw=data,
        )

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": [m.to_api() for m in messages],
            "temperature": kwargs.pop("temperature", self.default_temperature),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        if self.default_max_tokens is not None and "max_tokens" not in kwargs:
            payload["max_tokens"] = self.default_max_tokens
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        endpoint = f"{self.base_url}/chat/completions"

        for attempt in range(1, self.max_attempts + 1):
            final = attempt == self.max_attempts
            try:
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=self.proxies,
                )
            except requests.RequestException:
                if final:
                    raise
                self._wait_before_retry(attempt, None)
                continue

            if resp.status_code in self.RETRY_STATUSES and not final:
                self._wait_before_retry(attempt, resp)
                continue

            resp.raise_for_status()
            parsed = self._to_response(resp.json())

            # An empty completion is a provider hiccup, not an answer — retry it
            # like a 503. On the last attempt hand it back anyway: the caller's
            # diagnostics quote `finish_reason`, which explains an empty reply
            # far better than an exception raised on its behalf would.
            if parsed.content or parsed.tool_calls or final:
                return parsed
            self._wait_before_retry(attempt, resp)

        raise AssertionError("unreachable: the loop returns or raises")


class MockLLMClient(LLMClient):
    """Deterministic client for tests and offline demos.

    Two modes:
      * `script=[LLMResponse, ...]` — pop one response per call.
      * `handler=lambda messages, tools, **kw -> LLMResponse` — full control.

    Both record every call in `.calls` for assertions.
    """

    def __init__(
        self,
        script: list[LLMResponse] | None = None,
        handler: Callable[..., LLMResponse] | None = None,
        name: str = "mock",
    ) -> None:
        self.script = list(script or [])
        self.handler = handler
        self.name = name
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        if self.handler is not None:
            return self.handler(messages, tools, **kwargs)
        if self.script:
            return self.script.pop(0)
        return LLMResponse(content="(mock: script exhausted)")


def tool_call(name: str, arguments: dict[str, Any] | None = None, call_id: str | None = None) -> ToolCall:
    """Convenience for building scripted responses."""
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments or {})


__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAICompatClient",
    "MockLLMClient",
    "tool_call",
    "json",
]
