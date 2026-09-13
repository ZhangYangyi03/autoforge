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


class LLMResponseError(RuntimeError):
    """The endpoint answered 2xx with a body that is not a completion.

    Raised instead of letting the malformation surface later as
    ``AttributeError: 'str' object has no attribute 'get'`` -- a message that
    names neither the endpoint nor the body, and lands three frames away from
    the reply that caused it. Retrying cannot help: the payload and the
    credentials were accepted, and what came back is not the promised shape.
    """


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    #: The model's private scratchpad, when the provider exposes one
    #: (`reasoning_content` on reasoning models). Not part of the answer, but it
    #: is not noise either: it is where the `max_tokens` budget goes when a
    #: reply comes back empty, so diagnostics that count it can say *why*.
    reasoning: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def finish_reason(self) -> str | None:
        """The provider's stop reason: 'stop', 'length', 'tool_calls', ...

        Worth a name rather than three copies of the same dict walk, because it
        is the only thing that separates "the model finished" from "the model was
        cut off" — a distinction the error paths live or die by.
        """
        raw = self.raw if isinstance(self.raw, dict) else {}
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        # A gateway that fails out-of-band sometimes answers HTTP 200 with
        # ``choices: ["<something went wrong>"]``. Reading the shape rather than
        # assuming it keeps that failure at the body, where it can be reported,
        # instead of at this property, where all it can be is an AttributeError.
        return first.get("finish_reason") if isinstance(first, dict) else None

    @property
    def ran_out_of_budget_thinking(self) -> bool:
        """True when the cap was spent reasoning and no answer ever started.

        Distinguishes a deterministic wall from a transient hiccup: retrying
        this is guaranteed to burn the same budget again for the same nothing.
        """
        if self.content or self.tool_calls:
            return False
        return bool(self.reasoning) and self.finish_reason == "length"

    def describe_shortfall(self) -> str:
        """One line explaining an unusable reply, for the caller's error path."""
        if self.ran_out_of_budget_thinking:
            return (f"the model spent the whole max_tokens budget on "
                    f"reasoning ({len(self.reasoning)} chars of "
                    f"reasoning_content) and never began the answer; raise "
                    f"max_tokens or pick a model that answers directly")
        return f"finish_reason={self.finish_reason!r}, {len(self.content)} chars of content"


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
        max_attempts: int = 5,
        retry_backoff: float = 2.0,
        retry_max_delay: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.proxies = proxies
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_headers = extra_headers or {}
        # Five attempts at a base of two seconds is roughly 30 seconds of riding
        # out a 503 burst. The gateway's outages are bursty and outlast a short
        # ladder: a real forge died on `503 Service Unavailable` in round two
        # even with the three-attempt, 1.5x default, and a probe reproduced the
        # same 503 twice in a row while every cap returned 200 seconds later.
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff = retry_backoff
        self.retry_max_delay = retry_max_delay
        self.name = f"openai-compat:{model}"

    def _wait_before_retry(self, attempt: int, resp: Any) -> None:
        """Exponential backoff, but never earlier than the server asked for."""
        delay = min(self.retry_backoff ** attempt, self.retry_max_delay)
        if resp is not None:
            asked = (getattr(resp, "headers", None) or {}).get("Retry-After")
            if asked:
                try:
                    delay = max(delay, float(asked))
                except ValueError:
                    pass
        _sleep(delay)

    def _to_response(self, data: Any) -> LLMResponse:
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"{self.name} answered with a JSON {type(data).__name__}, "
                f"not an object: {str(data)[:200]!r}"
            )
        choices = data.get("choices")
        if not isinstance(choices, list):
            choices = []
        if choices and not isinstance(choices[0], dict):
            # 200 with choices[0] as a bare string is how an OpenAI-compatible
            # gateway reports a downstream failure. Name it; do not crash on it.
            raise LLMResponseError(
                f"{self.name} returned choices[0] of type "
                f"{type(choices[0]).__name__}, not an object: "
                f"{str(choices[0])[:200]!r}"
            )
        choice = choices[0] if choices else {}
        msg = choice.get("message")
        if not isinstance(msg, dict):
            msg = {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=[ToolCall.from_api(tc) for tc in (msg.get("tool_calls") or [])],
            raw=data,
            reasoning=msg.get("reasoning_content") or "",
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
            try:
                body = resp.json()
            except ValueError as exc:
                # A proxy or gateway in front of the endpoint can answer 200
                # with an HTML error page. Decoding that is not the caller's
                # problem to guess at, and the decode failure names nothing.
                raise LLMResponseError(
                    f"{self.name} returned a non-JSON body "
                    f"(HTTP {resp.status_code}): {(resp.text or '')[:200]!r}"
                ) from exc
            parsed = self._to_response(body)

            # An empty completion is usually a provider hiccup, and hiccups are
            # worth a second send. One exception: if the whole budget went to a
            # reasoning trace, the wall is deterministic and resending just
            # spends the same tokens for the same empty reply.
            if parsed.content or parsed.tool_calls or final:
                return parsed
            if parsed.ran_out_of_budget_thinking:
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
    "LLMResponseError",
    "OpenAICompatClient",
    "MockLLMClient",
    "tool_call",
    "json",
]
