"""Retrying transient provider failures, one layer under the forge round.

The trigger was real: a forge round against aiping.cn came back
`HTTPError: 503 Server Error`, and because a round is the unit of budget, that
one blip cost the entire tool. Nothing was wrong with the request, the key, or
the model -- the same payload succeeds on the next send.

So the client retries. What must stay true:

* a transient status is invisible to the caller if a later attempt works;
* a *permanent* failure (400/401/403/404) is surfaced immediately, because
  retrying a malformed request just delays the message that would fix it;
* the last attempt's outcome is handed back, never swallowed.
"""
from __future__ import annotations

import requests

from autoforge.core import llm
from autoforge.core.llm import OpenAICompatClient
from autoforge.core.message import Message


class _Resp:
    def __init__(self, status: int = 200, content: str | None = "ok",
                 headers: dict | None = None) -> None:
        self.status_code = status
        self._content = content
        self.headers = headers or {}

    def json(self) -> dict:
        message: dict = {}
        if self._content is not None:
            message["content"] = self._content
        return {"choices": [{"message": message, "finish_reason": "stop"}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class _Sender:
    """Stands in for `requests.post`, replaying a scripted list of responses."""

    def __init__(self, *steps) -> None:
        self.steps = list(steps)
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        step = self.steps.pop(0) if self.steps else _Resp()
        if isinstance(step, Exception):
            raise step
        return step

    @property
    def attempts(self) -> int:
        return len(self.calls)


def _client(**kw) -> OpenAICompatClient:
    return OpenAICompatClient("m", "https://gw.invalid/v1", "sk-test",
                              max_attempts=3, retry_backoff=0.0, **kw)


def _wire(monkeypatch, *steps) -> _Sender:
    sender = _Sender(*steps)
    monkeypatch.setattr(llm.requests, "post", sender)
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)   # never really wait
    return sender


def _ask(client: OpenAICompatClient) -> str:
    return client.chat([Message(role="user", content="hi")]).content


def test_a_transient_503_is_retried_and_the_caller_never_notices(monkeypatch):
    sender = _wire(monkeypatch, _Resp(503), _Resp(200, "recovered"))
    assert _ask(_client()) == "recovered"
    assert sender.attempts == 2


def test_a_permanent_400_is_not_retried(monkeypatch):
    """Retrying a bad payload only postpones the message that fixes it."""
    sender = _wire(monkeypatch, _Resp(400))
    try:
        _ask(_client())
        raise AssertionError("a 400 should have raised")
    except requests.HTTPError:
        pass
    assert sender.attempts == 1


def test_an_empty_completion_is_treated_as_a_hiccup(monkeypatch):
    sender = _wire(monkeypatch, _Resp(200, ""), _Resp(200, "second time"))
    assert _ask(_client()) == "second time"
    assert sender.attempts == 2


def test_a_dropped_connection_is_retried(monkeypatch):
    sender = _wire(monkeypatch, requests.ConnectionError("reset"), _Resp(200, "back"))
    assert _ask(_client()) == "back"
    assert sender.attempts == 2


def test_exhausting_the_attempts_re_raises_the_real_error(monkeypatch):
    sender = _wire(monkeypatch, _Resp(503), _Resp(503), _Resp(503))
    try:
        _ask(_client())
        raise AssertionError("three 503s should have raised")
    except requests.HTTPError as exc:
        assert "503" in str(exc)
    assert sender.attempts == 3


def test_the_last_empty_answer_is_returned_rather_than_raised(monkeypatch):
    """An empty body still carries `finish_reason`, which explains it better."""
    sender = _wire(monkeypatch, _Resp(200, ""), _Resp(200, ""), _Resp(200, ""))
    assert _ask(_client()) == ""
    assert sender.attempts == 3


def test_retry_after_is_honoured_when_the_server_asks_for_more(monkeypatch):
    waited: list[float] = []
    recipient = _Sender(_Resp(429, headers={"Retry-After": "7"}), _Resp(200, "ok"))
    monkeypatch.setattr(llm.requests, "post", recipient)
    monkeypatch.setattr(llm, "_sleep", waited.append)
    assert _ask(_client()) == "ok"
    assert waited == [7.0]                 # server's 7s beats the 0.0 backoff


def test_backoff_grows_between_attempts(monkeypatch):
    waited: list[float] = []
    monkeypatch.setattr(llm.requests, "post",
                        _Sender(_Resp(500), _Resp(500), _Resp(200, "ok")))
    monkeypatch.setattr(llm, "_sleep", waited.append)
    client = OpenAICompatClient("m", "https://gw.invalid/v1", "sk",
                                max_attempts=3, retry_backoff=2.0)
    assert _ask(client) == "ok"
    assert waited == [2.0, 4.0]            # 2**1, then 2**2


def test_max_attempts_of_one_disables_retrying(monkeypatch):
    sender = _wire(monkeypatch, _Resp(503))
    client = OpenAICompatClient("m", "https://gw.invalid/v1", "sk", max_attempts=1)
    try:
        _ask(client)
        raise AssertionError("should have raised on the first failure")
    except requests.HTTPError:
        pass
    assert sender.attempts == 1


def test_a_tool_call_only_reply_counts_as_an_answer(monkeypatch):
    """No text but a tool call is a complete answer, not a hiccup."""
    class _Tool(_Resp):
        def json(self):
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [{"id": "1", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}],
            }, "finish_reason": "tool_calls"}]}

    sender = _wire(monkeypatch, _Tool())
    resp = _client().chat([Message(role="user", content="hi")])
    assert resp.content == "" and len(resp.tool_calls) == 1
    assert sender.attempts == 1
