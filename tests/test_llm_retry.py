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
from autoforge.core.llm import LLMError, OpenAICompatClient
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
    kw.setdefault("max_attempts", 3)
    kw.setdefault("retry_backoff", 0.0)
    return OpenAICompatClient("m", "https://gw.invalid/v1", "sk-test", **kw)


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
    except LLMError:
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
    except LLMError as exc:
        assert "503" in str(exc)
        # "handed back, never swallowed" is the property this test is named for,
        # and chaining is how it stays true now that the client translates: the
        # last attempt's HTTPError is `__cause__`, so the status and the response
        # are still reachable by a caller that wants them.
        assert isinstance(exc.__cause__, requests.HTTPError)
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


def test_backoff_is_capped_so_a_long_outage_still_gets_retried_soon(monkeypatch):
    """Uncapped doubling would park the next attempt minutes away."""
    waited: list[float] = []
    monkeypatch.setattr(llm.requests, "post",
                        _Sender(*[_Resp(503)] * 5, _Resp(200, "ok")))
    monkeypatch.setattr(llm, "_sleep", waited.append)
    client = OpenAICompatClient("m", "https://gw.invalid/v1", "sk",
                                max_attempts=6, retry_backoff=10.0,
                                retry_max_delay=25.0)
    assert _ask(client) == "ok"
    assert waited == [10.0, 25.0, 25.0, 25.0, 25.0]   # 100s and 1000s clipped


def test_a_long_503_burst_is_ridden_out(monkeypatch):
    """The live failure: round two died on a burst that outlasted 3 attempts.

    Reproduced against aiping.cn — a forge round raised `503 Service
    Unavailable` after the old 3-attempt/1.5x ladder (~7s total), while the same
    calls returned 200 a minute later. Riding out ~30s is what turns that into a
    success instead of a lost round.
    """
    sender = _wire(monkeypatch, *[_Resp(503)] * 4, _Resp(200, "survived"))
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    assert _ask(_client(max_attempts=6)) == "survived"
    assert sender.attempts == 5


def test_the_default_ladder_outlasts_a_burst(monkeypatch):
    """Defaults are part of the contract: nobody passes retry flags in the CLI."""
    sender = _wire(monkeypatch, *[_Resp(503)] * 3, _Resp(200, "ok"))
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    client = OpenAICompatClient("m", "https://gw.invalid/v1", "sk")
    assert _ask(client) == "ok"
    assert sender.attempts == 4            # the default budget covers 4 tries


def test_max_attempts_of_one_disables_retrying(monkeypatch):
    sender = _wire(monkeypatch, _Resp(503))
    client = OpenAICompatClient("m", "https://gw.invalid/v1", "sk", max_attempts=1)
    try:
        _ask(client)
        raise AssertionError("should have raised on the first failure")
    except LLMError:
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

# ---------------------------------------------------------------------------
# why these expect LLMError
#
# A terminal HTTP failure leaves the client as LLMError, with the original
# requests.HTTPError chained as __cause__. That translation is deliberate: the
# failover chain and the run loop need one thing to catch, and it was added
# after a live forge round died on a raw "HTTPError: 503" that the chain could
# not see. These three assertions were written before the translation and had
# been failing ever since -- on any machine, from the checked-in code. What they
# are about (one attempt for a permanent 400, three for a transient 503, the
# last failure's status preserved) is unchanged.
