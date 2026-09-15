"""The output budget: why a small cap failed, and what guards a generous one.

Found live. Against aiping.cn / `DeepSeek-V4.1-Flash`, the generator's own prompt
at a cap of 3000 returned HTTP 200 with

    finish_reason='length'   content=0 chars   reasoning_content=10901 chars

-- the reasoning trace spent the whole budget and the answer never began, which
reaches the operator as "the model produced no answer".

`tests/test_reasoning_budget.py` drew a wider conclusion from two such samples, at
caps of 3000 and 8000, both empty: that the trace "grows to fill whatever cap it is
given", and therefore that raising the cap cannot help. Two empty samples at 3000
and 8000 do not distinguish "scales to fill" from "is naturally longer than 8000",
and only the first reading makes the cap unfixable. The evidence for the second
reading is that the same gateway, same model, same key, has been answering requests
that carry 65536-128000 all along (202 request dumps under the author's profile,
every one of them this same model) -- so the budget was the lever, not the model.

Hence two changes, and each is tested below:

* the default is generous enough to clear a trace (`DEFAULT_MAX_TOKENS`), and every
  caller that had its own number -- the generator, the CLI, the wizard -- uses it;
* because generous is a 400 on a stricter provider (api.deepseek.com stops at
  8192), a rejection that names `max_tokens` is answered with a smaller number
  rather than rethrown. That is what makes a big default safe to ship.
"""
from __future__ import annotations

import argparse

import pytest
import requests

from autoforge import cli
from autoforge.core import llm
from autoforge.core.llm import DEFAULT_MAX_TOKENS, OpenAICompatClient
from autoforge.core.message import Message
from autoforge.forge.generator import LLMToolGenerator

MSGS = [Message(role="user", content="hi")]


class _Resp:
    """A response good enough for the client's status, body and error paths."""

    def __init__(self, status: int = 200, content: str | None = "ok",
                 text: str = "") -> None:
        self.status_code = status
        self._content = content
        self.text = text
        self.headers: dict = {}

    def json(self) -> dict:
        message = {} if self._content is None else {"content": self._content}
        return {"choices": [{"message": message, "finish_reason": "stop"}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class _Sender:
    def __init__(self, *steps) -> None:
        self.steps = list(steps)
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        # Snapshot the body: the client mutates `payload` in place when it clamps,
        # so holding the dict itself would rewrite history and hide the clamp.
        self.calls.append({"url": url, **kwargs,
                           "json": dict(kwargs.get("json") or {})})
        step = self.steps.pop(0) if self.steps else _Resp()
        if isinstance(step, Exception):
            raise step
        return step

    @property
    def attempts(self) -> int:
        return len(self.calls)

    @property
    def caps(self) -> list:
        return [c["json"].get("max_tokens") for c in self.calls]


def _wire(monkeypatch, *steps) -> _Sender:
    sender = _Sender(*steps)
    monkeypatch.setattr(llm.requests, "post", sender)
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    return sender


def _client(**kw) -> OpenAICompatClient:
    kw.setdefault("max_attempts", 3)
    kw.setdefault("retry_backoff", 0.0)
    return OpenAICompatClient("m", "https://gw.invalid/v1", "sk-test", **kw)


# ======================================================================
# the default, and everyone who had their own copy of it
# ======================================================================
def test_the_default_clears_the_trace_that_ate_the_small_cap():
    """The 8000 sample was empty too, so the default must be well past it."""
    assert DEFAULT_MAX_TOKENS is not None
    assert DEFAULT_MAX_TOKENS >= 2 * 8000


def test_the_forge_generator_ships_that_cap():
    """The path that failed: no explicit cap of its own to drift from."""
    assert LLMToolGenerator(llm.MockLLMClient()).max_tokens == DEFAULT_MAX_TOKENS


def test_the_cli_default_is_the_same_cap():
    """One number, one place. Three copies is how 3000 survived the first fix.

    `strict=False` because the resolver refuses to run without a key and a missing
    key is not what this asserts: the cap is resolved before that check either way.
    """
    args = argparse.Namespace(model=None, base_url=None, api_key=None, fast=False,
                              max_tokens=None, no_proxy=False, proxy=None)
    cfg, src = cli._resolve(args, strict=False)
    assert cfg["max_tokens"] == DEFAULT_MAX_TOKENS
    assert src["max_tokens"] == "default"


def test_an_explicit_flag_still_wins():
    """A generous default is not an override of the operator's own number."""
    args = argparse.Namespace(model=None, base_url=None, api_key=None, fast=False,
                              max_tokens=1234, no_proxy=False, proxy=None)
    cfg, src = cli._resolve(args, strict=False)
    assert cfg["max_tokens"] == 1234 and src["max_tokens"] == "flag"


# ======================================================================
# a provider whose ceiling is below ours must not cost the round
# ======================================================================
DEEPSEEK_400 = ("Invalid max_tokens value, the valid range of max_tokens "
                "is [1, 8192]")
APING_LIKE_400 = "max_tokens must be <= 16384 for this model"


def test_a_rejected_cap_is_resent_at_the_ceiling_the_error_names(monkeypatch):
    sender = _wire(monkeypatch, _Resp(400, text=DEEPSEEK_400), _Resp(200, "hi"))
    assert _client().chat(MSGS).content == "hi"
    assert sender.caps == [DEFAULT_MAX_TOKENS, 8192]
    assert sender.attempts == 2


def test_the_ceiling_is_read_from_either_wording(monkeypatch):
    sender = _wire(monkeypatch, _Resp(400, text=APING_LIKE_400), _Resp(200, "hi"))
    assert _client().chat(MSGS).content == "hi"
    assert sender.caps == [DEFAULT_MAX_TOKENS, 16384]


def test_a_message_naming_no_ceiling_halves_the_cap(monkeypatch):
    sender = _wire(monkeypatch, _Resp(400, text="invalid max_tokens"), _Resp(200, "hi"))
    assert _client().chat(MSGS).content == "hi"
    assert sender.caps == [DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS // 2]


def test_the_replacement_attempt_is_not_taken_from_the_retry_budget(monkeypatch):
    """Nothing has been learned about the endpoint; only about its ceiling."""
    sender = _wire(monkeypatch, _Resp(400, text=DEEPSEEK_400),
                   _Resp(503), _Resp(200, "hi"))
    assert _client(max_attempts=3).chat(MSGS).content == "hi"
    assert sender.caps == [DEFAULT_MAX_TOKENS, 8192, 8192]
    assert sender.attempts == 3


def test_the_clamp_is_not_an_escape_hatch_from_the_attempt_budget(monkeypatch):
    """One clamp only. A second rejection is a real answer, not a puzzle."""
    sender = _wire(monkeypatch, _Resp(400, text=DEEPSEEK_400),
                   _Resp(400, text=DEEPSEEK_400), _Resp(200, "hi"))
    with pytest.raises(requests.HTTPError):
        _client(max_attempts=2).chat(MSGS)
    assert sender.attempts == 2


def test_an_explicit_caller_cap_is_clamped_too(monkeypatch):
    """The CLI can push a bigger number than the provider allows."""
    sender = _wire(monkeypatch, _Resp(400, text=DEEPSEEK_400), _Resp(200, "hi"))
    assert _client().chat(MSGS, max_tokens=99999).content == "hi"
    assert sender.caps == [99999, 8192]


# ======================================================================
# ... and must not become a general retry of every 4xx
# ======================================================================
def test_a_400_about_anything_else_is_still_fatal(monkeypatch):
    sender = _wire(monkeypatch, _Resp(400, text="unknown model: m"))
    with pytest.raises(requests.HTTPError):
        _client().chat(MSGS)
    assert sender.attempts == 1


def test_a_401_that_mentions_max_tokens_is_still_fatal(monkeypatch):
    """The status decides, not the wording: credentials are not a cap problem."""
    sender = _wire(monkeypatch, _Resp(401, text=DEEPSEEK_400))
    with pytest.raises(requests.HTTPError):
        _client().chat(MSGS)
    assert sender.attempts == 1


def test_a_ceiling_below_the_floor_is_reported_rather_than_chased(monkeypatch):
    """Under this the reply cannot hold an envelope, so keep the provider's word."""
    sender = _wire(monkeypatch, _Resp(400, text="max_tokens is in [1, 256]"))
    with pytest.raises(requests.HTTPError):
        _client().chat(MSGS)
    assert sender.attempts == 1


def test_no_cap_on_the_wire_means_nothing_to_clamp(monkeypatch):
    """`default_max_tokens=None` opts out of the field entirely; leave it alone."""
    sender = _wire(monkeypatch, _Resp(400, text=DEEPSEEK_400))
    with pytest.raises(requests.HTTPError):
        _client(default_max_tokens=None).chat(MSGS)
    assert sender.caps == [None]
