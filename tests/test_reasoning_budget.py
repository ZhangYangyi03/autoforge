"""The reasoning-content wall: a model that thinks until the budget is gone.

Found live, not imagined. Against aiping.cn / `DeepSeek-V4.1-Flash`, a
forge-shaped prompt came back HTTP 200 with `content` empty and
`finish_reason='length'` — twice in a row, two rounds, nothing usable. The
cause was invisible until the raw body was inspected (`probes/probe_gateway_empty.py`):

    cap 3000   finish='length'  content=0c  reasoning_content=10767c
    cap 8000   finish='length'  content=0c  reasoning_content=27092c

The reasoning trace grows to fill whatever cap it is given, so the answer never
starts. Raising the cap is not a fix, and none of six documented suppression
flags changed it (`probes/probe_gateway_thinking.py`).

Four things must therefore hold, and each is a test below:

1. the trace is *captured* — a reply is not described as "0 chars" when 10k
   characters of explanation came with it;
2. the wall is *recognised* — empty content plus a trace plus `length` is not
   confused with a transient empty reply;
3. it is not *retried*, because resending spends the same budget for the same
   nothing;
4. it stops the *round loop*, because round two relives round one exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from autoforge.core import llm
from autoforge.core.llm import LLMResponse, MockLLMClient, OpenAICompatClient
from autoforge.core.message import Message
from autoforge.forge.generator import LLMToolGenerator, UnrecoverableGeneration
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.core.llm import MockLLMClient as _Mock
from autoforge.tools.registry import ToolRegistry


def thinking_only(trace: str = "reasoning " * 40, finish: str = "length") -> LLMResponse:
    """The reply that started all this: 200 OK, no answer, trace wide open."""
    return LLMResponse(
        content="",
        raw={"choices": [{"message": {"content": "", "reasoning_content": trace},
                          "finish_reason": finish}]},
        reasoning=trace,
    )


# -- 1. the trace is captured ------------------------------------------
class _Resp:
    status_code = 200

    def __init__(self, payload: dict, headers: dict | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


def test_reasoning_content_is_kept_not_discarded(monkeypatch):
    payload = {"choices": [{"message": {"content": "", "reasoning_content": "hmm"},
                            "finish_reason": "length"}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(payload))
    resp = OpenAICompatClient("m", "https://gw.invalid/v1", "sk").chat(
        [Message(role="user", content="hi")])
    assert resp.reasoning == "hmm"


def test_a_budget_wall_is_described_by_what_it_actually_did():
    msg = thinking_only("x" * 10767).describe_shortfall()
    assert "10767" in msg and "reasoning" in msg
    assert "never began the answer" in msg
    # It must NOT be described as a truncation of an answer that never existed.
    assert "cut off mid-JSON" not in msg


def test_a_normal_truncated_answer_is_still_described_as_truncation():
    resp = LLMResponse(content='{"name": "x", "code": "def f(',
                       raw={"choices": [{"message": {}, "finish_reason": "length"}]})
    assert "finish_reason" in resp.describe_shortfall()
    assert not resp.ran_out_of_budget_thinking


def test_a_short_prompt_that_answers_is_not_a_wall():
    resp = LLMResponse(content="READY", reasoning="brief",
                       raw={"choices": [{"message": {}, "finish_reason": "stop"}]})
    assert not resp.ran_out_of_budget_thinking


def test_finish_stop_with_a_trace_is_not_a_wall():
    """Reasoning that *ended* is a working model, not a wall."""
    resp = LLMResponse(content="", reasoning="long trace",
                       raw={"choices": [{"message": {}, "finish_reason": "stop"}]})
    assert not resp.ran_out_of_budget_thinking


# -- 3 and 4. not retried, and it stops the loop ------------------------
def test_the_client_does_not_resend_a_budget_wall(monkeypatch):
    """Resending is guaranteed to burn the same budget for the same nothing."""
    attempts = []

    def post(*a, **k):
        attempts.append(1)
        return _Resp({"choices": [{"message": {"content": "",
                                               "reasoning_content": "y" * 500},
                                   "finish_reason": "length"}]})

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    resp = OpenAICompatClient("m", "https://gw.invalid/v1", "sk",
                              max_attempts=3).chat([Message(role="user", content="hi")])
    assert len(attempts) == 1                 # gave up immediately, told the truth
    assert resp.ran_out_of_budget_thinking


def test_a_plain_empty_reply_is_still_retried(monkeypatch):
    """The wall must not disable ordinary retrying."""
    attempts = []

    def post(*a, **k):
        attempts.append(1)
        empty = {"choices": [{"message": {}, "finish_reason": "stop"}]}
        if len(attempts) == 1:
            return _Resp(empty)
        return _Resp({"choices": [{"message": {"content": "second time"},
                                   "finish_reason": "stop"}]})

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    resp = OpenAICompatClient("m", "https://gw.invalid/v1", "sk").chat(
        [Message(role="user", content="hi")])
    assert len(attempts) == 2 and resp.content == "second time"


def test_the_generator_refuses_to_retry_a_budget_wall():
    gen = LLMToolGenerator(MockLLMClient(handler=lambda m, t=None, **k: thinking_only()),
                           max_tokens=3000)
    try:
        gen.generate("normalise ISBN identifiers")
        raise AssertionError("should not have produced a tool")
    except UnrecoverableGeneration as exc:
        assert "reasoning" in str(exc)
        assert "never began the answer" in str(exc)


def test_the_round_loop_stops_after_a_budget_wall():
    """max_rounds=3, but round one proved the model cannot answer at all."""
    llm_client = MockLLMClient(handler=lambda m, t=None, **k: thinking_only())
    sandbox = Sandbox(timeout=8)
    pipeline = ForgePipeline(
        LLMToolGenerator(llm_client, max_tokens=3000),
        ToolVerifier(llm_client, sandbox=sandbox), ToolRegistry(),
        sandbox=sandbox,
        config=ForgeConfig(promote_on_pass=True, max_rounds=3),
    )
    res = pipeline.forge("I need to normalise ISBN identifiers")
    assert not res.ok
    assert len(res.attempts) == 1
    assert "UnrecoverableGeneration" in res.attempts[0].error
    assert len(llm_client.calls) == 1


def test_an_ordinary_bad_answer_still_uses_the_whole_round_budget():
    """The wall's early exit must not become a general give-up."""
    calls = []

    def handler(messages, tools=None, **kwargs):
        calls.append(1)
        return LLMResponse(content="not json at all",
                           raw={"choices": [{"message": {}, "finish_reason": "stop"}]})

    llm_client = MockLLMClient(handler=handler)
    sandbox = Sandbox(timeout=8)
    pipeline = ForgePipeline(
        LLMToolGenerator(llm_client, max_tokens=3000),
        ToolVerifier(llm_client, sandbox=sandbox), ToolRegistry(),
        sandbox=sandbox,
        config=ForgeConfig(promote_on_pass=True, max_rounds=3),
    )
    res = pipeline.forge("I need to normalise ISBN identifiers")
    assert len(res.attempts) == 3            # all three rounds were spent
    assert len(calls) >= 3
