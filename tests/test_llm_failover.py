"""The failover chain: what it must do, and the four ways it must not lie.

The failure this exists for was measured, not imagined. aiping.cn answered six
consecutive calls in ~1s each, then went 503 for minutes; a single blip inside a
forge round costs the whole tool, because the round is the unit of budget. Retry
alone rides out a burst. It cannot ride out an outage, and no amount of ladder
tuning changes that -- only a second endpoint does.

The subtle part is not "try the other one". It is what must NOT happen:

  * a failing endpoint must not be tried first on every subsequent call, or the
    cooldown achieves nothing and each call pays a dead endpoint's timeout;
  * the operator's stop must not be reinterpreted as an outage and re-asked on
    another endpoint -- that is precisely what should_abort is for;
  * a bug in our own payload must not be reported as "every endpoint failed",
    which would make this class the place real defects go to look like weather;
  * a chain of one endpoint must not change behaviour at all.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import (FailoverClient, LLMAborted, LLMError, LLMResponse,
                                LLMResponseError)
from autoforge.core.message import Message


class FakeClient:
    """A client whose only behaviour is the one under test."""

    def __init__(self, name, *, answer="ok", raises=None, raises_once=None):
        self.name = name
        self._answer = answer
        self._raises = raises
        self._raises_once = raises_once
        self.calls = 0
        self.abort_check = None
        self.usage_total = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                            "completion_tokens": 0}

    @property
    def cache_hit_rate(self):
        return 0.0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self._raises_once is not None and self.calls == 1:
            raise self._raises_once
        if self._raises is not None:
            raise self._raises
        self.usage_total["prompt_tokens"] += 10
        return LLMResponse(content=self._answer, raw={})


def _msgs():
    return [Message.user("hi")]


# -- the plain case ------------------------------------------------------
def test_a_healthy_first_endpoint_is_used_and_the_second_is_not_touched():
    a, b = FakeClient("a"), FakeClient("b")
    chain = FailoverClient([a, b])
    assert chain.chat(_msgs()).content == "ok"
    assert (a.calls, b.calls) == (1, 0)


def test_a_dead_first_endpoint_fails_over_and_the_answer_survives():
    dead = FakeClient("dead", raises=LLMError("503"))
    live = FakeClient("live", answer="from the backup")
    chain = FailoverClient([dead, live])
    assert chain.chat(_msgs()).content == "from the backup"
    assert (dead.calls, live.calls) == (1, 1)


# -- health is remembered, not recomputed --------------------------------
def test_a_failed_endpoint_is_not_retried_until_its_cooldown_expires():
    """The arithmetic is the point: at ~3 minutes per failed attempt, trying the
    dead endpoint first on every call IS the outage."""
    now = [1000.0]
    dead = FakeClient("dead", raises=LLMError("503"))
    live = FakeClient("live")
    chain = FailoverClient([dead, live], cooldown=120.0, clock=lambda: now[0])
    chain.chat(_msgs())
    assert dead.calls == 1
    for _ in range(5):
        chain.chat(_msgs())
    assert dead.calls == 1, "the cooling endpoint was tried again inside cooldown"
    now[0] += 121.0
    chain.chat(_msgs())
    assert dead.calls == 2, "after cooldown it must be given another chance"


def test_the_configured_order_is_kept_and_the_primary_comes_back():
    """The primary is the primary for a reason, and a chain that promoted the
    backup for good would be a different system from the configured one.

    The same property is what keeps the prompt cache warm: while the primary
    answers it is tried first on every call, so one provider serves the whole
    conversation instead of the prefix being re-paid across two.
    """
    now = [0.0]
    primary = FakeClient("primary", raises_once=LLMError("503 once"))
    backup = FakeClient("backup")
    chain = FailoverClient([primary, backup], cooldown=60.0, clock=lambda: now[0])
    assert chain.chat(_msgs()).content == "ok"      # fell over to backup
    assert (primary.calls, backup.calls) == (1, 1)
    assert chain.clients[0] is primary, "the backup must not be promoted"
    chain.chat(_msgs())
    assert (primary.calls, backup.calls) == (1, 2), "the primary is still cooling"
    now[0] += 61.0
    chain.chat(_msgs())
    assert (primary.calls, backup.calls) == (2, 2), "recovered primary was not retried"
    chain.chat(_msgs())
    assert backup.calls == 2, "once the primary is healthy, the backup is idle again"


def test_when_everything_is_cooling_it_still_tries_the_least_recently_closed():
    """Refusing to call at all would turn a failover chain into an outage."""
    dead1 = FakeClient("d1", raises=LLMError("down"))
    dead2 = FakeClient("d2", raises=LLMError("down"))
    chain = FailoverClient([dead1, dead2], cooldown=600.0, clock=lambda: 0.0)
    with pytest.raises(LLMError):
        chain.chat(_msgs())
    assert (dead1.calls, dead2.calls) == (1, 1)
    with pytest.raises(LLMError):
        chain.chat(_msgs())          # every endpoint cooling, still attempted
    assert (dead1.calls, dead2.calls) == (2, 2)


# -- the three things it must not do -------------------------------------
def test_abort_is_a_decision_not_an_outage():
    """The operator asking to stop must never be re-asked on another endpoint."""
    stopper = FakeClient("stopper", raises=LLMAborted("operator spoke"))
    live = FakeClient("live")
    chain = FailoverClient([stopper, live])
    with pytest.raises(LLMAborted):
        chain.chat(_msgs())
    assert live.calls == 0, "the stop was reinterpreted as an endpoint failure"


def test_a_bug_in_our_own_payload_is_not_reported_as_an_outage():
    class Buggy(FakeClient):
        def chat(self, messages, tools=None, **kwargs):
            raise TypeError("payload bug")

    live = FakeClient("live")
    chain = FailoverClient([Buggy("buggy"), live])
    with pytest.raises(TypeError):
        chain.chat(_msgs())
    assert live.calls == 0, "a programming error was swallowed as network weather"


def test_all_endpoints_failing_names_every_one_of_them():
    chain = FailoverClient([FakeClient("d1", raises=LLMError("403 bad key")),
                            FakeClient("d2", raises=LLMError("503 unavailable"))])
    with pytest.raises(LLMError) as ei:
        chain.chat(_msgs())
    msg = str(ei.value)
    assert "d1" in msg and "d2" in msg
    assert "bad key" in msg and "unavailable" in msg


# -- and the seams it shares with the rest of the run --------------------
def test_the_abort_predicate_reaches_every_endpoint():
    a, b = FakeClient("a"), FakeClient("b")
    chain = FailoverClient([a, b])
    stop = lambda: False
    chain.abort_check = stop
    assert a.abort_check is stop and b.abort_check is stop


def test_usage_and_cache_rate_aggregate_over_the_chain():
    class Counted(FakeClient):
        def chat(self, messages, tools=None, **kwargs):
            self.usage_total["prompt_tokens"] += 10
            self.usage_total["cached_tokens"] += 5
            return LLMResponse(content="x", raw={})

    chain = FailoverClient([Counted("a"), Counted("b")])
    chain.chat(_msgs())
    assert chain.cache_hit_rate == pytest.approx(0.5)


def test_a_single_endpoint_chain_behaves_like_the_endpoint():
    only = FakeClient("only", answer="solo")
    assert FailoverClient([only]).chat(_msgs()).content == "solo"


def test_an_empty_chain_is_refused_at_construction():
    with pytest.raises(ValueError):
        FailoverClient([])


# -- the CLI seam: config -> endpoints, and the credential rule -----------
def test_config_fallbacks_are_read_and_one_provider_inherits_the_key(tmp_path):
    from argparse import Namespace

    from autoforge.cli import _resolve

    cfg = tmp_path / "config.json"
    cfg.write_text("""{
      "base_url": "https://aiping.cn/api/v1",
      "model": "primary-model",
      "api_key": "k1",
      "fallbacks": [
        {"model": "backup-model", "base_url": "https://aiping.cn/api/v1"},
        {"model": "remote", "base_url": "https://api.other.example/v1"}
      ]
    }""", encoding="utf-8")
    args = Namespace(model=None, base_url=None, api_key=None, proxy=None,
                     no_proxy=True, fast=True, max_tokens=None, policy=None)
    import autoforge.cli as cli
    saved = cli.configfile.load(cfg)
    import os
    os.environ["AUTOFORGE_CONFIG"] = str(cfg)
    try:
        resolved, src = _resolve(args)
    finally:
        os.environ.pop("AUTOFORGE_CONFIG", None)
    eps = resolved["endpoints"]
    assert len(eps) == 2, eps
    assert eps[1]["api_key"] == "k1", "same-provider fallback must inherit the key"
    assert "api.other.example" in src.get("fallbacks_dropped", ""), (
        "a different host without its own key must be dropped and SAID SO, not "
        "silently handed the wrong credential")
