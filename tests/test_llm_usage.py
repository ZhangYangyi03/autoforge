"""The token accounting a cached conversation is invisible without.

Two calls that differ only in whether the provider served them from its cache
return identical replies. So the one number that decides whether a long
conversation costs a tenth of what it looks like or exactly what it looks like
is not in the reply at all — it is in `usage`, which the client used to throw
away with the rest of the body.

These tests pin the parse (both shapes the gateways use), the accumulation
across a session, and that a provider sending nonsense loses a number rather
than the reply.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import (
    LLMResponse,
    MockLLMClient,
    OpenAICompatClient,
    account_usage,
    new_usage_totals,
)
from autoforge.core.message import Message


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise
        pass

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def posted(monkeypatch):
    """Post a canned body, and capture what the client sent."""
    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None, proxies=None):  # noqa: A002
        seen["body"] = json
        return _FakeResponse(seen["reply"])

    monkeypatch.setattr("autoforge.core.llm.requests.post", fake_post)
    return seen


def _client(**kw) -> OpenAICompatClient:
    return OpenAICompatClient(model="m", base_url="https://example.test/v1",
                              api_key="k", **kw)


MSGS = [Message(role="user", content="hello")]


def _reply(content: str = "hi", usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


# ======================================================================
# reading the split off the wire
# ======================================================================
class TestTheUsageIsRead:
    def test_deepseek_openai_shape(self, posted):
        posted["reply"] = _reply(usage={
            "prompt_tokens": 10_000, "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 9_600},
        })
        resp = _client().chat(MSGS)

        assert resp.prompt_tokens == 10_000
        assert resp.cached_tokens == 9_600
        assert resp.completion_tokens == 40

    def test_the_flat_gateway_shape(self, posted):
        """Some gateways report the cache hit as a top-level field."""
        posted["reply"] = _reply(usage={"prompt_tokens": 500,
                                        "prompt_cache_hit_tokens": 480})
        assert _client().chat(MSGS).cached_tokens == 480

    def test_absent_usage_reads_as_zero(self, posted):
        posted["reply"] = _reply()
        resp = _client().chat(MSGS)
        assert (resp.prompt_tokens, resp.cached_tokens) == (0, 0)

    def test_a_provider_sending_junk_loses_a_number_not_the_reply(self, posted):
        """Usage reports work already done; it must not be able to fail a call."""
        posted["reply"] = _reply("the answer", usage={
            "prompt_tokens": "1200",                     # a string, as some send
            "prompt_tokens_details": {"cached_tokens": None},
            "completion_tokens": {"nonsense": True},
        })
        resp = _client().chat(MSGS)

        assert resp.content == "the answer"
        assert resp.prompt_tokens == 1_200
        assert resp.cached_tokens == 0
        assert resp.completion_tokens == 0


# ======================================================================
# adding it up, which is where a session's cost actually lives
# ======================================================================
class TestItAccumulates:
    def test_a_session_reports_its_own_cache_hit_rate(self, posted):
        client = _client()
        posted["reply"] = _reply("a", usage={
            "prompt_tokens": 1_000,
            "prompt_tokens_details": {"cached_tokens": 0}})
        client.chat(MSGS)
        posted["reply"] = _reply("b", usage={
            "prompt_tokens": 1_200,
            "prompt_tokens_details": {"cached_tokens": 1_000}})
        client.chat(MSGS)

        assert client.usage_total["calls"] == 2
        assert client.usage_total["prompt_tokens"] == 2_200
        assert client.usage_total["cached_tokens"] == 1_000
        assert client.cache_hit_rate == pytest.approx(1_000 / 2_200)

    def test_no_calls_is_a_zero_rate_not_a_division_error(self):
        assert _client().cache_hit_rate == 0.0

    def test_the_mock_client_accounts_too(self):
        """Offline tests get the same arithmetic, or they cannot pin it."""
        mock = MockLLMClient(script=[
            LLMResponse(content="a", usage={"prompt_tokens": 100,
                                            "prompt_tokens_details":
                                                {"cached_tokens": 80}}),
            LLMResponse(content="b", usage={"prompt_tokens": 100}),
        ])
        mock.chat(MSGS)
        mock.chat(MSGS)

        assert mock.usage_total["prompt_tokens"] == 200
        assert mock.usage_total["cached_tokens"] == 80
        assert mock.cache_hit_rate == pytest.approx(0.4)


def test_the_accumulator_starts_empty():
    assert new_usage_totals() == {"calls": 0, "prompt_tokens": 0,
                                  "cached_tokens": 0, "completion_tokens": 0}


def test_accounting_folds_one_response_in():
    totals = new_usage_totals()
    account_usage(totals, LLMResponse(content="x", usage={
        "prompt_tokens": 7, "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 5}}))

    assert totals == {"calls": 1, "prompt_tokens": 7, "cached_tokens": 5,
                      "completion_tokens": 3}


def test_the_failover_chain_reports_the_spend_it_carried():
    """A chain in front of the clients must not swallow the counters.

    The ledger writes token spend with `getattr(llm, "usage_total", {})`.
    `FailoverClient` had no `usage_total`, so the getattr's default hid the
    hole: every run behind a `fallbacks` list recorded `usage: {}` from
    2026-09-17 on, and an empty counter reads exactly like a run that cost
    nothing. The attribute is the fix; this test is what stops it being
    dropped again, because the failure mode is silent by construction.
    """
    from autoforge.core.llm import FailoverClient

    a = MockLLMClient(script=[LLMResponse(content="a", usage={
        "prompt_tokens": 100, "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 90}})])
    b = MockLLMClient(script=[LLMResponse(content="b", usage={
        "prompt_tokens": 50, "completion_tokens": 5})])
    chain = FailoverClient([a, b])

    a.chat(MSGS)
    b.chat(MSGS)

    assert chain.usage_total == {"calls": 2, "prompt_tokens": 150,
                                 "cached_tokens": 90, "completion_tokens": 15}
    assert chain.cache_hit_rate == pytest.approx(90 / 150)


def test_an_empty_chain_is_zero_rather_than_missing():
    from autoforge.core.llm import FailoverClient

    assert FailoverClient([MockLLMClient()]).usage_total == new_usage_totals()
