"""The outgoing request body, asserted on the wire.

These exist because of a concrete failure: aiping.cn answers 503 to a
completion body that omits `max_tokens`, while the same body with a modest cap
returns 200. The client used to forward whatever the caller passed and nothing
else, so every call path that did not think about `max_tokens` -- the agent
loop, the demos -- silently carried a landmine. The fix belongs at the client,
and these tests keep it there.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import OpenAICompatClient
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
def captured(monkeypatch):
    """Capture the JSON body posted, without touching the network."""
    seen: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None, proxies=None):  # noqa: A002
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = json
        return _FakeResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("autoforge.core.llm.requests.post", fake_post)
    return seen


def _client(**kw) -> OpenAICompatClient:
    return OpenAICompatClient(model="m", base_url="https://example.test/v1",
                              api_key="k", **kw)


MSGS = [Message(role="user", content="hello")]


# ======================================================================
# max_tokens is always on the wire
# ======================================================================
class TestMaxTokensAlwaysSent:

    def test_omitted_by_caller_still_lands_in_the_body(self, captured):
        """The whole point: a caller that says nothing still sends the cap."""
        _client().chat(MSGS)
        assert captured["body"]["max_tokens"] == OpenAICompatClient.DEFAULT_MAX_TOKENS

    def test_explicit_caller_value_wins(self, captured):
        _client().chat(MSGS, max_tokens=512)
        assert captured["body"]["max_tokens"] == 512

    def test_constructor_default_is_used_over_the_class_default(self, captured):
        _client(default_max_tokens=1024).chat(MSGS)
        assert captured["body"]["max_tokens"] == 1024

    def test_can_be_disabled_for_providers_that_reject_the_field(self, captured):
        """Opt-out must be possible: some backends 400 on the parameter."""
        _client(default_max_tokens=None).chat(MSGS)
        assert "max_tokens" not in captured["body"]

    def test_default_sits_below_the_known_bad_ceiling(self):
        """4096 was flaky and 8192 failed outright in the probe."""
        assert OpenAICompatClient.DEFAULT_MAX_TOKENS is not None
        assert OpenAICompatClient.DEFAULT_MAX_TOKENS < 4096


# ======================================================================
# the rest of the body is unchanged by the fix
# ======================================================================
class TestRestOfBody:

    def test_temperature_and_model_defaults(self, captured):
        _client().chat(MSGS)
        assert captured["body"]["model"] == "m"
        assert captured["body"]["temperature"] == 0.0
        assert "tools" not in captured["body"]

    def test_tools_are_forwarded_with_a_choice(self, captured):
        _client().chat(MSGS, tools=[{"type": "function"}])
        assert captured["body"]["tools"] == [{"type": "function"}]
        assert captured["body"]["tool_choice"] == "auto"

    def test_extra_kwargs_still_pass_through(self, captured):
        _client().chat(MSGS, top_p=0.9)
        assert captured["body"]["top_p"] == 0.9

    def test_url_and_auth_header(self, captured):
        _client().chat(MSGS)
        assert captured["url"] == "https://example.test/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer k"

    def test_response_content_is_returned(self, captured):
        out = _client().chat(MSGS)
        assert out.content == "hi"
        assert out.raw == {"choices": [{"message": {"content": "hi"}}]}
