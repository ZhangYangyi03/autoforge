"""A malformed 2xx body must be named, never turned into an AttributeError.

Observed in ``evidence/run_20260912_163001.jsonl``: a forge round died with
``AttributeError: 'str' object has no attribute 'get'``. Both places that could
raise it assumed the endpoint's body had the shape the protocol promises --
``_to_response`` reading ``choices[0]`` and the ``finish_reason`` property
walking ``raw``. An OpenAI-compatible gateway that fails out-of-band answers
HTTP 200 with ``choices: ["<error text>"]``, and the assumption converted that
into an error naming neither the endpoint nor the body.

The client is also the only place that sees the raw reply, so it is the only
place that can report it. These tests keep the reporting there.
"""
from __future__ import annotations

import pytest

from autoforge.core.llm import LLMResponse, LLMResponseError, OpenAICompatClient
from autoforge.core.message import Message


def _client() -> OpenAICompatClient:
    return OpenAICompatClient(model="m", base_url="http://example.invalid/v1", api_key="k")


# -- the body itself is not an object -----------------------------------

@pytest.mark.parametrize("body", ["just a string", ["a", "list"], 42, None])
def test_non_object_body_is_named(body):
    with pytest.raises(LLMResponseError) as ei:
        _client()._to_response(body)
    msg = str(ei.value)
    assert "not an object" in msg
    assert type(body).__name__ in msg


# -- choices[0] is not an object ----------------------------------------

def test_string_choice_is_named_not_an_attributeerror():
    with pytest.raises(LLMResponseError) as ei:
        _client()._to_response({"choices": ["upstream provider unavailable"]})
    msg = str(ei.value)
    assert "choices[0]" in msg
    assert "str" in msg
    assert "upstream provider unavailable" in msg


def test_empty_choices_are_an_empty_reply_not_a_parse_error():
    """An empty reply is a hiccup the retry loop owns, so it must not raise."""
    resp = _client()._to_response({"choices": []})
    assert resp.content == ""
    assert resp.tool_calls == []
    assert resp.finish_reason is None


def test_choices_of_the_wrong_container_type_is_an_empty_reply():
    resp = _client()._to_response({"choices": "oops"})
    assert resp.content == ""


def test_message_of_the_wrong_type_is_ignored():
    resp = _client()._to_response({"choices": [{"message": "oops"}]})
    assert resp.content == ""
    assert resp.tool_calls == []


# -- finish_reason is total ---------------------------------------------

@pytest.mark.parametrize("raw", [
    None,
    "a string",
    ["a", "list"],
    {"choices": "not a list"},
    {"choices": []},
    {"choices": ["boom"]},
])
def test_finish_reason_never_raises(raw):
    got = LLMResponse(content="x", raw=raw).finish_reason
    assert got is None


def test_finish_reason_still_reads_a_well_formed_body():
    assert LLMResponse(raw={"choices": [{"finish_reason": "length"}]}).finish_reason == "length"


# -- the same two defects, through chat() -------------------------------

class _RawResponse:
    """A 200 whose body is whatever we say, including not-JSON."""

    status_code = 200

    def __init__(self, *, payload=None, text="", decode_error=None) -> None:
        self._payload = payload
        self.text = text
        self._decode_error = decode_error

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise
        pass

    def json(self):
        if self._decode_error is not None:
            raise self._decode_error
        return self._payload


def test_chat_names_a_non_json_body(monkeypatch):
    resp = _RawResponse(text="<html>502 Bad Gateway</html>",
                        decode_error=ValueError("Expecting value"))
    monkeypatch.setattr("autoforge.core.llm.requests.post", lambda *a, **kw: resp)
    with pytest.raises(LLMResponseError) as ei:
        _client().chat([Message.user("hi")])
    msg = str(ei.value)
    assert "non-JSON body" in msg
    assert "502 Bad Gateway" in msg


def test_chat_names_a_string_choice(monkeypatch):
    resp = _RawResponse(payload={"choices": ["downstream provider is down"]})
    monkeypatch.setattr("autoforge.core.llm.requests.post", lambda *a, **kw: resp)
    with pytest.raises(LLMResponseError) as ei:
        _client().chat([Message.user("hi")])
    assert "downstream provider is down" in str(ei.value)


def test_chat_still_returns_a_healthy_body(monkeypatch):
    resp = _RawResponse(payload={"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.setattr("autoforge.core.llm.requests.post", lambda *a, **kw: resp)
    assert _client().chat([Message.user("hi")]).content == "hi"
