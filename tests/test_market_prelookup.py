"""The forge must look on the shelf BEFORE it builds, not only after.

`_sync_to_market` has always checked for an existing entry -- but it runs
after the tool already exists. The forge path itself never looked, so the
principle "check the market before forging" lived in the prompt for a day
with zero executions: 158 forges, not one pre-forge lookup. A principle is
not a gate.

These tests pin the gate, not the prose: the lookup happens on the path the
forge takes, it runs before `pipeline.forge`, and -- the subtle half -- a
market that cannot answer is NOT read as "the shelf does not have it".
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import urllib.request

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient


def _agent() -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), policy=FULL_FREEDOM)


def _shelf(items):
    class _Resp:
        def __init__(self, payload):
            self._b = io.BytesIO(json.dumps(payload).encode("utf-8"))

        def read(self):
            return self._b.read()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        return _Resp(items)

    return _open


class TestPreLookupRunsAndIsHonest:
    def test_unreachable_market_returns_no_answer_not_no_entry(self, monkeypatch):
        """A dead market must not read as an empty one."""
        def _boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        out = _agent()._prelookup_market("read the last lines of a board file")
        assert out == "", (
            "an unreachable shelf must not be read as permission to forge: " + repr(out))

    def test_empty_shelf_says_so(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _shelf([]))
        out = _agent()._prelookup_market("read the last lines of a board file")
        assert "no entry" in out.lower() or "no" in out.lower()
        assert "duplicate" in out.lower()

    def test_overlapping_entry_is_named(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", _shelf([
            {"id": "tool:read_bus_ndjson", "name": "read_bus_ndjson",
             "description": "Read the last N lines of an agent-bus ndjson board file"},
        ]))
        out = _agent()._prelookup_market("read bus ndjson board lines")
        assert "read_bus_ndjson" in out
        assert "Do not re-forge" in out

    def test_lookup_lands_before_the_forge(self, monkeypatch):
        """The point of the whole change: it is on the forge path."""
        monkeypatch.setattr(urllib.request, "urlopen", _shelf([]))
        a = _agent()
        seen = {}

        def _fake_forge(need, context=None, should_abort=None, **kw):
            seen["context"] = context or ""
            seen["need"] = need
            return SimpleNamespace(ok=False, aborted=False, rounds=1, spec=None)

        a.pipeline = SimpleNamespace(forge=_fake_forge)
        a.registry.get("forge_tool").fn("read the last lines of a board file")
        assert "Tool-market pre-lookup" in seen.get("context", ""), (
            "the shelf was never consulted on the forge path: " + repr(seen))
