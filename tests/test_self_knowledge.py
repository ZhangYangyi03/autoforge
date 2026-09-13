"""The self-model must come from records, not from priors.

Companion to test_reach.py. That file pins the *reach* half — forged code has
the host filesystem and the network. This one pins the rest of the self-report,
each part read from the object that owns the fact rather than asserted in prose:

  * memory  — the ledger is sqlite on disk and outlives the process, and a
    forged tool is written to it, so the agent does not "forget" what it made
  * history — `my_history` reads that ledger back, amendments included
  * prompt  — the measured block rides in every request, so the description
    cannot drift away from the machine

The observed failure, again: asked what it could not do, the agent produced a
table — "no persistent memory, every call a fresh snapshot", "no continuous
self, only results" — while a sqlite ledger holding every run sat on disk
behind it, unread. Both halves were false; the tools to read them were missing.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM, SUPERVISED, AutonomyPolicy
from autoforge.core.llm import MockLLMClient
from autoforge.store import ToolStore
from autoforge.tools.spec import ToolSpec


@pytest.fixture()
def db_path():
    with tempfile.TemporaryDirectory() as td:
        yield os.path.join(td, "af.db")


@pytest.fixture()
def store(db_path):
    s = ToolStore(db_path)
    yield s
    s.close()


def _agent(store=None, policy=None) -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), store=store, policy=policy or FULL_FREEDOM)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"the {name} tool",
                    parameters={}, fn=lambda **_: name)


# ======================================================================
# store.report() tells the truth about what is on disk
# ======================================================================
class TestStoreReportIsMeasured:
    def test_report_counts_match_what_was_written(self, store):
        store.save_tool(_spec("alpha"))
        store.log_event("run", {"task": "x"})
        store.log_event("forge", {"need": "y"})
        r = store.report()
        assert r["backend"] == "sqlite"
        assert r["survives_restart"] is True
        assert os.path.isabs(r["db_path"])
        assert r["tools"] == 1
        assert r["events"] == 2
        assert r["event_kinds"] == {"run": 1, "forge": 1}

    def test_report_still_counts_after_a_reopen(self, store, db_path):
        store.save_tool(_spec("beta"))
        store.close()
        reopened = ToolStore(db_path)
        try:
            assert reopened.report()["tools"] == 1
        finally:
            reopened.close()


# ======================================================================
# a forged tool is persisted — it must not die with the process
# ======================================================================
class TestForgePersists:
    def test_forged_tool_is_on_the_ledger(self, store, db_path):
        a = _agent(store)
        res = SimpleNamespace(ok=True, rounds=1, spec=_spec("read_magic"))
        a.pipeline.forge = lambda need, context="": res   # no LLM needed

        out = a.registry.call("forge_tool", {"need": "read a file"}).output
        assert "read_magic" in out

        store.close()
        after_restart = ToolStore(db_path)
        try:
            assert "read_magic" in after_restart.load_all_tools()
        finally:
            after_restart.close()

    def test_failed_forge_persists_nothing(self, store):
        a = _agent(store)
        a.pipeline.forge = lambda need, context="": SimpleNamespace(
            ok=False, rounds=2, spec=None)
        a.registry.call("forge_tool", {"need": "impossible"})
        assert store.report()["tools"] == 0


# ======================================================================
# my_history reads the ledger back
# ======================================================================
class TestMyHistory:
    def test_history_shows_logged_events(self, store):
        a = _agent(store)
        store.log_event("amendment", {"target": "system_prompt"})
        out = a.registry.call("my_history", {}).output
        assert "amendment" in out
        assert "system_prompt" in out

    def test_history_is_honest_when_nothing_is_attached(self):
        out = _agent(None).registry.call("my_history", {}).output
        assert "no store" in out.lower()

    def test_history_is_registered_with_a_ledger_description(self):
        spec = _agent().registry.get("my_history")
        assert spec is not None
        assert "ledger" in spec.description.lower()


# ======================================================================
# the measurement rides in every request
# ======================================================================
class TestSelfReportIsInEveryRequest:
    def test_report_names_reach_memory_and_its_own_source(self, store):
        txt = _agent(store)._self_report()
        assert "filesystem" in txt and "network" in txt
        assert "survives restart" in txt
        assert store.db_path in txt
        assert "agent.py" in txt          # its own source, named, readable

    def test_effective_prompt_carries_the_measurement(self, store):
        a = _agent(store)
        prompt = a._effective_prompt()
        assert a.system_prompt in prompt          # base prompt preserved
        assert "MEASURED SELF-REPORT" in prompt
        assert store.db_path in prompt

    def test_report_admits_when_there_is_no_store(self):
        assert "nothing persists" in _agent(None)._self_report()


# ======================================================================
# the capability report never prints a section that has no rows
# ======================================================================
class TestNoPhantomSections:
    @pytest.mark.parametrize("policy", [FULL_FREEDOM, SUPERVISED])
    def test_shipped_presets_print_no_empty_section(self, policy):
        lines = _agent(policy=policy).registry.call("my_capabilities", {}).output.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Switched off"):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                assert nxt.startswith("  - "), f"header with no rows: {line!r}"

    def test_declared_only_off_prints_only_the_inert_section(self):
        # Only a declared-only freedom is off, so the *enforced* and *partial*
        # sections are empty. The old code printed their headers anyway, which
        # reads as a limit that does not exist — the exact failure this whole
        # report exists to prevent.
        policy = AutonomyPolicy(may_access_network=False)
        out = _agent(policy=policy).registry.call("my_capabilities", {}).output
        sections = [ln for ln in out.splitlines() if ln.startswith("Switched off")]
        assert sections == ["Switched off, but nothing obeys it (do not rely on these):"]
        assert "may_access_network" in out
        assert "network    — outbound" in out      # the reach line still tells the truth
