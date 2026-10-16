"""Tests for the exposure backport into the installed autoforge 0.4.2 build.

The installed wheel predates upstream's schema-exposure mechanism, so every
request carried every visible tool schema in full -- measured on this machine
at 26,504 chars (~8,282 tokens) on every turn, whether or not a tool was used.

This suite pins the four things the backport has to get right:

1. `tools/exposure.py` splits by *behaviour* (the ledger), not by a count, and
   the floor tools are never demoted.
2. `ToolRegistry` answers "when was this last called" from the ledger when one
   is bound, falls back to the spec's own stat when not, and caches the split
   so two no-call turns build a byte-identical prompt prefix.
3. `ToolStore.last_called_by_tool` reads the real `forge_events` ledger.
4. `Agent._request` sends only the hot schemas as `tools` and injects the cold
   ones as a block after the system prompt -- and reverts to sending
   everything when no exposure callback is handed over.

Run:
    C:/autoforge/venv/Scripts/python.exe -m unittest test_exposure_backport -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from typing import Any

from autoforge.core.agent import Agent
from autoforge.core.llm import LLMClient, LLMResponse
from autoforge.core.message import Message
from autoforge.tools import exposure as exposure_mod
from autoforge.tools.exposure import HOT_FLOOR, cold_line, manifest_header, split
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec
from autoforge.store import ToolStore


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def spec(name: str, desc: str, params: list[str], *, source: str = "agent") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=desc,
        parameters={"type": "object",
                    "properties": {p: {} for p in params}},
        fn=lambda **kw: "ok",
        source=source,
    )


def names_of(schemas: list[dict]) -> list[str]:
    return [(s.get("function") or s).get("name") for s in schemas]


def schema(name: str, desc: str, params: list[str], *, source: str = "agent") -> dict:
    """The wire form `split()` consumes -- not a ToolSpec."""
    return spec(name, desc, params, source=source).schema


def registry_with(*specs: ToolSpec) -> ToolRegistry:
    reg = ToolRegistry()
    for s in specs:
        reg.register(s)
        reg.promote(s.name)
    return reg


class StubLLM(LLMClient):
    """Records the exact request the loop built, then answers in plain text."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], list[dict]]] = []

    def chat(self, msgs, *, tools=None, should_abort=None):
        self.calls.append((list(msgs), tools or []))
        return LLMResponse(content="done", tool_calls=[])


# --------------------------------------------------------------------------
# 1. the split itself
# --------------------------------------------------------------------------
class TestTheSplit(unittest.TestCase):
    def test_the_floor_is_never_demoted_even_with_no_record(self):
        schemas = [schema(n, f"{n} does a thing.", ["a"]) for n in sorted(HOT_FLOOR)]
        hot, cold, stats = split(schemas, lambda n: None)
        self.assertEqual(names_of(hot), sorted(HOT_FLOOR))
        self.assertEqual(cold, [])
        self.assertEqual(stats["cold"], 0)

    def test_a_tool_never_called_is_cold(self):
        hot, cold, stats = split([schema("draw", "Make an image.", ["prompt"])],
                                 lambda n: None)
        self.assertEqual(hot, [])
        self.assertEqual(stats, {"hot": 0, "cold": 1, "total": 1})
        self.assertIn("draw(prompt)", cold[0])

    def test_the_window_is_inclusive_at_seven_days(self):
        s = [schema("edge", "An edge tool.", ["x"])]
        hot, _, _ = split(s, lambda n: 7.0, window_days=7.0)
        self.assertEqual(names_of(hot), ["edge"], "7.0 days must still be hot")
        hot, _, _ = split(s, lambda n: 7.01, window_days=7.0)
        self.assertEqual(hot, [], "a hair over the window must be cold")

    def test_protect_is_separate_from_the_floor(self):
        s = [schema("animate", "Make a clip.", ["prompt"])]
        hot, _, _ = split(s, lambda n: None, protect={"animate"})
        self.assertEqual(names_of(hot), ["animate"])
        # and it does not leak into the floor
        self.assertNotIn("animate", HOT_FLOOR)

    def test_a_cold_line_is_one_line_and_caps_its_description(self):
        long_desc = "x" * 400
        line = cold_line("thing", {"function": {
            "name": "thing", "description": long_desc,
            "parameters": {"type": "object",
                           "properties": {p: {} for p in "abcdef"}}}})
        self.assertNotIn("\n", line)
        self.assertTrue(line.startswith("  - thing(a, b, c, d, ...)"))
        self.assertLess(len(line), exposure_mod.COLD_DESC_CHARS + 40)
        self.assertTrue(line.endswith("\u2026"), "truncation must be visible")

    def test_the_manifest_says_the_tools_are_callable(self):
        _, _, stats = split([schema("draw", "d", ["p"])], lambda n: None)
        text = manifest_header(stats, 7.0)
        self.assertIn("1 of 1", text)
        self.assertIn("callable", text)
        self.assertIn("describe_tool", text)


# --------------------------------------------------------------------------
# 2. the registry
# --------------------------------------------------------------------------
class TestTheRegistry(unittest.TestCase):
    def test_no_ledger_and_never_called_is_none_not_zero(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        self.assertIsNone(reg.days_since_called("draw"))

    def test_a_call_bumps_the_counter_and_makes_the_tool_hot(self):
        reg = registry_with(spec("run_python", "r", ["code"]),
                            spec("draw", "d", ["p"]))
        before = reg._calls_seen
        reg.call("draw", {"p": "x"})
        self.assertEqual(reg._calls_seen, before + 1)
        hot, cold, stats = reg.exposure()
        self.assertIn("draw", names_of(hot))
        self.assertEqual(stats["hot"], 2)

    def test_a_blocked_call_does_not_count_as_use(self):
        """A tool kept hot by a call it never made is the whole bug this avoids."""
        reg = registry_with(spec("draw", "d", ["p"]))
        reg._gate = lambda *a, **k: None            # no gate here
        reg.get("draw").state = __import__(
            "autoforge.tools.spec", fromlist=["ToolState"]).ToolState.QUARANTINED
        reg.call("draw", {"p": "x"})                # refused: quarantined
        self.assertEqual(reg._calls_seen, 0)

    def test_the_split_is_cached_so_two_no_call_turns_are_identical(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        first = reg.exposure()
        second = reg.exposure()
        self.assertIs(first, second, "no call happened: must be the same object")

    def test_the_cache_is_dropped_when_a_call_happens(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        first = reg.exposure()
        reg.call("draw", {"p": "x"})
        self.assertIsNot(first, reg.exposure())

    def test_a_ledger_answers_instead_of_the_spec_stat(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        reg.get("draw").stats.last_called = time.time()      # says "just now"
        reg.last_called_source = lambda n: time.time() - 30 * 86400  # ledger: 30d
        hot, cold, _ = reg.exposure()
        self.assertEqual(hot, [], "the ledger must win over the spec stat")
        self.assertEqual(len(cold), 1)

    def test_the_ledger_is_preferred_but_a_dead_one_falls_back(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        reg.get("draw").stats.last_called = time.time()

        def boom(name):
            raise RuntimeError("ledger is gone")

        reg.last_called_source = boom
        hot, _, _ = reg.exposure()
        self.assertIn("draw", names_of(hot), "must fall back, not explode")

    def test_draft_tools_are_not_exposed_at_all(self):
        """`exposure()` reads `schemas()`, so the state filter still applies."""
        reg = ToolRegistry()
        reg.register(spec("drafty", "d", ["p"]))     # left in DRAFT
        hot, cold, stats = reg.exposure()
        self.assertEqual(stats["total"], 0)


# --------------------------------------------------------------------------
# 3. the ledger query
# --------------------------------------------------------------------------
class TestTheLedgerQuery(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), "_tmp_ledger.db")
        if os.path.exists(self.path):
            os.remove(self.path)
        self.store = ToolStore(self.path)

    def tearDown(self):
        try:
            self.store._conn.close()
        except Exception:
            pass
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_newest_call_per_tool_wins(self):
        self.store.log_event("call", {"tool": "draw"})
        time.sleep(0.01)
        self.store.log_event("call", {"tool": "draw"})
        time.sleep(0.01)
        self.store.log_event("call", {"tool": "animate"})
        m = self.store.last_called_by_tool()
        self.assertEqual(set(m), {"draw", "animate"})
        self.assertGreater(m["animate"], m["draw"], "newest must win")

    def test_only_call_events_count(self):
        self.store.log_event("run", {"task": "something"})
        self.store.log_event("result", {"tool": "draw", "ok": True})
        self.assertEqual(self.store.last_called_by_tool(), {},
                         "a result is not a call")

    def test_a_malformed_payload_does_not_kill_the_query(self):
        self.store._conn.execute(
            "INSERT INTO forge_events (timestamp, kind, payload) VALUES (?,?,?)",
            (time.time(), "call", "not json at all"))
        self.store._conn.commit()
        self.store.log_event("call", {"tool": "draw"})
        m = self.store.last_called_by_tool()
        self.assertEqual(set(m), {"draw"})

    def test_an_empty_ledger_returns_empty(self):
        self.assertEqual(self.store.last_called_by_tool(), {})


# --------------------------------------------------------------------------
# 4. the request the loop actually builds
# --------------------------------------------------------------------------
class TestTheRequest(unittest.TestCase):
    def setUp(self):
        self.reg = registry_with(
            spec("run_python", "Run python code.", ["code"]),
            spec("draw", "Make an image from a prompt and save it.", ["prompt", "size"]),
            spec("animate", "Make a short clip from a prompt.", ["prompt"]),
        )

    def _run(self, *, exposure):
        llm = StubLLM()
        Agent(llm, self.reg, exposure=exposure).run("do a thing")
        return llm.calls[0]

    def test_without_exposure_every_schema_is_sent_and_nothing_is_injected(self):
        msgs, tools = self._run(exposure=None)
        self.assertEqual(len(tools), len(self.reg.schemas()))
        self.assertEqual([m.role for m in msgs], ["system", "user"])

    def test_with_exposure_only_the_hot_schemas_are_sent(self):
        msgs, tools = self._run(exposure=self.reg)
        sent = names_of(tools)
        self.assertIn("run_python", sent, "the floor must always ride in full")
        self.assertNotIn("draw", sent)
        self.assertNotIn("animate", sent)
        self.assertLess(len(tools), len(self.reg.schemas()))

    def test_the_cold_tools_ride_as_one_block_after_the_system_prompt(self):
        msgs, _ = self._run(exposure=self.reg)
        self.assertEqual([m.role for m in msgs], ["system", "user", "user"])
        block = msgs[1].content
        self.assertIn("draw(prompt, size)", block)
        self.assertIn("animate(prompt)", block)
        self.assertIn("describe_tool", block)
        self.assertEqual(msgs[2].content, "do a thing",
                         "the task must still be the last thing said")

    def test_the_injected_block_never_enters_the_transcript(self):
        """A resume would replay it as something the operator typed."""
        llm = StubLLM()
        result = Agent(llm, self.reg, exposure=self.reg).run("do a thing")
        for m in result.messages:
            self.assertNotIn("not carrying in full", m.content)

    def test_a_broken_exposure_callback_falls_back_to_sending_everything(self):
        class Exploding:
            window_days = 7.0

            def exposure(self):
                raise RuntimeError("the ledger is on fire")

        msgs, tools = self._run(exposure=Exploding())
        self.assertEqual(len(tools), len(self.reg.schemas()),
                         "a cost optimisation must never lose the run")

    def test_no_cold_tools_means_no_injected_block(self):
        reg = registry_with(spec("run_python", "Run python code.", ["code"]))
        llm = StubLLM()
        Agent(llm, reg, exposure=reg).run("do a thing")
        msgs, _ = llm.calls[0]
        self.assertEqual([m.role for m in msgs], ["system", "user"])

    def test_the_prompt_prefix_is_byte_identical_across_two_no_call_turns(self):
        """This is what the provider's prefix cache is paid on."""
        first_msgs, first_tools = self._run(exposure=self.reg)
        second_msgs, second_tools = self._run(exposure=self.reg)
        self.assertEqual(first_msgs[1].content, second_msgs[1].content)
        self.assertEqual(json.dumps(first_tools, sort_keys=True),
                         json.dumps(second_tools, sort_keys=True))


# --------------------------------------------------------------------------
# 5. the ForgeAgent wiring
# --------------------------------------------------------------------------
class TestForgeAgentWiring(unittest.TestCase):
    """The wiring, on a real ForgeAgent backed by a throwaway store.

    The store lives in a fresh temp dir per test and is never deleted by hand:
    sqlite holds the file open, and a delete that fights a lock is a flaky test,
    not a finding.
    """

    def setUp(self):
        self._dirs: list[str] = []

    def tearDown(self):
        for agent in getattr(self, "_agents", []):
            try:
                agent.store._conn.close()
            except Exception:
                pass

    def _agent(self):
        from autoforge.agent import ForgeAgent
        from autoforge.core.llm import MockLLMClient

        d = tempfile.mkdtemp(prefix="exposure_wiring_")
        self._dirs.append(d)
        agent = ForgeAgent(llm=MockLLMClient(),
                           store=ToolStore(os.path.join(d, "agent.db")))
        if not hasattr(self, "_agents"):
            self._agents = []
        self._agents.append(agent)
        return agent

    def test_the_ledger_is_bound_when_a_store_is_present(self):
        agent = self._agent()
        self.assertIsNotNone(agent.registry.last_called_source)
        self.assertIs(agent.registry.last_called_source.__self__, agent)

    def test_a_bare_registry_keeps_none_and_still_works(self):
        reg = registry_with(spec("draw", "d", ["p"]))
        self.assertIsNone(reg.last_called_source)
        hot, cold, _ = reg.exposure()
        self.assertEqual(len(cold), 1)

    def test_hot_protect_matches_whole_words_only(self):
        agent = self._agent()
        agent._current_task = "please check whether you have seen the page"
        protected = agent._hot_protect()
        self.assertNotIn("see", protected,
                         "'seen' must not protect a tool called 'see'")
        agent._current_task = "now see the page"
        self.assertIn("see", agent._hot_protect())

    def test_hot_protect_needs_a_contiguous_run_for_a_multi_word_name(self):
        agent = self._agent()
        agent._current_task = "browser goto the example"
        self.assertIn("browser_goto", agent._hot_protect())
        agent._current_task = "goto the browser"
        self.assertNotIn("browser_goto", agent._hot_protect())

    def test_an_empty_task_protects_nothing(self):
        agent = self._agent()
        agent._current_task = ""
        self.assertEqual(agent._hot_protect(), set())

    def test_protect_is_set_for_the_run_and_cleared_after_it(self):
        agent = self._agent()
        seen: list[set[str]] = []
        real = agent._run_locked

        def spy(task, history, emit, **kw):
            seen.append(set(agent.registry.protect))
            return real(task, history, emit, **kw)

        agent._run_locked = spy
        agent.run("draw an image")
        self.assertTrue(seen, "the run must have started")
        self.assertIn("draw", seen[0], "the task named the tool")
        self.assertEqual(agent.registry.protect, set(),
                         "a leftover protect would keep tools hot forever")
        self.assertEqual(agent._current_task, "")

    def test_the_budget_report_counts_only_the_agents_own_tools(self):
        from autoforge.agent import TOOL_SCHEMA_BUDGET_CHARS
        agent = self._agent()
        self.assertEqual(TOOL_SCHEMA_BUDGET_CHARS, 6000)
        own = agent._tool_schema_chars(own_only=True)
        allof = agent._tool_schema_chars(own_only=False)
        self.assertLessEqual(own, allof)
        self.assertGreater(allof, 0, "the builtin baseline is not empty")

    def test_the_agent_hands_its_registry_to_the_loop_as_exposure(self):
        """Without this line the split exists and nothing ever asks for it."""
        agent = self._agent()
        captured: dict[str, Any] = {}
        real = Agent.__init__

        def spy(self_, llm, registry, **kw):
            captured.update(kw)
            return real(self_, llm, registry, **kw)

        Agent.__init__ = spy
        try:
            agent.run("do a thing")
        finally:
            Agent.__init__ = real
        self.assertIs(captured.get("exposure"), agent.registry)


# --------------------------------------------------------------------------
# 6. the measurement, against the real store
# --------------------------------------------------------------------------
class TestTheRealSaving(unittest.TestCase):
    """The number the whole exercise is about, on this machine's own data."""

    DB = os.path.expanduser("~/AppData/Local/autoforge/autoforge.db")

    def setUp(self):
        if not os.path.exists(self.DB):
            self.skipTest(f"no real store at {self.DB}")
        from autoforge.agent import ForgeAgent
        from autoforge.core.llm import MockLLMClient
        self.agent = ForgeAgent(llm=MockLLMClient(), store=ToolStore(self.DB))

    def test_exposure_shrinks_the_per_request_schema_payload(self):
        reg = self.agent.registry
        everything = reg.schemas()
        full_chars = sum(len(json.dumps(s, ensure_ascii=False)) for s in everything)
        hot, cold, stats = reg.exposure()
        hot_chars = sum(len(json.dumps(s, ensure_ascii=False)) for s in hot)
        cold_chars = sum(len(line) for line in cold)

        print(f"\n    visible schemas : {len(everything)}")
        print(f"    full payload    : {full_chars:,} chars "
              f"(~{full_chars // 4:,} tokens)")
        print(f"    hot             : {len(hot)} schemas, {hot_chars:,} chars")
        print(f"    cold            : {len(cold)} lines, {cold_chars:,} chars")
        print(f"    after the split : {hot_chars + cold_chars:,} chars "
              f"(~{(hot_chars + cold_chars) // 4:,} tokens)")

        self.assertLess(hot_chars, full_chars,
                        "the hot set must be smaller than everything")
        self.assertLess(hot_chars + cold_chars, full_chars,
                        "the split must actually save characters")
        self.assertEqual(stats["hot"] + stats["cold"], stats["total"])

    def test_the_floor_tools_are_hot_on_real_data(self):
        hot, _, _ = self.agent.registry.exposure()
        sent = names_of(hot)
        for name in HOT_FLOOR:
            if name in self.agent.registry:
                self.assertIn(name, sent, f"{name} is floor and must ride in full")


if __name__ == "__main__":
    unittest.main(verbosity=2)
