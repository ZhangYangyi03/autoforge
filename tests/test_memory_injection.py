"""Kept facts arrive on their own.

`remember`/`recall` existed and the memory still did not work, for a reason no
test caught: a fact the agent has to remember to *ask for* is one it will forget
to ask for. From the model's side, a kept fact that is not in the request is
indistinguishable from a fact that was never kept, so the store could be full
and the agent would describe itself as having no memory — correctly, from what
it could see.

This file pins the half that was missing, and pins the three ways the fix could
quietly go wrong:

  * the injection must not count as a recall — it runs every turn, so counting
    it would turn `recalls` into "age in turns" and destroy the one signal for
    which facts the agent reaches for on purpose;
  * the block must be bounded, and must *say* when it truncated rather than
    silently dropping the tail;
  * the three states — no store, empty, populated — must read differently, or
    the agent misdescribes its memory in one direction or the other.

The last test is the load-bearing one: not "the function returns a string" but
"the string is in the messages the model actually receives".
"""
from __future__ import annotations

import os
import tempfile

import pytest

from autoforge.agent import MEMORY_BUDGET_CHARS, MEMORY_ENTRY_CHARS, ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.store import ToolStore


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as td:
        s = ToolStore(os.path.join(td, "af.db"))
        yield s
        s.close()


def _agent(store=None) -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)


def _block(agent: ForgeAgent) -> str:
    return "\n".join(agent._memory_lines())


# ======================================================================
# reading for the prompt is not the same act as recalling on purpose
# ======================================================================
class TestTheInjectionDoesNotPolluteTheRecallCounter:
    def test_injection_leaves_recalls_at_zero(self, store):
        store.remember("hq", "the launch directory is D:/work")
        for _ in range(5):
            store.memory_for_injection()

        rows = store.memory_for_injection()
        assert rows[0]["recalls"] == 0

    def test_a_real_recall_still_counts(self, store):
        # Read the counter through the non-counting path: `recall` returns rows
        # it selected *before* incrementing them, so its own return value lags
        # the stored count by one. That off-by-one is pre-existing and left
        # alone here; observing via memory_for_injection tests the property
        # without depending on it.
        store.remember("hq", "D:/work")
        store.recall("hq")
        store.recall("hq")
        assert store.memory_for_injection()[0]["recalls"] == 2

    def test_injection_and_a_recall_do_not_compound(self, store):
        store.remember("hq", "D:/work")
        store.recall("hq")
        for _ in range(4):
            store.memory_for_injection()
        assert store.memory_for_injection()[0]["recalls"] == 1

    def test_a_recalled_fact_climbs_above_a_fresher_one(self, store):
        store.remember("old", "reached for constantly")
        store.remember("new", "written later, never looked up")
        store.recall("old")

        order = [r["key"] for r in store.memory_for_injection()]
        assert order == ["old", "new"]

    def test_a_fresh_store_falls_back_to_most_recent(self, store):
        store.remember("first", "a")
        store.remember("second", "b")
        order = [r["key"] for r in store.memory_for_injection()]
        assert order == ["second", "first"]


# ======================================================================
# three states, three sentences
# ======================================================================
class TestTheThreeStatesReadDifferently:
    def test_no_store_says_memory_is_off_not_empty(self):
        text = _block(_agent(store=None))
        assert "no store this session" in text
        assert "nothing persists" in text
        assert "none kept yet" not in text       # off is not the same as empty

    def test_empty_store_says_nothing_kept_yet(self, store):
        text = _block(_agent(store))
        assert "none kept yet" in text
        assert "no store" not in text

    def test_a_kept_fact_is_rendered_with_its_key(self, store):
        store.remember("db", "postgres lives on port 5433 here")
        text = _block(_agent(store))
        assert "db: postgres lives on port 5433 here" in text
        assert "none kept yet" not in text


# ======================================================================
# the block is bounded, and honest about what it dropped
# ======================================================================
class TestTheBlockIsBounded:
    def test_a_long_value_is_elided_not_dropped(self, store):
        store.remember("big", "x" * (MEMORY_ENTRY_CHARS * 3))
        text = _block(_agent(store))
        assert "big:" in text
        assert "…" in text
        assert len("x" * (MEMORY_ENTRY_CHARS * 3)) > len(text)

    def test_one_oversized_entry_still_appears(self, store):
        # total > budget, but it is the only entry: showing an empty section
        # would read as "I kept nothing", which is the opposite of the truth.
        store.remember("huge", "y" * (MEMORY_BUDGET_CHARS * 2))
        text = _block(_agent(store))
        assert "huge:" in text
        assert "none kept yet" not in text

    def test_many_entries_truncate_and_say_so(self, store):
        for i in range(20):
            store.remember(f"k{i}", "v" * 200)
        text = _block(_agent(store))
        assert "more kept" in text
        assert "recall() reads the rest" in text
        assert len(text) < MEMORY_BUDGET_CHARS + MEMORY_ENTRY_CHARS + 200

    def test_within_budget_nothing_is_dropped(self, store):
        store.remember("a", "one")
        store.remember("b", "two")
        text = _block(_agent(store))
        assert "more kept" not in text
        assert "a: one" in text and "b: two" in text


# ======================================================================
# the load-bearing one: it reaches the model
# ======================================================================
class TestItReachesTheModel:
    def test_the_kept_fact_is_in_the_system_message_of_a_real_run(self, store):
        store.remember("host", "this box has an RTX 4080")
        agent = _agent(store)
        agent.run("say hi")

        messages, _tools = agent.llm.calls[0]
        system = messages[0]
        assert system.role == "system"
        assert "this box has an RTX 4080" in system.content

    def test_a_memory_kept_mid_run_is_there_on_the_next_request(self, store):
        agent = _agent(store)
        # Turn 1: the model uses the real remember tool. Turn 2 must see it
        # without asking for it — that is the whole point of the change.
        agent.registry.call("remember", {"key": "lesson", "value": "evolve beats retry"})
        agent.run("carry on")

        messages, _tools = agent.llm.calls[0]
        assert "evolve beats retry" in messages[0].content

    def test_the_prompt_says_memory_is_off_when_there_is_no_store(self):
        agent = _agent(store=None)
        agent.run("say hi")
        messages, _tools = agent.llm.calls[0]
        assert "no store this session" in messages[0].content

    def test_building_the_prompt_does_not_write_a_ledger_event(self, store):
        # A per-turn write would grow the ledger by the turn count and make
        # `my_history` unreadable. Reading for the prompt is not an event.
        # (Building the prompt is what runs every turn; `run` also records its
        # own run events, which are real and stay.)
        store.remember("k", "v")
        agent = _agent(store)
        before = store.report()["events"]
        for _ in range(5):
            agent._effective_prompt()
        assert store.report()["events"] == before
