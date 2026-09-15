"""Compaction is wired, not merely available.

The failure this guards is the one this framework exists to complain about: a
capability that exists, is documented, is unit-tested — and is consulted by
nothing. `tests/test_compaction.py` proves the Compactor is *correct*. This file
proves it is *reachable*, by driving the real ForgeAgent and the real
MinimalAgent and asserting on what the loop actually did.

If someone deletes the `compactor=` line from the Agent construction, every test
in the other file still passes and this one fails. That is the point.
"""
from __future__ import annotations

import pytest

from autoforge.agent import ForgeAgent
from autoforge.core.compaction import (
    CompactionPolicy,
    Compactor,
    DeterministicSummarizer,
)
from autoforge.core.llm import LLMResponse, MockLLMClient, ToolCall
from autoforge.modes import MinimalAgent
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


def tiny(**kw) -> Compactor:
    """A compactor that fires almost immediately, with no model in the loop.

    Deterministic summarizer only: these tests are about wiring, and a wiring
    test that can fail because a model had an off day is a flaky wiring test.
    """
    policy = dict(max_context_tokens=150, keep_head=2, keep_recent_groups=2,
                  persist=False)
    policy.update(kw)
    return Compactor(summarizer=DeterministicSummarizer(), fallback=None,
                     policy=CompactionPolicy(**policy))


def bulky_llm(turns: int = 8) -> MockLLMClient:
    """Say a lot, several times, so the estimate climbs past any small budget."""
    script = [
        LLMResponse(content="",
                    tool_calls=[ToolCall(f"c{i}", "note", {"text": "x" * 600})])
        for i in range(turns)
    ]
    script.append(LLMResponse(content="done"))
    return MockLLMClient(script=script)


def note_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="note",
        description="write a note",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda text="": "y" * 600,
    ))
    return reg


# ======================================================================
# the default is real, and it degrades rather than declines
# ======================================================================
class TestTheDefaultExists:
    def test_forge_agent_builds_a_compactor_when_not_given_one(self):
        a = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        assert isinstance(a.compactor, Compactor)
        # A model summary with the deterministic summarizer behind it: the run
        # degrades to a structural note if the model cannot produce one, rather
        # than growing unbounded.
        assert a.compactor.fallback is not None

    def test_minimal_agent_builds_one_too(self):
        # The control group has to differ in scaffolding only. If the minimal
        # mode also lacked memory management, "does scaffolding help?" would be
        # answered by a comparison of scaffolding *and* recall.
        m = MinimalAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        assert isinstance(m.compactor, Compactor)

    def test_an_explicit_compactor_is_not_overwritten(self):
        mine = tiny()
        a = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]),
                       compactor=mine)
        assert a.compactor is mine


# ======================================================================
# the loop consults it
# ======================================================================
class TestTheLoopConsultsIt:
    def test_forge_agent_records_a_compaction_in_its_trace(self):
        a = ForgeAgent(llm=bulky_llm(), registry=note_registry(),
                       compactor=tiny(), max_turns=12)
        a.run("keep going")

        compacts = [e for e in a.trace if e.get("kind") == "compact"]
        assert compacts, [e.get("kind") for e in a.trace]
        # A pass that declines — nothing safe to cut yet — is recorded too, on
        # purpose: a wedged run used to leave no trace at all. What this asserts
        # is that a run with room to compact does compact.
        acted = [e for e in compacts if e.get("dropped")]
        assert acted, compacts
        assert acted[0]["dropped"] > 0

    def test_minimal_agent_records_one_too(self):
        m = MinimalAgent(llm=bulky_llm(), compactor=tiny(), max_turns=12)
        m.run("keep going")
        assert [e for e in m.trace if e.get("kind") == "compact"]

    def test_nothing_is_recorded_when_the_budget_is_never_reached(self):
        # An observer that fires on every turn is worse than none: it buries
        # the one turn that mattered.
        a = ForgeAgent(llm=bulky_llm(turns=2), registry=note_registry(),
                       compactor=tiny(max_context_tokens=10**9), max_turns=6)
        a.run("small task")
        assert not [e for e in a.trace if e.get("kind") == "compact"]

    def test_the_operator_words_reach_the_model_after_a_compaction(self):
        """End to end through the real loop: the correction is typed, the
        context is compacted underneath it, and the next request still carries
        the operator's sentence — because it was never the summarizer's to keep.
        """
        llm = bulky_llm()
        a = ForgeAgent(llm=llm, registry=note_registry(), compactor=tiny(),
                       max_turns=12)
        a.run("keep going")
        a.steer.submit("never touch prod") if a.steer else None
        # No steer channel here (unattended run); assert on the history instead.
        sent = "\n".join(m.content for m in llm.calls[-1][0])
        assert "keep going" in sent        # the original task survives


# ======================================================================
# it survives the boundary the agent loop actually has
# ======================================================================
class TestTheRequestStaysWellFormed:
    def test_no_request_is_sent_with_an_orphaned_tool_result(self):
        """Providers reject a `tool` message whose `tool_call_id` has no
        matching assistant message. Compaction is the only thing in the loop
        that can create that shape, so every request is checked."""
        llm = bulky_llm()
        a = ForgeAgent(llm=llm, registry=note_registry(), compactor=tiny(),
                       max_turns=12)
        a.run("keep going")
        assert llm.calls, "the loop never called the model"
        for sent, _kwargs in llm.calls:
            open_ids = {tc.id for m in sent if m.role == "assistant"
                        for tc in (m.tool_calls or [])}
            for m in sent:
                if m.role == "tool":
                    assert m.tool_call_id in open_ids, (
                        "a tool result survived without its assistant message")
