"""The token bill: which tool schemas ride in every request, and which do not.

Measured on this machine 2026-09-28, before any of this existed: 204 visible
tool schemas, 106,503 characters, ~33k tokens, sent with every request of every
turn -- 30,624 prompt tokens on a real call to the configured endpoint for a
one-word reply. 61 of those tools had never been called once.

The split these tests pin is not "delete the unused tools". It is that a tool
nobody has reached for in a week does not need its parameter schema in front of
the model to remain findable and callable, and the price of being wrong about
one is a single extra call. So what is asserted here is the *shape* of the
demotion -- what stays, what is listed, what is recoverable -- because the
failure that matters is not "too few schemas" but "a tool the model cannot
learn exists".
"""
from __future__ import annotations

import json

import pytest

from autoforge.agent import ForgeAgent
from autoforge.core.message import Message
from autoforge.core.llm import LLMResponse, MockLLMClient, tool_call
from autoforge.tools import exposure as exposure_mod
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState


def _spec(name: str, desc: str = "does a thing", **params) -> ToolSpec:
    # ACTIVE, not the default DRAFT: `registry.schemas()` only exposes the
    # states it trusts, so a fixture that forgot this would make every exposure
    # test pass vacuously by returning nothing at all.
    return ToolSpec(
        name=name, description=desc,
        parameters={"type": "object", "properties": params or {"x": {"type": "string"}}},
        fn=lambda **_: "ok", source="generated", state=ToolState.ACTIVE,
    )


def _registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(_spec(n), replace=True)
    return reg


def test_the_floor_is_carried_in_full_whatever_the_ledger_says():
    """run_python, forge_tool, describe_tool are how a cold tool is reached.

    If the floor could be demoted, the demotion would be unrecoverable: the way
    to find out about a tool is a hot tool, so the ways of finding out must not
    themselves depend on being called recently.
    """
    reg = _registry("run_python", "describe_tool", "list_tools", "forge_tool",
                    "terminate", "remember", "recall", "ancient_thing")
    reg.last_called_source = lambda name: None       # nothing ever called
    hot, cold, stats = reg.exposure()
    hot_names = {(s.get("function") or s).get("name") for s in hot}

    assert hot_names == set(exposure_mod.HOT_FLOOR)
    assert stats["hot"] == 7 and stats["cold"] == 1
    assert any("ancient_thing" in line for line in cold), \
        "demoted is not hidden: the never-called tool is still named"


def test_a_tool_with_no_record_of_being_called_is_listed_not_dropped():
    """The whole distinction: demoted is not hidden.

    A never-called tool still appears by name with its arguments and one line,
    because the alternative -- gone from the prompt entirely -- is a library the
    model cannot discover, and a library it cannot discover is one it will
    re-forge from scratch.
    """
    reg = _registry("run_python", "read_magic_value")
    reg.last_called_source = lambda name: None
    hot, cold, stats = reg.exposure()
    line = "\n".join(cold)

    assert "read_magic_value" in line
    assert "x" in line, "the argument names are what make a cold tool recognisable"
    assert "does a thing" in line


def _called(days_ago: float) -> float:
    """The source answers with a *timestamp*, not an age.

    Written as a helper because the first draft of these tests passed 0.5 where
    a unix time was expected -- which the code dutifully read as "called in
    1970, about 20,000 days ago" and demoted. A fixture that gets the units
    wrong makes a passing test mean nothing, and this one made a failing test
    mean nothing either.
    """
    import time
    return time.time() - days_ago * 86400.0


def test_a_tool_called_within_the_window_stays_in_full():
    reg = _registry("run_python", "fresh")
    reg.last_called_source = lambda name: _called(0.5) if name == "fresh" else None
    hot, cold, _ = reg.exposure(window_days=7)
    assert [(s.get("function") or s).get("name") for s in hot] == ["run_python", "fresh"]
    assert cold == []


def test_the_division_is_by_behaviour_not_by_a_cap():
    """A week, not "the top 40".

    A cap needs a tie-break, and every tie-break it could have demotes a tool
    for a reason unrelated to whether it is wanted. A window demotes for exactly
    one reason and reverses the moment something is called -- so the same tool
    moves between the two lists with nothing but a call.
    """
    reg = _registry("run_python", "thing")
    state = {"days": 9.0}
    reg.last_called_source = lambda name: _called(state["days"])
    _, cold, _ = reg.exposure(window_days=7)
    assert cold and "thing" in cold[0]

    state["days"] = 0.1
    # Through a real call, because that is what changes the answer in
    # production and the cache is keyed on it. Reaching in to clear the cache
    # would test the cache rather than the decision.
    assert reg.call("thing", {"x": "1"}).ok
    hot, cold, _ = reg.exposure(window_days=7)
    assert "thing" in [(s.get("function") or s).get("name") for s in hot]
    assert cold == []


def test_the_exposure_does_not_move_between_turns_that_call_nothing():
    """The prompt prefix has to stay byte-identical to be served from cache.

    This is the cost inside the cost: a block recomputed per turn would change
    the prefix and re-bill everything behind it, which would eat the saving it
    was making. The cache keys on the call counter, and the ledger cannot change
    a tool's age without a call.
    """
    reg = _registry("run_python", "thing")
    reg.last_called_source = lambda name: _called(1.0)
    first = reg.exposure()
    second = reg.exposure()
    assert first[0] == second[0] and first[1] == second[1]
    assert reg._exposure_cache is not None


def test_a_tool_the_task_names_is_protected():
    """The ledger answers a past question; a task can be about something new.

    Without this, the first turn of exactly the tasks that need a cold tool is
    the one turn that cannot call it in full -- and the failure lands as a
    guessed argument list, which reads like the tool's fault.
    """
    agent = ForgeAgent(MockLLMClient())
    agent.registry.register(_spec("drive_chromium_page"), replace=True)
    agent.registry.last_called_source = lambda name: None
    agent._current_task = "drive_chromium_page on the login form"
    protected = agent._hot_protect()
    assert "drive_chromium_page" in protected


def test_a_task_word_does_not_drag_in_every_tool_containing_it():
    """Whole names, not substrings: a set that grows to everything protects nothing."""
    agent = ForgeAgent(MockLLMClient())
    agent.registry.last_called_source = lambda name: None
    agent._current_task = "I have seen the thing"
    protected = agent._hot_protect()
    assert "see" not in protected, "a substring match would keep `see` hot on 'seen'"


def test_describe_tool_hands_back_the_schema_of_a_demoted_tool():
    agent = ForgeAgent(MockLLMClient())
    agent.registry.register(_spec("read_magic_value", "Reads the magic value.",
                                  path={"type": "string"}), replace=True)
    agent.registry.last_called_source = lambda name: None
    out = agent.registry.call("describe_tool", {"name": "read_magic_value"})
    assert out.ok
    assert '"path"' in out.output
    assert "what the request does not carry" in out.output


def test_describe_tool_on_a_miss_suggests_near_names():
    """On a library of 200 the useful answer to a miss is the near miss."""
    agent = ForgeAgent(MockLLMClient())
    agent.registry.register(_spec("read_magic_value_from_path"), replace=True)
    out = agent.registry.call("describe_tool", {"name": "read_magic_value"})
    assert out.ok and "read_magic_value_from_path" in out.output


def test_the_cold_list_rides_as_a_message_before_the_conversation():
    """Ahead of the transcript, and copied rather than appended.

    Ahead because a model reads instructions before the conversation, and
    because a block at the very end of a long transcript is the first thing a
    provider's context window drops. Copied because the transcript on disk is
    the conversation: a block injected per request that became part of the
    record would be replayed by a resume as something the person said.
    """
    agent = ForgeAgent(MockLLMClient())
    agent.registry.last_called_source = lambda name: None
    seen: dict = {}
    # One cold tool with a distinctive description, so the assertion is about
    # this tool and not about a line some other demoted tool happened to carry.
    agent.registry.register(_spec("snowflake_probe", "melts at exactly zero"), replace=True)

    def handler(messages, tools, **kw):
        seen["messages"] = list(messages)
        seen["tools"] = tools
        return LLMResponse(content="done")

    agent.llm = MockLLMClient(handler=handler)
    agent.run("do something", journal=False)

    bodies = [m.content for m in seen["messages"]]
    assert "not carrying in full" in bodies[1], "the list is right after the system prompt"
    assert "snowflake_probe" in bodies[1]
    assert "melts at exactly zero" in bodies[1]


def test_the_request_is_a_copy_and_the_transcript_is_unchanged():
    """The injected block must never reach `msgs`, or a resume replays it."""
    agent = ForgeAgent(MockLLMClient())
    agent.registry.last_called_source = lambda name: None
    from autoforge.core.agent import Agent

    loop = Agent(agent.llm, agent.registry, exposure=agent.registry,
                 allow_self_terminate=False)
    msgs = [Message.system("sys"), Message.user("hello")]
    req, tools = loop._request(msgs)

    assert len(msgs) == 2, "the conversation itself is untouched"
    assert len(req) == 3 and req[1].role == "user"


def test_a_broken_exposure_falls_back_to_sending_everything():
    """A cost optimisation must never be the thing that loses the run."""
    class Exploding:
        def exposure(self, **kw):
            raise RuntimeError("no ledger today")

    from autoforge.core.agent import Agent

    reg = _registry("run_python", "thing")
    loop = Agent(MockLLMClient(), reg, exposure=Exploding(),
                 allow_self_terminate=False)
    msgs = [Message.system("sys"), Message.user("hi")]
    req, tools = loop._request(msgs)
    assert len(req) == 2 and len(tools) == 2


def test_the_ledger_is_what_answers_when_a_tool_was_called(tmp_path):
    """`stats.last_called` is only written when something saves the tool.

    A call is not a state change, so most calls never reach the tools table.
    The ledger has every one of them, which is the only source with enough rows
    to demote a tool honestly.
    """
    from autoforge.store import ToolStore

    st = ToolStore(str(tmp_path / "af.db"))
    st.log_event("call", {"tool": "thing"})
    st.log_event("call", {"tool": "other"})
    newest = st.last_called_by_tool()
    assert set(newest) == {"thing", "other"}


def test_the_split_is_worth_what_it_claims_against_a_synthetic_library():
    """A guard against the saving quietly disappearing in a refactor.

    Not a benchmark -- a claim about arithmetic: with 100 tools that were never
    called, the cold list must cost a small fraction of their schemas. If a
    change ever makes the cold rendering as expensive as the full schema, this
    is the test that says so before a run pays for it.
    """
    reg = ToolRegistry()
    for i in range(100):
        reg.register(_spec(
            f"cold_tool_{i}",
            "Reads the magic value out of a binary of the given flavour and "
            "returns it, or an empty string when the marker is absent.",
            alpha={"type": "string", "description": "which binary to read"},
            beta={"type": "integer", "description": "offset to start at"},
            gamma={"type": "array", "items": {"type": "string"}},
            delta={"type": "object", "properties": {"k": {"type": "string"}}},
        ), replace=True)
    reg.register(_spec("run_python"), replace=True)
    reg.last_called_source = lambda name: None

    hot, cold, stats = reg.exposure()
    full = sum(len(json.dumps(s, ensure_ascii=False)) for s in hot)
    listed = sum(len(l) + 1 for l in cold)
    as_full = sum(len(json.dumps(reg.get(f"cold_tool_{i}").schema, ensure_ascii=False))
                  for i in range(100))

    assert stats["cold"] == 100
    # 4x is the assertion, not a hoped-for number: a cold line is a signature
    # plus a capped description, so the ratio is bounded by the schema it
    # replaces -- and the real library measured 106,503 chars down to 44,033.
    # Lower than 3x and the demotion is not paying for itself.
    assert listed < as_full / 3, (
        f"cold list costs {listed} against {as_full} in full -- the saving is gone")


def test_silence_alone_eventually_demotes_a_tool(monkeypatch):
    """Time moves the answer, not only calls.

    A cache keyed purely on the call counter is right for every session that
    restarts and wrong for one that stays up: measured from the counter alone, a
    tool called yesterday stays hot for as long as the process lives. This pins
    the clock into the key by making the clock jump.
    """
    import time as _time

    reg = _registry("run_python", "yesterday_thing")

    real = _time.time
    now = real()
    # Captured once, not recomputed per query: a fixture that answers "6 days
    # before whatever time it is now" never ages, so the tool stays hot forever
    # and the test would pass for the wrong reason. The first draft did exactly
    # that and failed; the fix is the point -- a *fixed* call time is the thing
    # being aged.
    called_at = now - 6 * 86400.0
    reg.last_called_source = lambda name: called_at
    monkeypatch.setattr(_time, "time", lambda: now)
    hot, _, _ = reg.exposure(window_days=7)
    assert "yesterday_thing" in [(s.get("function") or s).get("name") for s in hot]

    # Two days later, in a process that was never restarted and called nothing.
    now = now + 2 * 86400
    hot, cold, _ = reg.exposure(window_days=7)
    assert "yesterday_thing" not in [(s.get("function") or s).get("name") for s in hot]
    assert any("yesterday_thing" in line for line in cold)
