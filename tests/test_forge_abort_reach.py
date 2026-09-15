"""The four things that made one interruption look like a stuck repeat.

The operator's report was: "it repeats an operation, it looks stuck, and it
ignores me when I interrupt". Nothing in that was a network fault. It was four
separate defects whose symptoms were identical on screen:

  1. The operator's line only reached the *boundaries* of a forge round -- the
     round check and the sandbox. The model calls that write, attack and probe
     the tool were made by clients that had no predicate at all, so a round
     could spend minutes deaf. `ForgePipeline.forge` now attaches the predicate
     to every client it can reach, for the duration of the round.
  2. The heartbeat printed the *need* and nothing else. The need is the same
     string in round 1 and round 4, so every 30s line was byte-identical and a
     run making real progress read as a run repeating itself. The line now
     carries the round and the seconds spent in it.
  3. A round's failure line said what broke but not how long it took, so "this
     need is hard" and "that round took eleven minutes" were the same line.
  4. An aborted forge was read by the model as a call that *failed*, so its
     next act was the identical call -- the repeated operation itself. The
     agent now refuses that one verbatim retry.

Each is asserted here against the part that owns it.
"""
from __future__ import annotations

import io
import time

import pytest

from autoforge.cli import _LiveRun
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMAborted, LLMResponse, MockLLMClient
from autoforge.core.steering import Steering
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.tools.registry import ToolRegistry


# -- 1. the operator's line reaches the calls inside a round -----------------

class _Verifier:
    """Passes whatever it is handed, and holds a client like the real one."""

    class _Report:
        passed = True

        def to_dict(self):
            return {"passed": True}

        @property
        def failed(self):
            return []

    def __init__(self, llm=None):
        self.llm = llm
        self.sandbox = None
        self.adversary = None

    def verify(self, spec, sample_args=None):
        return self._Report()


def _pipeline_with_clients():
    from autoforge.forge.generator import GeneratedTool

    gen_client = MockLLMClient()
    ver_client = MockLLMClient()
    adv_client = MockLLMClient()

    seen: list[tuple[str, object]] = []

    class _Gen:
        def __init__(self):
            self.llm = gen_client

        def generate(self, need, context=""):
            # The moment that used to be deaf: a model call inside a round.
            seen.append(("generator", gen_client.abort_check))
            return GeneratedTool(
                name="t1", description="d",
                code="def t1(x=0):\n    return x\n", parameters={},
            )

    verifier = _Verifier(ver_client)
    verifier.adversary = type("_Adv", (), {"llm": adv_client})()
    pipeline = ForgePipeline(
        _Gen(), verifier, ToolRegistry(),
        config=ForgeConfig(max_rounds=1, require_execution=False,
                           require_trigger=False, require_negative=False),
        sandbox=Sandbox(timeout=5.0),
    )
    return pipeline, seen, (gen_client, ver_client, adv_client)


def test_every_client_in_a_round_can_hear_the_operator():
    """The generator's own call is the one that used to block for two minutes.

    `should_abort` was only ever consulted at round boundaries and by the
    sandbox, so the model call that *writes* the tool -- the longest single
    step -- was outside the operator's reach.
    """
    pipeline, seen, clients = _pipeline_with_clients()
    said = {"now": False}

    pipeline.forge("a need", should_abort=lambda: said["now"])

    assert seen, "the generator was never called"
    for stage, predicate in seen:
        assert predicate is not None, (
            f"the {stage}'s model call had no abort predicate: an operator "
            "speaking during it would have been ignored until the round ended"
        )


def test_the_clients_are_handed_back_afterwards():
    """The predicate is borrowed, not donated.

    These clients outlive the forge -- the verifier owns its client and hands
    the same one to the probe agents -- so leaving a stale predicate attached
    would let a long-gone operator abort somebody else's work.
    """
    pipeline, seen, clients = _pipeline_with_clients()

    def naive() -> bool:
        return False

    pipeline.forge("a need", should_abort=naive)

    for client in clients:
        assert client.abort_check is None, (
            "a forge left its abort predicate on a client that outlives it"
        )


def test_a_nested_loop_gives_the_abort_back_instead_of_spinning():
    """An inherited predicate is not this loop's to answer.

    A probe agent runs on the client its parent owns, so it inherits the
    parent's predicate and can be woken by a line addressed to the parent. Its
    own channel is empty, so `_absorb` has nothing to fold in and the round it
    tries to re-ask aborts instantly -- forever. The honest move is to raise,
    which hands the line back to the loop that can read it.
    """
    calls = {"n": 0}

    def handler(messages, tools, **kw):
        calls["n"] += 1
        if calls["n"] > 6:
            raise AssertionError(
                "the nested loop spun: it re-asked a question its own channel "
                "had no answer for"
            )
        raise LLMAborted("the parent's operator spoke")

    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=Steering(printer=lambda t: None),
                  allow_self_terminate=False)

    with pytest.raises(LLMAborted):
        agent.run("a task")

    assert calls["n"] == 1, (
        f"the abort was re-asked {calls['n']} times instead of being handed up"
    )


# -- 2 & 3. the progress line says something new each time -------------------

def _live() -> tuple[_LiveRun, io.StringIO]:
    buf = io.StringIO()
    live = _LiveRun(stream=buf)          # not a tty: the durable-log case
    return live, buf


def test_the_forge_announces_which_round_of_how_many():
    live, buf = _live()
    live("forge_start", {"need": "probe the network", "max_rounds": 4})

    assert "round 1 of 4" in buf.getvalue(), (
        f"the opening line does not say how long the wait may be: "
        f"{buf.getvalue()!r}"
    )


def test_a_failed_round_says_how_long_it_took():
    """Without the duration, "hard need" and "slow round" are one line."""
    live, buf = _live()
    live("forge_start", {"need": "n", "max_rounds": 4})
    live("forge_attempt", {
        "round": 2, "accepted": False, "error": "verification failed",
        "duration_ms": 41000.0,
    })

    line = [ln for ln in buf.getvalue().splitlines() if "round 2" in ln][-1]
    assert "(41s)" in line, f"the round's duration is missing from {line!r}"


def test_the_heartbeat_line_changes_as_the_work_advances():
    """The defect in one assertion: two beats must not be identical lines.

    Before, the beat printed only the need, which does not change from round to
    round -- so every 30s line in the transcript was the same characters, and a
    forge making real progress was indistinguishable from one repeating itself.
    """
    live, buf = _live()
    live("forge_start", {"need": "probe the network", "max_rounds": 4})
    live._report()

    before = [ln for ln in buf.getvalue().splitlines() if "still forging" in ln]
    assert before, "the heartbeat printed nothing while a forge was running"
    assert "round 1" in before[-1], (
        f"the beat does not say which round it is in: {before[-1]!r}"
    )

    live("forge_attempt", {
        "round": 3, "accepted": False, "error": "probe failed",
        "duration_ms": 1200.0,
    })
    live._report()

    beats = [ln for ln in buf.getvalue().splitlines() if "still forging" in ln]
    assert len(beats) >= 2
    assert beats[-1] != beats[-2], (
        "two heartbeat lines in different rounds are byte-identical -- this is "
        "the 'it looks stuck repeating itself' the operator reported"
    )
    assert "round 3" in beats[-1]
    assert "in this round" in beats[-1], (
        "the beat does not say how long the current round has been running"
    )


def test_the_round_clock_is_cleared_when_the_forge_ends():
    """A stale round must not leak into whatever runs next."""
    live, buf = _live()
    live("forge_start", {"need": "n", "max_rounds": 4})
    live("forge_attempt", {"round": 2, "accepted": False, "error": "x"})
    live("forge_done", {"need": "n", "ok": True, "rounds": 2, "name": "t1"})

    assert live._forge_round is None and live._forge_since is None

    live._last_report = 0.0
    live._report()
    tail = [ln for ln in buf.getvalue().splitlines() if "still" in ln][-1]
    assert "forging" not in tail, (
        f"the beat still claims to be forging after the forge finished: {tail!r}"
    )


# -- 4. an interrupted forge is not repeated verbatim ------------------------

class _AbortedPipeline:
    """A pipeline whose every round the operator interrupts."""

    def __init__(self):
        self.needs: list[str] = []

    def forge(self, need, context="", should_abort=None):
        self.needs.append(need)

        class _R:
            ok = False
            aborted = True
            rounds = 0
            spec = None

        return _R()


def _forge_tool_fn():
    from autoforge.agent import ForgeAgent

    agent = ForgeAgent(MockLLMClient())
    stub = _AbortedPipeline()
    agent.pipeline = stub
    tool = agent.registry.get("forge_tool")
    assert tool is not None, "forge_tool was not registered"
    return agent, stub, tool.fn


def test_an_interrupted_need_is_not_immediately_re_asked():
    """The repeated operation itself, at its source.

    An abandoned tool call reads to the model as a call that *failed*, so its
    next act was the same tool with the same need -- one forge, the operator's
    line, then the identical forge again. The second attempt is refused once,
    because the person who spoke is still waiting for an answer.
    """
    agent, stub, forge_tool = _forge_tool_fn()

    first = forge_tool(need="probe the local network")
    assert "Stopped at the operator's request" in first
    assert stub.needs == ["probe the local network"]

    second = forge_tool(need="probe the local network")
    assert "Do not start it again" in second, (
        f"the interrupted need was re-forged verbatim: {second!r}"
    )
    assert stub.needs == ["probe the local network"], (
        "the forge ran a second time despite the operator still waiting"
    )


def test_the_refusal_is_consumed_and_a_changed_need_gets_through():
    """One interruption buys one re-ask. This must not become a wall.

    The operator's line is usually a correction -- "no, the other host" -- so
    the changed need has to work, and the guard must not harden into a
    permanent ban on a need that was interrupted once.
    """
    agent, stub, forge_tool = _forge_tool_fn()

    forge_tool(need="probe the local network")          # interrupted
    forge_tool(need="probe the local network")          # refused, mark spent

    forge_tool(need="probe the remote network")
    assert stub.needs[-1] == "probe the remote network", (
        "a changed need was blocked by the interruption guard"
    )

    # Only the need that is *currently* interrupted is held. The refused one
    # was released, so it is forgeable again -- the guard answers the one
    # unanswered interruption, it does not accumulate.
    assert agent._interrupted_needs == {"probe the remote network"}
    forge_tool(need="probe the local network")
    assert stub.needs[-1] == "probe the local network", (
        "the guard outlived the interruption it was there to answer"
    )


def test_the_abort_message_tells_the_model_not_to_repeat_it():
    """The refusal above is a backstop; the message is the primary fix."""
    _agent, _stub, forge_tool = _forge_tool_fn()
    msg = forge_tool(need="probe the local network")

    assert "Do not call forge_tool again with this same need" in msg, (
        f"the abort message does not tell the model what not to do: {msg!r}"
    )
