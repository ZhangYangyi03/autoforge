"""The two promises, asserted end to end rather than at the seams.

`test_interrupt_responsiveness.py` checks the parts (the polled wait, the
sandbox's kill path, the heartbeat's cadence). This checks the promise the
operator actually made: *speak during a long step and the run reacts to you, in
seconds, and it keeps telling you where it is while it works.*

Both were broken in the same shape -- the only place the operator's line could
land was a boundary that was minutes away -- so both are asserted here against a
deliberately slow model, which is the only way the defect is visible. On a fast
model there is no gap to fall into, which is why this survived so long.
"""
from __future__ import annotations

import io
import threading
import time

from autoforge.cli import _LiveRun
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMAborted, LLMResponse, MockLLMClient, tool_call
from autoforge.core.steering import Steering
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec

#: Long enough that the old boundary-only behaviour could not possibly pass,
#: short enough to keep the suite quick.
STALL = 20.0
#: Well inside the operator's 30s -- the assertion is on the guarantee, not on
#: the measured 13ms, so it stays true on a loaded machine.
DEADLINE = 5.0


def _slow_model(stall: float = STALL):
    """A client that blocks like a real endpoint and honours the contract."""
    sent: list[float] = []
    prompts: list[str] = []
    aborted: list[float] = []

    def handler(messages, tools, **kw):
        sent.append(time.monotonic())
        prompts.append([m.content for m in messages if m.role == "user"][-1])
        should_abort = kw.get("should_abort")
        while time.monotonic() - sent[-1] < stall:
            time.sleep(0.02)
            if should_abort is not None and should_abort():
                aborted.append(time.monotonic())
                raise LLMAborted("the operator spoke")
        return LLMResponse(content="answered without interruption")

    return handler, sent, prompts, aborted


def test_a_running_call_yields_to_the_operator_and_reasks_informed():
    handler, sent, prompts, aborted = _slow_model()
    said: list[str] = []
    steer = Steering(printer=said.append)
    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=steer, allow_self_terminate=False)

    worker = threading.Thread(
        target=lambda: agent.run("a task that takes a while"), daemon=True)
    worker.start()
    time.sleep(1.0)

    spoke = time.monotonic()
    steer.submit("actually, do it the other way round")
    # Two stalls, not one: the line buys a reply turn of its own before the
    # work resumes (see
    # `test_a_line_typed_mid_run_buys_a_reply_not_just_a_slot_in_the_context`).
    # That reply is a request the operator's line costs, and it is the whole
    # reason they speak -- "it hears me and says nothing" is the complaint this
    # pair of tests exists to answer.
    worker.join(timeout=STALL * 2 + 10.0)

    assert not worker.is_alive(), "the run never came back for the correction"
    # It was heard, and said so, immediately.
    assert said and ("heard" in said[0])
    # The step that was already running gave way rather than being waited out.
    assert aborted and (aborted[0] - spoke) < DEADLINE
    # And the correction is what the next question carries -- otherwise the
    # interruption bought a retry of the same uninformed question.
    assert len(sent) > 2 and (sent[1] - spoke) < DEADLINE
    assert "the other way round" in prompts[2]


def test_a_line_typed_mid_run_buys_a_reply_not_just_a_slot_in_the_context():
    """The operator's complaint in full: it says "heard", then works for
    fifteen more minutes without answering.

    A line folded into the list is invisible to the person who typed it -- the
    model is free to fold a question into its next tool call and say nothing,
    which is exactly what the ledger shows: the line was absorbed at 00:00:27
    and the run made 60 more tool calls without a word. So a line that arrives
    mid-run also buys one question asked with the tool list empty, where
    answering is the only thing that can happen.
    """
    seen: list[list[str]] = []

    def handler(messages, tools, **kw):
        seen.append([t["function"]["name"] for t in (tools or [])])
        if not tools:
            return LLMResponse(content="checking the bus; two minutes left")
        return LLMResponse(tool_calls=[tool_call("poke", {})])

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="poke", description="Do one small piece of the work.",
        parameters={"type": "object", "properties": {}},
        fn=lambda **kw: "poked"))
    registry.promote("poke")                  # draft tools are not exposed
    replies: list[str] = []
    steer = Steering(printer=lambda t: None)
    agent = Agent(MockLLMClient(handler=handler), registry, steer=steer,
                  max_turns=6, allow_self_terminate=False,
                  on_reply=replies.append)

    steer.submit("what are you doing?")
    result = agent.run("poke the bus until it answers")

    # A turn was asked with no tools at all: that is the reply.
    assert [] in seen, f"no tool-free turn was ever asked: {seen}"
    assert replies == ["checking the bus; two minutes left"]
    # ...and it is part of the conversation, so the work that resumes knows
    # what the operator was already told.
    assert any(m.role == "assistant" and "checking the bus" in (m.content or "")
               for m in result.messages)


def test_a_stop_also_reaches_inside_a_running_call():
    """`/stop` that waits for a boundary is a note, not a stop."""
    handler, sent, prompts, aborted = _slow_model()
    steer = Steering(printer=lambda t: None)
    agent = Agent(MockLLMClient(handler=handler), ToolRegistry(),
                  steer=steer, allow_self_terminate=False)

    box: dict = {}
    worker = threading.Thread(
        target=lambda: box.update(r=agent.run("a long task")), daemon=True)
    worker.start()
    time.sleep(1.0)
    steer.submit("/stop")
    worker.join(timeout=STALL + 10.0)

    assert not worker.is_alive(), "stop did not take effect until the call ended"
    assert aborted, "the in-flight call was not abandoned"
    assert box["r"].stopped_by_operator is True


def test_a_quiet_run_keeps_saying_where_it_is():
    """Silence is the defect: a working run must not read as a dead one."""
    buf = io.StringIO()
    live = _LiveRun(stream=buf)          # not a tty: the durable-log case
    live.REPORT_EVERY = 0.5
    live.start()
    live("request", {"turn": 3})
    # The beat wakes once a second, so the cadence is `REPORT_EVERY` rounded up
    # to the next beat. Two full beats is the smallest honest window here.
    time.sleep(2.6)
    live._running = False
    live.stop()

    reports = [ln for ln in buf.getvalue().splitlines() if "still" in ln]
    assert len(reports) >= 2, f"a silent run produced {len(reports)} report(s)"
    assert "turn 3" in reports[0]
    # The every-second tick is a rewrite in place, which means nothing in a
    # log: one line per second, all identical. A tick line leads with the
    # ellipsis; an event line merely ends with one ("asking the model…").
    ticks = [ln for ln in buf.getvalue().splitlines() if ln.strip().startswith("…")]
    assert ticks == [], f"tick lines leaked into a log: {ticks[:2]}"


def _elapsed_of(line: str) -> float:
    """The `+12.3s` every CLI line carries, as a number."""
    at = line.index("+")
    return float(line[at + 1: line.index("s", at)])


def test_the_default_cadence_is_the_one_the_operator_asked_for():
    """The config, not just the mechanism. Two numbers, both the operator's.

    30s is the cadence they asked for -- "one line every 30 seconds, one step
    at a time, like you do" -- and 100s is what they will tolerate before a
    run reads as hung. The first is the one that has to hold, and a 60s
    ceiling did not: it sat *above* the pauses a real run produces (a 41s
    forge round, a 58s model wait), so it never fired when it mattered.
    """
    assert _LiveRun.REPORT_EVERY <= 30.0
    assert _LiveRun.REPORT_EVERY + _LiveRun.BEAT_SECONDS <= 100.0


def test_no_silence_outlives_the_configured_cadence():
    """The bound, measured on the real beat thread rather than asserted.

    `REPORT_EVERY` is a number in a class body; whether the thread acts on it
    is a separate claim, and it is the one that matters. Driven here on a fast
    beat so the check costs a second instead of a minute.
    """
    buf = io.StringIO()
    live = _LiveRun(stream=buf)
    live.BEAT_SECONDS = 0.02
    live.REPORT_EVERY = 0.10
    live.start()
    try:
        live("request", {"turn": 1})       # a run is underway; now say nothing
        time.sleep(0.8)
    finally:
        live._running = False
        live.stop()

    stamps = [_elapsed_of(ln) for ln in buf.getvalue().splitlines() if "still" in ln]
    assert len(stamps) >= 3, f"0.8s of silence at a 0.1s cadence gave {stamps}"
    gaps = [b - a for a, b in zip([0.0] + stamps, stamps)]
    bound = live.REPORT_EVERY + live.BEAT_SECONDS
    worst = max(gaps)
    assert worst <= bound + 0.25, (
        f"a {worst:.2f}s silence against a {bound:.2f}s bound")
