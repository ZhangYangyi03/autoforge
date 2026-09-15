"""The summary is a long step too, so it has to be able to yield.

The loop asks the steering channel before every one of *its* model calls, and
a running forge asks from inside itself. The summarizer asked nobody, so the
one call in the framework that fires exactly when the operator is waiting at a
turn boundary was also the one call they could not interrupt.

Two things had to be true for the promise to hold, and neither was:

  * the summary call has to carry the abort question (`LLMSummarizer`), and
  * an abort raised by it has to travel *out* rather than being caught as a
    summarizer failure. It was caught -- `LLMAborted` is an `InterruptedError`,
    so it is an `Exception`, so the `except Exception` that exists to fall back
    to the deterministic summarizer swallowed it. The operator's interruption
    was then spent waiting for a second summarizer, which rewrote the history
    they had just contradicted.

What the operator is owed is the last test here: their words on the next
question the model is asked, without waiting for a summary nobody wanted.
"""
from __future__ import annotations

import threading
import time

from autoforge.agent import ForgeAgent
from autoforge.core.agent import Agent
from autoforge.core.compaction import (
    CompactionPolicy,
    Compactor,
    DeterministicSummarizer,
    LLMSummarizer,
)
from autoforge.core.llm import LLMAborted, LLMResponse, MockLLMClient, ToolCall
from autoforge.core.message import Message
from autoforge.core.steering import Steering
from autoforge.modes import MinimalAgent
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


def compactor(summarizer, fallback=DeterministicSummarizer()) -> Compactor:
    """A compactor that fires almost immediately, with a chosen summarizer.

    `dump_transcripts` off: a test that writes 400KB transcripts into the
    operator's real `~/.autoforge/transcripts` is a test that leaves litter
    behind, and the litter is indistinguishable from a run that actually
    wedged.
    """
    return Compactor(
        summarizer=summarizer,
        fallback=fallback,
        policy=CompactionPolicy(max_context_tokens=150, keep_head=2,
                                keep_recent_groups=2, persist=False,
                                dump_transcripts=False),
    )


def bulky_transcript(notes: int = 8) -> list[Message]:
    return ([Message.system("s")]
            + [Message.user("x" * 400) for _ in range(notes)])


def bulky_llm(turns: int = 8) -> MockLLMClient:
    script = [
        LLMResponse(content="",
                    tool_calls=[ToolCall(f"c{i}", "note", {"text": "x" * 400})])
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
        fn=lambda text="": "y" * 400,
    ))
    return reg


# ======================================================================
# the call carries the question
# ======================================================================
class TestTheSummaryIsAskedToYield:

    def test_the_summarizer_asks_what_the_loop_asks(self):
        agent = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        summarizer = agent.compactor.summarizer
        assert isinstance(summarizer, LLMSummarizer)
        # The same callable, not a copy of its logic: a summarizer that asked
        # the question slightly differently would be the one step that failed
        # to yield, which is the bug this whole file is about.
        assert summarizer.should_abort == agent._operator_wants_the_floor

    def test_the_minimal_mode_asks_it_too(self):
        # The control group gets the same context management, so it has to get
        # the same interruptibility, or the comparison measures patience.
        mode = MinimalAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        summarizer = mode.compactor.summarizer
        assert isinstance(summarizer, LLMSummarizer)
        assert summarizer.should_abort == mode._operator_wants_the_floor

    def test_the_question_reaches_the_client(self):
        seen: list[bool] = []

        def handler(messages, tools, **kwargs):
            seen.append("should_abort" in kwargs)
            return LLMResponse(content="a summary")

        client = MockLLMClient(handler=handler)
        LLMSummarizer(client, should_abort=lambda: True).summarize(
            bulky_transcript())
        assert seen == [True]

    def test_without_a_channel_the_call_is_unchanged(self):
        # A compactor built by hand -- a library user, a script -- has nobody
        # to ask, and must not be asked to supply a question it cannot answer.
        client = MockLLMClient(script=[LLMResponse(content="a summary")])
        text = LLMSummarizer(client).summarize(bulky_transcript())
        assert text == "a summary"
        assert client.calls, "the summarizer never called the model"


# ======================================================================
# the abort travels out, and the fallback is not consulted
# ======================================================================
class TestAnAbortedSummaryDefers:

    def test_it_does_not_fall_back_to_the_other_summarizer(self):
        calls: list[str] = []

        class Yielding:
            name = "yielding"

            def summarize(self, msgs):
                calls.append("summarize")
                raise LLMAborted("the operator spoke")

        class Fallback:
            name = "fallback"

            def summarize(self, msgs):
                calls.append("fallback")
                return "the deterministic note"

        c = compactor(Yielding(), fallback=Fallback())
        msgs = bulky_transcript()
        before = list(msgs)

        assert c.maybe_compact(msgs, turn=3) is None
        # Making the operator wait out a second summarizer, only for the range
        # to be folded away by the one they were talking over, is the worst of
        # both outcomes. Nothing is summarized, so nothing is dropped.
        assert calls == ["summarize"]
        assert msgs == before, "an aborted pass dropped history"
        assert c.events == [], "an aborted pass was recorded as a compaction"

    def test_a_summarizer_that_merely_fails_still_degrades(self):
        # The guard above must not have eaten the degrade path: a model that
        # errors is a summarizer that failed, not an operator who spoke.
        class Broken:
            name = "broken"

            def summarize(self, msgs):
                raise ValueError("the model returned junk")

        c = compactor(Broken())
        event = c.maybe_compact(bulky_transcript(), turn=1)
        assert event is not None
        assert event.summarizer == "deterministic", (
            "a broken summarizer stopped the fallback from running")


# ======================================================================
# the operator gets the run back, mid-summary
# ======================================================================
class _WaitsForTheOperator:
    """A summary that never finishes on its own: it waits for the operator.

    The real one is a model call that can run for minutes, so the case worth
    testing is the one where it is *in flight* when the operator types. Once
    it has yielded, it stops standing in for a slow model -- the test is about
    the one call that was in flight, and a fake that blocks on every later
    pass would be measuring itself rather than the loop.
    """

    name = "waits"

    def __init__(self, steer, patience: float = 5.0) -> None:
        self.steer = steer
        self.patience = patience
        self.attempts = 0
        self.gave_way = False

    def summarize(self, msgs):
        self.attempts += 1
        if self.gave_way:
            return "a summary written after the interruption"
        deadline = time.monotonic() + self.patience
        while time.monotonic() < deadline:
            if self.steer.has_pending() or self.steer.stop_requested():
                self.gave_way = True
                raise LLMAborted("the operator spoke mid-summary")
            time.sleep(0.01)
        return "a summary nobody interrupted"


class TestTheLoopComesBackMidSummary:

    def test_the_correction_reaches_the_next_question(self):
        steer = Steering(printer=lambda _t: None)
        summarizer = _WaitsForTheOperator(steer)
        prompts: list[str] = []

        def handler(messages, tools, **kwargs):
            prompts.append("\n".join((m.content or "") for m in messages))
            if len(prompts) > 6:
                return LLMResponse(content="done")
            return LLMResponse(content="",
                               tool_calls=[ToolCall(f"c{len(prompts)}", "note",
                                                    {"text": "x" * 400})])

        agent = Agent(MockLLMClient(handler=handler), note_registry(),
                      compactor=compactor(summarizer), steer=steer,
                      max_turns=20, allow_self_terminate=False)
        worker = threading.Thread(
            target=lambda: agent.run("a long task"), daemon=True)
        worker.start()
        time.sleep(0.5)

        spoke = time.monotonic()
        steer.submit("actually, use csv not parquet")
        worker.join(timeout=30)
        came_back = time.monotonic() - spoke

        assert not worker.is_alive(), "the run never came back for the correction"
        assert summarizer.attempts >= 1, "no compaction pass was ever reached"
        assert summarizer.gave_way, "the summary was never asked to yield"
        assert any("use csv not parquet" in p for p in prompts), (
            "the operator's words never reached the model")
        # The whole point of yielding: from that moment it was the operator's
        # turn, not the summarizer's, and the run did not sit out its patience.
        assert came_back < summarizer.patience, (
            f"the run waited out the summary anyway ({came_back:.1f}s)")

