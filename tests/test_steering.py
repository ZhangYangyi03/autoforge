"""Steering: talking to a run while it is happening.

Two halves, tested against real objects. The channel itself, and — the part
that matters more — that the agent *loop* actually reads from it. A channel
nobody drains is the same fiction as a policy field nobody consults.
"""
from __future__ import annotations

import io
import queue
import threading

import pytest

from autoforge.autonomy.policy import AutonomyPolicy
from autoforge.core.agent import Agent, AgentResult
from autoforge.core.llm import LLMResponse, MockLLMClient, ToolCall
from autoforge.core.message import Message
from autoforge.core.steering import (OPERATOR_PREFIX, Steering,
                                    operator_message)
from autoforge.modes import MinimalAgent
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec

from _terminal import FakeTerm, Out


class Tty(io.StringIO):
    """A stream that claims to be a terminal, so the reader thread starts."""

    def isatty(self) -> bool:
        return True


def _steering(text: str = "", **kw) -> tuple[Steering, list[str]]:
    printed: list[str] = []
    s = Steering(Tty(text), printer=printed.append, **kw)
    return s, printed


# ======================================================================
# the channel
# ======================================================================
class TestTheChannel:
    def test_a_plain_sentence_is_queued_for_the_model(self):
        s, printed = _steering()
        s.submit("actually, use csv not parquet")
        assert s.take_supplements() == [
            operator_message("actually, use csv not parquet")]
        assert any("will reach the agent" in p for p in printed)

    def test_a_supplement_says_it_came_mid_run(self):
        # Without the marker the model reads a correction as a new task and
        # starts over, which is the failure this feature exists to prevent.
        s, _ = _steering()
        s.submit("stop rewriting that file")
        (text,) = s.take_supplements()
        assert text.startswith(OPERATOR_PREFIX)
        assert "not as a new task" in text
        assert "stop rewriting that file" in text

    @pytest.mark.parametrize("word", ["/status", "/progress", "/where", "?"])
    def test_asking_for_progress_never_enters_the_conversation(self, word):
        # Asking where it is must not become part of the task.
        s, printed = _steering(status=lambda: "3m elapsed  ·  turn 4")
        s.submit(word)
        assert s.take_supplements() == []
        assert any("turn 4" in p for p in printed)

    def test_status_with_no_run_says_so(self):
        s, printed = _steering()
        s.submit("/status")
        assert any("no run in progress" in p for p in printed)

    def test_status_survives_a_broken_snapshot(self):
        def boom():
            raise RuntimeError("gone")

        s, printed = _steering(status=boom)
        s.submit("/status")
        assert any("RuntimeError" in p for p in printed)

    @pytest.mark.parametrize("word", ["/stop", "/halt", "/abort"])
    def test_stop_sets_the_flag_and_is_not_sent_to_the_model(self, word):
        s, _ = _steering()
        assert s.stop_requested() is False
        s.submit(word)
        assert s.stop_requested() is True
        assert s.take_supplements() == []

    def test_an_unknown_command_is_refused_loudly_not_sent(self):
        # Silently dropping it loses the operator's line; silently sending it
        # feeds the model a command. Neither is acceptable, so it says so.
        s, printed = _steering()
        s.submit("/reprot")
        assert s.take_supplements() == []
        assert s.refused == ["/reprot"]
        assert any("unknown mid-run command" in p for p in printed)

    def test_a_leading_space_sends_a_slash_line_as_text(self):
        # The escape hatch: "/usr/bin/env is missing" must be sayable.
        s, _ = _steering()
        s.submit(" /usr/bin/env is missing")
        assert s.take_supplements() == [
            operator_message(" /usr/bin/env is missing")]

    def test_blank_lines_are_ignored(self):
        s, printed = _steering()
        s.submit("\n")
        s.submit("   \n")
        assert s.take_supplements() == []
        assert printed == []

    def test_a_word_containing_a_question_mark_is_text(self):
        s, _ = _steering()
        s.submit("why did that fail?")
        assert len(s.take_supplements()) == 1

    def test_delivery_is_counted_when_taken_not_when_typed(self):
        # "queued" and "delivered" are different facts; only the second one
        # means the model saw it.
        s, _ = _steering()
        s.submit("hello")
        assert s.delivered == 0
        s.take_supplements()
        assert s.delivered == 1

    def test_a_broken_printer_does_not_take_the_run_down(self):
        def boom(_text):
            raise OSError("stdout closed")

        s = Steering(Tty(), printer=boom)
        s.submit("/status")
        s.submit("a real note")
        assert len(s.take_supplements()) == 1


class TestTheReaderThread:
    def test_a_non_tty_stream_starts_no_thread(self):
        # A pipe or a cron job has nobody at the other end. Pretending
        # otherwise would block a run on a keyboard that does not exist.
        s = Steering(io.StringIO("hello\n"))
        s.start()
        assert s.interactive is False
        assert s._thread is None

    def test_a_tty_stream_is_read_without_blocking_the_caller(self):
        s = Steering(Tty("first\nsecond\n"), printer=lambda _t: None)
        s.start()
        assert s._thread is not None
        s._thread.join(timeout=5)
        assert s.take_supplements() == [operator_message("first"),
                                        operator_message("second")]

    def test_one_reader_owns_the_stream_for_the_session(self):
        # During a run the lines are steering; between runs the same queue is
        # the prompt. That is what stops a line typed mid-run from vanishing.
        s = Steering(Tty(""), printer=lambda _t: None).start()
        s.submit("typed while the agent worked")
        assert s.take_line("you> ") == "typed while the agent worked"

    def test_take_line_reads_a_piped_stream_directly(self):
        s = Steering(io.StringIO("piped task\n"), printer=lambda _t: None)
        assert s.take_line("you> ") == "piped task\n"

    def test_take_line_blocks_for_the_reader_when_interactive(self):
        s = Steering(Tty(""), printer=lambda _t: None).start()

        def type_later():
            import time
            time.sleep(0.05)
            s.submit("from the keyboard")

        threading.Thread(target=type_later, daemon=True).start()
        assert s.take_line("you> ") == "from the keyboard"

    def test_take_line_reports_end_of_input_instead_of_blocking(self):
        # A queue has no EOFError. Returning "" is what lets the REPL exit on a
        # closed stdin; without it control-D hangs the shell.
        s = Steering(Tty("bye\n"), printer=lambda _t: None)
        s.start()
        s._thread.join(timeout=5)
        assert s.take_line("you> ") == "bye"     # queued lines arrive normalized
        assert s.take_line("you> ") == ""

    def test_a_piped_stream_also_reports_end_of_input(self):
        s = Steering(io.StringIO("only line\n"), printer=lambda _t: None)
        assert s.take_line("you> ") == "only line\n"
        assert s.take_line("you> ") == ""

    def test_watch_points_replies_at_the_run_in_progress(self):
        class Live:
            def __init__(self):
                self.said = []

            def say(self, text):
                self.said.append(text)

            def snapshot(self):
                return "turn 2"

        s, printed = _steering()
        live = Live()
        s.watch(live)
        s.submit("a note")
        s.submit("/status")
        assert any("will reach the agent" in t for t in live.said)
        assert "turn 2" in live.said
        assert printed == []           # replies moved to the live run

    def test_feed_routes_several_lines(self):
        s, _ = _steering()
        s.feed(["/status", "one", "/stop", "two"])
        assert s.stop_requested() is True
        assert s.take_supplements() == [operator_message("one"),
                                        operator_message("two")]


# ======================================================================
# the loop drains it — a channel nobody reads is not a feature
# ======================================================================
def _tool_registry(calls: list) -> ToolRegistry:
    reg = ToolRegistry()

    def note(text: str = "") -> str:
        calls.append(text)
        return f"noted {text}"

    reg.register(ToolSpec(name="note", description="write something down",
                          parameters={"type": "object",
                                      "properties": {"text": {"type": "string"}}},
                          fn=note, source="builtin", effect_signature="read_only"))
    return reg


def _scripted() -> MockLLMClient:
    """Two tool turns then a final answer."""
    return MockLLMClient(script=[
        LLMResponse(content="", tool_calls=[ToolCall("c1", "note", {"text": "a"})]),
        LLMResponse(content="", tool_calls=[ToolCall("c2", "note", {"text": "b"})]),
        LLMResponse(content="done"),
    ])


class TestTheEditorIsTheReader:
    """The seam between the line editor and the channel.

    The pieces are tested apart; what matters is that they are bolted
    together — a keystroke has to travel from the terminal's byte stream,
    through the editor, into the queue the loop drains, while the run's own
    progress is being printed over the top of it.
    """

    @staticmethod
    def _wired(chunks, columns=40, paste_to=None):
        from autoforge.core.lineedit import LineEditor

        term = FakeTerm(chunks, columns=columns)
        out = Out()
        editor = LineEditor(stream=object(), out=out, term=term, paste_to=paste_to)
        editor.start()
        editor.set_prompt("you> ")
        # The stream is only a fallback: the editor wins, so it is never read.
        s = Steering(Tty(), printer=lambda _t: None, editor=editor)
        assert s.interactive is True
        started = s.start()
        assert started.editing is True, "the editor took the terminal"
        return started, editor, out

    def test_a_typed_correction_reaches_the_agent(self):
        s, _editor, _out = self._wired(["use ", "csv not parquet", "\r"])
        s._thread.join(timeout=5)
        assert s.take_supplements() == [operator_message("use csv not parquet")]

    def test_progress_printed_mid_typing_keeps_the_input_intact(self):
        """The whole point of owning the bottom line.

        The run narrates itself over the top of a half-typed correction, and
        both survive: the input row is redrawn underneath, still holding what
        was typed.
        """
        s, editor, out = self._wired(["half a sen", None])
        out.parts.clear()
        editor.write("  -> running bash")
        editor.tick("  ~ turn 3   12s")

        assert out.screen().lines() == ["  -> running bash",
                                        "  ~ turn 3   12s",
                                        "you> half a sen"]
        assert "".join(editor._buf) == "half a sen"

    def test_a_pasted_block_reaches_the_agent_expanded(self, tmp_path):
        """What the model is sent is the text, not the placeholder."""
        block = "\n".join(f"line {i}" for i in range(9))
        s, _editor, _out = self._wired([block, "\r"], paste_to=tmp_path)
        s._thread.join(timeout=5)
        got = s.take_supplements()

        assert len(got) == 1
        assert "line 0\nline 1" in got[0]          # the text, not the placeholder
        assert "line 8" in got[0]                   # the tail survived the file
        assert "[Pasted text" not in got[0]         # expanded before it is sent
        assert list(tmp_path.glob("paste_*.txt")), "the text was kept on disk"


class TestTheLoopAbsorbs:
    def test_a_note_typed_before_the_first_model_call_reaches_it(self):
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("the file is in /tmp not here")
        agent = Agent(MockLLMClient(script=[LLMResponse(content="ok")]),
                      _tool_registry(calls), steer=steer)
        agent.run("do the thing")

        sent = agent.llm.calls[0][0]
        assert sent[-1].role == "user"
        assert "the file is in /tmp not here" in sent[-1].content
        assert sent[-2].role == "user"          # the original task is still there

    def test_a_note_typed_while_a_tool_runs_lands_before_the_next_request(self):
        # This is the point of the feature: a forge can take minutes, and a
        # correction that only arrives after the task ends is useless.
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        llm = MockLLMClient(script=[
            LLMResponse(content="", tool_calls=[ToolCall("c1", "note", {"text": "a"})]),
            LLMResponse(content="done"),
        ])

        def mid_run(registry, name, args):
            steer.submit("actually use csv")
            return "noted a"

        reg = _tool_registry(calls)
        reg.get("note").fn = lambda text="": mid_run(reg, "note", text)
        agent = Agent(llm, reg, steer=steer)
        agent.run("do the thing")

        second = llm.calls[1][0]
        assert "actually use csv" in second[-1].content
        assert second[-1].role == "user"
        # ...and it arrives *after* the tool result it followed, not before.
        assert second[-2].role == "tool"

    def test_the_injection_is_reported_to_the_observer(self):
        seen: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("heads up")
        agent = Agent(MockLLMClient(script=[LLMResponse(content="ok")]),
                      _tool_registry([]), steer=steer, on_steer=seen.append)
        agent.run("task")
        assert len(seen) == 1 and "heads up" in seen[0]

    def test_stop_ends_the_run_and_is_labelled_as_the_operators_choice(self):
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        llm = MockLLMClient(script=[
            LLMResponse(content="", tool_calls=[ToolCall("c1", "note", {"text": "a"})]),
            LLMResponse(content="should never be reached"),
        ])
        reg = _tool_registry(calls)
        # Typed while the tool is running, which is the only moment a stop can
        # arrive mid-run: the loop is blocked in the call, not reading input.
        reg.get("note").fn = lambda text="": (steer.submit("/stop"), "noted")[1]
        result = Agent(llm, reg, steer=steer).run("task")

        assert result.stopped_by_operator is True
        assert result.self_terminated is False     # different facts, kept apart
        assert len(llm.calls) == 1                 # the second turn never happened
        assert result.turns == 1                   # ...and the run says where it got to

    def test_a_stop_requested_before_the_run_stops_before_the_first_call(self):
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("/stop")
        llm = MockLLMClient(script=[LLMResponse(content="never")])
        result = Agent(llm, _tool_registry(calls), steer=steer).run("task")
        assert result.stopped_by_operator is True
        assert llm.calls == []

    def test_stop_is_not_reported_as_self_termination(self):
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("/stop")
        result = Agent(MockLLMClient(script=[]), _tool_registry(calls),
                       steer=steer).run("task")
        assert result.stopped_by_operator and not result.self_terminated
        assert "operator" in result.termination_reason

    def test_a_note_and_a_stop_together_do_both(self):
        # The note still gets delivered; stopping does not swallow it.
        calls: list = []
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("one last thing: use csv")
        steer.submit("/stop")
        agent = Agent(MockLLMClient(script=[LLMResponse(content="ok")]),
                      _tool_registry(calls), steer=steer)
        result = agent.run("task")
        assert result.stopped_by_operator is True
        assert agent.llm.calls == []               # stopped before the call
        assert "use csv" in result.messages[-1].content

    def test_with_no_steering_the_loop_is_what_it_was(self):
        calls: list = []
        agent = Agent(MockLLMClient(script=[LLMResponse(content="ok")]),
                      _tool_registry(calls))
        result = agent.run("task")
        assert result.content == "ok"
        assert result.stopped_by_operator is False

    def test_a_steering_channel_that_lies_about_stopping_is_ignored(self):
        # Duck-typed on purpose: anything with the two methods works, and a
        # channel returning a truthy non-bool must not be read as a stop.
        class Odd:
            def take_supplements(self):
                return []

            def stop_requested(self):
                return 0

        calls: list = []
        llm = MockLLMClient(script=[LLMResponse(content="ok")])
        result = Agent(llm, _tool_registry(calls), steer=Odd()).run("task")
        assert result.content == "ok" and result.stopped_by_operator is False


# ======================================================================
# the modes, end to end
# ======================================================================
class TestModesCarrySteering:
    def test_minimal_agent_delivers_a_note_mid_run(self):
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("use the workspace, not /tmp")
        llm = MockLLMClient(script=[LLMResponse(content="ok")])
        agent = MinimalAgent(llm=llm, cwd=".", steer=steer)
        agent.run("list the files")
        assert "use the workspace, not /tmp" in llm.calls[0][0][-1].content

    def test_minimal_agent_records_the_note_in_its_trace(self):
        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("a note")
        agent = MinimalAgent(
            llm=MockLLMClient(script=[LLMResponse(content="ok")]),
            cwd=".", steer=steer)
        agent.run("task")
        kinds = [e["kind"] for e in agent.trace]
        assert "steer" in kinds

    def test_forge_agent_delivers_a_note_and_logs_it(self):
        from autoforge.agent import ForgeAgent

        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("lead with the conclusion")
        llm = MockLLMClient(script=[LLMResponse(content="ok")])
        agent = ForgeAgent(llm, enable_evolution=False, enable_meta_cognition=False,
                           steer=steer)
        result = agent.run("summarise the repo")

        assert "lead with the conclusion" in llm.calls[0][0][-1].content
        steer_events = [e for e in agent.trace if e["kind"] == "steer"]
        assert len(steer_events) == 1
        assert "lead with the conclusion" in steer_events[0]["text"]
        assert result.stopped_by_operator is False

    def test_forge_agent_records_a_stop_on_the_finish_event(self):
        from autoforge.agent import ForgeAgent

        steer = Steering(Tty(), printer=lambda _t: None)
        steer.submit("/stop")
        agent = ForgeAgent(MockLLMClient(script=[]), enable_evolution=False,
                           enable_meta_cognition=False, steer=steer)
        result = agent.run("a task nobody will let finish")

        assert result.stopped_by_operator is True
        finish = [e for e in agent.trace if e["kind"] == "finish"]
        assert finish and finish[-1]["stopped_by_operator"] is True
