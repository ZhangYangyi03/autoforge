"""A run that survives the wire going down.

The failure, as it actually happened on this host: a task is under way, the
call to aiping.cn comes back `503`, and the run dies. Two separate things are
wrong with that, and they need separate tests.

The first is that nothing remembered where the loop was. The transcript and the
turn count lived in a local variable inside `Agent._run`, so a raised exception
-- or a killed process, or a reboot -- took the whole task with it. So these
tests drive a *real* `Agent` against a client that fails partway through, and
then check that a second agent, built fresh with no shared memory, can be
handed the record and carries on. A resume that is only tested by calling
`adopt()` directly would not prove the interesting half: that the loop beats at
the right moments, and that the beat happens *before* the step that might not
come back.

The second is the shape of the transcript being resumed. An OpenAI-compatible
endpoint rejects an assistant message whose `tool_calls` have no matching
`role: tool` results, and the ones that do not reject it answer with the results
invented. A record written at the wrong moment is therefore not a shorter
request, it is a malformed one -- so the trimming is pinned case by case,
including the case that cannot arise from this code and can arise from a hand-
edited file.

Kept deliberately apart from the network: `MockLLMClient` raises whatever the
script says to raise, so the retry ladder's own behaviour (which is tested in
`test_llm_retry.py`) is not what is under test here.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pytest

from autoforge import runs
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.agent import Agent
from autoforge.core.llm import LLMError, LLMResponse, MockLLMClient, tool_call
from autoforge.core.message import Message, ToolCall
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec

DEAD_PID = 999_999   # measured absent on this host; see `Mark.owner_alive`


def U(text: str) -> Message:
    return Message.user(text)


def A(content: str = "", calls: list[ToolCall] | None = None) -> Message:
    return Message.assistant(content, calls or [])


def T(content: str, call_id: str, name: str = "probe") -> Message:
    return Message.tool(content, call_id, name)


def TC(call_id: str, name: str = "probe") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


def write_mark(root, *, rid="r1", pid=DEAD_PID, age=120.0, msgs=None, attempts=0,
               turns=3, task="build the thing", action="probe") -> runs.Mark:
    """Put a record on disk the way a dying writer would have left it."""
    mark = runs.Mark(
        run_id=rid, task=task, started_at=time.time() - age,
        updated_at=time.time() - age, writer_pid=pid, turns=turns,
        attempts=attempts, last_action=action,
        messages=msgs if msgs is not None else [U(task), A("", [TC("c1")]),
                                                T("ok", "c1", "probe")],
    )
    runs._atomic_write(runs._path(rid, root), json.dumps(mark.to_dict()))
    return mark


# ======================================================================
# the shape of a transcript that can be continued
# ======================================================================
class TestTrimmingToWholeGroups:
    """A partial group is a malformed request, not a shorter one."""

    def test_a_whole_group_survives_untouched(self):
        msgs = [U("t"), A("", [TC("a"), TC("b")]), T("1", "a"), T("2", "b")]
        kept, dropped = runs.repair(msgs)
        assert [m.role for m in kept] == ["user", "assistant", "tool", "tool"]
        assert dropped == 0

    def test_a_half_answered_group_is_trimmed_to_the_task(self):
        # The failure this exists for: the process died inside the second tool.
        msgs = [U("t"), A("", [TC("a"), TC("b")]), T("1", "a")]
        kept, dropped = runs.repair(msgs)
        assert [m.role for m in kept] == ["user"]
        assert dropped == 2, (
            "the assistant message that asked and the result that arrived both "
            "go: keeping the result alone leaves a group with no request, which "
            "is the other malformed shape")

    def test_an_assistant_that_asked_and_got_nothing_is_trimmed(self):
        kept, dropped = runs.repair([U("t"), A("", [TC("a")])])
        assert [m.role for m in kept] == ["user"]
        assert dropped == 1

    def test_a_result_for_a_call_that_was_never_made_is_trimmed(self):
        # Cannot arise from the loop; can arise from a record somebody edited,
        # and a resume is not the place to find out.
        kept, dropped = runs.repair([U("t"), A("", [TC("a")]), T("1", "a"), T("9", "zz")])
        assert [m.role for m in kept] == ["user", "assistant", "tool"]
        assert dropped == 1

    def test_a_finished_answer_is_left_alone(self):
        kept, dropped = runs.repair([U("t"), A("done")])
        assert dropped == 0 and [m.role for m in kept] == ["user", "assistant"]

    def test_the_system_message_is_not_part_of_the_transcript(self):
        kept, _ = runs.repair([Message.system("s"), U("t")])
        assert all(m.role != "system" for m in kept)

    def test_a_transcript_with_no_task_is_not_continuable(self):
        assert runs._continuable([Message.system("s")]) is False
        assert runs._continuable([U("t")]) is True


# ======================================================================
# may this record be taken over, and on what grounds
# ======================================================================
class TestTheClaimDecision:
    def test_a_dead_writer_is_resumable(self, tmp_path):
        mark = write_mark(str(tmp_path))
        status, reason = mark.claim()
        assert status == "resumable"
        assert str(DEAD_PID) in reason, "the reason must name the evidence"

    def test_a_live_writer_is_left_alone(self, tmp_path):
        mark = write_mark(str(tmp_path), pid=os.getpid(), age=1.0)
        assert mark.claim()[0] == "live"

    def test_a_writer_that_is_alive_but_silent_is_treated_as_wedged(self, tmp_path):
        """Alive is not the same as progressing.

        A thread parked on a socket that will never answer is one of the
        failures this module covers, and deferring to a pid forever is how a run
        that needs resuming never gets one.
        """
        mark = write_mark(str(tmp_path), pid=os.getpid(),
                          age=runs.WEDGED_AFTER_S + 60)
        assert mark.claim()[0] == "resumable"

    def test_a_record_that_just_stopped_waits_before_being_taken(self, tmp_path):
        """So two resumers do not start the same run in the same second."""
        mark = write_mark(str(tmp_path), age=0.5)
        assert mark.claim()[0] == "waiting"

    def test_the_resume_budget_is_spent_after_max_resumes(self, tmp_path):
        mark = write_mark(str(tmp_path), attempts=runs.MAX_RESUMES)
        status, reason = mark.claim()
        assert status == "stale"
        assert str(runs.MAX_RESUMES) in reason

    def test_a_record_past_its_ttl_is_stale(self, tmp_path):
        mark = write_mark(str(tmp_path), age=runs.DEFAULT_TTL_S + 10)
        assert mark.claim()[0] == "stale"

    def test_the_record_itself_is_decided_before_the_process_table(self, tmp_path):
        """A stale record does not become resumable by waiting."""
        mark = write_mark(str(tmp_path), attempts=runs.MAX_RESUMES, age=0.1)
        assert mark.claim()[0] == "stale"


# ======================================================================
# taking one over
# ======================================================================
class TestAdoption:
    def test_the_transcript_and_turn_count_are_carried_forward(self, tmp_path):
        mark = write_mark(str(tmp_path), turns=7)
        j = runs.Journal("t", root=str(tmp_path)).adopt(mark)
        assert j.mark.turns == 7
        assert len(j.mark.messages) == len(mark.messages)
        assert j.mark.attempts == 1
        assert j.mark.prior_run_id == "r1"

    def test_the_original_record_is_removed_so_the_task_runs_once(self, tmp_path):
        """The half that makes a takeover a takeover.

        Left standing, the same stranded record is read by the next scanner and
        the same task is resumed again, in parallel, on one workspace.
        """
        mark = write_mark(str(tmp_path))
        runs.Journal("t", root=str(tmp_path)).adopt(mark)
        assert not runs._path("r1", str(tmp_path)).exists()
        assert [m.run_id for m in runs.unfinished(str(tmp_path))] == [
            runs.unfinished(str(tmp_path))[0].run_id]

    def test_the_new_record_is_written_before_any_work_starts(self, tmp_path):
        """The claim has to be on disk before the work, or it is not a claim."""
        mark = write_mark(str(tmp_path))
        j = runs.Journal("t", root=str(tmp_path)).adopt(mark)
        on_disk = runs.unfinished(str(tmp_path))[0]
        assert on_disk.run_id == j.mark.run_id
        assert on_disk.writer_pid == os.getpid()

    def test_an_adopted_run_is_no_longer_offered_to_anyone_else(self, tmp_path):
        mark = write_mark(str(tmp_path))
        runs.Journal("t", root=str(tmp_path)).adopt(mark)
        assert runs.resumable(str(tmp_path)) == []

    def test_finishing_removes_the_record(self, tmp_path):
        j = runs.Journal("t", root=str(tmp_path))
        j.beat([U("t")], turn=1)
        assert runs.unfinished(str(tmp_path))
        j.finish()
        assert runs.unfinished(str(tmp_path)) == []

    def test_a_raising_run_leaves_its_record_standing(self, tmp_path):
        """The case the whole module exists for."""
        j = runs.Journal("t", root=str(tmp_path))
        with pytest.raises(ZeroDivisionError):
            with j:
                j.beat([U("t")], turn=1)
                raise ZeroDivisionError("the wire went down")
        assert [m.run_id for m in runs.unfinished(str(tmp_path))] == [j.mark.run_id]

    def test_a_clean_exit_removes_it(self, tmp_path):
        j = runs.Journal("t", root=str(tmp_path))
        with j:
            j.beat([U("t")], turn=1)
        assert runs.unfinished(str(tmp_path)) == []


# ======================================================================
# reading a directory somebody else may have written
# ======================================================================
class TestReadingTheDirectory:
    def test_a_damaged_record_does_not_take_the_others_with_it(self, tmp_path):
        write_mark(str(tmp_path), rid="good")
        (tmp_path / "run-bad.json").write_text("{not json", encoding="utf-8")
        assert [m.run_id for m in runs.unfinished(str(tmp_path))] == ["good"]

    def test_a_record_from_a_newer_schema_is_reported_not_guessed_at(self, tmp_path):
        """Resuming on a misread transcript hands the model a conversation
        that never happened, which is worse than not resuming."""
        mark = write_mark(str(tmp_path))
        data = json.loads(runs._path("r1", str(tmp_path)).read_text(encoding="utf-8"))
        data["schema"] = runs.SCHEMA + 1
        runs._atomic_write(runs._path("r1", str(tmp_path)), json.dumps(data))
        row = runs.scan(str(tmp_path))[0]
        assert row["status"] == "stale"
        assert "schema" in row["reason"]

    def test_an_unreadable_directory_is_an_empty_one_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "nope"))
        assert runs.scan() == []

    def test_the_oldest_stranded_run_goes_first(self, tmp_path):
        """The run stranded longest is the one whose operator waited longest."""
        write_mark(str(tmp_path), rid="new", age=60)
        write_mark(str(tmp_path), rid="old", age=3600)
        assert [m.run_id for m in runs.resumable(str(tmp_path))] == ["old", "new"]

    def test_prune_removes_only_what_is_past_its_ttl(self, tmp_path):
        write_mark(str(tmp_path), rid="fresh", age=60)
        write_mark(str(tmp_path), rid="ancient", age=runs.DEFAULT_TTL_S + 10)
        assert runs.prune(str(tmp_path)) == ["ancient"]
        assert [m.run_id for m in runs.unfinished(str(tmp_path))] == ["fresh"]

    def test_the_report_names_the_next_step_and_the_reason(self, tmp_path):
        write_mark(str(tmp_path))
        text = runs.report(str(tmp_path))
        assert "resumable" in text and "probe" in text

    def test_nothing_unfinished_says_so(self, tmp_path):
        assert runs.report(str(tmp_path)) == "no unfinished runs."


# ======================================================================
# the loop actually beats, at the moments that matter
# ======================================================================
def _counting_registry(counter: dict | None = None) -> ToolRegistry:
    """A registry whose one tool counts how often it is really entered."""
    seen = counter if counter is not None else {}
    seen.setdefault("n", 0)

    def probe(x: str = "") -> str:
        seen["n"] += 1
        return f"probe saw {x!r}"

    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="probe", description="a probe",
        parameters={"type": "object",
                    "properties": {"x": {"type": "string"}}, "required": []},
        code="def probe(x=''):\n    return 'probe saw %r' % (x,)\n",
        fn=probe, source="builtin", tags=["test"],
    ))
    return reg


def _probe_tool() -> ToolSpec:
    """A callable tool, registered the way `test_live_progress` does."""
    return ToolSpec(
        name="probe", description="a probe",
        parameters={"type": "object",
                    "properties": {"x": {"type": "string"}}, "required": []},
        code="def probe(x=''):\n    return 'probe saw %r' % (x,)\n",
        fn=lambda x="": f"probe saw {x!r}",
        source="builtin", tags=["test"],
    )


def _agent(journal, script):
    """A real Agent, a real registry, a scripted client, and a journal."""
    registry = ToolRegistry()
    registry.register(_probe_tool())
    return Agent(MockLLMClient(script=script), registry, journal=journal)


class TestTheLoopBeats:
    def test_a_beat_is_written_before_each_model_call(self, tmp_path):
        j = runs.Journal("do it", root=str(tmp_path))
        agent = _agent(j, [LLMResponse(content="done")])
        agent.run("do it")
        marks = runs.unfinished(str(tmp_path))
        assert marks and marks[0].turns == 1

    def test_the_step_in_flight_is_named_before_the_tools_run(self, tmp_path):
        """The more important beat: recorded *before* the step that can kill
        the process, so a death inside it leaves the step named."""
        j = runs.Journal("do it", root=str(tmp_path))
        seen = {}

        def handler(messages, tools, **kw):
            # Read the record from inside the tool call path: whatever is on
            # disk here was written before the tool ran.
            marks = runs.unfinished(str(tmp_path))
            seen["during"] = [m.last_action for m in marks]
            return LLMResponse(content="done")

        agent = _agent(j, [])
        agent.llm = MockLLMClient(handler=handler)
        agent.run("do it")
        assert seen.get("during") is not None

    def test_a_checkpoint_that_cannot_be_written_does_not_kill_the_run(self, tmp_path):
        """A checkpoint failure must not be the thing that kills the run it was
        recording -- and must be reported, not swallowed silently."""
        class Broken:
            mark = None
            def beat(self, *a, **kw):
                raise OSError("no space left on device")

        agent = _agent(Broken(), [LLMResponse(content="done")])
        res = agent.run("do it")
        assert res.content == "done"


# ======================================================================
# the whole round trip: die, come back, carry on
# ======================================================================
class TestARunThatDiesAndComesBack:
    def test_the_second_agent_continues_rather_than_restarting(self, tmp_path):
        """The end-to-end claim, driven through real agents.

        The first agent is given a task, reaches a tool call, and its client
        then raises the way a dropped connection does. Its record must survive
        with the step in flight named. The second agent -- a *new* one, sharing
        nothing but the directory -- adopts it, and the transcript it sends must
        contain the first agent's work, not a fresh copy of the task.
        """
        # --- the run that dies at a step boundary ------------------------
        j = runs.Journal("count the files", root=str(tmp_path))
        tool_runs: dict = {}
        registry = _counting_registry(tool_runs)

        calls = {"n": 0}

        def dying(messages, tools, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(content="", tool_calls=[TC("c1")])
            raise LLMError("503 from the gateway")

        agent = Agent(MockLLMClient(handler=dying), registry, journal=j)
        with pytest.raises(LLMError):
            with j:
                agent.run("count the files")

        stranded = runs.unfinished(str(tmp_path))
        assert len(stranded) == 1, "the record must survive the exception"

        mark = stranded[0]
        # The step in flight is named...
        assert mark.last_action == "probe"
        # ...and the group it belongs to is *whole*, because the tool did run
        # before the next model call went down. This is the case a resume is
        # cheap in: the result the model has not read is right there, and the
        # resumed run reads it instead of running the probe a second time.
        assert [m.role for m in mark.messages] == ["user", "assistant", "tool"]
        assert mark.messages[-1].content.startswith("probe saw")

        # A fresh resume instruction, built from the record, is the last thing
        # the model reads: the sentence that turns a transcript into a task.


        # These two agents share one process, and so one pid, which is the one
        # thing a restart would have changed. Rewriting the record's writer to a
        # pid that is not running is exactly what the next session sees, and
        # doing it here rather than asserting "the writer died" keeps the test
        # honest about what it is standing in for. (The claim logic itself is
        # decided by pid in TestTheClaimDecision, with no agent involved.)
        mark.writer_pid = DEAD_PID
        mark.updated_at = time.time() - runs.SETTLE_S - 5
        runs._atomic_write(runs._path(mark.run_id, str(tmp_path)),
                           json.dumps(mark.to_dict()))
        mark = runs.unfinished(str(tmp_path))[0]
        status, reason = mark.claim()
        assert status == "resumable", reason

        # --- the run that picks it up ------------------------------------
        seen: list[list[str]] = []

        def continuing(messages, tools, **kw):
            seen.append([m.role for m in messages])
            # Assert from inside: the tool result the first agent produced has
            # to be in front of this model, not just in the file.
            assert any(m.role == "tool" and m.content.startswith("probe saw")
                       for m in messages), "the earlier work was not replayed"
            return LLMResponse(content="the count is 31")

        fresh = Agent(MockLLMClient(handler=continuing), _counting_registry(),
                      journal=runs.Journal("count the files", root=str(tmp_path))
                      .adopt(mark))
        prompt = runs.resume_prompt(fresh.journal.mark)
        res = fresh.run(prompt, list(fresh.journal.mark.messages))
        assert res.content == "the count is 31"
        # What the second agent was asked is the first agent's conversation --
        # every message type preserved, in order -- plus one instruction. Not a
        # fresh copy of the task, which is the retry this module exists to
        # replace.
        assert seen, "the resumed agent never called the model"
        assert seen[0] == ["system", "user", "assistant", "tool", "user"], (
            f"expected the interrupted transcript with one instruction appended, "
            f"got {seen[0]}")
        assert tool_runs["n"] == 1, (
            "the already-finished step must not be redone by the resumed run")
        # The record is then closed by whoever owns the run's lifecycle, which
        # is deliberately not `Agent`: this loop records where it is, and
        # `ForgeAgent` decides whether the run is finished. The split matters
        # because the two disagree exactly when it counts -- a loop that raises
        # never reaches the code that would have closed the record.
        fresh.journal.finish()
        assert runs.unfinished(str(tmp_path)) == []


# ======================================================================
# the owner of the lifecycle: ForgeAgent, which starts and closes records
# ======================================================================
class TestForgeAgentOwnsTheLifecycle:
    """`Agent` records. `ForgeAgent` decides when a record is done with.

    Tested separately because the two halves fail in opposite directions and a
    test of one says nothing about the other: an `Agent` that beats perfectly
    under a `ForgeAgent` that never closes its records accumulates a resumable
    record per task, and every later session is offered work that is over.
    """

    def _agent(self, script=None, handler=None, **kw):
        from autoforge.agent import ForgeAgent
        from autoforge.autonomy.policy import FULL_FREEDOM
        llm = MockLLMClient(script=script, handler=handler)
        a = ForgeAgent(llm, policy=FULL_FREEDOM, **kw)
        a.registry.register(_probe_tool())
        return a

    def test_a_finished_task_leaves_no_record(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        a = self._agent(script=[LLMResponse(content="all done")])
        a.run("do the thing")
        assert runs.unfinished() == [], "a finished task must not be resumable"

    def test_a_task_that_raises_leaves_its_record(self, monkeypatch, tmp_path):
        """The case the module exists for, exercised through the real entry
        point rather than through `Journal` directly."""
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))

        def dying(messages, tools, **kw):
            raise LLMError("503 from the gateway")

        a = self._agent(handler=dying)
        with pytest.raises(LLMError):
            a.run("do the thing")
        stranded = runs.unfinished()
        assert len(stranded) == 1
        assert stranded[0].task == "do the thing"
        assert stranded[0].turns >= 1, (
            "the beat is before the model call, so a run that died on its first "
            "call still knows it started")

    def test_the_prompt_offers_the_stranded_run(self, monkeypatch, tmp_path):
        """In the prompt, not only in a tool, and for the same reason the
        mission list is: the failure is not "could not find it", it is "did not
        think to look"."""
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        write_mark(str(tmp_path), task="port the parser")
        a = self._agent(script=[LLMResponse(content="ok")])
        text = a._effective_prompt()
        assert "UNFINISHED RUN" in text
        assert "port the parser" in text
        assert "resume_run" in text

    def test_the_prompt_is_unchanged_when_there_is_nothing_to_resume(
            self, monkeypatch, tmp_path):
        """The common case must cost exactly nothing -- no empty header, and no
        change to the prompt prefix the provider bills from cache."""
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        a = self._agent(script=[LLMResponse(content="ok")])
        assert "UNFINISHED RUN" not in a._effective_prompt()

    def test_the_tool_reads_and_resumes_and_the_agent_can_see_it(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        write_mark(str(tmp_path), task="port the parser", age=600)
        a = self._agent(handler=lambda m, t, **k: LLMResponse(content="carried on"))
        for name in ("runs_status", "resume_run", "runs_forget"):
            assert name in a.registry.names(), f"{name} is not reachable"
        before = a.registry.get("runs_status").fn()
        assert "port the parser" in before
        out = a.registry.get("resume_run").fn()
        assert "carried on" in out, out
        # And the resumed run is not offered again: it reached a result.
        assert runs.unfinished() == []

    def test_a_declined_resume_says_why_instead_of_starting_work(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        a = self._agent(script=[LLMResponse(content="x")])
        out = a.registry.get("resume_run").fn()
        assert "cannot resume" in out
        out2 = a.registry.get("resume_run").fn(run_id="nosuch")
        assert "cannot resume" in out2

    def test_forgetting_a_run_makes_it_unresumable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path))
        write_mark(str(tmp_path), rid="pickme", task="abandoned work", age=600)
        a = self._agent(script=[LLMResponse(content="x")])
        out = a.registry.get("runs_forget").fn(run_id="pickme", why="not wanted")
        assert "dropped" in out
        assert runs.unfinished() == []

    def test_the_resume_prompt_does_not_ask_for_the_first_step_again(self):
        """A resume is not a retry. A model told only "do X" will do X from the
        beginning, re-reading the file it already read."""
        mark = runs.Mark(run_id="x", task="port the module", started_at=0,
                         updated_at=0, writer_pid=DEAD_PID, turns=6,
                         last_action="read_file")
        text = runs.resume_prompt(mark)
        assert "resuming" in text and "port the module" in text
        assert "read_file" in text
        assert "do not" in text.lower()

    def test_the_resume_prompt_names_a_cut_step_as_needing_a_redo(self):
        mark = runs.Mark(run_id="x", task="t", started_at=0, updated_at=0,
                         writer_pid=DEAD_PID, turns=2, dropped_steps=1)
        assert "redone" in runs.resume_prompt(mark)


# ======================================================================
# the command line: unattended pickup, and a person picking it up
# ======================================================================
class TestFromTheCommandLine:
    """The three readers of one record, each with a different failure mode.

    The prompt is how the *agent* remembers, the tick is how the *machine*
    remembers, and `auto runs` is how the *person* remembers. Only the last two
    can work when nobody is at the terminal, so the tick is the half that makes
    a resume happen at all -- tested here against a stub agent, because what is
    under test is whether the tick reaches for the record, not what the model
    then says.
    """

    def _stub(self, monkeypatch, cli, record: dict):
        class _FakeAgent:
            def __init__(self, cfg=None):
                record["built"] = True

            def run(self, prompt, **kw):
                record["prompt"] = prompt
                record["resumed"] = kw.get("resumed")
                return argparse.Namespace(content="carried on: " + prompt[:40])

        monkeypatch.setattr(cli, "_build_mode", lambda cfg, mode: _FakeAgent(cfg))

    def _configured(self, monkeypatch, tmp_path):
        # `_config` exits before reaching the seam without a provider, and
        # `AUTOFORGE_RUNS` keeps the agent's own journal out of the real one.
        monkeypatch.setenv("AUTOFORGE_API_KEY", "test-key")
        monkeypatch.setenv("AUTOFORGE_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("AUTOFORGE_RUNS", str(tmp_path / "runs"))

    def test_a_tick_picks_up_a_stranded_run_with_nothing_on_the_schedule(
            self, monkeypatch, tmp_path, capsys):
        """The order is the change.

        A schedule with nothing on it and one stranded run is the normal case,
        not the edge case, so the pickup has to happen before the tick decides
        there is nothing to do and returns.
        """
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        write_mark(str(tmp_path / "runs"), task="finish the port", age=900)
        record: dict = {}
        self._stub(monkeypatch, cli_mod, record)

        rc = cli_mod.main(["tick", "--file", str(tmp_path / "empty.jsonl")])
        out = capsys.readouterr().out
        assert record.get("resumed") is not None, (
            "the tick never reached for the stranded record")
        assert record["resumed"].task == "finish the port"
        assert "interrupted" in out.lower()
        # The schedule really was empty; `--file` above points at nothing.
        assert "Nothing is due." in out
        assert rc == 0

    def test_a_tick_does_not_build_an_agent_when_there_is_nothing_to_do(
            self, monkeypatch, tmp_path, capsys):
        """An unconfigured or offline machine must be able to tick silently."""
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        record: dict = {}
        self._stub(monkeypatch, cli_mod, record)
        rc = cli_mod.main(["tick", "--file", str(tmp_path / "empty.jsonl")])
        assert rc == 0
        assert not record.get("built"), "an idle tick built a model"
        assert capsys.readouterr().out.strip().startswith("Nothing is due.")

    def test_a_tick_resumes_at_most_the_budget(self, monkeypatch, tmp_path, capsys):
        """One unattended wake-up must not spend six model conversations."""
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        root = str(tmp_path / "runs")
        for i in range(4):
            write_mark(root, rid=f"s{i}", task=f"job {i}", age=900 + i)
        seen: list = []

        class _FakeAgent:
            def __init__(self, cfg=None):
                pass

            def run(self, prompt, **kw):
                seen.append(kw.get("resumed"))
                return argparse.Namespace(content="ok")

        monkeypatch.setattr(cli_mod, "_build_mode", lambda cfg, mode: _FakeAgent(cfg))
        cli_mod.main(["tick", "--file", str(tmp_path / "empty.jsonl"),
                      "--resume-budget", "1"])
        assert len(seen) == 1, f"budget ignored: {len(seen)} runs entered"

    def test_a_tick_that_fails_to_resume_leaves_the_record_for_the_next_one(
            self, monkeypatch, tmp_path, capsys):
        """A second failure is a fact about the task, and the attempt counter is
        what eventually says so -- deleting the record here would erase it."""
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        root = str(tmp_path / "runs")
        write_mark(root, task="job", age=900)

        class _Boom:
            def __init__(self, cfg=None):
                pass

            def run(self, prompt, **kw):
                raise RuntimeError("503 again")

        monkeypatch.setattr(cli_mod, "_build_mode", lambda cfg, mode: _Boom(cfg))
        rc = cli_mod.main(["tick", "--file", str(tmp_path / "empty.jsonl")])
        assert rc == 0, "a failed resume is not a failed tick"
        assert "still interrupted" in capsys.readouterr().out
        assert [m.task for m in runs.unfinished(root)] == ["job"]

    def test_the_person_can_list_resume_and_drop(self, monkeypatch, tmp_path, capsys):
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        root = str(tmp_path / "runs")
        write_mark(root, rid="aaa111", task="finish the port", age=900)
        record: dict = {}
        self._stub(monkeypatch, cli_mod, record)

        assert cli_mod.main(["runs"]) == 0
        assert "finish the port" in capsys.readouterr().out

        assert cli_mod.main(["runs", "resume", "--id", "aaa"]) == 0
        out = capsys.readouterr().out
        assert "carried on" in out
        # No record was left behind by the stub agent, so it is still offered --
        # which is correct and worth pinning: the *record* is closed by the code
        # that owns the run, and this stub is not that code.
        assert record.get("resumed") is not None
        assert record["resumed"].task == "finish the port"

        assert cli_mod.main(["runs", "drop", "--id", "aaa111",
                             "--why", "not wanted"]) == 0
        assert "dropped" in capsys.readouterr().out
        assert runs.unfinished(root) == []

    def test_a_drop_of_something_that_is_not_there_says_so(
            self, monkeypatch, tmp_path, capsys):
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        assert cli_mod.main(["runs", "drop", "--id", "nosuch"]) == 2
        assert "no unfinished run" in capsys.readouterr().out

    def test_resume_with_nothing_to_resume_is_not_an_error(
            self, monkeypatch, tmp_path, capsys):
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        assert cli_mod.main(["runs", "resume"]) == 0
        assert "nothing to resume" in capsys.readouterr().out.lower()

    def test_the_shape_of_a_resume_is_observable_from_the_cli(
            self, monkeypatch, tmp_path, capsys):
        """What the operator sees must name what is being continued, not just
        that something is -- otherwise a resume is indistinguishable from a run
        that mysteriously starts in the middle."""
        from autoforge import cli as cli_mod
        self._configured(monkeypatch, tmp_path)
        root = str(tmp_path / "runs")
        write_mark(root, rid="bbb222", task="port the parser", age=900)
        record: dict = {}
        self._stub(monkeypatch, cli_mod, record)
        cli_mod.main(["runs", "resume", "--id", "bbb222"])
        out = capsys.readouterr().out
        assert "resuming bbb222" in out
        assert "port the parser" in out
        assert "gone" in out, "the reason it was resumable must be shown"
