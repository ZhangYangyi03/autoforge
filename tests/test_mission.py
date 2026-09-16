"""What I owe arrives on its own, and cannot be quietly dropped.

`remember` handles facts and `schedule` handles times. Neither handles the
thing that actually got lost: after a session with many requests, the sentence
saying what they are all FOR had been replaced by the most recent one. The
store could be accurate and the agent would still describe the work wrongly --
correctly, from what it could see.

So this file pins a table that is printed into every prompt, and pins the ways
that could quietly go wrong:

  * it must not write a ledger event per turn, or `my_history` becomes the turn
    count;
  * bounded, and honest when it truncates;
  * the three states -- no store, no missions, missions open -- must read
    differently, or the agent misdescribes what it owes;
  * close must not delete: "finished" and "abandoned" must be tellable apart
    afterwards, and "still owed" must never be closable away.

The last test is the load-bearing one: not "the store returns a string" but
"the string is in the messages the model actually receives".
"""
from __future__ import annotations

import os
import tempfile

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.mission import DONE, DROPPED, OPEN, MissionError, MissionStore
from autoforge.store import ToolStore


@pytest.fixture()
def db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield os.path.join(td, "af.db")


@pytest.fixture()
def missions(db):
    s = MissionStore(db)
    yield s
    s.close()


@pytest.fixture()
def store(db):
    s = ToolStore(db)
    yield s
    s.close()


def _agent(store=None) -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), store=store, policy=FULL_FREEDOM)


def _block(agent: ForgeAgent) -> str:
    return "\n".join(agent._mission_lines())


# ======================================================================
# an obligation is not a fact and not a time
# ======================================================================
class TestItIsItsOwnThing:
    def test_opening_a_mission_writes_no_kept_fact(self, db):
        st = ToolStore(db)
        ms = MissionStore(db, conn=st._conn)
        ms.open("ship the audit fix", next_step="write the test")
        assert st.memory_for_injection() == []
        st.close()

    def test_a_mission_does_not_appear_in_the_schedule(self, missions, db):
        missions.open("ship the audit fix")
        from autoforge.schedule import Schedule
        # Reported from a store with no schedule in it at all. Separation is
        # the claim: reading what is owed must not require the schedule, or one
        # of the two becomes a view of the other.
        assert missions.report().count("ship the audit fix") == 1
        assert "ship the audit fix" not in Schedule(getattr(missions, "_sched_path", None)).report()

    def test_a_mission_survives_the_process_that_opened_it(self, db):
        m = MissionStore(db)
        m.open("survive a restart")
        m.close()
        again = MissionStore(db)
        assert [x.text for x in again.all(OPEN)] == ["survive a restart"]


# ======================================================================
# "still owed" cannot be closed away
# ======================================================================
class TestClosingIsGuarded:
    def test_a_parent_with_open_children_refuses_to_close(self, missions):
        parent = missions.open("the whole job")
        missions.open("one piece", parent=parent.id)
        with pytest.raises(MissionError) as e:
            missions.finish(parent.id)
        assert "still has open sub-missions" in str(e.value)

    def test_once_the_children_are_closed_the_parent_closes(self, missions):
        parent = missions.open("the whole job")
        child = missions.open("one piece", parent=parent.id)
        missions.finish(child.id, "piece done")
        assert missions.finish(parent.id, "all done").status == DONE

    def test_a_closed_mission_cannot_be_reopened_under_a_new_parent(self, missions):
        done = missions.open("finished thing")
        missions.finish(done.id, "done")
        other = missions.open("later thing")
        with pytest.raises(MissionError) as e:
            missions.open("revival", parent=done.id)
        assert "open a new mission instead" in str(e.value)

    def test_closing_twice_is_reported_not_hidden(self, missions):
        m = missions.open("x")
        missions.finish(m.id, "done")
        with pytest.raises(MissionError):
            missions.finish(m.id, "done again")

    def test_dropping_says_dropped_not_done(self, missions):
        m = missions.open("abandoned thing")
        assert missions.drop(m.id, "operator changed the goal").status == DROPPED


# ======================================================================
# close is not delete
# ======================================================================
class TestTheRecordOutlivesTheWork:
    def test_a_closed_mission_is_still_readable_with_its_note(self, missions):
        m = missions.open("ship it")
        missions.finish(m.id, "shipped as 022f230")
        row = missions.get(m.id)
        assert row.closed_at is not None
        assert "022f230" in row.close_note

    def test_history_keeps_the_opening_and_the_notes(self, missions):
        m = missions.open("long one")
        missions.note(m.id, "first attempt failed")
        missions.finish(m.id, "done after all")
        kinds = [h["kind"] for h in missions.history(m.id)]
        assert kinds == ["close", "note", "open"]

    def test_closed_missions_stay_in_the_all_view(self, missions):
        a = missions.open("a")
        missions.finish(a.id, "done")
        assert [x.status for x in missions.all(None)] == [DONE]

    def test_the_open_report_does_not_list_closed_work(self, missions):
        a = missions.open("this-work-is-finished")
        missions.finish(a.id, "done")
        assert "this-work-is-finished" not in missions.report()

    def test_the_report_counts_what_was_closed(self, missions):
        a = missions.open("a")
        b = missions.open("b")
        missions.finish(a.id, "done")
        missions.drop(b.id, "not wanted")
        text = missions.report()
        # The count is the point: a report that says "nothing owed" while two
        # missions were closed is still telling the truth, and it must not read
        # like a store that was never written to.
        assert "nothing open" in text
        assert "2 closed and on the record" in text


# ======================================================================
# three states, three sentences
# ======================================================================
class TestTheThreeStatesReadDifferently:
    def test_no_store_says_nothing_is_tracked(self):
        text = _block(_agent(store=None))
        assert "no store this session" in text
        assert "nothing is tracked" in text

    def test_empty_says_a_request_is_a_mission(self, store):
        text = _block(_agent(store))
        assert "nothing open" in text
        assert "mission_open" in text          # tells the agent what to do next
        assert "no store" not in text

    def test_open_missions_are_named_with_their_next_step(self, store):
        agent = _agent(store)
        ms = agent._mission_store()
        ms.open("fix the console window storm", next_step="gate each .bat")
        text = _block(agent)
        assert "fix the console window storm" in text
        assert "gate each .bat" in text


# ======================================================================
# bounded, and honest about truncation
# ======================================================================
class TestTheBlockIsBounded:
    def test_many_missions_truncate_and_say_so(self, missions):
        for i in range(40):
            missions.open(f"mission number {i} with a reasonably long sentence")
        text = missions.report()
        assert "more open" in text

    def test_one_huge_mission_still_appears(self, missions):
        missions.open("x" * 3000)
        text = missions.report()
        assert "xxxx" in text                  # elided nowhere, shown somewhere

    def test_the_block_is_small_enough_to_ride_every_turn(self, missions):
        for i in range(40):
            missions.open("y" * 200)
        assert len(missions.report()) < 2500


# ======================================================================
# the load-bearing one: it reaches the model
# ======================================================================
class TestItReachesTheModel:
    def test_an_open_mission_is_in_the_system_message_of_a_real_run(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "fix the black window storm"})
        agent.run("say hi")

        messages, _tools = agent.llm.calls[0]
        system = messages[0]
        assert system.role == "system"
        assert "fix the black window storm" in system.content

    def test_the_sub_mission_is_in_the_system_message_too(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "the whole job"})
        agent.registry.call("mission_open",
                            {"text": "one named piece", "parent": 1})
        agent.run("say hi")
        messages, _tools = agent.llm.calls[0]
        assert "one named piece" in messages[0].content

    def test_a_closed_mission_leaves_the_prompt(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "already finished thing"})
        agent.registry.call("mission_close", {"mid": 1, "note": "done"})
        agent.run("say hi")
        messages, _tools = agent.llm.calls[0]
        assert "already finished thing" not in messages[0].content

    def test_the_prompt_says_there_is_no_store_when_there_is_none(self):
        agent = _agent(store=None)
        agent.run("say hi")
        messages, _tools = agent.llm.calls[0]
        assert "no store this session" in messages[0].content

    def test_building_the_prompt_writes_no_ledger_event(self, store):
        # Per-turn writes would grow the ledger by the turn count and make
        # `my_history` unreadable. Reading for the prompt is not an event.
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "something owed"})
        before = store.report()["events"]
        for _ in range(5):
            agent._effective_prompt()
        assert store.report()["events"] == before


# ======================================================================
# the tool surface refuses the moves that would hide work
# ======================================================================
class TestTheToolsRefuse:
    def test_closing_a_parent_through_the_tool_reports_the_child(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "parent"})
        agent.registry.call("mission_open", {"text": "child", "parent": 1})
        out = agent.registry.call("mission_close", {"mid": 1, "note": "done"}).output
        assert "Not closed." in out
        assert "M2" in out

    def test_a_bad_parent_id_is_refused_not_guessed(self, store):
        agent = _agent(store)
        out = agent.registry.call("mission_open", {"text": "x", "parent": 99}).output
        assert out.startswith("Not recorded.")

    def test_mission_list_defaults_to_what_is_open(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "open one"})
        agent.registry.call("mission_close", {"mid": 1, "note": "done"})
        out = agent.registry.call("mission_list", {}).output
        assert "open one" not in out
        assert "No open missions" in out

    def test_mission_show_has_the_history(self, store):
        agent = _agent(store)
        agent.registry.call("mission_open", {"text": "traceable"})
        agent.registry.call("mission_note", {"mid": 1, "note": "step one"})
        out = agent.registry.call("mission_show", {"mid": 1}).output
        assert "step one" in out and "open:" in out


# ======================================================================
# the terminal path: readable without a model, so a stuck session can look
# ======================================================================
class TestTheCliReadsItWithoutAModel:
    def _run(self, argv, db):
        from autoforge import cli
        parser = cli.build_parser()
        args = parser.parse_args(["mission", *argv, "--file", db])
        return cli.cmd_mission(args)

    def test_a_mission_can_be_opened_and_listed_from_the_shell(self, db, capsys):
        assert self._run(["add", "ship it"], db) == 0
        assert "M1 open" in capsys.readouterr().out
        assert self._run(["list"], db) == 0
        assert "ship it" in capsys.readouterr().out

    def test_the_flag_after_free_text_is_not_eaten_as_the_note(self, db, capsys):
        # `rest` is REMAINDER, so "--file X" typed after the note lands inside
        # the note unless it is pulled back out. The first version of this
        # recorded a note of "done --file C:\...\autoforge.db", which is a
        # corrupted record that looks like a working command.
        self._run(["add", "parent thing"], db)
        capsys.readouterr()
        self._run(["close", "1", "finished"], db)
        out = capsys.readouterr().out
        assert "--file" not in out
        assert out.strip().endswith("finished")

    def test_it_refuses_to_close_a_parent_over_an_open_child(self, db, capsys):
        self._run(["add", "the whole job"], db)
        self._run(["add", "one piece", "-p", "1"], db)
        capsys.readouterr()
        assert self._run(["close", "1", "premature"], db) == 1
        assert "not closed" in capsys.readouterr().out

    def test_the_default_report_is_what_the_prompt_gets(self, db, capsys):
        self._run(["add", "owed thing"], db)
        capsys.readouterr()
        assert self._run([], db) == 0
        assert "MISSION — what I owe" in capsys.readouterr().out
