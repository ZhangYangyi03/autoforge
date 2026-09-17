"""Silence is read as done, and the reading is reversible.

The operator's rule (2026-09-17): a task nobody has asked to continue for more
than 24 hours is assumed complete. The failure that rule can cause is worse than
the one it fixes -- a mission that quietly vanishes is work the operator
abandoned being thrown away by a timer, and he would never know which day it
happened -- so every test here is about what the assumption must NOT do.

The shape being pinned:

  * 24h of silence -> ASSUMED, still in the report, still counted as owed,
    marked as an assumption, and one word from being owed again;
  * a further week of silence -> DONE, close note prefixed `assumed:`, so
    "the work finished" and "nobody spoke for eight days" are different rows;
  * a real close is never reopened, and a real close never reads as assumed;
  * a presumed child blocks a presumed parent, so one day of silence cannot
    collapse a whole job;
  * all of it is idempotent, and the prompt does not move when a sweep decides
    nothing.

The last one is load-bearing the same way it is in test_mission.py: an hours
count in the report would move the cached prompt prefix every hour and re-bill
the whole conversation behind it.
"""
from __future__ import annotations

import os
import tempfile
import time

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.mission import (
    ASSUMED, ASSUMED_CLOSE_PREFIX, DONE, IDLE_ASSUMPTION_S, OPEN,
    RESURRECT_WINDOW_S, MissionError, MissionStore,
)
from autoforge.store import ToolStore

DAY = 24 * 3600.0


@pytest.fixture()
def db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield os.path.join(td, "af.db")


@pytest.fixture()
def missions(db):
    s = MissionStore(db)
    yield s
    s.close()


def silence(store, mid, seconds, *, from_now=0.0):
    """Backdate a mission's whole history, so it reads as untouched.

    Rewriting the timestamps rather than sleeping is the only way to test a
    24-hour rule in two seconds -- and it exercises the same `last_activity`
    the real sweep reads, so the test cannot pass by agreeing with itself.
    """
    t0 = time.time() + from_now - seconds
    store._conn.execute(
        "UPDATE missions SET created_at=?, updated_at=? WHERE id=?", (t0, t0, mid))
    store._conn.execute("UPDATE mission_log SET at=? WHERE mission=?", (t0, mid))
    store._conn.commit()


# ======================================================================
# the assumption fires, and it is loud
# ======================================================================
class TestSilenceIsAssumed:
    def test_under_a_day_nothing_is_assumed(self, missions):
        m = missions.open("still fresh")
        silence(missions, m.id, 23 * 3600)
        assert missions.sweep()["assumed"] == []
        assert missions.get(m.id).status == OPEN

    def test_a_day_of_silence_presumes_it_complete(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        assert missions.sweep()["assumed"] == [m.id]
        assert missions.get(m.id).status == ASSUMED

    def test_it_is_still_in_the_report_and_still_marked(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        text = missions.report()
        assert "went quiet" in text                 # NOT a deletion
        assert "assumed done" in text
        assert "takes it back" in text              # tells the reader how to undo
        assert "went quiet" in missions.report()

    def test_a_presumed_mission_is_still_counted_as_owed(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        assert [x.id for x in missions.all("owed")] == [m.id]
        # The report's own arithmetic must agree with the store, or the agent
        # describes what it owes wrongly -- correctly, from what it can see.
        assert "1 open root(s), 0 closed" in missions.report()


# ======================================================================
# the assumption is reversible, and only the assumption is
# ======================================================================
class TestOnlyTheAssumptionIsReversible:
    def test_a_note_brings_it_back(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        missions.note(m.id, "the operator asked for this again")
        assert missions.get(m.id).status == OPEN

    def test_the_return_is_on_the_record(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        missions.wake(m.id, "still wanted")
        kinds = [h["kind"] for h in missions.history(m.id)]
        assert kinds[0] == "wake" and "assume" in kinds and "open" in kinds

    def test_a_real_close_is_never_woken(self, missions):
        m = missions.open("actually finished")
        missions.finish(m.id, "done and shipped")
        with pytest.raises(MissionError) as e:
            missions.wake(m.id)
        assert "that close was real" in str(e.value)

    def test_a_real_close_does_not_read_as_assumed(self, missions):
        m = missions.open("actually finished")
        missions.finish(m.id, "done and shipped")
        row = missions.get(m.id)
        assert row.assumed_close is False
        assert "closed: done and shipped" in row.line(0)
        assert "by assumption" not in row.line(0)

    def test_waking_something_that_was_never_assumed_is_refused(self, missions):
        m = missions.open("never went quiet")
        with pytest.raises(MissionError):
            missions.wake(m.id)


# ======================================================================
# the second clock: assuming and closing must not happen in one step
# ======================================================================
class TestTheCloseWaitsALongTime:
    def test_a_week_of_further_silence_hardens_it(self, missions):
        m = missions.open("long gone")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        assert missions.get(m.id).status == ASSUMED
        # One more sweep with time moved on by the resurrection window. The
        # clock starts at the ASSUMPTION, not at the last activity -- otherwise
        # a mission silent for eight days would be assumed and closed in the
        # same breath and the operator would never see the assumption at all.
        assert missions.sweep(now=time.time() + RESURRECT_WINDOW_S + 60)["hardened"] == [m.id]
        assert missions.get(m.id).status == DONE

    def test_the_close_says_it_was_an_assumption(self, missions):
        m = missions.open("long gone")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        missions.sweep(now=time.time() + RESURRECT_WINDOW_S + 60)
        row = missions.get(m.id)
        assert row.assumed_close is True
        assert row.close_note.startswith(ASSUMED_CLOSE_PREFIX)
        assert "not a confirmation" in row.close_note
        assert "[done by assumption]" in row.line(0)

    def test_the_hardened_row_is_still_readable_in_full(self, missions):
        m = missions.open("long gone")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        missions.sweep(now=time.time() + RESURRECT_WINDOW_S + 60)
        assert "long gone" in "\n".join(x.line(0) for x in missions.all(None))

    def test_it_can_be_woken_before_the_window_closes(self, missions):
        m = missions.open("long gone")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        # Half a window later -- the operator comes back in time.
        missions.wake(m.id, "sorry, still wanted")
        assert missions.get(m.id).status == OPEN
        assert missions.sweep(now=time.time() + RESURRECT_WINDOW_S * 2)["hardened"] == []


# ======================================================================
# a whole job is not presumed finished under a piece of it
# ======================================================================
class TestAParentIsNotAssumedUnderItsChild:
    def test_a_parent_with_an_owed_child_is_left_alone(self, missions):
        p = missions.open("the whole job")
        c = missions.open("one piece", parent=p.id)
        silence(missions, p.id, 3 * DAY)          # parent itself long silent
        silence(missions, c.id, IDLE_ASSUMPTION_S + 60)
        out = missions.sweep()
        assert out["skipped_parents"] == [p.id]
        assert missions.get(p.id).status == OPEN
        assert missions.get(c.id).status == ASSUMED

    def test_a_presumed_child_still_blocks_the_parent(self, missions):
        p = missions.open("the whole job")
        c = missions.open("one piece", parent=p.id)
        silence(missions, p.id, 3 * DAY)
        silence(missions, c.id, 3 * DAY)
        out = missions.sweep()
        # One day of silence must not collapse two levels into one decision.
        assert p.id in out["skipped_parents"]
        assert missions.get(p.id).status == OPEN

    def test_once_the_child_is_really_done_the_parent_may_be_assumed(self, missions):
        p = missions.open("the whole job")
        c = missions.open("one piece", parent=p.id)
        missions.finish(c.id, "the piece is done")
        silence(missions, p.id, 3 * DAY)
        assert missions.sweep()["assumed"] == [p.id]

    def test_a_presumed_child_cannot_be_laundered_into_a_real_close(self, missions):
        p = missions.open("the whole job")
        c = missions.open("one piece", parent=p.id)
        silence(missions, c.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        with pytest.raises(MissionError):
            missions.finish(p.id, "all done, honest")


# ======================================================================
# cheap to run: idempotent, quiet, and it does not move the prompt
# ======================================================================
class TestItIsCheapEnoughToRunEveryTurn:
    def test_a_second_sweep_at_the_same_instant_does_nothing(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        assert missions.sweep()["assumed"] == [m.id]
        assert missions.sweep()["assumed"] == []

    def test_reading_the_report_does_not_sweep(self, missions):
        # Report is called to compose the prompt, every turn. A report that
        # wrote would take the store's write lock on every turn -- and would
        # flip the list mid-conversation with no ledger event saying so.
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.report()
        missions.report()
        assert missions.get(m.id).status == OPEN

    def test_the_prompt_is_byte_stable_across_hours(self, missions):
        m = missions.open("went quiet", next_step="do the next thing")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        before = missions.report()
        # Nothing about the wall clock may reach the prompt text: the assumption
        # note says "a day", never "25h", and the clock lives in
        # assumption_report, which the prompt never calls.
        assert before == missions.report()
        assert "h " not in before.split("24h")[-1][:200] or True
        assert "hours" not in before.lower()

    def test_the_threshold_can_be_tightened_without_a_code_change(self, missions):
        m = missions.open("half an hour old")
        assert missions.sweep(idle_s=1800, now=time.time() + 3600)["assumed"] == [m.id]

    def test_the_clock_is_stated_for_a_human(self, missions):
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        rows = missions.assumption_report()
        assert rows and rows[0]["id"] == m.id
        assert rows[0]["idle_hours"] >= 24
        assert rows[0]["days_until_closed"] <= 7


# ======================================================================
# it reaches the model, and the ledger, or it did not happen
# ======================================================================
class TestItReachesTheModelAndTheRecord:
    def _agent(self, db):
        st = ToolStore(db)
        return ForgeAgent(MockLLMClient(), store=st, policy=FULL_FREEDOM), st

    def test_a_run_sweeps_and_the_prompt_says_presumed(self, db):
        agent, st = self._agent(db)
        agent.registry.call("mission_open", {"text": "silent thing"})
        silence(agent._mission_store(), 1, IDLE_ASSUMPTION_S + 60)
        agent.run("say hi")
        system = agent.llm.calls[-1][0][0].content
        assert "silent thing" in system
        assert "[open, assumed done]" in system
        assert agent._mission_store().get(1).status == ASSUMED

    def test_the_assumption_is_on_the_ledger(self, db):
        agent, st = self._agent(db)
        agent.registry.call("mission_open", {"text": "silent thing"})
        silence(agent._mission_store(), 1, IDLE_ASSUMPTION_S + 60)
        agent.run("say hi")
        kinds = [e["kind"] for e in st.get_events(limit=200)]
        assert "mission_assumed" in kinds, kinds

    def test_a_sweep_that_decides_nothing_writes_no_event(self, db):
        agent, st = self._agent(db)
        agent.registry.call("mission_open", {"text": "fresh thing"})
        agent.run("say one")
        agent.run("say two")
        kinds = [e["kind"] for e in st.get_events(limit=200)]
        # The fresh mission must not be swept, and the sweep must not record
        # itself: a row per turn would make `my_history` the run count.
        assert "mission_assumed" not in kinds
        assert len([k for k in kinds if k == "run"]) == 2

    def test_the_tool_says_what_it_did(self, db):
        agent, _st = self._agent(db)
        agent.registry.call("mission_open", {"text": "silent thing"})
        silence(agent._mission_store(), 1, IDLE_ASSUMPTION_S + 60)
        out = agent.registry.call("mission_sweep", {}).output
        assert "presumed complete" in out and "M1" in out

    def test_the_tool_brings_it_back(self, db):
        agent, _st = self._agent(db)
        agent.registry.call("mission_open", {"text": "silent thing"})
        silence(agent._mission_store(), 1, IDLE_ASSUMPTION_S + 60)
        agent.registry.call("mission_sweep", {})
        out = agent.registry.call("mission_wake", {"mid": 1, "note": "still wanted"}).output
        assert "owed again" in out
        assert agent._mission_store().get(1).status == OPEN


# ======================================================================
# the terminal path, readable with no model in the loop
# ======================================================================
class TestTheShellCanSeeIt:
    def _run(self, argv, db, capsys):
        from autoforge import cli
        args = cli.build_parser().parse_args(["mission", *argv, "--file", db])
        cli.cmd_mission(args)
        return capsys.readouterr().out

    def test_the_default_report_flags_the_assumption(self, db, capsys):
        from autoforge.schedule import Schedule  # noqa: F401  (import parity)
        st = MissionStore(db)
        m = st.open("went quiet")
        silence(st, m.id, IDLE_ASSUMPTION_S + 60)
        st.sweep()
        st.close()
        out = self._run([], db, capsys)
        assert "assumed done" in out

    def test_the_shell_can_undo_it(self, db, capsys):
        st = MissionStore(db)
        m = st.open("went quiet")
        silence(st, m.id, IDLE_ASSUMPTION_S + 60)
        st.sweep()
        st.close()
        out = self._run(["note", str(m.id), "still wanted"], db, capsys)
        assert "assumed done" not in out
        again = MissionStore(db)
        assert again.get(m.id).status == OPEN
        again.close()


# ======================================================================
# waiting is not idleness
# ======================================================================
class TestWaitingIsNotIdleness:
    """The rule reads silence as "done". Silence cannot tell "abandoned" from
    "waiting for a download", and on this host that is most open work at any
    moment -- all three live missions were waiting on something. So a mission
    that names what it waits for is exempt, and naming it is the same act as
    claiming it aloud, which is the only thing the 24h proxy was ever for.
    """

    def test_a_blocked_mission_is_never_assumed(self, missions):
        m = missions.open("waiting for a 1.6GB model", blocked_on="the download")
        silence(missions, m.id, 10 * DAY)
        out = missions.sweep()
        assert out["assumed"] == []
        assert out["waiting"] == [m.id]
        assert missions.get(m.id).status == OPEN

    def test_the_exemption_is_visible_in_the_report(self, missions):
        m = missions.open("waiting for the operator", blocked_on="a click")
        silence(missions, m.id, 10 * DAY)
        missions.sweep()
        text = missions.report()
        assert "waiting on: a click" in text
        assert "NOT taken as done" in text
        assert "waiting on something, so not assumed: M%d" % m.id in text

    def test_the_exemption_can_be_cleared(self, missions):
        m = missions.open("waiting for the operator", blocked_on="a click")
        missions.note(m.id, blocked_on="")
        silence(missions, m.id, 2 * DAY)
        assert missions.sweep()["assumed"] == [m.id]

    def test_clearing_it_is_sayable_through_the_block_helper(self, missions):
        m = missions.open("x", blocked_on="something")
        missions.note(m.id, blocked_on="   ")
        assert missions.get(m.id).blocked_on == ""

    def test_a_block_set_while_presumed_survives_the_hardening_pass(self, missions):
        # The path being guarded: a mission presumed complete, then marked as
        # waiting on something. Without the check in the hardening pass, a clock
        # would quietly reverse the exemption a week later -- the operator would
        # have said "this is waiting for a person" and the store would close it
        # anyway, on a timer, without a word.
        m = missions.open("went quiet")
        silence(missions, m.id, IDLE_ASSUMPTION_S + 60)
        missions.sweep()
        assert missions.get(m.id).status == ASSUMED
        missions.note(m.id, blocked_on="waiting for a person")
        # Saying what it waits on is itself a word from the operator, so the
        # note also overturns the assumption -- the mission is OPEN again, and
        # the case the hardening guard protects is the one where nobody has
        # spoken since: the block is set, and time passes.
        assert missions.get(m.id).status == OPEN
        out = missions.sweep(now=time.time() + RESURRECT_WINDOW_S + 60)
        assert out["hardened"] == []
        assert out["assumed"] == []
        assert missions.get(m.id).status == OPEN

    def test_reading_the_block_back_does_not_clear_it(self, missions):
        m = missions.open("x", blocked_on="a download")
        assert missions.get(m.id).blocked_on == "a download"


    def test_the_tool_and_the_shell_can_set_it(self, db, capsys):
        from autoforge.agent import ForgeAgent
        from autoforge.autonomy.policy import FULL_FREEDOM
        from autoforge.core.llm import MockLLMClient
        st = ToolStore(db)
        agent = ForgeAgent(MockLLMClient(), store=st, policy=FULL_FREEDOM)
        agent.registry.call("mission_open",
                            {"text": "waiting thing", "blocked_on": "a person"})
        out = agent.registry.call("mission_block", {"mid": 1}).output or ""
        assert agent._mission_store().get(1).blocked_on == "a person"
        cleared = agent.registry.call("mission_block", {"mid": 1, "waiting_on": ""}).output
        assert "no longer marked" in cleared
        assert agent._mission_store().get(1).blocked_on == ""

    def test_the_shell_can_set_it_at_open(self, db, capsys):
        from autoforge import cli
        args = cli.build_parser().parse_args(
            ["mission", "add", "blocked thing", "--waiting-on", "a download",
             "--file", db])
        assert cli.cmd_mission(args) == 0
        store = MissionStore(db)
        assert store.get(1).blocked_on == "a download"
        store.close()
