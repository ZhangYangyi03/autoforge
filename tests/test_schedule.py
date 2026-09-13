"""An agenda the agent can keep, and a wake-up that actually exists.

Two things here are load-bearing beyond the data structure.

**Repeats are anchored, not drifted.** A recurring task advances from its
*scheduled* time, never from the moment it happened to run. Advance from now
and a nightly job run ten minutes late becomes 00:10 tomorrow, then 00:20, and
inside a month "every night at midnight" is "sometime after lunch" -- and no
single run looks wrong.

**The wake-up command is real.** `wake_command` exists to be handed to the
operating system, so it is tested by taking the exact argv it emits, stripping
the interpreter, and feeding the rest to the real parser. That is the shape of
the bug this pins down: the module emitted `python -m autoforge tick --quiet`
for `install_system_task` to register, and `autoforge` had no `__main__` and no
`tick` subcommand. Every scheduled task would have failed every thirty minutes,
for as long as the machine lived, without ever once running.
"""

from __future__ import annotations

import importlib
import json
import time

import pytest

from autoforge.schedule import (
    PERIODS,
    Schedule,
    ScheduleError,
    _console_text,
    as_clock,
    human_delta,
    install_system_task,
    parse_when,
    system_task_command,
    wake_command,
)

BASE = 1_700_000_000.0        # a fixed epoch, so every assertion is exact


# ----------------------------------------------------------------------
# reading a time the way a person writes one
# ----------------------------------------------------------------------


class TestParseWhen:

    @pytest.mark.parametrize("text,expected", [
        ("+90s", 90), ("90s", 90),
        ("+10m", 600), ("10m", 600),
        ("+2h", 7200), ("2h", 7200),
        ("+1d", 86400), ("1d", 86400),
        ("+1w", 604800),
        ("0s", 0), ("+0m", 0),
    ])
    def test_durations(self, text, expected):
        assert parse_when(text, BASE) == BASE + expected

    def test_units_are_not_assumed(self):
        """Every unit in PERIODS must be reachable by its own letter."""
        for unit, size in PERIODS.items():
            assert parse_when(f"3{unit}", BASE) == BASE + 3 * size

    def test_negative_duration_is_refused(self):
        with pytest.raises(ScheduleError, match="negative"):
            parse_when("-5m", BASE)

    def test_garbage_names_the_expected_forms(self):
        with pytest.raises(ScheduleError, match="ISO 8601"):
            parse_when("someday", BASE)

    def test_duration_shaped_garbage_says_it_is_the_duration_that_is_wrong(self):
        """'next tuesday-ish' ends in 'h', so it reads as a broken duration.

        Worth pinning: the message must be about the thing the parser actually
        tried, or the person fixes the wrong half of their input.
        """
        with pytest.raises(ScheduleError, match="as a duration"):
            parse_when("next tuesday-ish", BASE)

    def test_empty_is_refused(self):
        with pytest.raises(ScheduleError, match="no time"):
            parse_when("", BASE)
        with pytest.raises(ScheduleError, match="no time"):
            parse_when(None, BASE)

    def test_naive_iso_is_local_time(self):
        """'nine tomorrow morning' means nine where the person is."""
        from datetime import datetime
        got = parse_when("2026-09-14T09:00:00", BASE)
        assert datetime.fromtimestamp(got).strftime("%H:%M") == "09:00"

    def test_z_suffix_is_utc(self):
        from datetime import datetime, timezone
        got = parse_when("2026-09-14T09:00:00Z", BASE)
        assert datetime.fromtimestamp(got, timezone.utc).hour == 9

    def test_explicit_offset_is_honoured(self):
        assert parse_when("2026-09-14T09:00:00+02:00", BASE) == \
            parse_when("2026-09-14T07:00:00+00:00", BASE)

    def test_bare_epoch_passes_through(self):
        assert parse_when(1_700_000_123.5) == 1_700_000_123.5
        assert parse_when(1_700_000_123) == 1_700_000_123.0

    def test_a_bool_is_not_an_epoch(self):
        """True is 1 in Python; accepting it would be a silent nonsense date."""
        with pytest.raises(ScheduleError):
            parse_when(True, BASE)


class TestHumanising:

    @pytest.mark.parametrize("secs,expected", [
        (0, "0s"), (45, "45s"), (60, "1m"), (5400, "1.5h"), (86400, "1d"),
        (172800, "2d"), (-10, "0s"),
    ])
    def test_human_delta(self, secs, expected):
        assert human_delta(secs) == expected

    def test_as_clock_reads_back_as_the_same_instant(self):
        """Readable, and in local time -- not a UTC reading with the label shaved.

        Parsing the output back and comparing the instant catches a wrong
        offset, which a format-only assertion would happily accept.
        """
        from datetime import datetime
        parsed = datetime.strptime(as_clock(BASE), "%Y-%m-%d %H:%M:%S")
        assert abs(parsed.timestamp() - BASE) < 1.0


# ----------------------------------------------------------------------
# the table
# ----------------------------------------------------------------------


@pytest.fixture
def table(tmp_path):
    return Schedule(tmp_path / "schedule.jsonl")


class TestAdding:

    def test_add_returns_a_task_with_an_id(self, table):
        t = table.add("water the plants", "1h", now=BASE)
        assert t.id and t.text == "water the plants"
        assert t.due_at == BASE + 3600
        assert t.kind == "once"

    def test_empty_text_is_refused(self, table):
        with pytest.raises(ScheduleError, match="empty"):
            table.add("   ", "1h", now=BASE)

    def test_a_repeating_task_records_its_interval(self, table):
        t = table.add("check the queue", "1h", repeat="1d", now=BASE)
        assert t.kind == "repeating"
        assert t.repeat == 86400

    def test_a_non_positive_repeat_is_refused(self, table):
        with pytest.raises(ScheduleError, match="positive"):
            table.add("x", "1h", repeat="0s", now=BASE)

    def test_ids_are_distinct(self, table):
        ids = {table.add(f"task {i}", "1h", now=BASE).id for i in range(20)}
        assert len(ids) == 20


class TestQuerying:

    def test_due_is_a_query_not_a_trigger(self, table):
        table.add("later", "1h", now=BASE)
        assert table.due(BASE) == []
        assert len(table.due(BASE + 3600)) == 1

    def test_due_is_oldest_first(self, table):
        table.add("second", "2h", now=BASE)
        table.add("first", "1h", now=BASE)
        assert [t.text for t in table.due(BASE + 7200)] == ["first", "second"]

    def test_cancelled_tasks_are_not_due(self, table):
        t = table.add("x", "1h", now=BASE)
        table.cancel(t.id)
        assert table.due(BASE + 7200) == []
        assert table.get(t.id).cancelled

    def test_next_due_ignores_the_past(self, table):
        table.add("overdue", "1h", now=BASE)
        table.add("coming", "5h", now=BASE)
        assert table.next_due(BASE + 7200).text == "coming"

    def test_next_due_is_none_on_an_empty_table(self, table):
        assert table.next_due(BASE) is None


class TestCompleting:

    def test_a_one_shot_closes(self, table):
        t = table.add("x", "1h", now=BASE)
        done = table.complete(t.id, now=BASE + 3600)
        assert done.runs == 1
        assert done.cancelled, "a finished one-shot is not still on the agenda"
        assert table.due(BASE + 7200) == []

    def test_a_failure_is_counted_and_named(self, table):
        t = table.add("x", "1h", now=BASE)
        done = table.complete(t.id, ok=False, note="connection refused",
                              now=BASE + 3600)
        assert done.failures == 1
        assert "connection refused" in done.notes[-1]

    def test_an_unknown_id_is_refused(self, table):
        with pytest.raises(ScheduleError, match="no task"):
            table.complete("nope")

    def test_history_and_notes_are_bounded(self, table):
        t = table.add("x", "1h", repeat="1h", now=BASE)
        for i in range(40):
            table.complete(t.id, note=f"run {i}", now=BASE + 3600 * (i + 1))
        got = table.get(t.id)
        assert len(got.history) <= 20
        assert len(got.notes) <= 10

    def test_a_long_note_is_truncated(self, table):
        t = table.add("x", "1h", now=BASE)
        got = table.complete(t.id, note="z" * 5000, now=BASE + 1)
        assert len(got.notes[-1]) <= 500


class TestRepeatsDoNotDrift:
    """The property that keeps 'every night at midnight' meaning that."""

    def _grid(self, base: float, step: float, at: float) -> bool:
        """Is `at` on the series that started at `base`, stepping `step`?"""
        return abs(((at - base) / step) - round((at - base) / step)) < 1e-9

    def test_an_on_time_run_advances_one_period(self, table):
        t = table.add("nightly", "1h", repeat="1h", now=BASE)
        got = table.complete(t.id, now=BASE + 3600)
        assert got.due_at == BASE + 7200

    def test_a_late_run_does_not_shift_the_series(self, table):
        """Lateness must not move the next occurrence."""
        t = table.add("nightly", "1h", repeat="1h", now=BASE)
        # Runs 100 seconds late, having missed nothing else.
        got = table.complete(t.id, now=BASE + 3600 + 100)
        assert self._grid(BASE, 3600, got.due_at)
        assert got.due_at > BASE + 3600 + 100

    def test_many_late_runs_never_accumulate(self, table):
        """Twenty consecutive late runs; the series is still the original."""
        t = table.add("nightly", "1h", repeat="1h", now=BASE)
        now = BASE
        for _ in range(20):
            now += 3600 + 137          # 137s late, every time
            table.complete(t.id, now=now)
            assert self._grid(BASE, 3600, table.get(t.id).due_at), \
                "a late run moved the series"

    def test_a_long_outage_skips_whole_periods_only(self, table):
        """After a week asleep, the next run is on the grid and in the future."""
        t = table.add("nightly", "1h", repeat="1h", now=BASE)
        now = BASE + 7 * 86400 + 123
        got = table.complete(t.id, now=now)
        assert self._grid(BASE, 3600, got.due_at)
        assert got.due_at > now
        assert got.due_at - now <= 3600, "it over-skipped past the next slot"

    def test_a_repeating_task_is_not_cancelled_by_completing(self, table):
        t = table.add("nightly", "1h", repeat="1h", now=BASE)
        table.complete(t.id, now=BASE + 3600)
        assert not table.get(t.id).cancelled


class TestPersistence:

    def test_a_task_survives_a_restart(self, tmp_path):
        path = tmp_path / "s.jsonl"
        first = Schedule(path)
        first.add("survive me", "1h", repeat="1d", now=BASE)
        again = Schedule(path)
        assert [t.text for t in again.all()] == ["survive me"]

    def test_completion_state_survives(self, tmp_path):
        path = tmp_path / "s.jsonl"
        first = Schedule(path)
        t = first.add("x", "1h", now=BASE)
        first.complete(t.id, ok=False, note="boom", now=BASE + 1)
        again = Schedule(path)
        assert again.get(t.id).failures == 1
        assert "boom" in again.get(t.id).notes

    def test_one_damaged_line_costs_only_its_own_task(self, tmp_path):
        """A crash mid-append must not take every future task with it."""
        path = tmp_path / "s.jsonl"
        good = json.dumps({"id": "aaaa", "text": "keep me", "due_at": BASE,
                           "created_at": BASE})
        path.write_text(good + "\n{ this is not json\n", encoding="utf-8")
        table = Schedule(path)
        assert [t.text for t in table.all()] == ["keep me"]
        assert "1 unreadable line" in table.load_error

    def test_the_load_problem_is_visible_in_the_report(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text("garbage\n", encoding="utf-8")
        assert "unreadable" in Schedule(path).report(BASE)

    def test_an_absent_file_is_not_an_error(self, tmp_path):
        table = Schedule(tmp_path / "never-written.jsonl")
        assert table.all() == []
        assert table.load_error == ""

    def test_a_missing_directory_is_created(self, tmp_path):
        path = tmp_path / "deep" / "deeper" / "s.jsonl"
        Schedule(path).add("x", "1h", now=BASE)
        assert path.exists()

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        path = tmp_path / "s.jsonl"
        Schedule(path).add("x", "1h", now=BASE)
        assert not list(tmp_path.glob("*.tmp"))


class TestRemoving:

    def test_forget_removes_and_reports(self, table):
        t = table.add("x", "1h", now=BASE)
        assert table.forget(t.id) is True
        assert table.forget(t.id) is False
        assert table.get(t.id) is None

    def test_cancel_keeps_the_record(self, table):
        t = table.add("x", "1h", now=BASE)
        table.cancel(t.id)
        assert table.get(t.id) is not None
        assert table.get(t.id).cancelled

    def test_cancelling_an_unknown_id_is_refused(self, table):
        with pytest.raises(ScheduleError):
            table.cancel("nope")


class TestReport:

    def test_empty_says_so(self, table):
        assert "Nothing scheduled" in table.report(BASE)

    def test_overdue_and_upcoming_are_separated(self, table):
        table.add("late one", "1h", now=BASE)
        table.add("later one", "5h", now=BASE)
        out = table.report(BASE + 7200)
        assert "OVERDUE" in out
        assert "1 upcoming" in out

    def test_it_says_what_it_cannot_do(self, table):
        """An agenda that cannot wake anyone must say so, not imply it can."""
        table.add("x", "1h", now=BASE)
        out = table.report(BASE)
        assert "install_system_task" in out


# ----------------------------------------------------------------------
# handing it to the operating system
# ----------------------------------------------------------------------


class TestWakeCommand:

    def test_it_runs_the_module(self):
        argv = wake_command(30)
        assert argv[1:] == ["-m", "autoforge", "tick", "--quiet"]

    def test_the_module_entry_point_exists(self):
        """`python -m autoforge` needs autoforge/__main__.py to exist.

        Without it the registered task dies with "No module named
        autoforge.__main__", thirty minutes at a time, forever.
        """
        module = importlib.import_module("autoforge.__main__")
        assert hasattr(module, "main")

    def test_the_emitted_command_is_accepted_by_the_real_parser(self):
        """Take the exact argv, drop the interpreter, and parse the rest.

        This is the test that would have caught the original bug: the module
        promised a command the CLI did not have.
        """
        from autoforge.cli import build_parser

        argv = wake_command(30)
        rest = argv[2:]                       # drop the python executable
        assert rest[0] == "autoforge"
        args = build_parser().parse_args(rest[1:])
        assert args.command == "tick"
        assert args.quiet is True

    def test_it_uses_the_running_interpreter(self):
        import sys
        assert wake_command(30)[0] == sys.executable

    def test_a_script_override_replaces_the_module_invocation(self):
        assert wake_command(30, script="/opt/af.py")[1:] == ["/opt/af.py"]


class TestSystemTaskCommand:

    def test_windows_registers_a_minute_schedule(self):
        argv = system_task_command(30, "af-tick", platform="win32")
        assert argv[0] == "schtasks"
        assert "/SC" in argv and argv[argv.index("/SC") + 1] == "MINUTE"
        assert argv[argv.index("/MO") + 1] == "30"
        # /F so a second install repairs rather than prompts.
        assert "/F" in argv

    def test_the_registered_action_quotes_its_arguments(self):
        argv = system_task_command(30, "af-tick", platform="win32")
        action = argv[argv.index("/TR") + 1]
        assert action.startswith('"') and "autoforge" in action

    def test_posix_returns_a_crontab_line(self):
        argv, line = system_task_command(30, "af-tick", platform="linux")
        assert argv == ["crontab", "-l"]
        assert "*/30 * * * *" in line
        assert line.rstrip().endswith("# af-tick")

    def test_an_hourly_interval_is_expressed_in_hours(self):
        _, line = system_task_command(120, "af-tick", platform="linux")
        assert line.startswith("0 */2 * * *")

    def test_macos_uses_the_same_mechanism(self):
        argv, _ = system_task_command(15, platform="darwin")
        assert argv == ["crontab", "-l"]

    def test_a_zero_interval_is_clamped_to_one(self):
        argv = system_task_command(0, platform="win32")
        assert argv[argv.index("/MO") + 1] == "1"

    def test_an_unknown_platform_is_refused_loudly(self):
        with pytest.raises(ScheduleError, match="no system-task mechanism"):
            system_task_command(30, platform="plan9")


class TestConsoleText:
    """The console code page is not UTF-8 on the machines this runs on."""

    def test_gbk_bytes_decode(self):
        """schtasks on a Chinese Windows answers in GBK, not UTF-8."""
        assert _console_text("错误: 拒绝访问。".encode("gbk")) == "错误: 拒绝访问。"

    def test_utf8_bytes_decode(self):
        assert _console_text("access denied".encode("utf-8")) == "access denied"

    def test_undecodable_bytes_do_not_raise(self):
        """The point of the helper: a bad byte is a replacement char."""
        out = _console_text(b"\xb3\xff\xfe not a string")
        assert isinstance(out, str) and out

    def test_empty_and_none_are_empty(self):
        assert _console_text(None) == ""
        assert _console_text(b"") == ""


class _FakeProc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class TestInstallSystemTask:
    def test_an_unknown_platform_raises_rather_than_pretending(self):
        with pytest.raises(ScheduleError):
            install_system_task(30, platform="plan9")

    def test_posix_prints_a_line_instead_of_rewriting_a_crontab(self):
        """Editing someone's crontab blind takes their other jobs with it."""
        out = install_system_task(30, platform="linux")
        assert "crontab -e" in out
        assert "Not installed automatically" in out

    def test_windows_success_names_the_undo(self):
        """A config change the agent made must come with its reversal."""
        import subprocess as sp
        import autoforge.schedule as sched
        original = sp.run
        sp.run = lambda *a, **k: _FakeProc(0, stdout=b"SUCCESS\r\n")
        try:
            out = sched.install_system_task(30, name="af-test", platform="win32")
        finally:
            sp.run = original
        assert "af-test" in out
        assert "schtasks /Delete /TN af-test /F" in out

    def test_a_refusal_in_gbk_is_reported_as_text_not_a_decode_crash(self):
        """The bug this pins: `text=True` decoded GBK as UTF-8 and blew up.

        The operator would have seen a UnicodeDecodeError from a reader thread
        instead of the scheduler's actual complaint.
        """
        import subprocess as sp
        import autoforge.schedule as sched
        refusal = "错误: 无法创建计划任务。".encode("gbk")
        original = sp.run
        sp.run = lambda *a, **k: _FakeProc(1, stderr=refusal)
        try:
            with pytest.raises(ScheduleError) as caught:
                sched.install_system_task(30, name="af-test", platform="win32")
        finally:
            sp.run = original
        assert "refused the task" in str(caught.value)
        assert "无法创建计划任务" in str(caught.value)

    def test_bytes_are_requested_from_subprocess_not_text(self):
        """`text=True` is what broke on a non-UTF-8 console; assert it is gone."""
        import subprocess as sp
        import autoforge.schedule as sched
        seen: dict = {}
        original = sp.run
        def spy(*a, **k):
            seen.update(k)
            return _FakeProc(0, stdout=b"ok")
        sp.run = spy
        try:
            sched.install_system_task(30, name="af-test", platform="win32")
        finally:
            sp.run = original
        assert not seen.get("text"), "text=True re-introduces the decode crash"
        assert "encoding" not in seen
