"""The bus is a file, so the tests are about the two things a file gets wrong.

First: "have I read this already". A reader that tracks that in its head loses
the answer at the end of its turn, which is the whole problem the bus exists to
solve, so the cursor has to survive being written down and read back. Second:
addressing. A message to someone else must not leave a permanent unread backlog
behind it, or the next real message arrives buried.

Everything else here is the surface the CLI exposes, because the CLI is what a
session will actually call.
"""
from __future__ import annotations

import io
import json

import pytest

from autoforge import bus as bus_mod


@pytest.fixture()
def bus(tmp_path):
    return bus_mod.Bus(tmp_path / "bus")


# -- the cursor --------------------------------------------------------------
def test_a_message_survives_being_written_and_read_back(bus):
    bus.send(sender="a", board="work", body="holding cli.py")

    got = bus.read("work", "b")

    assert [e["body"] for e in got] == ["holding cli.py"]
    assert got[0]["from"] == "a"
    assert got[0]["to"] == "*"


def test_reading_advances_so_the_second_read_is_quiet(bus):
    """The point of a cursor: a reader that comes back sees only what is new."""
    bus.send(sender="a", board="work", body="one")

    assert len(bus.read("work", "b")) == 1
    assert bus.read("work", "b") == []

    bus.send(sender="a", board="work", body="two")

    assert [e["body"] for e in bus.read("work", "b")] == ["two"]


def test_peek_looks_without_advancing(bus):
    bus.send(sender="a", board="work", body="one")

    assert len(bus.read("work", "b", advance=False)) == 1
    assert len(bus.read("work", "b", advance=False)) == 1
    assert len(bus.read("work", "b")) == 1


def test_all_ignores_the_cursor_entirely(bus):
    bus.send(sender="a", board="work", body="one")
    bus.read("work", "b")

    got = bus.read("work", "b", everything=True)

    assert [e["body"] for e in got] == ["one"]


def test_cursors_are_per_reader(bus):
    """Two sessions reading one board must not consume each other's mail."""
    bus.send(sender="a", board="work", body="for both")

    assert len(bus.read("work", "b")) == 1
    assert len(bus.read("work", "c")) == 1


def test_a_board_that_does_not_exist_reads_as_empty(bus):
    assert bus.read("nosuch", "b") == []


# -- addressing --------------------------------------------------------------
def test_mine_hides_other_peoples_mail_but_still_moves_past_it(bus):
    bus.send(sender="a", board="work", body="for b", to="b")
    bus.send(sender="a", board="work", body="for c", to="c")

    got = bus.read("work", "b", only_mine=True)

    assert [e["body"] for e in got] == ["for b"]
    # The `for c` entry must not come back forever.
    assert bus.read("work", "b", only_mine=True) == []


def test_a_broadcast_reaches_everyone(bus):
    bus.send(sender="a", board="work", body="everyone", to="*")

    assert len(bus.read("work", "b", only_mine=True)) == 1
    assert len(bus.read("work", "c", only_mine=True)) == 1


# -- the file is the contract ------------------------------------------------
def test_one_entry_per_line_so_a_reader_never_splits_a_record(bus):
    bus.send(sender="a", board="work", body="line one\nline two")

    raw = bus.board_path("work").read_text(encoding="utf-8")

    assert raw == json.dumps(bus.tail("work", 1)[0], ensure_ascii=False) + "\n"


def test_an_unparseable_line_is_reported_not_dropped(bus):
    """A channel that silently drops what it cannot parse loses the one
    message that needed explaining."""
    bus.send(sender="a", board="work", body="fine")
    with open(bus.board_path("work"), "a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    got = bus.read("work", "b")

    # Only the good line is an entry; the other is a report about a line.
    assert got[0]["body"] == "fine"
    assert got[1]["broken"] == "{not json"


def test_an_unknown_kind_is_refused(bus):
    with pytest.raises(ValueError, match="unknown kind"):
        bus.send(sender="a", board="work", body="x", kind="shout")


@pytest.mark.parametrize("board", ["../escape", "a/b", "a\\b", "", "a:b"])
def test_a_board_name_that_is_not_a_filename_is_refused(bus, board):
    with pytest.raises(ValueError):
        bus.board_path(board)


def test_tail_does_not_touch_the_cursor(bus):
    bus.send(sender="a", board="work", body="one")

    assert len(bus.tail("work")) == 1
    assert len(bus.read("work", "b")) == 1


# -- presence ----------------------------------------------------------------
def test_register_keeps_what_was_learned_before(bus):
    """A follow-up call without a session id lands on the same row.

    Keyed by session id, since several sessions share the name "autoforge"; the
    name is the label inside the record, so it is what a second call matches on
    when it has no id to give.
    """
    bus.register("peer", session_id="s1", note="holding skills.py")
    bus.register("peer", cwd="/repo")

    info = bus.agents()["s1"]


def test_who_is_empty_before_anyone_registers(bus):
    assert bus.agents() == {}


# -- the CLI a session will actually call ------------------------------------
def test_cli_round_trip(tmp_path, capsys):
    d = str(tmp_path / "bus")
    assert bus_mod.cmd_bus(["--dir", d, "register", "--as", "a"]) == 0
    assert bus_mod.cmd_bus(["--dir", d, "send", "--as", "a",
                            "--board", "work", "hello there"]) == 0

    assert bus_mod.cmd_bus(["--dir", d, "read", "--as", "b",
                            "--board", "work"]) == 0
    out = capsys.readouterr().out

    assert "hello there" in out
    assert "a -> all" in out


def test_cli_read_json_is_parseable(tmp_path, capsys):
    d = str(tmp_path / "bus")
    bus_mod.cmd_bus(["--dir", d, "send", "--as", "a", "--board", "work",
                     "--kind", "ask", "who owns it?"])
    capsys.readouterr()          # drop the send's own confirmation line

    bus_mod.cmd_bus(["--dir", d, "read", "--as", "b", "--board", "work",
                     "--json"])

    entry = json.loads(capsys.readouterr().out.splitlines()[0])
    assert entry["kind"] == "ask"


def test_cli_refuses_to_send_an_empty_message(tmp_path, capsys, monkeypatch):
    """An empty send is a bug in the caller, and silence would hide it."""
    # pytest hands stdin something that raises on read, so the empty body has
    # to be injected rather than assumed.
    monkeypatch.setattr(bus_mod.sys, "stdin", io.StringIO(""))
    code = bus_mod.cmd_bus(["--dir", str(tmp_path / "bus"), "send",
                            "--as", "a", "--board", "work"])

    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_the_directory_follows_the_autoforge_home(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOFORGE_BUS_DIR", raising=False)
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-mount-here"))
    assert bus_mod.default_bus_dir() == tmp_path / "bus"

    monkeypatch.setenv("AUTOFORGE_BUS_DIR", str(tmp_path / "elsewhere"))
    assert bus_mod.default_bus_dir() == tmp_path / "elsewhere"


def test_an_existing_mount_beats_a_fresh_home(tmp_path, monkeypatch):
    """A board someone already writes to wins over an empty directory of ours.

    The failure this pins: two agents that each default to their own empty
    directory can read each other's silence as having nothing to say, when the
    real cause is that neither is looking where the other writes. On Windows the
    sessions on this host mount the protocol under %LOCALAPPDATA%\\hermes.
    """
    monkeypatch.delenv("AUTOFORGE_BUS_DIR", raising=False)
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path / "home"))
    mount = tmp_path / "platform" / "hermes" / "agent-bus"
    mount.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "platform"))

    assert bus_mod.default_bus_dir() == mount


# -- presence: a fossil is not a peer ----------------------------------------
def test_a_dead_pid_is_not_alive():
    """The failure this exists for: a registration that outlives its process.

    A session that wrote an hour ago and died since looks exactly like one that
    is merely quiet, so liveness must be read off the process table, not off
    `last_seen`.
    """
    from autoforge.bus import pid_alive

    assert pid_alive(0) is False
    assert pid_alive(-1) is False
    assert pid_alive("not a pid") is False
    assert pid_alive(99999999) is False


def test_the_running_process_is_alive():
    import os
    from autoforge.bus import pid_alive

    assert pid_alive(os.getpid()) is True


def test_live_agents_drops_a_registration_whose_process_is_gone(bus):
    bus.register("ghost", session_id="s-ghost", pid=99999999)
    bus.register("me", session_id="s-me")
    # Keyed by session, a name-keyed lookup no longer applies; assert on values.

    live = bus.live_agents()

    ids = {v["session_id"] for v in live.values()}
    assert "s-ghost" not in ids
    assert "s-me" in ids          # registered with this process’s own pid


def test_send_stamps_the_session_so_a_name_is_not_the_identity(bus):
    bus.send(sender="autoforge", board="work", body="one", session="s-1")

    assert bus.tail("work", 1)[0]["session"] == "s-1"


def test_departing_hides_that_sessions_entries(bus):
    """What a session can honestly do about its own words in an append-only log."""
    bus.send(sender="a", board="work", body="from the dead", session="s-gone")
    bus.send(sender="a", board="work", body="still here", session="s-here")

    bus.depart("s-gone")
    got = [e["body"] for e in bus.read("work", "reader")]

    assert got == ["still here"]
    # And the log still holds it, for the question "what did it say".
    assert [e["body"] for e in bus.read("work", "reader", everything=True,
                                        include_departed=True)] == [
        "from the dead", "still here"]


def test_departing_unregisters_the_session(bus):
    bus.register("me", session_id="s-me")

    info = bus.depart("s-me")

    assert info["unregistered"] == ["s-me"]
    assert "s-me" not in bus.agents()


def test_who_hides_a_registration_whose_process_is_gone(tmp_path, capsys):
    d = str(tmp_path / "bus")
    b = bus_mod.Bus(d)
    b.register("ghost", session_id="s-ghost", pid=99999999)

    assert bus_mod.cmd_bus(["--dir", d, "who"]) == 0
    out = capsys.readouterr().out
    assert "ghost" not in out
    assert "--all" in out            # the hint, so the fossil is not simply lost

    assert bus_mod.cmd_bus(["--dir", d, "who", "--all"]) == 0
    assert "ghost" in capsys.readouterr().out


# -- currency: a dead author's words are not the operator's problem ----------
def test_a_dead_sessions_words_are_not_current(bus):
    """The failure this exists for, in the operator's own words: "you keep
    telling me what a session I killed hours ago said".

    `departed` only covers a session that left on purpose, and a session that
    was killed never gets to say goodbye -- so its words stay current forever
    and every startup reports them as mail. The process table is the only
    honest judge, and it is the same judge `live_agents` already uses.
    """
    bus.register("gone", session_id="s-gone", pid=99999999)
    bus.register("here", session_id="s-here")
    bus.send(sender="gone", board="work", body="from the grave", session="s-gone")
    bus.send(sender="here", board="work", body="still here", session="s-here")

    assert [e["body"] for e in bus.read("work", "me")] == ["still here"]
    # And the log still holds it, for the question "what did it say".
    kept = bus.read("work", "me", everything=True, include_dead=True)
    assert [e["body"] for e in kept] == ["from the grave", "still here"]


def test_an_entry_with_only_a_name_is_judged_by_that_name(bus):
    """The older writer stamped a name and no session id, and those entries are
    most of a board that has been running for a while. Leaving them unjudged is
    what kept the fossils alive: measured on this host, all 31 entries on the
    board carried no session at all, so no id-based rule could ever retire one.
    """
    bus.register("old-timer", session_id="s-old", pid=99999999)
    with open(bus.board_path("work"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "x", "ts": "2026-09-15T11:00:00+08:00",
                             "from": "old-timer", "to": "*", "kind": "msg",
                             "body": "written before sessions were stamped"})
                 + "\n")

    assert bus.read("work", "me") == []
    assert len(bus.read("work", "me", everything=True, include_dead=True)) == 1


def test_an_author_nobody_registered_keeps_its_words(bus):
    """A name with no registration cannot be judged, and the rule only ever
    fires on evidence: hiding a live peer's message is worse than showing a
    dead one's, because the message is the thing the channel exists to carry.
    """
    bus.send(sender="stranger", board="work", body="unjudgeable")

    assert [e["body"] for e in bus.read("work", "me")] == ["unjudgeable"]


def test_one_live_registration_keeps_the_shared_name_alive(bus):
    """Two sessions are called "autoforge" here as a rule, so a name-based
    verdict has to be the conservative one -- otherwise one session dying
    retires the words of the one still working."""
    bus.register("autoforge", session_id="s-dead", pid=99999999)
    bus.register("autoforge", session_id="s-alive")
    bus.send(sender="autoforge", board="work", body="who owns cli.py?",
             session="s-alive")

    assert [e["body"] for e in bus.read("work", "me")] == ["who owns cli.py?"]


# -- startup: only what a live peer actually needs --------------------------
def test_startup_counts_only_what_a_live_peer_needs(tmp_path):
    """The line the operator reads at every start. Whole history is not news:
    asks and claims from a live peer are, and so is the fact that something was
    passed over -- "ignored" and "nothing there" are different facts.
    """
    b = bus_mod.Bus(tmp_path / "bus")
    b.register("peer", session_id="s-peer")            # this process: alive
    b.register("ghost", session_id="s-ghost", pid=99999999)
    b.send(sender="peer", board="autoforge", body="who owns cli.py?",
           session="s-peer", kind="ask")
    b.send(sender="ghost", board="autoforge", body="I am editing agent.py",
           session="s-ghost", kind="claim")

    line = bus_mod.startup_check(b, "s-me")

    assert "1 other live session(s)" in line
    assert "1 message(s) needing you" in line
    assert "1 stale entry" in line


def test_startup_says_so_when_the_board_holds_nothing_current(tmp_path):
    """Nothing to report is a result, and it has to read as one: the operator's
    complaint was a board of fossils reported as unread mail."""
    b = bus_mod.Bus(tmp_path / "bus")
    b.register("ghost", session_id="s-ghost", pid=99999999)
    b.send(sender="ghost", board="autoforge", body="hours ago", session="s-ghost")

    line = bus_mod.startup_check(b, "s-me")

    assert "no other live session" in line
    assert "unread" not in line
    assert "stale" in line


def test_a_wrapper_message_is_stamped_with_the_session_it_belongs_to(tmp_path):
    """`auto bus send --as autoforge` from a shell used to stamp a brand-new
    throwaway id per invocation, so the session that owns those entries could
    never retire them: "clear my messages when I close" was unenforceable for
    exactly the messages a session writes through the CLI."""
    d = str(tmp_path / "bus")
    b = bus_mod.Bus(d)
    b.register("autoforge", session_id="s-me")          # alive: this process

    assert bus_mod.cmd_bus(["--dir", d, "send", "--as", "autoforge",
                            "--board", "work", "holding cli.py"]) == 0

    assert b.tail("work", 1)[0]["session"] == "s-me"


def test_an_explicit_session_beats_the_registration(tmp_path):
    d = str(tmp_path / "bus")
    b = bus_mod.Bus(d)
    b.register("autoforge", session_id="s-me")

    assert bus_mod.cmd_bus(["--dir", d, "send", "--as", "autoforge",
                            "--session", "s-mine", "--board", "work", "hi"]) == 0

    assert b.tail("work", 1)[0]["session"] == "s-mine"

