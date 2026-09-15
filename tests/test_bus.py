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
    bus.register("peer", session_id="s1", note="holding skills.py")
    bus.register("peer", cwd="/repo")

    info = bus.agents()["peer"]

    assert info["session_id"] == "s1"
    assert info["note"] == "holding skills.py"
    assert info["cwd"] == "/repo"


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
