"""The bus writes to one place, and says so when there are two.

The bug these pin: `default_bus_dir()` resolved the same mount name to two
different directories depending on whether the caller had inherited
LOCALAPPDATA. A scrubbed child fell back to the profile root, so a `bus send`
returned exit code 0 and `sent <id>` while landing on a board nobody read. That
is not a message that failed -- it is a message that looked delivered.
"""
import os
from pathlib import Path

import pytest

from autoforge import bus as bus_mod


def _scrub(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOFORGE_BUS_DIR", raising=False)
    monkeypatch.delenv("AUTOFORGE_HOME", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))


def test_no_localappdata_falls_back_to_the_appdata_local_root(tmp_path, monkeypatch):
    """Without LOCALAPPDATA the mount is still under .../AppData/Local.

    The old fallback was `expanduser("~")`, which put the mount one level up. On
    Windows `AppData/Local` is not a guess about layout -- it is what
    LOCALAPPDATA names -- and skipping it is what forked the board: the same
    mount name resolved to two directories, one per inheritance.
    """
    _scrub(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    base = Path(bus_mod._base_dir())
    assert base == tmp_path / "profile" / "AppData" / "Local"

    # With the mount present, that is where it resolves -- the real-machine case.
    mount = base / "hermes" / "agent-bus"
    (mount / "boards").mkdir(parents=True)
    (mount / "boards" / "work.ndjson").write_text('{"id": "a"}\n' * 5, encoding="utf-8")
    assert bus_mod.default_bus_dir() == mount


def test_the_chosen_mount_is_the_busiest_one(tmp_path, monkeypatch):
    """A leftover with one line must not outrank the board everyone writes to.

    "First that exists" let an earlier mistake keep winning: the abandoned
    directory satisfied the rule, so every later write went back into it. The
    two candidates below are exactly the pair the fork produced -- the
    LOCALAPPDATA mount and the profile-rooted one.
    """
    _scrub(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    base = Path(bus_mod._base_dir())                 # .../profile/AppData/Local
    live = base / "hermes" / "agent-bus"
    leftover = base / "autoforge" / "bus"
    for root, lines in ((leftover, 1), (live, 40)):
        (root / "boards").mkdir(parents=True)
        (root / "boards" / "work.ndjson").write_text(
            "".join('{"id": "%d"}\n' % i for i in range(lines)), encoding="utf-8")
    assert bus_mod.default_bus_dir() == live


def test_an_explicit_answer_beats_the_search(tmp_path, monkeypatch):
    """AUTOFORGE_BUS_DIR is followed even when it is not the busiest mount."""
    _scrub(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "platform"))
    busy = tmp_path / "platform" / "hermes" / "agent-bus"
    (busy / "boards").mkdir(parents=True)
    (busy / "boards" / "work.ndjson").write_text('{"id": "a"}\n' * 9, encoding="utf-8")
    wanted = tmp_path / "somewhere-else"
    monkeypatch.setenv("AUTOFORGE_BUS_DIR", str(wanted))
    assert bus_mod.default_bus_dir() == wanted


def test_a_split_is_reported_and_the_write_still_goes_where_it_was_told(tmp_path, monkeypatch):
    """Diagnosis is a report, never a redirect.

    Silently retargeting a write to the mount the tool thinks is right is how a
    caller loses track of its own message. The chosen mount keeps the write; the
    other one, holding more, is named.
    """
    _scrub(monkeypatch, tmp_path)
    profile = tmp_path / "profile"
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "platform"))
    ghost = profile / "hermes" / "agent-bus"          # the scrubbed-env mount
    (ghost / "boards").mkdir(parents=True)
    (ghost / "boards" / "work.ndjson").write_text('{"id": "a"}\n' * 12, encoding="utf-8")
    chosen = tmp_path / "platform" / "hermes" / "agent-bus"
    (chosen / "boards").mkdir(parents=True)
    (chosen / "boards" / "work.ndjson").write_text('{"id": "b"}\n', encoding="utf-8")

    rows = bus_mod.mount_report()
    labels = {Path(r["path"]): r["source"] for r in rows}
    assert labels[ghost] == "scrubbed-env only"

    reported = bus_mod.divergent_mounts(chosen)
    assert [Path(r["path"]) for r in reported] == [ghost]

    bus = bus_mod.Bus(chosen)
    bus.send(sender="me", board="work", body="hello")
    assert "hello" in (chosen / "boards" / "work.ndjson").read_text(encoding="utf-8")
    assert "hello" not in (ghost / "boards" / "work.ndjson").read_text(encoding="utf-8")


def test_one_mount_reports_no_split(tmp_path, monkeypatch):
    """Nothing to warn about when there is one mount: a warning that is always
    on is a warning that is never read."""
    _scrub(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "platform"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "nobody"))
    only = tmp_path / "platform" / "hermes" / "agent-bus"
    (only / "boards").mkdir(parents=True)
    (only / "boards" / "work.ndjson").write_text('{"id": "a"}\n' * 3, encoding="utf-8")
    assert bus_mod.default_bus_dir() == only
    assert bus_mod.divergent_mounts(only) == []
