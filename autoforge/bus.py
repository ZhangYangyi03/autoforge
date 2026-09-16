"""A mail channel between autoforge sessions, and between agent and operator.

Two `auto` processes started in the same repo cannot see each other. There is no
socket, no broadcast, and — this is the part that surprises — no store of theirs
the other one polls: each run reads the database when its own turn takes it
there. So a second session's existence is invisible exactly when it matters,
which is when both are editing the same tree.

The channel is therefore a file. The bus is an append-only log that any session
may write to and any session may read, and delivery is pull: a message waits
until someone looks. That is a weaker promise than a socket, and the honest one —
what makes it useful anyway is that both sides already have a shell.

The one thing a bare file gets wrong is "have I read this already". A reader who
tracks that mentally loses the answer at the end of its turn, which is the whole
problem restated. So each reader keeps a cursor on disk: a line count per board.
`read` advances it, `--peek` does not, and `--all` ignores it. That is what lets
someone come back an hour later and see only what arrived since.

Entries are one JSON object per line, appended in a single write call so a reader
never observes half a record:

    {"id": "c676487fb74e", "ts": "2026-09-15T11:53:33+08:00",
     "from": "hermes-ef3ad8", "to": "hermes-7f9860", "kind": "msg",
     "body": "...", "reply_to": null}

Layout under the bus directory: `$AUTOFORGE_BUS_DIR` if set, else the first of
the known mounts that already exists, else the per-user home the tool store uses:

    boards/<board>.ndjson           the log — one board per workstream
    cursors/<board>.<reader>        that reader's line count, as plain text
    agents.json                     who registered, with session id and pid

Because it is only a file, one implementation serves both mounts: this module is
what `auto bus` runs, and the same protocol is what the Hermes sessions on this
machine use under `%LOCALAPPDATA%\\hermes\\agent-bus`. Point either at the other's
directory and they interoperate.

The resolution of the home directory is spelled out here rather than imported
from `store._default_home` on purpose: that name is private to the store, and a
cross-module reach for a private helper couples two modules that are otherwise
free to move.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time as _time

#: The kinds a message may claim to be. A closed set, so a reader can filter on
#: it: an `ask` wants an `answer`, and a `claim` on a file is what stops two
#: sessions editing it. `msg` is the untyped default and stays the common case.
KINDS = ("msg", "ask", "answer", "claim", "done", "note")


#: Presence must be CHECKED, never remembered. `last_seen` is a fossil: it says
#: when someone last wrote, and a session that wrote an hour ago and has since
#: died looks exactly like one that is merely quiet. Only the process table
#: tells those two apart -- reading a board is not enough to know whether
#: anyone is still there to answer.
def pid_alive(pid) -> bool:
    "Is `pid` a live process right now? False for 0, junk and gone."
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True              # exists, owned by someone else
        except OSError:
            return False
        return True
    import ctypes
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)      # QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if not k.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == 259                 # STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def own_session_id() -> str:
    "A per-process identity, so two sessions sharing a name stay two people."
    env = os.environ.get("AUTOFORGE_SESSION_ID")
    if env:
        return env
    return "auto-%d-%d" % (os.getpid(), int(_time.time()))


def _ppid_of(pid) -> int:
    "Parent pid, or 0. A wrapper shell registers on behalf of its agent."
    if os.name != "nt":
        try:
            with open("/proc/%d/stat" % int(pid), encoding="utf-8") as fh:
                return int(fh.read().rsplit(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return 0
    import ctypes
    from ctypes import wintypes
    class _PE32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]
    k = ctypes.windll.kernel32
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    if not snap:
        return 0
    try:
        e = _PE32()
        e.dwSize = ctypes.sizeof(_PE32)
        if not k.Process32First(snap, ctypes.byref(e)):
            return 0
        while True:
            if e.th32ProcessID == int(pid):
                return int(e.th32ParentProcessID)
            if not k.Process32Next(snap, ctypes.byref(e)):
                return 0
    finally:
        k.CloseHandle(snap)

def _base_dir() -> str:
    """The per-user data root, however this platform spells it.

    On Windows the fallback is `%USERPROFILE%\AppData\Local`, not
    `%USERPROFILE%`. `LOCALAPPDATA` is one of the variables the sandbox does
    not pass to a forged child, and a scheduled task can run without it too --
    so a bare `expanduser("~")` fallback resolves the *same* mount name
    (`hermes/agent-bus`) to a *different* directory depending on who is asking.
    The board then splits in two: one with every live session on it, one with
    an empty registry, both called "autoforge". Measured on 2026-09-16: a
    `bus send` returned `sent <id>` with exit code 0 and landed on the board
    nobody was reading, because its parent process had no `LOCALAPPDATA`.
    """
    env = os.environ.get("LOCALAPPDATA")
    if env:
        return env
    if os.name == "nt":
        profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        return os.path.join(profile, "AppData", "Local")
    return os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~")


def default_home() -> str:
    """Where the bus lives when nothing overrides it.

    Same rule as the tool store, for the same reason: state that follows the
    current directory is state that silently resets, and a channel that resets
    reads as a peer that went quiet rather than as a path that moved.
    """
    env = os.environ.get("AUTOFORGE_HOME")
    if env:
        return env
    return os.path.join(_base_dir(), "autoforge")


#: Other mounts of the same protocol. An existing board beats a fresh empty
#: one: two agents that each default to their own empty directory look exactly
#: like two agents with nothing to say -- which is the failure this module
#: exists to prevent, so the default must not create it.
OTHER_MOUNTS = ("hermes/agent-bus",)


def _mount_score(root: Path) -> tuple[int, float]:
    """How much is *on* a mount: (entries, last write). Emptiness is not a claim."""
    entries = 0
    newest = 0.0
    boards = root / "boards"
    if boards.is_dir():
        for f in boards.glob("*.ndjson"):
            try:
                entries += sum(1 for ln in f.read_text(encoding="utf-8").splitlines()
                               if ln.strip())
                newest = max(newest, f.stat().st_mtime)
            except OSError:
                pass
    return entries, newest


def _dedupe(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    keys: set[str] = set()
    for q in paths:
        key = os.path.normcase(os.path.abspath(str(q)))
        if key not in keys:
            keys.add(key)
            out.append(q)
    return out


def env_mounts() -> list[Path]:
    """The mounts the *handed* environment justifies. Only these are chosen.

    Not a search of the disk. `default_bus_dir` has to be a function of what it
    was given, or a test that sets its own environment gets answered by whatever
    happens to be on the machine -- the same mistake one level up: a default
    that reads state the caller never named.
    """
    rels = [os.environ.get("AUTOFORGE_BUS_DIR")]
    rels += [os.path.join(_base_dir(), *rel.split("/")) for rel in OTHER_MOUNTS]
    rels.append(os.path.join(default_home(), "bus"))
    return _dedupe([Path(r) for r in rels if r])


def diagnostic_mounts() -> list[Path]:
    """Every directory the protocol might live in, for *diagnosis*.

    Adds where a child with a scrubbed environment resolves to. A process that
    does not inherit LOCALAPPDATA computes the profile-rooted mount, and that --
    not a bad habit -- is how one protocol came to have two boards on this host.
    Reported by `auto bus mounts`, never selected by default.
    """
    extra: list[Path] = []
    if os.name == "nt":
        profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        extra = [Path(profile).joinpath(*rel.split("/")) for rel in OTHER_MOUNTS]
    return _dedupe(env_mounts() + extra)


def candidate_mounts() -> list[Path]:
    """Every directory this protocol might live in, de-duplicated, best first.

    Kept as a list rather than a single answer because the *divergence* is the
    bug: one mount is where the live sessions are, the other is where a child
    with a scrubbed environment just wrote. Returning only the winner hides
    that, which is exactly how the split went unnoticed.
    """
    return _dedupe(diagnostic_mounts())


def mount_report() -> list[dict[str, Any]]:
    """What is in each candidate mount: the data that makes a split visible."""
    live = os.environ.get("AUTOFORGE_BUS_DIR")
    rows = []
    for q in candidate_mounts():
        entries, newest = _mount_score(q)
        rows.append({
            "path": str(q),
            "exists": q.is_dir(),
            "entries": entries,
            "newest": (datetime.fromtimestamp(newest).astimezone()
                       .isoformat(timespec="seconds") if newest else ""),
            "source": ("AUTOFORGE_BUS_DIR" if live and str(q) == live
                       else "default" if q in env_mounts()
                       else "scrubbed-env only"),
        })
    rows.sort(key=lambda r: (r["entries"], r["newest"]), reverse=True)
    return rows


def divergent_mounts(chosen: Path) -> list[dict[str, Any]]:
    """Mounts that hold entries while the chosen one is not among the busiest.

    A warning, not a redirect. Silently writing to a mount the caller did not
    name is how a message ends up somewhere nobody looked; the fix is to say so
    loudly and write where you were told.
    """
    rows = mount_report()
    if not rows:
        return []
    key = os.path.normcase(os.path.abspath(str(chosen)))
    top = rows[0]
    chosen_row = next((r for r in rows
                       if os.path.normcase(os.path.abspath(r["path"])) == key), None)
    if chosen_row is None or top["entries"] == 0:
        return []
    if chosen_row["entries"] >= top["entries"]:
        return []
    return [r for r in rows
            if r["entries"] > 0 and
            os.path.normcase(os.path.abspath(r["path"])) != key]


def default_bus_dir() -> Path:
    """`$AUTOFORGE_BUS_DIR` wins, else the *busiest* existing mount, else home/bus.

    "Busiest" replaced "first that exists" on 2026-09-16. A directory left over
    from one earlier mistake is not empty and therefore satisfied the old rule,
    and the mount every live session was using lost to it. An environment
    variable still outranks everything: an explicit answer is not a guess to be
    second-guessed.
    """
    env = os.environ.get("AUTOFORGE_BUS_DIR")
    if env:
        return Path(env)
    existing = [q for q in env_mounts() if q.is_dir()]
    if existing:
        return max(existing, key=_mount_score)
    return Path(default_home()) / "bus"


def _stamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Bus:
    """One bus directory: its boards, its cursors, its registry."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root is not None else default_bus_dir()
        self.boards_dir = self.root / "boards"
        self.cursors_dir = self.root / "cursors"
        for d in (self.boards_dir, self.cursors_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- boards ----------------------------------------------------------
    def board_path(self, board: str) -> Path:
        # A board name becomes a filename, so a name with a separator in it
        # would escape the bus. Reject rather than sanitise: a silently
        # rewritten path is a message that lands where nobody looks.
        if not board or any(c in board for c in "\\/:*?\"<>|"):
            raise ValueError(f"board name {board!r} is not usable as a filename")
        return self.boards_dir / f"{board}.ndjson"

    def boards(self) -> list[str]:
        return sorted(p.stem for p in self.boards_dir.glob("*.ndjson"))

    # -- writing ---------------------------------------------------------
    def send(self, *, sender: str, board: str, body: str, to: str = "*",
             kind: str = "msg", reply_to: str | None = None,
             session: str | None = None) -> dict[str, Any]:
        """Append one entry and return it.

        One `write` call, then flush: a line is far under the size at which an
        append can be observed half-written, and the alternative — locking a
        file a peer may be holding open — trades a theoretical interleave for a
        real deadlock.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r} (have: {', '.join(KINDS)})")
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "ts": _stamp(),
            "from": sender,
            "to": to,
            "kind": kind,
            "body": body,
            "reply_to": reply_to,
            # Which *process* said it, not which name. Everyone registers as
            # the same name, so a name cannot be retired and a session cannot
            # be told apart from its own predecessor without this.
            "session": session or own_session_id(),
        }
        with open(self.board_path(board), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
        return entry

    def departed_file(self) -> Path:
        return self.root / "departed.json"

    def departed(self) -> list[str]:
        """Session ids that have explicitly left. Their entries read as gone."""
        try:
            data = json.loads(self.departed_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [str(x) for x in data] if isinstance(data, list) else []

    def depart(self, session: str, *, name: str | None = None) -> dict[str, Any]:
        """Withdraw a session: drop its registration and bury its entries.

        The board is append-only, so a departure cannot delete a line -- and it
        should not, because the log is also the record of what happened. What a
        departure *can* do is make those entries stop counting as current: the
        session id joins the departed set, and every later read skips what carries
        it. That is the honest form of "clear my messages" in a channel made of a
        file nobody may rewrite.
        """
        gone = self.departed()
        if session and session not in gone:
            gone.append(session)
        self.departed_file().write_text(
            json.dumps(gone, indent=2, ensure_ascii=False), encoding="utf-8")
        # Remove exactly this session, never "everyone who shares my name".
        # Names are not identities here: two live sessions are both called
        # "autoforge", so a name-based removal lets whichever one leaves first
        # unregister the one still working -- the fossil bug wearing a smile.
        removed: dict[str, Any] = {}
        reg = self.agents()
        for key, info in list(reg.items()):
            if session and info.get("session_id") == session:
                removed[key] = reg.pop(key)
            elif not session and name is not None and key == name:
                removed[key] = reg.pop(key)
        self.agents_file().write_text(
            json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"session": session, "unregistered": sorted(removed)}

    # -- reading ---------------------------------------------------------
    def all_lines(self, board: str) -> list[str]:
        path = self.board_path(board)
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        return [ln for ln in text.splitlines() if ln.strip()]

    def cursor(self, board: str, reader: str) -> int:
        """How many lines this reader has already seen."""
        try:
            return int((self.cursors_dir / f"{board}.{reader}")
                       .read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    def read(self, board: str, reader: str, *, advance: bool = True,
             only_mine: bool = False, everything: bool = False,
             include_departed: bool = False, include_dead: bool = False) -> list[dict[str, Any]]:
        """Entries this reader has not seen, oldest first.

        `--mine` filters what is *shown*, but the cursor still moves past
        everything: an entry addressed to someone else is not one this reader
        will ever want later, and skipping it here is how a filtered read
        becomes a permanent unread backlog.

        What is shown is the *current* mail, not the whole log. An entry counts
        as no longer current when its author is no longer running, judged off
        the process table exactly as `live_agents` judges a registration -- the
        operator's complaint was being told what a session they killed hours
        ago had said, and `departed` cannot answer for a session that was
        killed, because a killed session never writes a departure. Pass
        `include_dead` (or `include_departed`, which means the same thing from
        the reader's side: "show me what is no longer current") to read the
        history instead.
        """
        lines = self.all_lines(board)
        departed = set(self.departed())
        gone_sessions: set[str] = set()
        gone_names: set[str] = set()
        if not (include_dead or include_departed):
            gone_sessions, gone_names = self.gone_authors()
        start = 0 if everything else min(self.cursor(board, reader), len(lines))
        out: list[dict[str, Any]] = []
        for raw in lines[start:]:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                # A torn or hand-edited line is reported, never skipped: a
                # channel that drops what it cannot parse loses exactly the
                # message that needed explaining.
                out.append({"broken": raw})
                continue
            if not include_departed and entry.get("session") in departed:
                # A departed session’s words are not current. Shown only when asked
                # for: "what did that dead session say" is a real question, and a
                # silent drop would answer it wrongly.
                continue
            if gone_sessions or gone_names:
                # Which *process* said it, by id when the entry carries one and
                # by name otherwise -- entries written before sessions were
                # stamped carry only the name, and on this host that was every
                # entry on the board, which is why no id-based rule could ever
                # retire one.
                said_by_id = str(entry.get("session") or "")
                said_by_name = str(entry.get("from") or entry.get("sender") or "")
                if (said_by_id and said_by_id in gone_sessions) or \
                        (said_by_name and said_by_name in gone_names):
                    continue
            if only_mine and entry.get("to") not in ("*", reader):
                continue
            out.append(entry)
        if advance:
            self.set_cursor(board, reader, len(lines))
        return out

    def set_cursor(self, board: str, reader: str, value: int) -> None:
        (self.cursors_dir / f"{board}.{reader}").write_text(str(value),
                                                            encoding="utf-8")

    def tail(self, board: str, n: int = 20) -> list[dict[str, Any]]:
        """The last `n` entries, cursor untouched — for catching up, not reading."""
        out: list[dict[str, Any]] = []
        for raw in self.all_lines(board)[-n:]:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append({"broken": raw})
        return out

    # -- presence --------------------------------------------------------
    def agents(self) -> dict[str, Any]:
        try:
            reg = json.loads(self.agents_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return reg if isinstance(reg, dict) else {}

    def agents_file(self) -> Path:
        return self.root / "agents.json"

    def register(self, name: str, *, session_id: str = "", cwd: str = "",
                 note: str = "", pid: int | None = None) -> dict[str, Any]:
        """Record that a session is alive, keyed by SESSION, not by name.

        Everyone on this machine registers as the same name ("autoforge"), so a
        name-keyed registry holds exactly one autoforge no matter how many are
        running: the second writer silently overwrites the first. That is how a
        live peer reads as absent -- the one question the registry exists to
        answer. The key is therefore the session id; the name is kept inside the
        record as a human label, which is all it ever was.

        A call without a session id falls back to the name as the key, so the
        single-session case still reads like a roster.

        The pid recorded is THIS process. Registration is meant to be called by
        the session itself (cli.py calls it at startup); a wrapper registering on
        behalf of someone else passes `pid` explicitly. Guessing the parent here
        would record the short-lived shell that ran the command, which is the
        fossil bug this whole section exists to fix.
        """
        reg = self.agents()
        # The key is the session id whenever one was given -- never the name.
        # Sharing a name across as many rows as there are sessions is the whole
        # point: matching by name here would fold two live sessions onto one row
        # and the second would erase the first, which is the fossil bug again.
        if session_id:
            key = session_id
        else:
            # No id, so the name is all we have. Reuse a row only if it is
            # unambiguous: two same-named rows means we cannot tell which one
            # this call belongs to, and guessing would erase a live session.
            same = [k for k, v in reg.items()
                    if v.get("name") == name or k == name]
            key = same[0] if len(same) == 1 else name
        prev = reg.get(key) or {}
        who_pid = int(pid) if pid is not None else os.getpid()
        reg[key] = {
            "name": name,
            "session_id": session_id or prev.get("session_id", ""),
            "pid": who_pid,
            "cwd": cwd or prev.get("cwd") or os.getcwd(),
            "note": note or prev.get("note", ""),
            "last_seen": _stamp(),
        }
        self.agents_file().write_text(
            json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        return reg[key]

    def live_agents(self) -> dict[str, Any]:
        """Registered sessions whose process is still running, checked now.

        `agents()` answers "who ever registered"; this answers "who is there",
        and they are different questions with different costs -- a fossil that
        reads as a peer is worse than no peer at all, because it invites a
        session to wait for an answer that cannot come.
        """
        out: dict[str, Any] = {}
        for name, info in self.agents().items():
            if pid_alive(info.get("pid", 0)):
                out[name] = info
        return out

    def gone_authors(self) -> tuple[set[str], set[str]]:
        """(session ids, names) whose process is gone, read off the process table.

        The failure this judges: a session that was killed never gets to write a
        departure, so `departed` cannot speak for it and its words keep reading
        as current mail -- measured on this host, three registrations whose pids
        had been gone for hours and a board whose entries were all written
        before sessions were stamped, so no id-based rule could retire one.

        A name is only ever called gone when *every* registration answering to
        it is gone. Several sessions are called "autoforge" here as a matter of
        course, so a name-based verdict has to be the conservative one: retiring
        a live peer's words would hide exactly the message the channel exists to
        carry, and a rule that hides mail is worse than one that shows a fossil.
        """
        gone_sessions: set[str] = set()
        gone_names: set[str] = set()
        live_sessions: set[str] = set()
        live_names: set[str] = set()
        for key, info in self.agents().items():
            session = str(info.get("session_id") or "")
            name = str(info.get("name") or key)
            if pid_alive(info.get("pid", 0)):
                live_names.add(name)
                if session:
                    live_sessions.add(session)
            else:
                gone_names.add(name)
                if session:
                    gone_sessions.add(session)
        return gone_sessions - live_sessions, gone_names - live_names

    def live_session_for(self, name: str) -> str:
        """The session id of the live registration answering to `name`, if any.

        A wrapper shell sends on behalf of its agent and has no session id of
        its own, so without this every `auto bus send` from a shell stamped a
        brand-new throwaway id -- which made "clear my messages when I close"
        unenforceable for exactly the entries a session writes through the CLI.
        """
        for info in self.live_agents().values():
            if str(info.get("name") or "") == name:
                return str(info.get("session_id") or "")
        return ""


# ---------------------------------------------------------------------------
def render(entry: dict[str, Any]) -> str:
    """One entry, laid out for a terminal rather than for a parser."""
    if "broken" in entry:
        return f"  [unparseable line] {entry['broken']}"
    to = entry.get("to", "*")
    return (f"  [{str(entry.get('ts', '?'))[:19].replace('T', ' ')}] "
            f"{entry.get('from', '?')} -> {to if to != '*' else 'all'} "
            f"({entry.get('kind', 'msg')})\n"
            f"      {entry.get('body', '')}")




def _addressed_to(entries: list[dict[str, Any]], name: str,
                  session: str) -> list[dict[str, Any]]:
    """Entries meant for this session or for everyone, and not written by it."""
    return [e for e in entries
            if e.get("to") in ("*", name, session)
            and e.get("from") != name
            and e.get("session") != session]


def startup_check(bus: "Bus", session: str, *, name: str = "autoforge") -> str:
    """Register this session and report who else is live, in one call.

    This is the piece that makes the bus proactive instead of remembered. A
    session that only reads the board when it happens to think of it will
    coordinate exactly as often as it remembers to -- which, measured, is never.
    Running this at startup means the answer to "is anyone else here" arrives
    unprompted, and is recomputed from the process table rather than from a
    registration someone wrote before they died.

    What is *reported* is only what a live peer needs from this session. A board
    keeps every line anyone ever wrote and a killed session keeps its words on
    it, so "unread" on its own is a pile of history -- the operator's complaint
    was being told what a session they killed hours ago had said. So: asks and
    claims from a live peer are counted as needing an answer, notes are reported
    as notes, and the number of entries passed over because their author is gone
    is stated rather than silently dropped. "Ignored" and "nothing there" are
    different facts, and only the second one is safe to trust.

    Returns the line to show the operator: never raises, because a session that
    cannot reach the bus is still a usable session and must not fail to start.
    """
    try:
        bus.register(name, session_id=session, note="started")
        others = {k: v for k, v in bus.live_agents().items()
                  if v.get("session_id") != session}
        now = _addressed_to(bus.read("autoforge", session, advance=False),
                            name, session)
        whole_board = _addressed_to(
            bus.read("autoforge", session, advance=False, include_dead=True,
                     include_departed=True), name, session)
    except Exception as exc:                                   # noqa: BLE001
        return f"bus unavailable ({type(exc).__name__}: {exc}) -- continuing alone"
    parts = [f"registered as {name} ({session})"]
    if others:
        who = ", ".join(
            f"{v.get('name') or k}({v.get('session_id') or k})"
            for k, v in others.items())
        parts.append(f"{len(others)} other live session(s): {who}")
        parts.append("declare your lane before editing a shared file")
    else:
        parts.append("no other live session on this bus")
    needs = [e for e in now if e.get("kind") in ("ask", "claim")]
    if needs:
        parts.append(f"{len(needs)} message(s) needing you on board `autoforge`")
    notes = len(now) - len(needs)
    if notes:
        parts.append(f"{notes} note(s) from live peers")
    stale = len(whole_board) - len(now)
    if stale:
        parts.append(f"{stale} stale entr{'y' if stale == 1 else 'ies'} from "
                     f"sessions that are gone, ignored")
    return "; ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto bus",
        description="A mail channel between autoforge sessions.")
    p.add_argument("--dir", dest="bus_dir", default=None,
                   help="bus directory (defaults to $AUTOFORGE_BUS_DIR, else "
                        "$AUTOFORGE_HOME/bus)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="say that this session is alive")
    r.add_argument("--as", dest="who", required=True)
    r.add_argument("--session-id", default="")
    r.add_argument("--cwd", default="")
    r.add_argument("--note", default="", help="what this session is holding")

    dp = sub.add_parser("depart", help="say that this session is leaving")
    dp.add_argument("--as", dest="who", default="")
    dp.add_argument("--session", default="", help="the session id to retire")

    s = sub.add_parser("send", help="append a message")
    s.add_argument("--as", dest="who", required=True)
    s.add_argument("--board", default="default")
    s.add_argument("--to", default="*", help="a reader's name, or * for all")
    s.add_argument("--kind", default="msg", choices=KINDS)
    s.add_argument("--reply-to", default=None)
    s.add_argument("--session", default="",
                   help="the session id to stamp; defaults to "
                        "$AUTOFORGE_SESSION_ID, else the live session "
                        "registered under --as")
    s.add_argument("body", nargs="*", help="the message; omit to read stdin")

    rd = sub.add_parser("read", help="print what is unread, then advance")
    rd.add_argument("--as", dest="who", required=True)
    rd.add_argument("--board", default="default")
    rd.add_argument("--peek", action="store_true", help="do not advance")
    rd.add_argument("--all", action="store_true", help="ignore the cursor")
    rd.add_argument("--mine", action="store_true",
                    help="show only entries addressed to me")
    rd.add_argument("--json", action="store_true")
    rd.add_argument("--departed", action="store_true",
                    help="show what is no longer current: sessions that left, "
                         "or whose process is gone")
    rd.add_argument("--dead", action="store_true",
                    help="the same, named for the case that catches people: "
                         "words from a session that was killed")

    t = sub.add_parser("tail", help="the last N entries, cursor untouched")
    t.add_argument("--board", default="default")
    t.add_argument("-n", type=int, default=20)
    t.add_argument("--json", action="store_true")

    st = sub.add_parser("startup", help="register and report who else is live")
    st.add_argument("--as", dest="who", default="autoforge")
    st.add_argument("--session", default="")

    sub.add_parser("boards", help="list boards and their depth")
    sub.add_parser("mounts", help="every directory this protocol might live in, "
                                  "and how much is on each")
    wh = sub.add_parser("who", help="list sessions that are still running")
    wh.add_argument("--all", action="store_true",
                    help="include registrations whose process is gone")
    return p


def cmd_bus(argv: list[str] | None = None) -> int:
    """Entry point for `auto bus`, and the function the CLI table routes to."""
    args = build_parser().parse_args(argv)
    bus = Bus(args.bus_dir)

    if args.cmd == "register":
        info = bus.register(args.who, session_id=args.session_id,
                            cwd=args.cwd, note=args.note)
        print(f"registered {args.who} (pid {info['pid']}) -> {bus.root}")
        return 0

    if args.cmd == "send":
        body = " ".join(args.body) if args.body else sys.stdin.read().strip()
        if not body:
            print("nothing sent: the message was empty", file=sys.stderr)
            return 2
        # Stamp a session the sender can actually retire. A shell that sends on
        # behalf of a live session has no id of its own, and the fallback
        # (`own_session_id`) mints a fresh one per invocation -- so those
        # entries could never be cleared by their session when it closed, which
        # is the one thing the operator asked this channel to do.
        session = (args.session or os.environ.get("AUTOFORGE_SESSION_ID", "")
                   or bus.live_session_for(args.who))
        e = bus.send(sender=args.who, board=args.board, body=body,
                     to=args.to, kind=args.kind, reply_to=args.reply_to,
                     session=session or None)
        print(f"sent {e['id']} to {args.board} "
              f"(to={e['to']}, kind={e['kind']})")
        # Say the *root* as well as the board. Exit code 0 and "sent <id>" were
        # true while the message sat on a mount nobody read, and a caller cannot
        # check a delivery it was never told the address of.
        print(f"  mount: {bus.root}")
        split = divergent_mounts(bus.root)
        if split:
            hottest = split[0]
            print(f"WARNING: {len(split)} other mount(s) hold entries and are "
                  f"busier than the one just written ({hottest['entries']} vs "
                  f"{[r['entries'] for r in mount_report() if r['path'] == str(bus.root)] or [0]}). "
                  f"Busiest: {hottest['path']}", file=sys.stderr)
            print("WARNING: another mount is busier; run `auto bus mounts`",
                  file=sys.stderr)
        return 0

    if args.cmd == "read":
        entries = bus.read(args.board, args.who, advance=not args.peek,
                           only_mine=args.mine, everything=args.all,
                           include_departed=args.departed,
                           include_dead=args.dead)
        if not entries:
            print(f"(nothing unread on {args.board} for {args.who})")
            return 0
        for e in entries:
            # `--json` is for the caller that will parse it; the default is for
            # the one that will read it.
            print(json.dumps(e, ensure_ascii=False) if args.json else render(e))
        if args.peek:
            print(f"\n  -- {len(entries)} entr(ies); cursor not advanced")
        return 0

    if args.cmd == "tail":
        for e in bus.tail(args.board, args.n):
            print(json.dumps(e, ensure_ascii=False) if args.json else render(e))
        return 0

    if args.cmd == "boards":
        found = bus.boards()
        for b in found:
            print(f"  {b}  ({len(bus.all_lines(b))} entries)")
        if not found:
            print(f"  (no boards yet under {bus.root})")
        return 0

    if args.cmd == "mounts":
        rows = mount_report()
        chosen = os.path.normcase(os.path.abspath(str(bus.root)))
        for r in rows:
            mark = "*" if os.path.normcase(os.path.abspath(r["path"])) == chosen else " "
            state = "exists" if r["exists"] else "absent"
            print(f" {mark} {r['entries']:>5} entries  {r['newest'] or '-':<25} "
                  f"{state:<6} {r['path']:<52} {r['source']}")
        print(f"\n  * = this invocation is writing there ({bus.root})")
        split = divergent_mounts(bus.root)
        if split:
            print(f"\nSPLIT: {len(split)} other mount(s) hold entries, "
                  f"busiest has {split[0]['entries']}.")
            print("  Two mounts of the same protocol means two boards, two "
                  "registries, and two sessions called the same name.")
        return 0

    if args.cmd == "depart":
        session = args.session or os.environ.get("AUTOFORGE_SESSION_ID", "")
        if not session and args.who:
            session = str(bus.agents().get(args.who, {}).get("session_id", ""))
        info = bus.depart(session, name=args.who or None)
        who = info["session"] or "(no session id)"
        print(f"departed {who} unregistered={info['unregistered'] or []}")
        return 0

    if args.cmd == "startup":
        session = args.session or own_session_id()
        line = startup_check(bus, session, name=args.who)
        print(line)
        return 0

    if args.cmd == "who":
        reg = bus.live_agents() if not args.all else bus.agents()
        for key, info in reg.items():
            name = info.get("name") or key
            print(f"  {name}  pid={info.get('pid', '?')}  "
                  f"{info.get('session_id') or '-'}  {info.get('note') or ''}"
                  f"  {info.get('last_seen', '')}")
        if not reg:
            hint = " (--all shows registrations whose process is gone)" if not args.all else ""
            print(f"  (no session is running under {bus.root}){hint}")
        return 0

    return 2


def main(argv: list[str] | None = None) -> int:
    """Kept so `python -m autoforge.bus` works where the CLI is not installed."""
    return cmd_bus(argv)


if __name__ == "__main__":
    raise SystemExit(main())
