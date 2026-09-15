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

#: The kinds a message may claim to be. A closed set, so a reader can filter on
#: it: an `ask` wants an `answer`, and a `claim` on a file is what stops two
#: sessions editing it. `msg` is the untyped default and stays the common case.
KINDS = ("msg", "ask", "answer", "claim", "done", "note")


def _base_dir() -> str:
    """The per-user data root, however this platform spells it."""
    return (os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_DATA_HOME")
            or os.path.expanduser("~"))


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


#: Other mounts of the same protocol, best-known first. An existing board beats
#: a fresh empty one: two agents that each default to their own empty directory
#: look exactly like two agents with nothing to say -- which is the failure this
#: module exists to prevent, so the default must not create it.
OTHER_MOUNTS = ("hermes/agent-bus",)


def default_bus_dir() -> Path:
    """`$AUTOFORGE_BUS_DIR` wins, else the first mount that exists, else home/bus."""
    env = os.environ.get("AUTOFORGE_BUS_DIR")
    if env:
        return Path(env)
    for rel in OTHER_MOUNTS:
        candidate = Path(os.path.join(_base_dir(), *rel.split("/")))
        if candidate.is_dir():
            return candidate
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
             kind: str = "msg", reply_to: str | None = None) -> dict[str, Any]:
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
        }
        with open(self.board_path(board), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
        return entry

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
             only_mine: bool = False, everything: bool = False
             ) -> list[dict[str, Any]]:
        """Entries this reader has not seen, oldest first.

        `--mine` filters what is *shown*, but the cursor still moves past
        everything: an entry addressed to someone else is not one this reader
        will ever want later, and skipping it here is how a filtered read
        becomes a permanent unread backlog.
        """
        lines = self.all_lines(board)
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
                 note: str = "") -> dict[str, Any]:
        """Record that `name` is alive, keeping anything learned earlier."""
        reg = self.agents()
        prev = reg.get(name, {})
        reg[name] = {
            "session_id": session_id or prev.get("session_id", ""),
            "pid": os.getpid(),
            "cwd": cwd or prev.get("cwd") or os.getcwd(),
            "note": note or prev.get("note", ""),
            "last_seen": _stamp(),
        }
        self.agents_file().write_text(
            json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        return reg[name]


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

    s = sub.add_parser("send", help="append a message")
    s.add_argument("--as", dest="who", required=True)
    s.add_argument("--board", default="default")
    s.add_argument("--to", default="*", help="a reader's name, or * for all")
    s.add_argument("--kind", default="msg", choices=KINDS)
    s.add_argument("--reply-to", default=None)
    s.add_argument("body", nargs="*", help="the message; omit to read stdin")

    rd = sub.add_parser("read", help="print what is unread, then advance")
    rd.add_argument("--as", dest="who", required=True)
    rd.add_argument("--board", default="default")
    rd.add_argument("--peek", action="store_true", help="do not advance")
    rd.add_argument("--all", action="store_true", help="ignore the cursor")
    rd.add_argument("--mine", action="store_true",
                    help="show only entries addressed to me")
    rd.add_argument("--json", action="store_true")

    t = sub.add_parser("tail", help="the last N entries, cursor untouched")
    t.add_argument("--board", default="default")
    t.add_argument("-n", type=int, default=20)
    t.add_argument("--json", action="store_true")

    sub.add_parser("boards", help="list boards and their depth")
    sub.add_parser("who", help="list registered sessions")
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
        e = bus.send(sender=args.who, board=args.board, body=body,
                     to=args.to, kind=args.kind, reply_to=args.reply_to)
        print(f"sent {e['id']} to {args.board} "
              f"(to={e['to']}, kind={e['kind']})")
        return 0

    if args.cmd == "read":
        entries = bus.read(args.board, args.who, advance=not args.peek,
                           only_mine=args.mine, everything=args.all)
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

    if args.cmd == "who":
        reg = bus.agents()
        for name, info in reg.items():
            print(f"  {name}  pid={info.get('pid', '?')}  "
                  f"{info.get('session_id') or '-'}  {info.get('note') or ''}"
                  f"  {info.get('last_seen', '')}")
        if not reg:
            print(f"  (nobody registered under {bus.root})")
        return 0

    return 2


def main(argv: list[str] | None = None) -> int:
    """Kept so `python -m autoforge.bus` works where the CLI is not installed."""
    return cmd_bus(argv)


if __name__ == "__main__":
    raise SystemExit(main())
