"""Missions: what I owe, kept where losing sight of it is visible.

The failure this exists for is not "I cannot remember a fact". It is that a
session with many tasks loses the one sentence saying what all of them are
FOR, and then every new request silently replaces it -- the work still gets
done, and the mission it was for is never closed, or never named. A schedule
entry is something that happens at a time; a kept fact is something I know; a
mission is something I owe. Kept apart on purpose, so that "what am I for"
answers with a mission instead of with a fact about a directory.

Storage is the same sqlite file as the tool store, so a mission survives a
restart for the same reason the tools do, and the ledger and the mission list
cannot drift apart by living in two places.

Two rules the shape of this store enforces:

* A mission is never deleted by `close` -- it is closed, with a note. "I
  stopped tracking it" and "it is finished" must not look the same afterwards.
* A new request does not overwrite a mission. It either becomes a new mission
  or a line under an existing one (`parent`), which is why the report prints
  the tree and not a flat list.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_DB = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
    "autoforge", "autoforge.db",
)

OPEN = "open"
DONE = "done"
DROPPED = "dropped"
STATUSES = (OPEN, DONE, DROPPED)

# The report is re-sent on every turn, so it is bounded like the kept-facts
# block is: an unbounded mission list would price the agenda at the cost of the
# work, and the agent would have no way to see that happen.
REPORT_BUDGET_CHARS = 1100

_DDL = """
CREATE TABLE IF NOT EXISTS missions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    parent      INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'open',
    next_step   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    closed_at   REAL,
    close_note  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS mission_log (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    mission INTEGER NOT NULL,
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,
    note    TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS mission_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MissionError(Exception):
    """Asked for a mission id that is not there, or an illegal move."""


@dataclass
class Mission:
    id: int
    text: str
    parent: int = 0
    status: str = OPEN
    next_step: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    closed_at: float | None = None
    close_note: str = ""
    children: list["Mission"] = field(default_factory=list)

    def line(self, indent: int = 0) -> str:
        pad = "  " * indent
        mark = {"open": "[open]", "done": "[done]", "dropped": "[dropped]"}[self.status]
        out = f"{pad}M{self.id} {mark} {self.text}"
        if self.status == OPEN and self.next_step:
            out += f"\n{pad}    next: {self.next_step}"
        if self.status != OPEN and self.close_note:
            out += f"\n{pad}    closed: {self.close_note}"
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "parent": self.parent,
            "status": self.status, "next_step": self.next_step,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "closed_at": self.closed_at, "close_note": self.close_note,
        }


class MissionStore:
    """A small, durable list of what is owed. Append-minded: close, never erase.

    Deliberately not a task scheduler. `autoforge/schedule.py` answers "when
    should this run"; this answers "what is this all for", and the answers must
    stay queryable at different moments -- which is exactly the distinction a
    session with many tasks loses first.
    """

    def __init__(self, db_path: str | None = None,
                 conn: sqlite3.Connection | None = None) -> None:
        """Open the store, or adopt a connection the caller already holds.

        The `conn` path exists because of a failure this store caused the first
        time it ran: a ToolStore and a MissionStore on the same sqlite file are
        two connections, and the second one's schema setup hit
        `database is locked` while the first was alive. That is not a rare
        interleaving -- the agent holds both at once by design, since a mission
        and the ledger entry for the work it caused should live in one file so
        they cannot drift apart.

        So the default wiring is one file, one connection. `db_path` is still
        honoured for the standalone reader (`auto mission`), which has no tool
        store to borrow from.
        """
        self._owns_conn = conn is None
        self.db_path = str(db_path or _DEFAULT_DB)
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.abspath(self.db_path)
        if conn is not None:
            self._conn = conn
        else:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            # A second reader (the `auto mission` CLI, or a peer session on the
            # same store) must wait rather than fail: the failure it would
            # otherwise produce is "the mission list is unreadable", which is
            # the one thing this store exists to prevent.
            self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        """Release the connection -- but never one it borrowed.

        Closing a connection it did not open would take the tool store's
        connection down with it, and the agent would lose its ledger mid-run.
        """
        if self._owns_conn:
            self._conn.close()

    # -- writes -----------------------------------------------------------
    def open(self, text: str, parent: int = 0, next_step: str = "") -> Mission:
        text = " ".join((text or "").split())
        if not text:
            raise MissionError("a mission needs a sentence; empty is not a mission")
        if parent:
            row = self._conn.execute(
                "SELECT status FROM missions WHERE id=?", (parent,)).fetchone()
            if row is None:
                raise MissionError(f"no mission M{parent} to hang this under")
            if row["status"] != OPEN:
                raise MissionError(
                    f"M{parent} is {row['status']}; open a new mission instead of "
                    f"reviving a closed one, so the record of closing it stays true")
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO missions (text, parent, status, next_step, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)", (text, int(parent), OPEN, next_step, now, now))
        self._conn.commit()
        mid = int(cur.lastrowid)
        self._log(mid, "open", text)
        return self.get(mid)

    def note(self, mid: int, note: str = "", next_step: str | None = None) -> Mission:
        m = self.get(mid)
        now = time.time()
        sets, args = ["updated_at=?"], [now]
        if next_step is not None:
            sets.append("next_step=?")
            args.append(next_step)
        args.append(mid)
        self._conn.execute(f"UPDATE missions SET {', '.join(sets)} WHERE id=?", args)
        if note:
            self._log(mid, "note", note)
        self._conn.commit()
        return self.get(mid)

    def finish(self, mid: int, note: str = "", ok: bool = True) -> Mission:
        m = self.get(mid)
        if m.status != OPEN:
            raise MissionError(f"M{mid} is already {m.status}")
        kids = [c for c in self.open_children(mid)]
        if kids:
            raise MissionError(
                "M%d still has open sub-missions: %s. Close those first, or they "
                "disappear from the report while still being owed."
                % (mid, ", ".join(f"M{k.id}" for k in kids)))
        status = DONE if ok else DROPPED
        now = time.time()
        self._conn.execute(
            "UPDATE missions SET status=?, closed_at=?, close_note=?, updated_at=?"
            " WHERE id=?", (status, now, note or "", now, mid))
        self._log(mid, "close", f"{status}: {note}" if note else status)
        self._conn.commit()
        if self.focused() == mid:
            self.set_focus(0)
        return self.get(mid)

    def drop(self, mid: int, note: str) -> Mission:
        return self.finish(mid, note=note, ok=False)

    def set_focus(self, mid: int) -> str:
        if mid:
            m = self.get(mid)
            if m.status != OPEN:
                raise MissionError(f"M{mid} is {m.status}, so it cannot be the focus")
        self._conn.execute(
            "INSERT INTO mission_state (key, value) VALUES ('focus', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(int(mid)),))
        self._conn.commit()
        return f"focus is now M{mid}" if mid else "focus cleared"

    def focused(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM mission_state WHERE key='focus'").fetchone()
        return int(row["value"]) if row else 0

    # -- reads ------------------------------------------------------------
    def get(self, mid: int) -> Mission:
        row = self._conn.execute("SELECT * FROM missions WHERE id=?", (mid,)).fetchone()
        if row is None:
            raise MissionError(f"no mission M{mid}")
        return self._row(row)

    def open_children(self, mid: int) -> list[Mission]:
        rows = self._conn.execute(
            "SELECT * FROM missions WHERE parent=? AND status='open' ORDER BY id",
            (mid,)).fetchall()
        return [self._row(r) for r in rows]

    def all(self, status: str | None = None) -> list[Mission]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM missions WHERE status=? ORDER BY id", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM missions ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def tree(self, status: str | None = OPEN) -> list[Mission]:
        """Roots with their open children attached -- the report's real shape."""
        rows = self.all(status)
        by_id = {m.id: m for m in rows}
        roots: list[Mission] = []
        for m in rows:
            if m.parent and m.parent in by_id:
                by_id[m.parent].children.append(m)
            else:
                roots.append(m)
        return roots

    def history(self, mid: int, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, at, kind, note FROM mission_log WHERE mission=?"
            " ORDER BY seq DESC LIMIT ?", (mid, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- the block that rides in the prompt -------------------------------
    def report(self, budget: int = REPORT_BUDGET_CHARS) -> str:
        roots = self.tree(OPEN)
        focused = self.focused()
        head = ("MISSION — what I owe. Read this before acting, and update it when "
                "the answer changes:")
        closed = len(self.all(DONE)) + len(self.all(DROPPED))
        if not roots:
            # The closed count stays in the empty case on purpose. "Nothing is
            # owed" and "this store was never written to" are different facts,
            # and the first version of this printed the same sentence for both
            # -- the same conflation, in miniature, as the bug being fixed.
            return (head
                    + "\n    (nothing open — if the operator asked for "
                      "something, that request IS a mission: record it with "
                      "mission_open before working on it)"
                    + (f"; {closed} closed and on the record" if closed else ""))
        lines: list[str] = []
        used = 0
        hidden = 0
        for m in roots:
            chunk = m.line(0) + "".join("\n" + c.line(1) for c in m.children)
            marker = "  <== focus" if m.id == focused else ""
            chunk += marker
            if lines and used + len(chunk) > budget:
                hidden += 1
                continue
            lines.append(chunk)
            used += len(chunk)
        if hidden:
            lines.append(f"    (+{hidden} more open — mission_list reads them)")
        tail = (f"    {len(roots)} open root(s), {closed} closed. "
                f"mission_note updates, mission_close ends one. A new request is a "
                f"line under this, not a replacement for it.")
        return "\n".join([head, *lines, tail])

    # -- internals --------------------------------------------------------
    def _log(self, mid: int, kind: str, note: str = "") -> None:
        self._conn.execute(
            "INSERT INTO mission_log (mission, at, kind, note) VALUES (?,?,?,?)",
            (mid, time.time(), kind, note))

    @staticmethod
    def _row(r: sqlite3.Row) -> Mission:
        return Mission(
            id=int(r["id"]), text=r["text"], parent=int(r["parent"]),
            status=r["status"], next_step=r["next_step"],
            created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
            closed_at=(float(r["closed_at"]) if r["closed_at"] is not None else None),
            close_note=r["close_note"],
        )


__all__ = ["Mission", "MissionStore", "MissionError", "OPEN", "DONE", "DROPPED",
           "REPORT_BUDGET_CHARS"]
