"""The ledger under concurrent writers.

This exists because of a measured loss, not a theory. Before the lock in
`ToolStore`, the probe below landed 4 of 16 rows and raised 12 exceptions --
and the rows that vanished were mostly *refusals* (`gate_deny`,
`declaration_deny`), the one kind of row an audit log exists to keep.

Two levels are tested, because they fail for different reasons:

  * threads in ONE process through ONE connection -- a sqlite3.Connection is
    not re-entrant, and while one thread iterates the head-hash SELECT another
    thread's statement makes that iteration fail;
  * several PROCESSES on one file -- different code path, the write lock, and
    the reason `ensure_schema` has to tolerate a lost race for a column.

The chain is verified after each, because "16 rows landed" is worth nothing if
the hash links between them are wrong: a fast wrong ledger is a worse outcome
than a slow correct one.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge.store import ToolStore  # noqa: E402


def _store(tmp_path, name="chain.db"):
    return ToolStore(str(tmp_path / name))


def test_threads_through_one_connection_land_every_row(tmp_path):
    """16 threads, one store, one connection. Every event must survive."""
    s = _store(tmp_path)
    n = 16
    landed, errors = [], []
    gate = threading.Barrier(n)

    def writer(i):
        gate.wait()  # make them collide on purpose rather than in sequence
        try:
            s.log_event("gate_deny", {"tool": "t%d" % i, "reason": "probe"})
            landed.append(i)
        except Exception as e:  # noqa: BLE001 -- the failure is the assertion
            errors.append("%s: %s" % (type(e).__name__, e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], "concurrent writes raised: %s" % errors[:5]
    assert len(landed) == n

    conn = sqlite3.connect(str(tmp_path / "chain.db"))
    try:
        rows = conn.execute(
            "SELECT COUNT(*) FROM forge_events WHERE kind='gate_deny'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert rows == n, "landed %d rows for %d writes" % (rows, n)

    v = s.verify_chain()
    assert v.get("breaks") == [], v


def test_the_chain_stays_unbroken_under_contention(tmp_path):
    """A row count is not enough: the links between the rows must hold.

    Losing the writer lock around `_head_hash` would let two threads read the
    same predecessor and both point at it -- 16 rows that form a fork instead
    of a chain. That is strictly worse than 4 rows, because it looks correct.
    """
    s = _store(tmp_path, "fork.db")
    gate = threading.Barrier(12)

    def writer(i):
        gate.wait()
        s.log_event("call", {"i": i})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    v = s.verify_chain()
    assert v.get("breaks") == [], v
    assert v.get("gaps") == [], v


def test_two_processes_on_one_file(tmp_path):
    """Cross-process: the case a thread lock cannot cover.

    Two interpreters write the same file at once. Each must succeed (the
    busy timeout is what buys that) and the resulting chain must verify --
    which it cannot if either process read a stale head.
    """
    db = str(tmp_path / "shared.db")
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from autoforge.store import ToolStore
        s = ToolStore(%r)
        for i in range(25):
            s.log_event("run", {"pid": __import__("os").getpid(), "i": i})
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code % (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=180) for p in procs]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, (out[-400:], err[-400:])

    s = ToolStore(db)
    v = s.verify_chain()
    assert v.get("breaks") == [], v

    conn = sqlite3.connect(db)
    try:
        kinds = conn.execute(
            "SELECT COUNT(*) FROM forge_events WHERE kind='run'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert kinds == 50, "cross-process writes lost rows: %d of 50" % kinds
