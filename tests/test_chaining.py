"""The chained event log: what it catches, and what it only reports.

Each test tampers with a database and asks the log one question: does this
show up as a break? The two negative cases matter as much as the positive ones
-- a verifier that flags everything is one nobody runs, and the "gaps are not
breaks" distinction is the difference between this tool surviving a rolling
restart and being deleted as noise.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from autoforge import chaining
from autoforge.store import ToolStore
from autoforge.tools.spec import ToolSpec, ToolState


def _legacy_db(tmp_path, legacy_rows: int = 3):
    """A store with rows written the way the old code wrote them: no hashes."""
    db = str(tmp_path / "chain.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE forge_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " timestamp REAL NOT NULL, kind TEXT NOT NULL,"
        " payload TEXT NOT NULL DEFAULT '{}')")
    for i in range(legacy_rows):
        conn.execute("INSERT INTO forge_events (timestamp, kind, payload)"
                     " VALUES (?,?,?)", (100.0 + i, "legacy", json.dumps({"i": i})))
    conn.commit()
    chaining.ensure_schema(conn)
    chaining.anchor(conn, note="baseline")
    chaining.append_event(conn, "genuine", {"x": 1})
    chaining.append_event(conn, "genuine", {"x": 2})
    return conn


class TestChainDetectsRewriting:
    def test_a_clean_chain_verifies(self, tmp_path):
        conn = _legacy_db(tmp_path)
        v = chaining.verify_chain(conn)
        assert v["ok"] is True
        assert v["chained_rows"] == 2
        assert v["breaks"] == []
        assert v["gaps"] == []

    def test_rewriting_a_chained_payload_is_caught(self, tmp_path):
        conn = _legacy_db(tmp_path)
        conn.execute("UPDATE forge_events SET payload=? WHERE payload_hash != ''",
                     (json.dumps({"x": 99}),))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        assert any(f["kind"] == "payload_hash_mismatch" for f in v["breaks"])

    def test_a_link_that_points_nowhere_is_caught(self, tmp_path):
        conn = _legacy_db(tmp_path)
        conn.execute("UPDATE forge_events SET prev_hash=? WHERE id=("
                     " SELECT MAX(id) FROM forge_events)", ("0" * 64,))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        assert any(f["kind"] == "broken_link" for f in v["breaks"])

    def test_editing_a_legacy_row_breaks_the_anchor(self, tmp_path):
        """The rows written before chaining cannot be chained, only anchored."""
        conn = _legacy_db(tmp_path)
        conn.execute("UPDATE forge_events SET payload=? WHERE id=2",
                     (json.dumps({"i": 99}),))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        assert any(f["kind"] == "legacy_region_modified" for f in v["breaks"])
        assert v["legacy_rows_anchored"] == 3

    def test_deleting_a_chained_row_in_the_middle_is_caught(self, tmp_path):
        conn = _legacy_db(tmp_path)
        rows = [r[0] for r in conn.execute(
            "SELECT id FROM forge_events WHERE payload_hash != '' ORDER BY id")]
        conn.execute("DELETE FROM forge_events WHERE id=?", (rows[0],))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        # Cutting the first chained row leaves the next one claiming a
        # predecessor that is not there, which reads as a bad chain start.
        assert any(f["kind"] in ("broken_link", "orphan_chain_start")
                   for f in v["breaks"])

    def test_truncating_the_tail_is_caught_by_an_anchor(self, tmp_path):
        """Walking forward cannot see a cut tail -- the new last row looks valid.

        An anchor is what makes it visible: it froze a head, and a frozen head
        that is gone means rows after it were removed.
        """
        conn = _legacy_db(tmp_path)
        head = chaining.verify_chain(conn)["head_hash"]
        chaining.anchor(conn, note="periodic")
        conn.execute("DELETE FROM forge_events WHERE payload_hash=?", (head,))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        assert any(f["kind"] == "anchored_head_missing" for f in v["breaks"])

    def test_a_truncation_with_no_anchor_is_the_known_blind_spot(self, tmp_path):
        """Without an anchor, cutting the tail verifies clean. Say so in a test.

        This is not a bug to fix here -- it is why the closed-loop caller takes
        an anchor every time it finishes a run, and why the store does not
        pretend a bare chain is enough.
        """
        conn = _legacy_db(tmp_path)
        rows = [r[0] for r in conn.execute(
            "SELECT id FROM forge_events WHERE payload_hash != '' ORDER BY id")]
        conn.execute("DELETE FROM forge_events WHERE id=?", (rows[-1],))
        conn.commit()
        assert chaining.verify_chain(conn)["ok"] is True


class TestGapsAreNotBreaks:
    """A row from a process still running the old code is the normal case here.

    Conflating it with a rewrite is what makes a verifier noisy enough to be
    ignored, and this store is shared with sessions that are mid-restart.
    """

    def test_an_unchained_row_after_chaining_began_is_a_gap(self, tmp_path):
        conn = _legacy_db(tmp_path)
        conn.execute("INSERT INTO forge_events (timestamp, kind, payload)"
                     " VALUES (?,?,?)", (200.0, "old_code", "{}"))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is True
        assert v["breaks"] == []
        assert len(v["gaps"]) == 1
        assert "not every row is chained" not in (v["gaps"][0]["detail"] or "")

    def test_the_ok_means_nothing_was_rewritten_not_that_all_is_chained(self, tmp_path):
        conn = _legacy_db(tmp_path)
        conn.execute("INSERT INTO forge_events (timestamp, kind, payload)"
                     " VALUES (?,?,?)", (200.0, "old_code", "{}"))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is True
        assert v["unchained_rows"] == 4          # 3 legacy + 1 gap
        assert v["unchained_after_chaining_began"] == 1

    def test_the_first_chained_row_must_start_somewhere_real(self, tmp_path):
        conn = _legacy_db(tmp_path)
        first = conn.execute("SELECT MIN(id) FROM forge_events WHERE payload_hash != ''"
                             ).fetchone()[0]
        conn.execute("UPDATE forge_events SET prev_hash=? WHERE id=?", ("a" * 64, first))
        conn.commit()
        v = chaining.verify_chain(conn)
        assert v["ok"] is False
        assert any(f["kind"] == "orphan_chain_start" for f in v["breaks"])


class TestStoreWiring:
    def test_log_event_writes_a_chained_row(self, tmp_path):
        store = ToolStore(str(tmp_path / "s.db"))
        store.log_event("run", {"task": "x"})
        row = store._conn.execute(
            "SELECT prev_hash, payload_hash, writer_id FROM forge_events"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert row["payload_hash"]
        assert row["prev_hash"] == chaining.GENESIS
        assert row["writer_id"].startswith("sessionless@pid")
        assert store.verify_chain()["ok"] is True
        store.close()

    def test_the_chain_continues_across_reopen(self, tmp_path):
        db = str(tmp_path / "s.db")
        first = ToolStore(db)
        first.log_event("run", {"n": 1})
        head = first.verify_chain()["head_hash"]
        first.close()
        second = ToolStore(db)
        second.log_event("run", {"n": 2})
        row = second._conn.execute(
            "SELECT prev_hash FROM forge_events ORDER BY id DESC LIMIT 1").fetchone()
        assert row["prev_hash"] == head
        assert second.verify_chain()["ok"] is True
        second.close()

    def test_a_builtin_is_not_saved_but_a_real_tool_is_versioned(self, tmp_path):
        store = ToolStore(str(tmp_path / "s.db"))
        spec = ToolSpec(name="t", description="d", parameters={}, fn=lambda: 1,
                        code="def f(): return 1", source="human")
        spec.state = ToolState.ACTIVE
        store.save_tool(spec)
        assert [v["version"] for v in store.versions_of("t")] == [1]
        store.close()

class TestTheChainSurvivesContention:
    """The four ways this ledger can lose an event without saying so.

    Every test here is a regression test for something measured on this host,
    not something imagined. The theme is identical in all four: the log looked
    fine afterwards -- `verify_chain` said ok -- while events were simply gone,
    and the events that go missing are disproportionately the refusals, which
    are the only reason the log exists.
    """

    def test_sixteen_threads_on_one_connection_land_sixteen_rows(self, tmp_path):
        """sqlite3.Connection is not re-entrant, and its cursors are shared.

        Measured before the fix: 16 threads through one connection landed 5
        rows, raised 11 exceptions (DatabaseError "another row available",
        IndexError from a cursor consumed by the other thread), and
        `verify_chain` still returned ok.
        """
        import threading

        db = str(tmp_path / "threads.db")
        conn = sqlite3.connect(db, check_same_thread=False, timeout=15)
        conn.execute("CREATE TABLE forge_events (id INTEGER PRIMARY KEY"
                     " AUTOINCREMENT, timestamp REAL, kind TEXT, payload TEXT)")
        landed, errors = [], []

        def write(i):
            try:
                chaining.append_event(conn, "gate_deny", {"i": i})
                landed.append(i)
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], errors
        assert len(landed) == 16
        assert conn.execute("SELECT COUNT(*) FROM forge_events").fetchone()[0] == 16
        assert chaining.verify_chain(conn)["ok"] is True

    def test_a_plain_connection_can_verify_and_anchor(self, tmp_path):
        """A reader with no row_factory must still be able to ask.

        Measured: `sqlite3.connect()` plus `verify_chain` raised
        `TypeError: tuple indices must be integers or slices, not str`,
        because the module indexed rows by name and only ToolStore set the
        row_factory that makes that work. The one question this function
        exists to answer was answered with a traceback.
        """
        db = str(tmp_path / "plain.db")
        conn = sqlite3.connect(db)                 # deliberately no row_factory
        conn.execute("CREATE TABLE forge_events (id INTEGER PRIMARY KEY"
                     " AUTOINCREMENT, timestamp REAL, kind TEXT, payload TEXT)")
        chaining.append_event(conn, "genuine", {"x": 1})

        assert not hasattr(conn, "row_factory") or conn.row_factory is None
        verdict = chaining.verify_chain(conn)
        assert verdict["ok"] is True
        assert verdict["chained_rows"] == 1
        assert chaining.anchor(conn, note="plain")["rows"] == 0
        assert chaining.verify_chain(conn)["ok"] is True
        assert chaining.lineage_of(conn, "nothing")["nodes"] == ["nothing"]

    def test_three_processes_land_every_row(self, tmp_path):
        """The default busy timeout is zero, and this ledger is shared.

        Three processes -- two agent sessions and the market server is the real
        case on this host -- each appending through a plain connection. Measured
        before the fix: 91 of 120 rows landed and 29 were raised as
        OperationalError("database is locked"), then discarded by the caller.
        The fix is a wait-on-lock set inside this module, because the callers
        are not one caller and the sqlite default is "give up immediately".
        """
        import subprocess
        import sys
        import textwrap

        db = str(tmp_path / "multi.db")
        worker = textwrap.dedent(
            """
            import sqlite3, sys
            from autoforge import chaining
            conn = sqlite3.connect(sys.argv[1])        # no timeout set by hand
            conn.execute("CREATE TABLE IF NOT EXISTS forge_events (id INTEGER"
                         " PRIMARY KEY AUTOINCREMENT, timestamp REAL, kind TEXT,"
                         " payload TEXT)")
            ok = 0
            for i in range(20):
                chaining.append_event(conn, "gate_deny", {"w": sys.argv[2], "i": i})
                ok += 1
            print(ok)
            """)
        script = tmp_path / "worker.py"
        script.write_text(worker, encoding="utf-8")

        procs = [subprocess.Popen([sys.executable, str(script), db, tag],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
                 for tag in ("A", "B", "C")]
        outs = []
        for p in procs:
            out, err = p.communicate(timeout=180)
            outs.append((out.strip(), err.strip()))

        landed = [int(o) for o, _ in outs if o.isdigit()]
        assert landed == [20, 20, 20], outs
        conn = sqlite3.connect(db)
        assert conn.execute("SELECT COUNT(*) FROM forge_events").fetchone()[0] == 60
        assert chaining.verify_chain(conn)["ok"] is True
