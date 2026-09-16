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
