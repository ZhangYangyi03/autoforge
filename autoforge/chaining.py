"""Hash-chained event log over the sqlite ledger.



The append-only event log recorded *what happened*. It could not answer a

different question: was any of it rewritten afterwards? Nothing in a row said

so, and 4535 rows were already on disk before anything could. This module adds

the field that makes each row self-witnessing.



Three columns are added to forge_events -- prev_hash, payload_hash, writer_id --

and nothing else about the table changes. Old code that still writes

`INSERT INTO forge_events (timestamp, kind, payload)` keeps working: those rows

land unchained and are reported as such, never silently trusted.



The region that already existed cannot be retroactively chained: the writer of

row 2000 was not recording prev_hash, and no amount of hashing now recovers

what it should have been. So the past is *anchored*, not chained. An anchor

freezes a digest of every legacy row as it stands at anchor time; if any legacy

row is edited later, that digest stops matching and verify() says so. The

distinction between "chained" and "anchored" is the whole point and is never

glossed: anchored rows are detectable-after-the-fact, chained rows are

detectable-individually.



Deliberately not a general log store: no query language, no compaction, no

pluggable backends. append, verify, anchor, lineage_of -- four verbs. Anything

resembling SQL belongs in the sqlite layer underneath.

"""

from __future__ import annotations



import hashlib

import json

import os

import sqlite3

import time

from typing import Any



GENESIS = "0" * 64

_CHAIN_COLUMNS = ("prev_hash", "payload_hash", "writer_id")





def _j(obj: Any) -> str:

    return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)





def writer_id(session: str | None = None) -> str:

    """Identify the writer process, not the agent name.



    Two sessions on this host both call themselves "autoforge" and share one

    store; a name is not an identity here. session+pid is.

    """

    sid = session or os.environ.get("AUTOFORGE_SESSION") or "sessionless"

    return f"{sid}@pid{os.getpid()}"





def _digest(kind: str, payload_json: str, timestamp: float, prev_hash: str,

            writer: str) -> str:

    """The exact bytes a row commits to.



    Sorted keys and a fixed separator mean two processes that build the same

    event agree on its hash. Anything not in this string is not covered, which

    is why the writer id is in it: "who wrote it" is part of the claim.

    """

    material = "\x1f".join([

        format(float(timestamp), ".6f"),

        kind,

        payload_json,

        prev_hash,

        writer,

    ])

    return hashlib.sha256(material.encode("utf-8")).hexdigest()





def ensure_schema(conn: sqlite3.Connection) -> list[str]:

    """Add the chain columns if absent. Returns what it added.



    Additive only, on purpose: another session may be running older code

    against this same file right now, and a rewrite of forge_events would

    break it without warning.

    """

    added: list[str] = []

    have = {r[1] for r in conn.execute("PRAGMA table_info(forge_events)")}

    for col in _CHAIN_COLUMNS:

        if col not in have:

            conn.execute(f"ALTER TABLE forge_events ADD COLUMN {col} TEXT DEFAULT ''")

            added.append(col)

    conn.execute(

        "CREATE TABLE IF NOT EXISTS chain_anchors ("

        " id INTEGER PRIMARY KEY AUTOINCREMENT,"

        " created_at REAL NOT NULL,"

        " up_to_id INTEGER NOT NULL,"

        " rows INTEGER NOT NULL,"

        " baseline_digest TEXT NOT NULL,"

        " head_hash TEXT NOT NULL,"

        " writer TEXT NOT NULL DEFAULT '',"

        " note TEXT NOT NULL DEFAULT ''"

        ")"

    )

    conn.commit()

    return added





def _row_material(row: sqlite3.Row) -> str:

    return "\x1f".join([str(row["id"]), format(float(row["timestamp"]), ".6f"),

                        row["kind"], row["payload"]])





def _baseline_digest(conn: sqlite3.Connection, up_to_id: int) -> tuple[str, int]:

    """One digest over every legacy row, in id order.



    Not a per-row chain: it cannot say *which* legacy row moved, only that the

    set it was taken over no longer matches. That is the honest limit of

    anchoring, and it is still strictly more than the table had before, which

    was nothing.

    """

    h = hashlib.sha256()

    n = 0

    for row in conn.execute(

        "SELECT id, timestamp, kind, payload FROM forge_events "

        "WHERE id <= ? AND (payload_hash IS NULL OR payload_hash = '') "

        "ORDER BY id", (up_to_id,)

    ):

        h.update(_row_material(row).encode("utf-8"))

        n += 1

    return h.hexdigest(), n





def anchor(conn: sqlite3.Connection, note: str = "", writer: str | None = None) -> dict:

    """Freeze the current state as a baseline. Idempotent in effect, not in rows."""

    ensure_schema(conn)

    up_to = conn.execute("SELECT COALESCE(MAX(id), 0) FROM forge_events").fetchone()[0]

    digest, rows = _baseline_digest(conn, up_to)

    head = _head_hash(conn)

    conn.execute(

        "INSERT INTO chain_anchors (created_at, up_to_id, rows, baseline_digest,"

        " head_hash, writer, note) VALUES (?,?,?,?,?,?,?)",

        (time.time(), up_to, rows, digest, head, writer or writer_id(), note),

    )

    conn.commit()

    return {"up_to_id": up_to, "rows": rows, "baseline_digest": digest, "head_hash": head}





def _head_hash(conn: sqlite3.Connection) -> str:

    """The hash the next chained row must point at.



    Falls back through: last chained row -> newest anchor's head -> GENESIS.

    """

    row = conn.execute(

        "SELECT payload_hash FROM forge_events WHERE payload_hash IS NOT NULL"

        " AND payload_hash != '' ORDER BY id DESC LIMIT 1"

    ).fetchone()

    if row and row[0]:

        return row[0]

    row = conn.execute(

        "SELECT head_hash FROM chain_anchors ORDER BY id DESC LIMIT 1"

    ).fetchone()

    if row and row[0]:

        return row[0]

    return GENESIS





def append_event(conn: sqlite3.Connection, kind: str, payload: dict,

                 session: str | None = None, writer: str | None = None) -> dict:

    """Append one chained event. The only write path into the log.



    BEGIN IMMEDIATE because reading the head and writing the row must not be

    interleaved with another process doing the same -- the hash of my row is a

    claim about the row before it, and a claim made against a stale head is a

    fork, not a chain.

    """

    ensure_schema(conn)

    w = writer or writer_id(session)

    payload_json = _j(payload)

    ts = time.time()

    conn.execute("BEGIN IMMEDIATE")

    try:

        prev = _head_hash(conn)

        digest = _digest(kind, payload_json, ts, prev, w)

        conn.execute(

            "INSERT INTO forge_events (timestamp, kind, payload, prev_hash,"

            " payload_hash, writer_id) VALUES (?,?,?,?,?,?)",

            (ts, kind, payload_json, prev, digest, w),

        )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    return {"timestamp": ts, "kind": kind, "prev_hash": prev,

            "payload_hash": digest, "writer_id": w}





def verify_chain(conn: sqlite3.Connection) -> dict:
    """Walk the whole log and report what is provable, row by row.

    Two kinds of bad news, deliberately kept apart because they mean different
    things and only one of them is a failure:

    breaks -- a row's contents do not match its own hash, a link does not point
        at its predecessor, or the anchored legacy region changed. Any of these
        is evidence the log was rewritten, and nothing about this store's
        multi-process setup excuses them.

    gaps -- a row with no hash at all, written after chaining began. On this
        host that is the expected state while another session is still running
        code from before the change, and conflating it with a break is what
        makes a verifier into noise nobody reads. Reported, counted, not failed.

    ok therefore means "nothing was rewritten", not "every row is chained".
    """
    ensure_schema(conn)
    anchors = list(conn.execute("SELECT * FROM chain_anchors ORDER BY id"))
    breaks: list[dict] = []
    gaps: list[dict] = []
    chained = 0
    checked_legacy = 0
    unchained: list[int] = []
    prev = ""
    started = False

    for row in conn.execute(
        "SELECT id, timestamp, kind, payload, prev_hash, payload_hash, writer_id"
        " FROM forge_events ORDER BY id"
    ):
        if not row["payload_hash"]:
            unchained.append(row["id"])
            if started:
                gaps.append({"row": row["id"], "kind": "unchained_row",
                             "writer": row["writer_id"] or "",
                             "detail": "written after chaining began, carries no hash"})
            continue
        if not started:
            # The first chained row starts from whatever it points at. Only two
            # things are legitimate: genesis, for a log with nothing before it,
            # or the head of an anchor that was already frozen. Anything else is
            # a row inserted with a fabricated starting point, which is a break.
            started = True
            head = row["prev_hash"] or ""
            allowed = {GENESIS, ""} | {a["head_hash"] for a in anchors}
            if head not in allowed:
                breaks.append({"row": row["id"], "kind": "orphan_chain_start",
                               "writer": row["writer_id"] or "",
                               "detail": "first chained row does not start at genesis"
                                         " or at any anchor head"})
            expected_prev = head
        else:
            expected_prev = prev
        expect = _digest(row["kind"], row["payload"], row["timestamp"],
                         row["prev_hash"] or "", row["writer_id"] or "")
        if expect != row["payload_hash"]:
            breaks.append({"row": row["id"], "kind": "payload_hash_mismatch",
                           "writer": row["writer_id"] or "",
                           "detail": "row contents do not match the hash stored on the row"})
        if (row["prev_hash"] or "") != expected_prev:
            breaks.append({"row": row["id"], "kind": "broken_link",
                           "writer": row["writer_id"] or "",
                           "detail": "prev_hash does not equal the previous row's payload_hash"})
        prev = row["payload_hash"]
        chained += 1

    for a in anchors:
        digest, n = _baseline_digest(conn, a["up_to_id"])
        checked_legacy += n
        if n and digest != a["baseline_digest"]:
            breaks.append({"row": a["up_to_id"], "kind": "legacy_region_modified",
                           "writer": "",
                           "detail": f"rows anchored at id {a['up_to_id']} differ from the"
                                     f" baseline taken there"})

    # Truncation: a chain walking forward cannot see its own tail being cut,
    # because whatever row is last always looks like a valid last row. An anchor
    # is what makes that visible -- it recorded a head, and a head that is no
    # longer in the log means rows after that point were removed. One anchor,
    # then, is not ceremony; it is the only defence against a silent truncate.
    for a in anchors:
        if a["head_hash"] and a["head_hash"] != GENESIS:
            still_there = conn.execute(
                "SELECT 1 FROM forge_events WHERE payload_hash=?", (a["head_hash"],)
            ).fetchone()
            if still_there is None:
                breaks.append({"row": a["up_to_id"], "kind": "anchored_head_missing",
                               "writer": "",
                               "detail": "the head recorded by an anchor is no longer in"
                                         " the log: rows were removed after it was taken"})

    writers = sorted({r[0] for r in conn.execute(
        "SELECT DISTINCT writer_id FROM forge_events"
        " WHERE payload_hash != '' AND writer_id != ''")})
    return {
        "ok": not breaks and chained > 0,
        "chained_rows": chained,
        "legacy_rows_anchored": checked_legacy,
        "anchors": len(anchors),
        "unchained_rows": len(unchained),
        "unchained_after_chaining_began": len(gaps),
        "chained_writers": writers,
        "head_hash": _head_hash(conn),
        "breaks": breaks,
        "gaps": gaps,
        "findings": breaks + gaps,
    }


def lineage_of(conn: sqlite3.Connection, tool: str, depth: int = 3) -> dict:

    """The derivation edges recorded in the log for one tool.



    Only what the writers actually wrote down: forge_done says what was built,

    an evolve event says what it was bred from, a veto says who refused it.

    Nothing is inferred from name or need-text similarity -- that would be a

    guess wearing the word "lineage".

    """

    edges: list[dict] = []

    nodes = {tool}

    frontier = {tool}

    for _ in range(max(1, depth)):

        nxt: set[str] = set()

        for t in frontier:

            for row in conn.execute(

                "SELECT id, timestamp, kind, payload FROM forge_events WHERE payload LIKE ?"

                " ORDER BY id", (f'%"{t}"%',)

            ):

                try:

                    p = json.loads(row["payload"])

                except Exception:

                    continue

                for key in ("derived_from", "parent_version", "derives_from"):

                    if p.get("tool") == t or p.get("name") == t:

                        src = p.get(key)

                        if isinstance(src, str) and src:

                            edges.append({"from": src, "to": t, "kind": row["kind"],

                                          "row": row["id"], "rel": key})

                            nxt.add(src)

                if p.get("tool") == t and p.get("vetoed"):

                    for v in p["vetoed"] or []:

                        edges.append({"from": str(v), "to": t, "kind": "vetoed",

                                      "row": row["id"], "rel": "vetoed_by"})

                        nxt.add(str(v))

        if not nxt:

            break

        nodes |= nxt

        frontier = nxt - nodes | nxt

    return {"tool": tool, "nodes": sorted(nodes), "edges": edges}

