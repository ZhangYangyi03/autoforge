"""PostgreSQL persistence — the same three tables, a different engine.

`store.py` says the store is deliberately dumb and that "a different backend
(Postgres, a file tree, an object store) can replace this without touching
anything above it". This module is that claim tested rather than asserted: it
implements the *same duck-typed surface* as `ResourceStore` and nothing above
the store changes. `ResourceRegistry` cannot tell the two apart, which is why
there is no interface to extract — the interface was always the method names.

Four differences from the SQLite store are deliberate:

  * `json` columns are **JSONB**, not TEXT. The substrate's records are queried,
    not just fetched (which resources are ACTIVE, which events belong to a
    resource), and JSONB makes that a database question instead of a Python one.
  * `append_event` is a plain `INSERT`, with no `ON CONFLICT` clause at all. The
    event log is append-only by contract, and *both* clauses are wrong here:
    `DO UPDATE` would be a silent tamper path, and `DO NOTHING` would silently
    drop a racing writer's event — the caller would be told it appended when it
    did not, which is the same data loss with better manners. A duplicate `seq`
    therefore raises `UniqueViolation`: the database refusing to fork the chain
    or to lose an event from it. The SQLite store keeps the identical contract
    with a bare `INSERT`.
  * `stats()["path"]` returns a **redacted DSN**. The SQLite store returns a file
    path because a file path cannot hold a password. A DSN can, and this dict is
    reachable from `GET /stats` — so the password is stripped here rather than
    left to every caller to remember.
  * The next `seq` and the chain head live in a one-row `event_head` table, which
    an append locks with `SELECT ... FOR UPDATE`. SQLite gets the same guarantee
    from `BEGIN IMMEDIATE`, which locks the whole database; Postgres has no
    coarse switch like that, so the lock has to be named. Without a lock the
    allocation is a read of a value another writer is about to move, and the
    `events` primary key catches the collision only after the client has been
    told its registration failed.

Connections come from a `ThreadedConnectionPool`: the API serves requests on a
threadpool and the Celery worker runs in another process entirely, so a single
shared connection with a lock would serialize both against each other.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

from toolmarket.protocol.events import Event, EventLog
from toolmarket.protocol.lineage import LineageGraph, LineageNode
from toolmarket.protocol.resources import ResourceRecord

try:  # psycopg2 is an optional extra; import failure is reported at construction
    import psycopg2
    from psycopg2 import pool as _pg_pool
    _HAVE_PG = True
    _PG_ERR: Optional[BaseException] = None
except Exception as _exc:  # noqa: BLE001
    psycopg2 = None  # type: ignore[assignment]
    _pg_pool = None  # type: ignore[assignment]
    _HAVE_PG = False
    _PG_ERR = _exc


_SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    id          TEXT PRIMARY KEY,
    json        JSONB NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY,
    resource_id  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    json         JSONB NOT NULL,
    at           DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_resource ON events(resource_id);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE TABLE IF NOT EXISTS lineage (
    node_id      TEXT PRIMARY KEY,
    resource_id  TEXT NOT NULL,
    json         JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lineage_resource ON lineage(resource_id);
-- The log's head, in a single row, so an append can lock the thing it is about
-- to move. `seq` is *allocated here*, not derived from a process's memory of the
-- log: with two writers — the API and the Celery worker — a length-derived `seq`
-- is a claim about a database that neither process can see, and the primary key
-- on `events` then turns the collision into a 500 rather than into a fact.
-- The seed backfills from the events already present, so deploying this onto a
-- database that has a log continues that chain instead of restarting it.
CREATE TABLE IF NOT EXISTS event_head (
    only_one   BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_one),
    next_seq   INTEGER NOT NULL,
    head_hash  TEXT
);
INSERT INTO event_head(only_one, next_seq, head_hash)
SELECT TRUE,
       COALESCE((SELECT MAX(seq) + 1 FROM events), 0),
       (SELECT json->>'hash' FROM events ORDER BY seq DESC LIMIT 1)
ON CONFLICT (only_one) DO NOTHING;
"""


def redact_dsn(dsn: str) -> str:
    """A DSN safe to put in an HTTP response: credentials removed, rest kept.

    Kept public and separate because it is the kind of thing that gets
    re-implemented inline, badly, the second time somebody adds a health field.
    """
    try:
        parts = urlsplit(dsn)
    except Exception:  # noqa: BLE001
        return "<unparseable dsn>"
    if not parts.netloc:
        return dsn
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, ""))


def _as_dict(value: Any) -> dict[str, Any]:
    """A JSONB column arrives already decoded; TEXT arrives as a string.

    Both are accepted so the same code path works against either column type —
    and so a store written by an older revision (TEXT) still loads.
    """
    if isinstance(value, (dict, list)):
        return value  # type: ignore[return-value]
    return json.loads(value)


class PostgresStore:
    """The SQLite store's surface, on PostgreSQL. `dsn` like
    `postgresql://user:pw@host:5432/db`."""

    # Read by `/ready` to name the connected backend. See the note on
    # `ResourceStore.backend`: without this the endpoint's default reported
    # "sqlite" for a stack that was in fact on Postgres.
    backend = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        minconn: int = 1,
        maxconn: int = 8,
        connect_timeout: int = 5,
    ) -> None:
        if not _HAVE_PG:
            raise RuntimeError(
                "PostgresStore needs psycopg2, which failed to import: "
                f"{_PG_ERR!r}. Install the extra: pip install 'tool-market[postgres]'"
            )
        self.dsn = dsn
        self._pool = _pg_pool.ThreadedConnectionPool(
            minconn, maxconn, dsn=dsn, connect_timeout=connect_timeout
        )
        self._init_schema()

    # -- plumbing ----------------------------------------------------------
    @contextmanager
    def _cursor(self, *, commit: bool = False) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
            if commit:
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _init_schema(self) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(_SCHEMA)

    def wait_ready(self, attempts: int = 30, delay: float = 1.0) -> None:
        """Block until the server accepts a trivial query.

        `depends_on: condition: service_healthy` covers the compose case, but a
        locally-run `uvicorn` against a just-started container hits this window
        constantly. Retrying here means the failure mode is a slow start rather
        than a traceback the reader has to decode.
        """
        last: Optional[BaseException] = None
        for _ in range(max(1, attempts)):
            try:
                with self._cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(delay)
        raise RuntimeError(f"postgres never became ready: {last!r}") from last

    def ping(self) -> bool:
        try:
            with self._cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        self._pool.closeall()

    def __enter__(self) -> "PostgresStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- resources --------------------------------------------------------
    def save_resource(self, rec: ResourceRecord) -> None:
        payload = json.dumps(rec.to_dict(), default=str)
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO resources(id, json, updated_at) VALUES(%s, %s::jsonb, %s) "
                "ON CONFLICT(id) DO UPDATE SET json=EXCLUDED.json, "
                "updated_at=EXCLUDED.updated_at",
                (rec.id, payload, time.time()),
            )

    def load_resource(self, resource_id: str) -> Optional[ResourceRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT json FROM resources WHERE id=%s", (resource_id,))
            row = cur.fetchone()
        return ResourceRecord.from_dict(_as_dict(row[0])) if row else None

    def max_updated_at(self) -> float:
        """Newest `updated_at` held here; 0.0 when empty. See `ResourceStore`.

        Present so `ResourceRegistry.refresh` has one call to make against either
        backend -- a store that cannot answer this would be a store the registry
        had to know the type of, which is the coupling the store layer exists to
        prevent.
        """
        with self._cursor() as cur:
            cur.execute("SELECT MAX(updated_at) FROM resources")
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def load_resources(self) -> list[ResourceRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT json FROM resources ORDER BY updated_at")
            rows = cur.fetchall()
        return [ResourceRecord.from_dict(_as_dict(r[0])) for r in rows]

    # -- events -----------------------------------------------------------
    def append_event(self, event: Event) -> None:
        # Deliberately dumb: the caller already sealed this event, so all that is
        # left is to store it or refuse. Allocation lives in `reserve_event`,
        # which is what the log is wired to; this is the primitive it and the
        # tests use directly.
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO events(seq, resource_id, kind, json, at) "
                "VALUES(%s, %s, %s, %s::jsonb, %s)",
                (event.seq, event.resource_id, event.kind.value,
                 json.dumps(event.to_dict(), default=str), event.at),
            )

    def reserve_event(
        self, build: Callable[[int, Optional[str]], Event]
    ) -> Event:
        """The `on_reserve` contract: the *database* decides `seq` and `prev`.

        It hands the builder the `(next_seq, head_hash)` it read, seals and
        inserts whatever comes back, and advances the head — all inside one
        transaction, with the head row locked for the duration. Two processes
        therefore serialize on that row instead of on their own guess at the
        tail, and `seq` stays contiguous because a rollback rolls the head back
        with it rather than burning a number.

        The client only ever sees this through `EventLog.append`, which is the
        point: nothing above the store had to learn that there is a second writer.
        """
        with self._cursor(commit=True) as cur:
            cur.execute(
                "SELECT next_seq, head_hash FROM event_head WHERE only_one FOR UPDATE"
            )
            seq, prev = cur.fetchone()
            event = build(seq, prev)
            cur.execute(
                "INSERT INTO events(seq, resource_id, kind, json, at) "
                "VALUES(%s, %s, %s, %s::jsonb, %s)",
                (event.seq, event.resource_id, event.kind.value,
                 json.dumps(event.to_dict(), default=str), event.at),
            )
            cur.execute(
                "UPDATE event_head SET next_seq=%s, head_hash=%s WHERE only_one",
                (seq + 1, event.hash),
            )
            return event

    def load_events(self) -> EventLog:
        with self._cursor() as cur:
            cur.execute("SELECT json FROM events ORDER BY seq")
            rows = cur.fetchall()
        return EventLog([Event.from_dict(_as_dict(r[0])) for r in rows])

    # -- lineage ----------------------------------------------------------
    def save_lineage_node(self, node: LineageNode) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO lineage(node_id, resource_id, json) "
                "VALUES(%s, %s, %s::jsonb) "
                "ON CONFLICT(node_id) DO UPDATE SET json=EXCLUDED.json",
                (node.node_id, node.resource_id,
                 json.dumps(node.to_dict(), default=str)),
            )

    def load_lineage(self) -> LineageGraph:
        with self._cursor() as cur:
            cur.execute("SELECT json FROM lineage")
            rows = cur.fetchall()
        graph = LineageGraph()
        # Insert parents-first so add() validation passes regardless of order;
        # orphans are force-added so a partial store still loads (same policy as
        # the SQLite store — the two must not disagree about what "loadable" is).
        pending = [LineageNode(**_as_dict(r[0])) for r in rows]
        while pending:
            progressed = False
            for node in list(pending):
                if all(p in graph for p in node.parents):
                    graph.add(node)
                    pending.remove(node)
                    progressed = True
            if not progressed:
                for node in pending:
                    graph._nodes[node.node_id] = node  # noqa: SLF001
                break
        return graph

    # -- aggregate --------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM resources")
            n_res = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM events")
            n_ev = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM lineage")
            n_ln = cur.fetchone()[0]
        return {"resources": n_res, "events": n_ev, "lineage_nodes": n_ln,
                "backend": "postgres", "path": redact_dsn(self.dsn)}
