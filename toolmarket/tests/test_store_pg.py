"""The Postgres backend: the parts testable without a server, and a live suite.

Two layers, deliberately separated.

  * Everything above `TestLivePostgres` runs everywhere, including on a machine
    with no PostgreSQL and no psycopg2. It covers the pure helpers and — more
    usefully — asserts that the two store implementations still present the *same
    surface*. That structural test is what catches the real risk of a second
    backend: a method added to one and forgotten in the other, which fails at
    runtime in the deployment that has the newer store and nowhere else.
  * `TestLivePostgres` runs only when `TOOLMARKET_TEST_PG_DSN` is set, which the
    compose verification does (`docker compose exec api python -m pytest`). It
    exercises the SQL, the JSONB round trip and cross-process visibility — none
    of which a fake cursor can honestly test.
"""
from __future__ import annotations

import json
import os

import pytest

from toolmarket.protocol.events import EventKind, EventLog
from toolmarket.protocol.lineage import LineageGraph, LineageNode
from toolmarket.protocol.resources import ResourceRecord, ResourceType, ToolContract
from toolmarket.store import ResourceStore, make_store
from toolmarket.store_pg import _SCHEMA, _as_dict, redact_dsn

try:
    import psycopg2  # noqa: F401
    HAVE_PG = True
except Exception:  # noqa: BLE001
    HAVE_PG = False

LIVE_DSN = os.environ.get("TOOLMARKET_TEST_PG_DSN", "").strip()


# ------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("dsn,expected", [
    ("postgresql://user:secret@db:5432/tm", "postgresql://db:5432/tm"),
    ("postgres://u:p@host/tm", "postgres://host/tm"),
    ("postgresql://u:p@host:6543/tm?sslmode=require",
     "postgresql://host:6543/tm?sslmode=require"),
    ("postgresql://host/tm", "postgresql://host/tm"),
])
def test_redact_dsn_removes_credentials_keeps_everything_else(dsn, expected):
    """A DSN in an HTTP response is the classic way a password leaks.

    The port, path and query are kept on purpose: `/ready` reporting
    `postgresql://db/tm` instead of `postgresql://db:5433/tm` would hide the
    misconfiguration the caller is trying to diagnose.
    """
    out = redact_dsn(dsn)
    assert out == expected
    assert "secret" not in out and "u:p" not in out


def test_redact_dsn_survives_junk():
    assert redact_dsn("not a dsn") == "not a dsn"
    assert redact_dsn("") == ""


def test_as_dict_accepts_jsonb_and_text():
    # JSONB arrives decoded; a column written by an older revision arrives as a
    # string. Both have to load or the migration path is a data-loss path.
    assert _as_dict({"a": 1}) == {"a": 1}
    assert _as_dict('{"a": 1}') == {"a": 1}
    assert _as_dict("[1, 2]") == [1, 2]


def test_schema_declares_the_three_tables_and_their_indexes():
    flat = " ".join(_SCHEMA.split())
    for table in ("resources", "events", "lineage"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in flat
    # `IF NOT EXISTS` on every statement is what makes this safe to run on every
    # process start, which is what lets the API and the worker race to boot
    # without a migration step or a lock.
    assert flat.count("IF NOT EXISTS") == flat.count("CREATE ")
    assert "idx_events_resource" in flat
    assert "idx_events_kind" in flat
    assert "idx_lineage_resource" in flat
    # The seq column is the hash chain's spine: it must be the primary key, so a
    # duplicate or reordered append is refused by the database, not merely by the
    # application layer that happens to be running.
    assert "seq INTEGER PRIMARY KEY" in flat


# ------------------------------------------------------- structural parity
def _public_methods(cls: type) -> set[str]:
    return {
        name for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
        and name not in {"close"}
    }


def test_postgres_store_has_every_method_the_sqlite_store_has():
    """The drift guard.

    A second backend is a promise that the two are interchangeable. Without this
    test, the way that promise breaks is somebody adding `delete_resource` to
    `ResourceStore`, shipping it, and finding out in production that the Postgres
    deployment silently lacks it — because the code that would call it catches
    `AttributeError` somewhere, or because nobody exercises that path in the
    environment that has Postgres.
    """
    from toolmarket.store_pg import PostgresStore

    sqlite_surface = _public_methods(ResourceStore)
    pg_surface = _public_methods(PostgresStore)
    # `backend` and `path` are attributes, not methods; compare the callables.
    missing = sqlite_surface - pg_surface
    assert not missing, f"PostgresStore is missing: {sorted(missing)}"


def test_sqlite_store_does_not_grow_a_postgres_only_method():
    from toolmarket.store_pg import PostgresStore

    extra = _public_methods(PostgresStore) - _public_methods(ResourceStore)
    # `wait_ready` and `ping` are the two that legitimately exist only on a
    # store with a server to wait for. Anything else appearing here means the
    # interface is drifting in the other direction.
    assert extra <= {"wait_ready", "ping", "redact"}, f"unexpected: {sorted(extra)}"


def test_make_store_dispatches_on_the_dsn(monkeypatch):
    monkeypatch.setenv("TOOLMARKET_STORE", ":memory:")
    assert isinstance(make_store(), ResourceStore)
    monkeypatch.setenv("TOOLMARKET_STORE", "sqlite:///tmp/does-not-matter.db")
    st = make_store()
    assert isinstance(st, ResourceStore)
    st.close()
    monkeypatch.delenv("TOOLMARKET_STORE", raising=False)
    monkeypatch.setenv("DATABASE_URL", ":memory:")
    assert isinstance(make_store(), ResourceStore)


def test_make_store_prefers_its_argument_over_the_environment(monkeypatch):
    monkeypatch.setenv("TOOLMARKET_STORE", ":memory:")
    st = make_store("sqlite:///tmp/arg-wins.db")
    assert isinstance(st, ResourceStore)
    st.close()


@pytest.mark.skipif(HAVE_PG, reason="psycopg2 is installed; the failure path is unreachable")
def test_missing_psycopg2_is_reported_at_construction(monkeypatch):
    """Without the extra, the error must name the extra — not raise an ImportError
    three frames deep from a module the caller never mentioned."""
    monkeypatch.setenv("TOOLMARKET_STORE", "postgresql://u:p@localhost:5432/tm")
    with pytest.raises(Exception) as excinfo:
        make_store()
    assert "psycopg2" in str(excinfo.value)


# ------------------------------------------------------------- live server
@pytest.mark.skipif(not LIVE_DSN, reason="set TOOLMARKET_TEST_PG_DSN for the live suite")
@pytest.mark.skipif(not HAVE_PG, reason="psycopg2 not installed")
class TestLivePostgres:
    """Runs against a real server. This is the suite the compose stack runs."""

    @pytest.fixture()
    def store(self):
        from toolmarket.store_pg import PostgresStore

        st = PostgresStore(LIVE_DSN)
        st.wait_ready(attempts=10, delay=0.5)
        # Start from a known state: the live suite may be pointed at the stack's
        # own database, which the API has been writing to. TRUNCATE rather than
        # DROP, so the schema — which `_init_schema` only creates — is left alone.
        with st._cursor(commit=True) as cur:
            cur.execute("TRUNCATE resources, events, lineage")
        yield st
        st.close()

    def _record(self, rid: str = "tool:live") -> ResourceRecord:
        return ResourceRecord(
            id=rid,
            type=ResourceType.TOOL,
            name="live",
            description="d",
            contract=ToolContract(parameters={}),
        )

    def test_ping(self, store):
        assert store.ping() is True

    def test_stats_report_the_backend(self, store):
        stats = store.stats()
        assert stats["backend"] == "postgres"
        assert "resources" in stats

    def test_resource_round_trip(self, store):
        rec = self._record()
        store.save_resource(rec)
        loaded = store.load_resource("tool:live")
        assert loaded is not None
        assert loaded.type == ResourceType.TOOL
        assert loaded.contract == rec.contract
        assert loaded.name == "live"
        assert [r.id for r in store.load_resources()] == ["tool:live"]

    def test_missing_resource_is_none_not_an_exception(self, store):
        assert store.load_resource("absent") is None

    def test_save_is_an_upsert(self, store):
        store.save_resource(self._record())
        rec = self._record()
        rec.ledger = {"times_invoked": 7}
        store.save_resource(rec)
        assert store.load_resource("tool:live").ledger["times_invoked"] == 7

    def test_event_chain_round_trips_and_verifies(self, store):
        log = EventLog()
        log.append(EventKind.REGISTER, "tool:live", data={"x": 1})
        log.append(EventKind.TRANSITION, "tool:live", data={"to": "probation"})
        for event in log:
            store.append_event(event)

        loaded = store.load_events()
        assert len(loaded) == 2
        # The chain is recomputed from the stored JSON, so the hash survives the
        # database unchanged — which is the property that makes the log
        # trustworthy rather than merely ordered.
        assert loaded.verify_chain()
        assert [e.seq for e in loaded] == [0, 1]
        assert loaded.for_resource("tool:live") == list(loaded)

    def test_duplicate_event_seq_is_refused_by_the_database(self, store):
        """The primary key is the guard, not the application.

        If two processes append concurrently, the database must reject the second
        write rather than let the chain fork. A test that only asserted the
        happy path would let a future "helpful" ON CONFLICT DO NOTHING rewrite the
        integrity guarantee into a silent data-loss bug.
        """
        log = EventLog()
        event = log.append(EventKind.REGISTER, "tool:live")
        store.append_event(event)
        with pytest.raises(psycopg2.errors.UniqueViolation):
            store.append_event(event)

    def test_lineage_round_trip(self, store):
        store.save_lineage_node(LineageNode("t@1.0.0", "t", "1.0.0"))
        store.save_lineage_node(
            LineageNode("t@1.0.1", "t", "1.0.1", parents=["t@1.0.0"]))
        graph = store.load_lineage()
        assert isinstance(graph, LineageGraph)
        assert [n.node_id for n in graph.roots()] == ["t@1.0.0"]
        assert {n.node_id for n in graph.ancestors("t@1.0.1")} == {"t@1.0.0"}

    def test_events_survive_a_second_connection(self, store):
        """Cross-process visibility — the reason Postgres is here at all.

        The API and the Celery worker are separate processes with separate
        `ResourceRegistry` objects. If the store were not shared, `POST
        /evolve/async` would hand the client a task id whose work the worker could
        not find, and the async path would be broken in exactly the configuration
        it was built for.
        """
        from toolmarket.store_pg import PostgresStore

        store.save_resource(self._record("tool:shared"))
        other = PostgresStore(LIVE_DSN)
        try:
            assert other.load_resource("tool:shared") is not None
            assert other.stats()["resources"] >= 1
        finally:
            other.close()
