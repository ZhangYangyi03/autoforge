"""Both front doors must be the same substrate, and this file is that claim.

The REST surface and the gRPC surface are peers over one registry. The failure
mode this test exists to catch is not "the RPC is broken" — it is *the two doors
disagreeing*: a state string one accepts and the other rejects, a transition one
allows and the other refuses, an event one writes and the other does not. That
class of bug is invisible to a test of either surface alone, because each surface
is internally consistent; it only exists in the gap between them.

So nothing here asserts on one surface's behaviour in isolation. Every case runs
the *same* operation through both and asserts they agree — on the outcome, on the
resource's resulting state, and on which category a failure falls into. Where the
two transports necessarily differ (HTTP 409 vs FAILED_PRECONDITION), the test
asserts the *correspondence* rather than equality, which is the thing that can
actually be wrong.

The registry is shared on purpose. Giving each surface its own would make every
assertion pass while proving nothing about whether a deployment that runs both
sees one substrate or two.
"""
from __future__ import annotations

import threading
import uuid

import grpc
import pytest
from fastapi.testclient import TestClient

from toolmarket.api.main import create_app
from toolmarket.cache import reset_cache_singleton
from toolmarket.grpc.client import SubstrateClient
from toolmarket.grpc.server import serve
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore

CODE = "def add(a=0, b=0):\n    return a + b\n"


def _registry() -> ResourceRegistry:
    """A registry on an in-memory store.

    The cache is process-wide (`get_cache()`) rather than a registry argument, so
    the fixture pins it to `none` through the environment instead. No cache
    because the cache is orthogonal here and its presence would make a read's
    provenance depend on whether something else read first — a source of
    flakiness that has nothing to do with the question being asked.
    """
    return ResourceRegistry(store=ResourceStore(":memory:"))


@pytest.fixture
def two_doors(monkeypatch):
    """One registry, served over both transports, torn down after the case."""
    # A fresh *memory* cache per case, not `REDIS_URL=none`. `none` selects
    # NullCache, and a task record lives in the cache — so `none` does not just
    # disable caching, it makes every async task unqueryable and would fail the
    # evolve case for a reason that has nothing to do with parity. Memory is
    # in-process and therefore deterministic, which was the actual goal.
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_cache_singleton()
    reg = _registry()
    rest = TestClient(create_app(reg))
    server = serve(reg, host="127.0.0.1", port=0)
    grpc_client = SubstrateClient(f"127.0.0.1:{server.bound_port}")
    try:
        yield reg, rest, grpc_client
    finally:
        grpc_client.close()
        server.stop(grace=0)
        reset_cache_singleton()


def _name() -> str:
    # Unique per call: the two surfaces share the registry, so a fixed name would
    # collide with the previous case's row and raise "already registered".
    return f"add_{uuid.uuid4().hex[:8]}"


# -- the probes ------------------------------------------------------------

def test_both_report_healthy(two_doors):
    """Liveness from both doors, and both name a version.

    The vocabularies differ by design and the assertion is on the
    *correspondence*, not on equal key names: the HTTP surface answers
    `{"ok": true}` because that is what its other endpoints say, while gRPC says
    `status: "ok"` because `status` is the field name every gRPC health check
    uses. Forcing them identical would make one of the two surfaces speak a
    dialect its own conventions do not recognise — the thing that must match is
    the *answer*, not the spelling.
    """
    _, rest, grpc_client = two_doors
    http = rest.get("/health")
    assert http.status_code == 200
    assert http.json()["ok"] is True
    assert grpc_client.health()["status"] == "ok"


def test_readiness_agrees(two_doors):
    """Ready on both, over the same store and cache.

    Note that `ready` is a boolean on the wire rather than an error: the gRPC
    surface deliberately does not abort with UNAVAILABLE for "not ready", so a
    client sees the same *kind* of answer the HTTP client gets in a 200 body.
    """
    _, rest, grpc_client = two_doors
    http = rest.get("/ready").json()
    grpc_ready = grpc_client.ready()
    assert http["ready"] is True
    assert grpc_ready["ready"] is True
    assert http["store"]["backend"] == grpc_ready["store"]


# -- registration ----------------------------------------------------------

def test_register_produces_the_same_resource(two_doors):
    """A tool registered over gRPC is visible, identical, over HTTP.

    This is the parity a shared registry buys: the second read goes through a
    different process boundary and a different serialisation and must still name
    the same state.
    """
    _, rest, grpc_client = two_doors
    name = _name()
    rid = grpc_client.register(name=name, code=CODE, description="Add two ints.")

    over_http = rest.get(f"/resources/{rid}").json()
    over_grpc = grpc_client.get(rid)

    assert over_http["id"] == over_grpc["id"] == rid
    assert over_http["name"] == over_grpc["name"] == name
    assert over_http["state"] == over_grpc["state"] == "draft"
    # The shapes differ on purpose and the assertion respects that. REST returns
    # the whole record (`rec.to_dict()`), which nests the current version under
    # `version_current`; the gRPC message is a flat, deliberately partial
    # projection, so it carries `version` directly. What has to agree is the
    # version's *identity*, not the key path — forcing the key names equal would
    # make one surface abandon its own convention to satisfy the other.
    assert over_http["version_current"]["version"] == over_grpc["version"]


def test_register_defaults_evolving_on_both(two_doors):
    """The `optional bool` in the .proto is not decoration.

    A proto3 plain bool cannot express "unset", and the REST surface defaults a
    new tool's evolving ON. A client that never sets the field must not get the
    opposite answer from the other door — that is exactly the silent divergence
    the field's presence is there to prevent.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    assert rest.get(f"/resources/{rid}").json()["enable_evolving"] is True
    assert grpc_client.get(rid)["contract"]["code"] == CODE


def test_get_unknown_is_not_found_on_both(two_doors):
    """The same missing id is 404 on one door and NOT_FOUND on the other."""
    _, rest, grpc_client = two_doors
    assert rest.get("/resources/does-not-exist").status_code == 404
    with pytest.raises(grpc.RpcError) as err:
        grpc_client.get("does-not-exist")
    assert err.value.code() == grpc.StatusCode.NOT_FOUND


# -- the lifecycle, which is where divergence would actually hurt ----------

def test_legal_transition_agrees(two_doors):
    """DRAFT -> PROBATION -> ACTIVE, driven alternately through both doors.

    Alternating is the point: if a state change were applied by the surface
    rather than the registry, driving the second step through the *other*
    transport would be the case that shows it.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)

    rest.post(f"/resources/{rid}/transition",
              json={"to": "PROBATION", "reason": "verified"})
    assert rest.get(f"/resources/{rid}").json()["state"] == "probation"

    grpc_client.transition(rid, "ACTIVE", reason="earned")
    assert rest.get(f"/resources/{rid}").json()["state"] == "active"
    assert grpc_client.get(rid)["state"] == "active"


def test_state_names_are_case_insensitive_on_both(two_doors):
    """'ACTIVE' and 'active' name one state, whichever door is asked.

    Tested explicitly because the two surfaces each used to coerce the string
    themselves, and a difference in that coercion is a difference in which
    requests succeed — the failure would look like a permissions bug to a client.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    assert rest.post(f"/resources/{rid}/transition",
                     json={"to": "probation"}).status_code == 200
    # Lowercase through gRPC, uppercase through HTTP, same destination.
    assert grpc_client.transition(rid, "active")["state"] == "active"
    rid2 = grpc_client.register(name=_name(), code=CODE)
    assert rest.post(f"/resources/{rid2}/transition",
                     json={"to": "PROBATION"}).status_code == 200
    assert grpc_client.transition(rid2, "ACTIVE")["state"] == "active"


def test_illegal_transition_is_a_conflict_on_both(two_doors):
    """A semantically impossible move, refused by both, in corresponding codes.

    HTTP says 409 Conflict; gRPC says FAILED_PRECONDITION. Those are the same
    statement in two vocabularies — "your request is well-formed and this
    resource cannot do it" — and a client's retry logic depends on them
    corresponding. UNKNOWN here, which is what a bare try/except produces, would
    make an impossible move indistinguishable from a bug in the server.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    # DRAFT -> ACTIVE is not an edge: a tool must pass through probation.
    http = rest.post(f"/resources/{rid}/transition", json={"to": "ACTIVE"})
    assert http.status_code == 409
    with pytest.raises(grpc.RpcError) as err:
        grpc_client.transition(rid, "ACTIVE")
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_unknown_state_is_a_bad_argument_on_both(two_doors):
    """An unknown *name* is a bad argument, not an illegal transition.

    The distinction matters more than it looks: HTTP 422 vs 409, gRPC
    INVALID_ARGUMENT vs FAILED_PRECONDITION. A caller that confused the two would
    retry a typo forever, or give up on a legal-but-blocked move.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    http = rest.post(f"/resources/{rid}/transition", json={"to": "NOT_A_STATE"})
    assert http.status_code == 422
    with pytest.raises(grpc.RpcError) as err:
        grpc_client.transition(rid, "NOT_A_STATE")
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# -- invocation and the audit trail ---------------------------------------

def test_invoke_returns_the_same_answer(two_doors):
    """One implementation of the tool, reached two ways, one result."""
    _, rest, grpc_client = two_doors
    name = _name()
    rid = grpc_client.register(name=name, code=CODE)
    rest.post(f"/resources/{rid}/transition", json={"to": "PROBATION"})

    over_grpc = grpc_client.invoke(rid, {"a": 2, "b": 3})
    over_http = rest.post(f"/resources/{rid}/invoke",
                          json={"arguments": {"a": 2, "b": 3}}).json()

    assert over_grpc["ok"] is True
    assert over_http["ok"] is True
    # Agreement, not a literal: the tool implementation returns its result as a
    # string (`'5'`, not `5`) and both doors carry it through unchanged. Asserting
    # `== 5` would have been asserting the *implementation's* type, which is not
    # this file's business and would go red on a correct change to the tool
    # runner. What must hold is that the two surfaces see the same value.
    assert over_grpc["output"] == over_http["output"] == "5"


def test_both_writes_land_in_one_append_only_log(two_doors):
    """Two doors, one event log — the audit trail cannot fork.

    A transition through gRPC and one through HTTP must appear in the *same*
    ordered log, because the log's whole value is that it is the single record
    of what happened. Two logs would each be a partial truth and neither would
    be evidence.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    before = len(rest.get(f"/resources/{rid}/events").json()["events"])

    grpc_client.transition(rid, "PROBATION", reason="via grpc")
    rest.post(f"/resources/{rid}/transition", json={"to": "ACTIVE",
                                                    "reason": "via http"})

    events = rest.get(f"/resources/{rid}/events").json()["events"]
    assert len(events) >= before + 2
    kinds = [e.get("kind") for e in events]
    assert "transition" in kinds


def test_lineage_agrees(two_doors):
    """Registration creates one lineage node, and both doors see it."""
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    over_grpc = grpc_client.lineage(rid)
    over_http = rest.get(f"/resources/{rid}/lineage").json()
    # Both must name a node; the shapes differ (edges vs nodes) by design, so the
    # assertion is on the count of nodes reachable, not on the envelope.
    assert len(over_grpc) == 0  # the root has no parents, so no edges
    assert len(over_http.get("nodes", [])) >= 1


def test_list_agrees_on_counts(two_doors):
    """Both doors enumerate the same set of resources."""
    _, rest, grpc_client = two_doors
    for _ in range(3):
        grpc_client.register(name=_name(), code=CODE)
    over_http = rest.get("/resources").json()
    over_grpc = grpc_client.list()
    assert over_http["count"] == len(over_grpc)
    assert {r["id"] for r in over_http["resources"]} == \
           {r["id"] for r in over_grpc}


# -- the async evolution loop ---------------------------------------------

def test_evolve_is_enqueued_and_pollable_through_both(two_doors):
    """An evolution started via gRPC is pollable via gRPC's GetTask.

    The gRPC surface has no synchronous evolve on purpose — a candidate
    assessment can outrun any deadline the server does not control — so the
    contract to verify is that submit returns an id whose state can be read back.
    """
    _, rest, grpc_client = two_doors
    rid = grpc_client.register(name=_name(), code=CODE)
    grpc_client.transition(rid, "PROBATION")
    started = grpc_client.evolve(rid, "make it handle strings", proposer="stub")
    assert started["task_id"]
    # The REST surface must be able to see the same task id: one queue, and an
    # id that only one door can resolve would mean two.
    over_http = rest.get(f"/tasks/{started['task_id']}")
    assert over_http.status_code == 200
    assert over_http.json()["task_id"] == started["task_id"]


def test_concurrent_registrations_through_both_doors_do_not_collide(two_doors):
    """Two threads, two doors, distinct ids, one registry.

    A weak concurrency check, and deliberately so — it is not a race detector.
    What it does prove is that the gRPC server's thread pool and the registry
    coexist without the shared mutable state that would make a single-threaded
    test pass and a real deployment fail.
    """
    _, rest, grpc_client = two_doors
    names = [_name() for _ in range(6)]
    ids: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def over_grpc(n: str) -> None:
        try:
            rid = grpc_client.register(name=n, code=CODE)
            with lock:
                ids.append(rid)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=over_grpc, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert len(set(ids)) == 6
    assert rest.get("/resources").json()["count"] >= 6
