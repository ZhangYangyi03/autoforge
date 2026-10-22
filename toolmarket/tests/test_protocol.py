"""Tests for the toolmarket substrate.

These run against real `autoforge`; they are not mocks. The point of the project
is that the enforcement is the real thing, so the tests exercise the real gate.
"""
from __future__ import annotations

import pytest

from toolmarket.protocol.events import EventKind, EventLog
from toolmarket.protocol.lifecycle import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    ResourceState,
    VersionStatus,
    can_transition,
    transition,
)
from toolmarket.protocol.lineage import LineageGraph, LineageNode
from toolmarket.protocol.resources import (
    ResourceRecord,
    ResourceType,
    ResourceVersion,
    ToolContract,
)
from toolmarket.protocol.sepl import (
    EvolutionOperator,
    ProposalRejected,
    StubProposer,
)
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


# ---------------------------------------------------------------- fixtures
def _sample_spec(name: str = "slugify"):
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    def slugify(text: str = "") -> str:
        return "-".join(text.lower().split())

    return ToolSpec(
        name=name,
        description="Slugify a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=slugify,
        code="def slugify(text=''):\n    return '-'.join(text.lower().split())\n",
        source="human",
        probes=[TriggerProbe(query="slugify this", expect="call")],
        invariances=["text"],
        # Declare a scope so the context-ceiling gate has something real to
        # reason about. "pure" => usable in any context (ceiling 4 in autoforge).
        effect_signature="pure",
    )


@pytest.fixture()
def registry():
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_sample_spec())
    return reg


# ---------------------------------------------------------------- lifecycle
def test_legal_and_illegal_transitions():
    assert can_transition(ResourceState.DRAFT, ResourceState.PROBATION)
    assert not can_transition(ResourceState.DRAFT, ResourceState.ACTIVE)
    assert transition(ResourceState.PROBATION, ResourceState.ACTIVE) == ResourceState.ACTIVE
    with pytest.raises(LifecycleError):
        transition(ResourceState.DRAFT, ResourceState.ACTIVE)
    with pytest.raises(LifecycleError):
        transition(ResourceState.RETIRED, ResourceState.ACTIVE)
    # no-ops are refused, not silently accepted
    with pytest.raises(LifecycleError):
        transition(ResourceState.ACTIVE, ResourceState.ACTIVE)


def test_retired_is_not_a_one_way_door():
    """RETIRED has exactly one way out, and it is the documented one.

    This asserted ``== frozenset()`` -- "retired is terminal" -- while the
    module's own state diagram draws ``retired --restore--> PROBATION`` and the
    table below it implements the back-edge. The test had been red since at
    least 9f23c3e (2026-09-15), which is why nobody noticed the two disagreed:
    an always-red test is read as "this file is broken", not as "the machine
    changed".

    Named for what the code actually promises rather than for terminality: a
    restore is a real operation (the recycle bin a retired tool goes to), and
    the invariant worth pinning is that it goes to PROBATION -- re-earning
    trust -- and never straight back to ACTIVE.
    """
    assert LEGAL_TRANSITIONS[ResourceState.RETIRED] == frozenset({
        ResourceState.PROBATION,
    })
    assert not can_transition(ResourceState.RETIRED, ResourceState.ACTIVE)


def test_version_status_has_exactly_three_states():
    # The claim is AGP's; assert it so nobody quietly widens it later.
    assert [s.value for s in VersionStatus] == ["active", "deprecated", "archived"]


# ---------------------------------------------------------------- events
def test_event_log_chain_and_tamper_detection():
    import dataclasses

    log = EventLog()
    log.append(EventKind.REGISTER, "tool:a", data={"x": 1})
    log.append(EventKind.TRANSITION, "tool:a", data={"to": "probation"})
    log.append(EventKind.INVOKE, "tool:a", data={"ok": True})
    assert len(log) == 3
    assert log.verify_chain()
    assert log.for_resource("tool:a") == list(log)
    # Tamper: swap a middle event for one whose content differs but whose stored
    # hash is left as-is. The chain must notice the mismatch.
    forged = dataclasses.replace(log._events[1], kind=EventKind.INVOKE)
    log._events[1] = forged
    assert not log.verify_chain()


def test_event_seq_is_monotonic():
    log = EventLog()
    for i in range(5):
        log.append(EventKind.INVOKE, "tool:a", data={"i": i})
    assert [e.seq for e in log] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- lineage
def test_lineage_graph_traversal():
    g = LineageGraph()
    g.add(LineageNode("t@1.0.0", "t", "1.0.0"))
    g.add(LineageNode("t@1.0.1", "t", "1.0.1", parents=["t@1.0.0"]))
    g.add(LineageNode("t@1.0.2", "t", "1.0.2", parents=["t@1.0.1"]))
    g.add(LineageNode("t@1.0.3", "t", "1.0.3", parents=["t@1.0.0"]))  # branch
    assert [n.node_id for n in g.roots()] == ["t@1.0.0"]
    assert {n.node_id for n in g.ancestors("t@1.0.2")} == {"t@1.0.0", "t@1.0.1"}
    assert {n.node_id for n in g.descendants("t@1.0.0")} == {
        "t@1.0.1", "t@1.0.2", "t@1.0.3"}
    assert g.path("t@1.0.0", "t@1.0.2") == ["t@1.0.0", "t@1.0.1", "t@1.0.2"]
    assert g.path("t@1.0.2", "t@1.0.0") is None  # no upward path


def test_lineage_rejects_unknown_parent():
    g = LineageGraph()
    with pytest.raises(ValueError):
        g.add(LineageNode("t@2", "t", "2.0.0", parents=["t@1"]))


# ---------------------------------------------------------------- resources
def test_toolspec_roundtrip_is_lossless():
    spec = _sample_spec()
    rec = ResourceRecord.from_toolspec(spec)
    assert rec.id == "tool:slugify"
    assert rec.type is ResourceType.TOOL
    assert rec.state is ResourceState.DRAFT
    assert rec.contract.invariances == ["text"]
    assert rec.contract.probes[0]["query"] == "slugify this"
    back = rec.to_toolspec()
    assert back.name == spec.name
    assert back.invariances == spec.invariances
    assert back.code == spec.code
    assert [p.query for p in back.probes] == [p.query for p in spec.probes]


def test_version_advance_archives_previous():
    rec = ResourceRecord.from_toolspec(_sample_spec())
    v0 = rec.version_current.version
    v1 = rec.advance_version(note="evolved")
    assert v1.version != v0
    assert v1.supersedes == v0
    assert v1.status is VersionStatus.ACTIVE
    archived = next(v for v in rec.versions if v.version == v0)
    assert archived.status is VersionStatus.ARCHIVED


def test_capability_schema_is_strict():
    rec = ResourceRecord.from_toolspec(_sample_spec())
    schema = rec.as_capability_schema()
    assert schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False


# ---------------------------------------------------------------- registry
def test_register_then_walk_lifecycle(registry):
    rid = "tool:slugify"
    assert registry.get(rid).state is ResourceState.DRAFT
    registry.transition(rid, ResourceState.PROBATION, reason="verified")
    assert registry.get(rid).state is ResourceState.PROBATION
    registry.promote(rid)
    assert registry.get(rid).state is ResourceState.ACTIVE
    registry.quarantine(rid, reason="decay")
    assert registry.get(rid).state is ResourceState.QUARANTINED
    registry.rehab(rid)
    assert registry.get(rid).state is ResourceState.PROBATION
    registry.retire(rid)
    assert registry.get(rid).state is ResourceState.RETIRED
    kinds = [e["kind"] for e in registry.event_log(rid)]
    assert kinds[0] == "register"
    assert "quarantine" in kinds and "rehab" in kinds and "retire" in kinds


def test_illegal_transition_is_refused(registry):
    with pytest.raises(LifecycleError):
        registry.transition("tool:slugify", ResourceState.ACTIVE)  # from DRAFT


def test_registry_persists_and_reloads():
    store = ResourceStore(":memory:")
    reg = ResourceRegistry(store)
    reg.register(_sample_spec())
    reg.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    # reload from the same store
    reg2 = ResourceRegistry(store)
    assert reg2.get("tool:slugify").state is ResourceState.PROBATION
    assert len(reg2.log) == 2
    assert reg2.log.verify_chain()


def test_invoke_logs_and_updates_ledger(registry):
    registry.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    result = registry.invoke("tool:slugify", {"text": "Hello World"})
    assert getattr(result, "ok", False) is True
    assert registry.get("tool:slugify").ledger["calls"] >= 1
    assert registry.event_log("tool:slugify")[-1]["kind"] == "invoke"


# ---------------------------------------------------------------- SEPL
def test_sepl_propose_assess_commit(registry):
    registry.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    registry.promote("tool:slugify")
    op = EvolutionOperator(registry, proposer=StubProposer())
    proposal = op.propose("tool:slugify", goal="handle unicode")
    assert len(proposal.candidates) == 3  # 2 good + 1 sabotaged
    report = op.assess(proposal.proposal_id)
    # The sabotaged candidate (dropped invariance) must be vetoed at the gate,
    # and the veto must happen BEFORE verification — it has no fitness.
    sab_idx = [i for i, c in enumerate(proposal.candidates)
               if "sabotage" in c.provenance.get("generator", "")]
    assert sab_idx, "expected a sabotaged candidate in the proposal"
    sabotaged = report.verdicts[sab_idx[0]]
    assert sabotaged["admissible"] is False
    assert sabotaged["fitness"] == 0.0
    assert sabotaged["verification"] is None
    assert "regression" in sabotaged["reason"]
    assert report.admissible
    before = registry.get("tool:slugify").version_current.version
    rec = op.commit(proposal.proposal_id)
    assert rec.version_current.version != before
    # A version change re-opens the trial: enforcement axis, not decoration.
    assert rec.state is ResourceState.PROBATION
    kinds = [e["kind"] for e in registry.event_log("tool:slugify")]
    assert "propose" in kinds and "assess" in kinds and "commit" in kinds


def test_sepl_rejects_when_all_candidates_vetoed(registry):
    """If every candidate drops a guardrail, the incumbent stays untouched."""
    registry.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    registry.promote("tool:slugify")

    class AllBad:
        def __call__(self, base, goal):
            from autoforge.tools.spec import ToolSpec
            return [ToolSpec(
                name=base.name, description=base.description + " bad",
                parameters=dict(base.parameters), fn=base.fn, code=base.code,
                source="generated", generator="bad", probes=[],
                invariances=[],  # drops the baseline invariance every time
            )]

    op = EvolutionOperator(registry, proposer=AllBad())
    prop = op.propose("tool:slugify", goal="regress everything")
    report = op.assess(prop.proposal_id)
    assert not report.admissible
    before = registry.get("tool:slugify").version_current.version
    with pytest.raises(ProposalRejected):
        op.commit(prop.proposal_id)
    assert registry.get("tool:slugify").version_current.version == before
    assert "reject" in [e["kind"] for e in registry.event_log("tool:slugify")]


def test_sepl_rollback_restores_archived_version(registry):
    registry.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    registry.promote("tool:slugify")
    op = EvolutionOperator(registry, proposer=StubProposer())
    first = registry.get("tool:slugify").version_current.version
    op.commit(op.propose("tool:slugify", goal="v2").proposal_id)
    assert registry.get("tool:slugify").version_current.version != first
    rolled = op.rollback("tool:slugify", first)
    assert rolled.version_current.version == first
    assert rolled.version_current.status is VersionStatus.ACTIVE
    assert "rollback" in [e["kind"] for e in registry.event_log("tool:slugify")]


# ---------------------------------------------------------------- store
def test_store_stats_and_chain_after_reload():
    store = ResourceStore(":memory:")
    reg = ResourceRegistry(store)
    reg.register(_sample_spec("a"))
    reg.register(_sample_spec("b"))
    reg.transition("tool:a", ResourceState.PROBATION, reason="v")
    s = store.stats()
    assert s["resources"] == 2
    assert s["events"] == 3
    assert s["lineage_nodes"] == 2
    reloaded = store.load_events()
    assert reloaded.verify_chain()


# ---------------------------------------------------------------- API
def test_api_smoke():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from toolmarket.api.main import create_app

    reg = ResourceRegistry(ResourceStore(":memory:"))
    app = create_app(reg)
    client = TestClient(app)

    assert client.get("/health").json()["ok"] is True

    r = client.post("/resources", json={
        "name": "shout",
        "description": "Uppercase text",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]},
        "code": "def shout(text=''):\n    return text.upper()\n",
    })
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "draft"

    assert client.get("/resources").json()["count"] == 1
    assert client.get("/resources/tool:shout").json()["name"] == "shout"

    # illegal transition -> 409
    bad = client.post("/resources/tool:shout/transition",
                      json={"to": "active"})
    assert bad.status_code == 409

    # legal transition
    ok = client.post("/resources/tool:shout/transition",
                     json={"to": "probation", "reason": "ok"})
    assert ok.status_code == 200 and ok.json()["state"] == "probation"

    inv = client.post("/resources/tool:shout/invoke",
                      json={"arguments": {"text": "hi"}})
    assert inv.status_code == 200 and inv.json()["ok"] is True

    lin = client.get("/resources/tool:shout/lineage").json()
    assert lin["nodes"] and lin["current"] == "tool:shout@0.1.0"

    ev = client.get("/events").json()
    assert ev["count"] >= 3


# ---------------------------------------------------------------- durability
def test_evolution_events_and_lineage_survive_persistence(tmp_path):
    """The operator's events + lineage must be durable, not just in-memory.

    Regression guard for the sink bug: before it, a full evolution left 8
    events in memory and only 4 in the store.
    """
    db = str(tmp_path / "substrate.db")

    store = ResourceStore(db)
    reg = ResourceRegistry(store)
    reg.register(_sample_spec())
    reg.transition("tool:slugify", ResourceState.PROBATION, reason="v")
    reg.promote("tool:slugify")
    op = EvolutionOperator(reg, proposer=StubProposer())
    op.commit(op.propose("tool:slugify", goal="unicode").proposal_id)
    op.rollback("tool:slugify", "0.1.0")

    live_events = len(reg.log)
    live_nodes = len(reg.lineage)
    live_resources = len(reg.list())
    # register, transition, promote, propose, assess, commit, rollback = 7
    assert live_events >= 7, f"expected the full trail, got {live_events}"
    store.close()

    # Reopen cold: nothing is carried over in Python objects.
    store2 = ResourceStore(db)
    reg2 = ResourceRegistry(store2)
    assert len(reg2.log) == live_events, "events did not persist"
    assert len(reg2.lineage) == live_nodes, "lineage did not persist"
    assert len(reg2.list()) == live_resources, "resources did not persist"
    assert reg2.log.verify_chain() is True
    kinds = [n.mutation for n in reg2.lineage.of_resource("tool:slugify")]
    assert "rollback" in kinds
    store2.close()


def test_two_writers_over_one_log_do_not_fork_the_chain(tmp_path):
    """The API and the Celery worker are two processes over one store.

    Both come up, both load the log, and only *then* do they write. With `seq`
    as `len(self._events)` — the length of a mirror that knows only this
    process's events — each writer independently concludes it is writing event
    N, and the second INSERT dies on the primary key. In the deployed stack that
    is a 500 on `POST /resources` the first time the worker's evolution appends
    anything: the first smoke run passes because the log is empty, and every run
    after it fails. `seq` has to be a fact about the database, not about one
    process's memory of it.

    Both writers are stood up *before* either writes, because that is the order
    that collides — two mirrors stale at boot, not a live race.
    """
    db = str(tmp_path / "shared.db")
    a = ResourceRegistry(ResourceStore(db))
    b = ResourceRegistry(ResourceStore(db))

    for i in range(5):
        a.log.append(EventKind.REGISTER, "tool:a", data={"i": i})
        b.log.append(EventKind.REGISTER, "tool:b", data={"i": i})

    reloaded = ResourceStore(db).load_events()
    assert [e.seq for e in reloaded] == list(range(10)), "seq must stay contiguous"
    # One chain, not two: every `prev` is the previous event's hash, no matter
    # which writer produced it.
    assert reloaded.verify_chain()
    assert len(reloaded.for_resource("tool:a")) == 5
    assert len(reloaded.for_resource("tool:b")) == 5
