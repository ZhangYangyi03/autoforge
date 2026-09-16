"""The control plane: gate, declaration, self-modification — each with a receipt.

The claim under test is narrow and checkable: a decision that stops something
from running survives the process that made it. So every test here writes
through a real chain and reads the row back, rather than asserting that a
callback fired. The three failures this is meant to catch are all invisible
from inside: a gate outcome that lived in a list, a ceiling nobody asked, and a
rejected amendment that left no trace.
"""
from __future__ import annotations

import json
import threading

import pytest

from autoforge import audit, chaining, egress
from autoforge.autonomy import controls as controls_mod
from autoforge.autonomy.controls import ControlPlane, null_plane
from autoforge.autonomy.policy import AutonomyPolicy
from autoforge.autonomy.selfmod import SelfModifier
from autoforge.forge.manifest import CapabilityManifest, ManifestRefused
from autoforge.store import ToolStore
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


@pytest.fixture()
def store(tmp_path):
    st = ToolStore(str(tmp_path / "chain.db"))
    yield st
    st.close()


@pytest.fixture()
def plane(store):
    return ControlPlane(conn=store._conn, agent="test")


def rows(store, event_type=None):
    return [r for r in audit._rows(store._conn, event_type=event_type)]


def chained(store):
    return chaining.verify_chain(store._conn)


# -- the gate ------------------------------------------------------------

def _gated_registry(plane, answer):
    """A registry whose policy has a freedom off, with a confirmer that answers."""
    spec = ToolSpec(name="net_tool", description="needs the network",
                    parameters={}, source="builtin", effect_signature="network",
                    fn=lambda: "ran")
    seen = []

    def confirmer(name, arguments, needed):
        seen.append(needed)
        return answer

    reg = ToolRegistry(policy=AutonomyPolicy(may_access_network=False),
                       confirmer=confirmer, auto_quarantine=False,
                       controls=plane)
    reg.register(spec)
    return reg, seen


def test_a_confirmed_gate_call_leaves_a_row(store, plane):
    reg, seen = _gated_registry(plane, True)
    result = reg.call("net_tool", {})
    assert result.ok and seen == [["may_access_network"]]
    got = rows(store, "gate_allow")
    assert len(got) == 1
    assert got[0]["allowed"] is True
    assert got[0]["rule"] == "autonomy.gate: confirmed"
    assert got[0]["freedoms"] == ["may_access_network"]
    assert got[0]["resource"] == "net_tool"


def test_a_refusal_leaves_a_row_and_names_the_operator(store, plane):
    reg, _ = _gated_registry(plane, False)
    result = reg.call("net_tool", {})
    assert not result.ok and result.awaiting_confirmation
    got = rows(store, "gate_deny")
    assert got[0]["rule"] == "autonomy.gate: refused"
    assert "operator refused" in got[0]["reason"]


def test_no_operator_is_not_recorded_as_a_refusal(store, plane):
    """"Nobody was there" and "somebody said no" are different facts."""
    spec = ToolSpec(name="net_tool", description="x", parameters={},
                    source="builtin", effect_signature="network", fn=lambda: "ran")
    reg = ToolRegistry(policy=AutonomyPolicy(may_access_network=False),
                       confirmer=None, auto_quarantine=False, controls=plane)
    reg.register(spec)
    assert not reg.call("net_tool", {}).ok
    got = rows(store, "gate_deny")
    assert got[0]["rule"] == "autonomy.gate: no_operator"
    assert "nobody was available" in got[0]["reason"]
    assert "silence is not consent" in got[0]["reason"]


def test_a_broken_confirmer_is_recorded_as_an_error_not_a_refusal(store, plane):
    def boom(*a):
        raise RuntimeError("stdin is closed")

    spec = ToolSpec(name="net_tool", description="x", parameters={},
                    source="builtin", effect_signature="network", fn=lambda: "ran")
    reg = ToolRegistry(policy=AutonomyPolicy(may_access_network=False),
                       confirmer=boom, auto_quarantine=False, controls=plane)
    reg.register(spec)
    assert not reg.call("net_tool", {}).ok
    got = rows(store, "gate_deny")
    assert got[0]["rule"] == "autonomy.gate: error"
    assert "stdin is closed" in got[0]["reason"]


def test_the_receipts_go_on_the_same_chain_as_everything_else(store, plane):
    reg, _ = _gated_registry(plane, True)
    reg.call("net_tool", {})
    report = chained(store)
    assert report.get("ok"), report
    # and a forged row is detectable, i.e. we are on the chain and not beside it
    assert rows(store, "gate_allow")[0]["chained"] is True


def test_a_plane_with_no_ledger_counts_but_does_not_lie(plane):
    bare = ControlPlane(conn=None)
    bare.gate(tool="t", needed=["may_access_network"], outcome="refused")
    report = bare.summary()
    assert report["writing"] is False
    assert report["counts"] == {"gate_deny": 1}
    assert report["record_failures"] == 0      # nothing failed; nothing was asked of it


def test_a_failed_write_is_reported_not_swallowed(store):
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("database is locked")

        def commit(self): pass

        def rollback(self): pass

    bad = ControlPlane(conn=Broken())
    bad.gate(tool="t", needed=[], outcome="refused")
    assert bad.summary()["record_failures"] == 1
    assert "locked" in bad.failures()[0]["error"]


# -- the declaration -----------------------------------------------------

def test_an_admitted_declaration_is_a_row(store, plane):
    m = CapabilityManifest(intent="sum two numbers", memory_mb=64,
                           max_processes=1, cpu_seconds=5.0, wall_s=5.0)
    plane.declaration(m, source="add_tool")
    got = rows(store, "declaration_allow")
    assert got[0]["rule"] == "ceiling: within limits"
    assert got[0]["source"] == "add_tool"
    assert json.loads(got[0]["extra"])["memory_mb"] == 64


def test_a_breach_is_recorded_with_the_numbers(store, plane):
    m = CapabilityManifest(intent="hog", memory_mb=10 ** 9,
                           max_processes=1, cpu_seconds=5.0, wall_s=5.0)
    plane.declaration(m, source="hog_tool")
    got = rows(store, "declaration_deny")
    assert got[0]["allowed"] is False
    assert "memory_mb" in got[0]["rule"]
    assert "ceiling" in got[0]["reason"]


def test_no_intent_line_is_a_refusal_not_a_pass(store, plane):
    m = CapabilityManifest(intent="   ", memory_mb=64, max_processes=1,
                           cpu_seconds=5.0, wall_s=5.0)
    plane.declaration(m, source="mystery")
    got = rows(store, "declaration_deny")
    assert got[0]["rule"] == "intent: required"
    assert "nothing to reconcile" in got[0]["reason"]


def test_general_purpose_is_recorded_as_exempt_not_as_admitted(store, plane):
    """`run_python` is an arbitrary shell by definition; the ceiling is for a
    forged tool's budget. The receipt has to say which, or a reader cannot tell
    an exemption from a check that ran and passed."""
    m = CapabilityManifest(intent="operator snippet", general_purpose=True,
                           memory_mb=10 ** 9, max_processes=99)
    plane.declaration(m, source="run_python")
    got = rows(store, "declaration_allow")
    assert "not applied" in got[0]["rule"]
    assert "general-purpose" in got[0]["reason"]


def test_a_refused_declaration_is_recorded_before_it_is_raised(store):
    """The verifier's promise is "refused before any code ran"."""
    from autoforge.forge import verifier as verifier_mod

    plane = ControlPlane(conn=store._conn)
    m = CapabilityManifest(intent="hog", memory_mb=10 ** 9, max_processes=1)
    with pytest.raises(ManifestRefused):
        try:
            m.admit()
        except ManifestRefused as exc:
            plane.declaration(m, source="hog", refused=str(exc))
            raise
    got = rows(store, "declaration_deny")
    assert got[0]["rule"] == "ceiling: refused before any code ran"
    assert "before any code ran" in got[0]["reason"]
    # and the verifier really does this, rather than this test simulating it
    src = open(verifier_mod.__file__, encoding="utf-8").read()
    assert "except ManifestRefused" in src and "refused=str(exc)" in src


def test_admit_is_actually_called_by_the_verifier():
    """`admit()` existed and nothing called it. Read the source, because that is
    exactly the kind of gap a passing test suite does not notice."""
    from autoforge.forge import verifier as verifier_mod

    src = open(verifier_mod.__file__, encoding="utf-8").read()
    assert "self.declaration.admit()" in src


# -- self-modification ---------------------------------------------------

def _modifier(plane):
    return SelfModifier(require_rationale=True, controls=plane)


def test_an_accepted_amendment_is_a_row(store, plane):
    host = type("H", (), {"weights": {"success": 1.0}})()
    a = _modifier(plane).amend(host, "weights", {"success": 1.5}, "success predicts usefulness")
    assert a.accepted
    got = rows(store, "selfmod_allow")
    assert got[0]["resource"] == "weights"
    # rationale (ran), no-op (ran, value differs so it passed). The veto is not
    # consulted at all, which is the number that matters: `checked` counts guards
    # that actually ran, and a guard nobody reached must not be reported as one.
    assert got[0]["checked"] == 2
    assert "success predicts" in got[0]["reason"]


def test_a_missing_rationale_is_a_row_with_the_count_of_checks_that_ran(store, plane):
    host = type("H", (), {"weights": {}})()
    a = _modifier(plane).amend(host, "weights", {"x": 1})
    assert not a.accepted
    got = rows(store, "selfmod_deny")
    assert "rationale required" in got[0]["rule"]
    # one check ran before it stopped, and the receipt says so: "rejected" reads
    # identically whether a guard looked at it or nothing did
    assert got[0]["checked"] == 1


def test_a_no_op_is_recorded_apart_from_a_refusal(store, plane):
    host = type("H", (), {"weights": {"a": 1}})()
    _modifier(plane).amend(host, "weights", {"a": 1}, "try to change nothing")
    got = rows(store, "selfmod_deny")
    assert got[0]["rule"] == "selfmod.rejected: no-op (value unchanged)"


def test_a_veto_is_recorded_with_its_reason(store, plane):
    host = type("H", (), {"weights": {}})()
    mod = SelfModifier(require_rationale=True, controls=plane,
                       veto=lambda am: "weights must sum to 1")
    a = mod.amend(host, "weights", {"a": 2}, "rebalance")
    assert not a.accepted
    got = rows(store, "selfmod_deny")
    assert "weights must sum to 1" in got[0]["rule"]
    assert got[0]["checked"] == 3


def test_every_verdict_is_recorded_exactly_once(store, plane):
    host = type("H", (), {"weights": {}})()
    mod = _modifier(plane)
    mod.amend(host, "weights", {"a": 1})                       # no rationale
    mod.amend(host, "weights", {"a": 1}, "ok")                 # lands
    mod.amend(host, "weights", {"a": 1}, "again")              # no-op
    assert len(mod.amendments) == 3
    assert len(rows(store, "selfmod_allow")) + len(rows(store, "selfmod_deny")) == 3


def test_a_selfmodifier_with_no_plane_behaves_as_before():
    host = type("H", (), {"weights": {}})()
    a = SelfModifier().amend(host, "weights", {"a": 1}, "because")
    assert a.accepted and host.weights == {"a": 1}


# -- the vocabulary and the wiring ---------------------------------------

def test_the_new_event_types_are_in_the_vocabulary():
    for t in ("gate_allow", "gate_deny", "declaration_allow", "declaration_deny",
              "selfmod_allow", "selfmod_deny"):
        assert t in audit.EVENT_TYPES


def test_denials_by_rule_covers_the_control_layer(store, plane):
    """`by_rule` is the report that turns receipts into an edit; it must see the
    control plane's refusals, not only the resource ones."""
    reg, _ = _gated_registry(plane, False)
    reg.call("net_tool", {})
    rules = {r["rule"] for r in audit.by_rule(store._conn)}
    assert "autonomy.gate: refused" in rules


def test_nothing_to_gate_means_no_row(store, plane):
    """With every freedom on, the gate has nothing to ask and must not invent one."""
    spec = ToolSpec(name="plain", description="x", parameters={},
                    source="builtin", effect_signature="read_only", fn=lambda: "ok")
    reg = ToolRegistry(policy=AutonomyPolicy(), confirmer=lambda *a: True,
                       auto_quarantine=False, controls=plane)
    reg.register(spec)
    assert reg.call("plain", {}).ok
    assert not rows(store, "gate_allow")
    assert not rows(store, "gate_deny")


def test_the_null_plane_records_nothing_at_all():
    assert null_plane.gate(tool="t", needed=[], outcome="refused") is None
    assert null_plane.summary()["counts"] == {}


def test_two_threads_gating_at_once_do_not_lose_a_count(store, plane):
    reg, _ = _gated_registry(plane, True)
    threads = [threading.Thread(target=lambda: reg.call("net_tool", {})) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(rows(store, "gate_allow")) == 8
    assert chained(store).get("ok")

class TestAnAgentCanBeBuilt:
    """The construction-time regression, caught by the least clever test here.

    Measured: the first version of `attach_store`'s counterpart in `__init__`
    called `control_plane(agent="autoforge")`, but `plane()` takes no argument
    and its `ControlPlane` already defaults the name. That raised TypeError at
    construction -- and because 21 test files build a real `ForgeAgent` to get
    at anything else, one wrong call in the constructor took 28 tests down and
    looked like 28 unrelated failures.

    The cheap test that would have caught it is this one: build the object the
    other tests are standing on, with an explicit llm so nothing reaches the
    network.
    """

    def test_forge_agent_constructs(self):
        from autoforge.agent import ForgeAgent
        from autoforge.autonomy.policy import FULL_FREEDOM

        class _LLM:
            def complete(self, *a, **k):        # pragma: no cover - never called
                raise AssertionError("construction must not call the model")

        agent = ForgeAgent(_LLM(), policy=FULL_FREEDOM)
        assert agent.controls is not None
        # The plane is process-wide, so a second agent resolves to the same
        # object -- the property the registry's lazy lookup depends on.
        other = ForgeAgent(_LLM(), policy=FULL_FREEDOM)
        assert other.controls is agent.controls

    def test_attaching_a_store_records_the_plane_binding(self, tmp_path):
        """`attach_store` is the one entry point, and it must not raise.

        A file, not ":memory:": `ToolStore` treats its path as a real path and
        joins it against the home directory, so ":memory:" becomes a filename
        and sqlite refuses it. Found by writing this test.
        """
        from autoforge.agent import ForgeAgent
        from autoforge.autonomy.policy import FULL_FREEDOM

        class _LLM:
            def complete(self, *a, **k):        # pragma: no cover
                raise AssertionError("construction must not call the model")

        store = ToolStore(str(tmp_path / "attach.db"))
        try:
            agent = ForgeAgent(_LLM(), policy=FULL_FREEDOM)
            agent.attach_store(store)              # must not raise
            assert agent.store is store
            assert agent.registry.controls is agent.controls
        finally:
            store.close()
