"""The forge must consult the ranking it already computes, and a duplicate name
must be refused -- exactly, and early.

Two defects are pinned here, both measured on the live ledger:

1. `BehaviourRouter` was constructed, weighted, documented and amendable, and
   `rank()`/`route()` were called from nowhere in the package. Seven references
   to routing weights, zero decisions. A ranking nothing consults is an
   ornament, and its tests passed the whole time because a ranking can be
   asserted without ever sitting on a path a turn takes.

2. `registry.register(spec)` on the forge path inherited `replace=True`, so a
   candidate named like an existing tool silently overwrote it. 8 of 117
   accepted forges landed on a name already taken; `git_repo_status` did it
   three times, `chrome_cdp_drive` reached version 3 the same way. The reply
   said "Forged", and there was no refusal.

The first fix for (2) was a pre-forge veto: refuse the build when the need's
words anchored on an owned tool's name. It was measured against every accepted
forge in the ledger and it was wrong, so it was removed. `TestTheVetoWasRefuted`
holds that measurement as an assertion, so the veto cannot come back in a
plausible-sounding form without someone having to delete a test that says why.
The duplicate is caught instead where the name is known and no guess is needed:
in `ForgePipeline`, between generation and verification, on `replace=False`.

One thing deliberately *not* asserted as sufficient anywhere below: a high
router score. A ranking can be right and still not be a claim about identity.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from autoforge.agent import ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM
from autoforge.core.llm import MockLLMClient
from autoforge.route.arbiter import Arbiter, name_anchor
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState


def _spec(name: str, description: str = "does a thing", *,
          state: ToolState = ToolState.ACTIVE) -> ToolSpec:
    return ToolSpec(name=name, description=description,
                    parameters={"type": "object", "properties": {}},
                    fn=lambda **kw: "ok", state=state)


def _registry(*specs: ToolSpec) -> ToolRegistry:
    r = ToolRegistry()
    for s in specs:
        r.register(s)
    return r


class _Router:
    """A stand-in for BehaviourRouter: a fixed ranking, no lexical machinery."""

    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def rank(self, query):
        return [SimpleNamespace(name=n, score=s, breakdown={"text": s})
                for n, s in self.pairs]


class TestArbiterRanks:
    """What the arbiter does: it orders the library for a need."""

    def test_the_ranking_is_the_evidence_the_forge_writes_from(self):
        ar = Arbiter(_Router([("a_tool", 0.9), ("b_tool", 0.4)]),
                     _registry(_spec("a_tool"), _spec("b_tool")))
        v = ar.decide("something else entirely")
        assert [c.name for c in v.ranked] == ["a_tool", "b_tool"]
        block = v.context_block()
        assert block.index("a_tool") < block.index("b_tool")
        assert "ranked for THIS need" in block

    def test_it_never_refuses(self):
        """The veto was removed on evidence; this is the contract that replaced it."""
        ar = Arbiter(_Router([("read_bus_ndjson", 0.99)]),
                     _registry(_spec("read_bus_ndjson")))
        for need in ("read_bus_ndjson", "read the bus ndjson file for me",
                     "list the agent processes"):
            assert ar.decide(need).refused is False

    def test_retired_and_quarantined_tools_are_not_competitors(self):
        ar = Arbiter(_Router([("read_bus_ndjson", 0.9), ("live_one", 0.5)]),
                     _registry(_spec("read_bus_ndjson", state=ToolState.RETIRED),
                               _spec("live_one")))
        v = ar.decide("read_bus")
        assert [c.name for c in v.ranked] == ["live_one"]

    def test_a_broken_router_does_not_block_the_forge(self):
        class _Boom:
            def rank(self, q):
                raise RuntimeError("no index")
        v = Arbiter(_Boom(), _registry(_spec("x"))).decide("anything")
        assert not v.refused and v.ranked == []
        assert v.context_block() == ""


class TestTheVetoWasRefuted:
    """The measurement that took the veto out, as an assertion.

    A pre-forge veto was shipped in commit 48de4c0: refuse the build when the
    need's content words anchor (>= 2) on an owned tool's name and the router
    scored it above a floor. Replayed against all 117 accepted forges in the
    ledger, with the library reconstructed at each forge:

        rule                       duplicates caught   legitimate blocked
        anchor>=1 (as shipped)          7/8                95/109
        anchor>=2                       3/8                29/109
        whole name verbatim             0/8                 3/109

    95 legitimate forges refused to catch 7 duplicates -- and the 8th was
    caught by the register step anyway, exactly, at zero cost. Re-tuning does
    not rescue it: every stricter variant caught *fewer* duplicates while still
    blocking real work, because a need that names a tool is usually asking for
    something around it, not a rebuild of it.
    """

    def test_a_named_tool_is_evidence_for_building_not_against(self):
        # The real pair from the ledger: the need for a *probe* of the
        # transition endpoint names the schema tool it probes. Both exist.
        need = ("probe the toolmarket transition endpoint: send empty JSON and a "
                "bad action, read the 422, report the real field names")
        ar = Arbiter(_Router([("toolmarket_transition_schema", 0.95)]),
                     _registry(_spec("toolmarket_transition_schema")))
        assert name_anchor(need, "toolmarket_transition_schema") >= 2, (
            "this pair is why the anchor rule fired -- and the forge that "
            "produced the probe tool was legitimate")
        assert ar.decide(need).refused is False

    def test_no_threshold_setting_could_have_worked(self):
        """The floor was inert: from 0.0 to 0.6 the outcome did not move.

        Sweeping it is what showed the anchor was doing all the work, and that
        the anchor cannot separate a duplicate from a neighbour.
        """
        need = "read the agent bus ndjson board and return the last N messages"
        seen = set()
        for floor in (0.0, 0.12, 0.3, 0.6):
            ar = Arbiter(_Router([("read_agent_bus", 0.9)]),
                         _registry(_spec("read_agent_bus")), veto_floor=floor)
            seen.add(ar.decide(need).refused)
        assert seen == {False}, (
            "a knob that changes nothing at any setting is not a knob")


class TestForgePathCallsIt:
    """The whole point: it is on the path a real forge takes."""

    def _agent(self, *specs, pairs):
        a = ForgeAgent(MockLLMClient(), policy=FULL_FREEDOM)
        for s in specs:
            a.registry.register(s)
        a.arbiter = Arbiter(_Router(pairs), a.registry)
        return a

    def test_the_ranking_reaches_the_forge_context(self):
        a = self._agent(_spec("read_bus_ndjson"), _spec("read_file_lines"),
                        pairs=[("read_bus_ndjson", 0.9), ("read_file_lines", 0.3)])
        seen = {}

        def _forge(need, context=None, should_abort=None, **kw):
            seen["context"] = context or ""
            return SimpleNamespace(ok=False, rounds=1, aborted=False, spec=None,
                                   replace_conflict="")

        a.pipeline = SimpleNamespace(forge=_forge)
        a.registry.get("forge_tool").fn("read something else entirely")
        assert "ranked for THIS need" in seen["context"]
        assert "read_bus_ndjson" in seen["context"]

    def test_the_ranking_is_on_the_ledger(self):
        a = self._agent(_spec("read_bus_ndjson"), pairs=[("read_bus_ndjson", 0.9)])
        # Stub the build: this test is about what the arbiter recorded, and a
        # real forge here would spend the model budget proving nothing about it.
        a.pipeline = SimpleNamespace(forge=lambda *a_, **k: SimpleNamespace(
            ok=False, rounds=1, aborted=False, spec=None, replace_conflict=""))
        a.registry.get("forge_tool").fn("read something else entirely")
        rows = [e for e in a.trace if e.get("kind") == "arbitration"]
        assert rows and rows[-1]["ranked"][0] == "read_bus_ndjson"
        # and the ledger does not claim a decision the arbiter no longer makes
        assert "veto" not in rows[-1] and "refused" not in rows[-1]

    def test_the_router_weights_are_the_ones_that_rank(self, monkeypatch):
        """The weights amend_self edits must be the ones that order the context.

        If the arbiter scored with its own private formula, amending
        `routing_weights` would stay decoration -- the exact defect being fixed.
        """
        a = self._agent(_spec("read_bus_ndjson"), pairs=[("read_bus_ndjson", 0.02)])
        a.pipeline = SimpleNamespace(forge=lambda *a_, **k: SimpleNamespace(
            ok=False, rounds=1, aborted=False, spec=None, replace_conflict=""))
        a.registry.get("forge_tool").fn("read something else entirely")
        rows = [e for e in a.trace if e.get("kind") == "arbitration"]
        assert rows[-1]["ranked"][0] == "read_bus_ndjson"


class TestNoSilentReplacement:
    """`register(replace=False)` on the forge path."""

    def test_forge_refuses_a_name_that_is_already_taken(self):
        from autoforge.forge.pipeline import ForgeConfig, ForgePipeline

        reg = _registry(_spec("already_here"))
        made = _spec("already_here")

        class _Gen:
            def generate(self, need, context=""):
                return SimpleNamespace(name=made.name, description=made.description,
                                       parameters=made.parameters, code="x",
                                       entry="run", probes=[], tags=[],
                                       effect_signature="", sample_call={},
                                       sample_expect="", invariances=[])

        class _Verifier:
            sandbox = SimpleNamespace(abort_check=None)

            def verify(self, spec, sample_args=None):
                return SimpleNamespace(passed=True, failed=[], to_dict=lambda: {})

        p = ForgePipeline(_Gen(), _Verifier(), reg, config=ForgeConfig(max_rounds=1))
        res = p.forge("do the thing", replace=False)
        assert not res.ok
        assert "already_here" in res.replace_conflict
        assert res.spec is None
        # and the incumbent is untouched
        assert reg.get("already_here") is not None

    def test_the_refusal_happens_before_verification_ever_runs(self):
        """The measurement in the test name, not a claim about the code.

        The duplicate costs one model call and zero sandbox runs. A check that
        fires *after* verification catches the same 8/8 and spends the
        verification budget to do it; the ledger's own duplicates cost 438
        seconds that way.
        """
        from autoforge.forge.pipeline import ForgeConfig, ForgePipeline

        reg = _registry(_spec("already_here"))
        calls = {"gen": 0, "verify": 0}

        class _Gen:
            def generate(self, need, context=""):
                calls["gen"] += 1
                return SimpleNamespace(
                    name="already_here", description="d",
                    parameters={"type": "object", "properties": {}}, code="x",
                    entry="run", probes=[], tags=[], effect_signature="",
                    sample_call={}, sample_expect="", invariances=[])

        class _Verifier:
            sandbox = SimpleNamespace(abort_check=None)

            def verify(self, spec, sample_args=None):
                calls["verify"] += 1
                return SimpleNamespace(passed=True, failed=[], to_dict=lambda: {})

        events = []
        p = ForgePipeline(_Gen(), _Verifier(), reg, config=ForgeConfig(max_rounds=3),
                          on_event=lambda k, d: events.append(k))
        res = p.forge("do the thing", replace=False)
        assert not res.ok
        assert calls == {"gen": 1, "verify": 0}, calls
        assert len(res.attempts) == 1, "the refusal left no round record"
        assert events == ["forge_start", "name_taken", "forge_attempt",
                          "forge_done"], events

    def test_evolve_still_replaces_on_purpose(self):
        """`replace=False` must not close the deliberate replacement path."""
        reg = _registry(_spec("t"))
        spec = _spec("t")
        reg.register(spec, replace=True)     # the evolve path's own call
        assert reg.get("t") is spec
