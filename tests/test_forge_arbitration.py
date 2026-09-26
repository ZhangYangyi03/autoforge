"""The forge must consult the ranking it already computes, and a refusal is
not a failure.

Two defects are pinned here, both measured on the live ledger before the fix:

1. `BehaviourRouter` was constructed, weighted, documented and amendable, and
   `rank()`/`route()` were called from nowhere in the package. Seven references
   to routing weights, zero decisions. A ranking nothing consults is an
   ornament, and its tests passed the whole time because a ranking can be
   asserted without ever sitting on a path a turn takes.

2. `registry.register(spec)` on the forge path inherited `replace=True`, so a
   candidate named like an existing tool silently overwrote it. 8 of 116
   accepted forges landed on a name already taken; `chrome_cdp_drive` reached
   version 3 that way. The reply said "Forged", and there was no refusal.

The veto is deliberately not rank-alone. It needs a *name* claim, because a name
is a statement about what a tool does and a description is a sentence that
happens to contain words -- a lesson this codebase already paid for twice.
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


class TestNameAnchor:
    def test_the_whole_name_verbatim_anchors(self):
        assert name_anchor("read_bus_ndjson the board", "read_bus_ndjson") >= 1

    def test_shared_content_words_anchor(self):
        # list + agent + process("processes" canonically reduces to "process")
        assert name_anchor("list the agent processes here",
                           "list_agent_processes") == 3
        assert name_anchor("where is the agent", "list_agent_processes") == 1

    def test_stopwords_do_not_count_toward_the_anchor(self):
        """"and"/"the"/"with" are how two unrelated names reach two words."""
        assert name_anchor("transcode the video and report the bitrate",
                           "inspect_repo_the_and_gitignore_dir") == 0

    def test_a_prefix_is_not_a_word(self):
        """`repo` must not match `report` -- the 2026-09-16 lesson."""
        assert name_anchor("transcode a video and report the bitrate",
                           "inspect_repo_gitignore") == 0

    def test_a_description_coincidence_does_not_anchor(self):
        """`host_artifact_scan` was named for a need about processes.

        The words were in its description. That is a coincidence of English, and
        the anchor is computed on the *need against the name* alone.
        """
        assert name_anchor("count the running processes on this windows machine",
                           "host_artifact_scan") == 0


class TestArbiterDecides:
    def test_it_vetoes_when_the_name_is_the_needs(self):
        ar = Arbiter(_Router([("list_agent_processes", 0.9)]),
                     _registry(_spec("list_agent_processes")))
        v = ar.decide("list the agent processes running here")
        assert v.refused and v.veto.name == "list_agent_processes"

    def test_a_high_score_alone_is_not_a_veto(self):
        """Rank without a name claim is how a gate cries duplicate every time."""
        ar = Arbiter(_Router([("host_artifact_scan", 0.95)]),
                     _registry(_spec("host_artifact_scan", "scan host processes")))
        v = ar.decide("count the running processes on this machine")
        assert not v.refused

    def test_a_name_claim_alone_is_not_enough_either(self):
        """A dead tool must not block the forge that replaces it."""
        ar = Arbiter(_Router([("read_bus_ndjson", 0.01)]),
                     _registry(_spec("read_bus_ndjson")))
        assert not ar.decide("read_bus_ndjson lines").refused

    def test_retired_and_quarantined_tools_are_not_coverage(self):
        ar = Arbiter(_Router([("read_bus_ndjson", 0.9)]),
                     _registry(_spec("read_bus_ndjson", state=ToolState.RETIRED)))
        assert not ar.decide("read_bus_ndjson").refused

    def test_a_broken_router_does_not_block_the_forge(self):
        class _Boom:
            def rank(self, q):
                raise RuntimeError("no index")
        v = Arbiter(_Boom(), _registry(_spec("x"))).decide("anything")
        assert not v.refused and v.ranked == []

    def test_the_refusal_names_the_tool_and_says_how_to_get_through(self):
        ar = Arbiter(_Router([("read_bus_ndjson", 0.9)]),
                     _registry(_spec("read_bus_ndjson")))
        text = ar.decide("read_bus_ndjson").refusal()
        assert "read_bus_ndjson" in text
        assert "evolve_tool" in text
        assert "re-state the need" in text

    def test_the_context_block_is_a_ranked_list(self):
        ar = Arbiter(_Router([("a_tool", 0.9), ("b_tool", 0.4)]),
                     _registry(_spec("a_tool"), _spec("b_tool")))
        block = ar.decide("something else entirely").context_block()
        assert block.index("a_tool") < block.index("b_tool")


class TestForgePathCallsIt:
    """The whole point: it is on the path a real forge takes."""

    def _agent(self, *specs, pairs):
        a = ForgeAgent(MockLLMClient(), policy=FULL_FREEDOM)
        for s in specs:
            a.registry.register(s)
        a.arbiter = Arbiter(_Router(pairs), a.registry)
        return a

    def test_a_covered_need_is_refused_before_building(self):
        a = self._agent(_spec("read_bus_ndjson"), pairs=[("read_bus_ndjson", 0.9)])
        built = {"n": 0}

        def _forge(*args, **kw):
            built["n"] += 1
            return SimpleNamespace(ok=True, rounds=1, aborted=False, spec=None,
                                   replace_conflict="")

        a.pipeline = SimpleNamespace(forge=_forge)
        out = a.registry.get("forge_tool").fn("read_bus_ndjson")
        assert built["n"] == 0, "the forge ran despite an owned need"
        assert "read_bus_ndjson" in out

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

    def test_the_verdict_is_on_the_ledger(self):
        a = self._agent(_spec("read_bus_ndjson"), pairs=[("read_bus_ndjson", 0.9)])
        a.registry.get("forge_tool").fn("read_bus_ndjson")
        rows = [e for e in a.trace if e.get("kind") == "arbitration"]
        assert rows and rows[-1]["refused"] is True
        assert rows[-1]["veto"] == "read_bus_ndjson"

    def test_the_router_weights_are_the_ones_that_decide(self, monkeypatch):
        """The weights amend_self edits must be the ones that route.

        If the arbiter scored with its own private formula, amending
        `routing_weights` would stay decoration -- which is the exact defect
        being fixed.
        """
        a = self._agent(_spec("read_bus_ndjson"),
                        pairs=[("read_bus_ndjson", 0.02)])
        assert not a.arbiter.decide("read_bus_ndjson").refused
        a.arbiter.veto_floor = 0.01            # the floor lives on the arbiter
        assert a.arbiter.decide("read_bus_ndjson").refused


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

    def test_evolve_still_replaces_on_purpose(self):
        """`replace=False` must not close the deliberate replacement path."""
        reg = _registry(_spec("t"))
        spec = _spec("t")
        reg.register(spec, replace=True)     # the evolve path's own call
        assert reg.get("t") is spec
