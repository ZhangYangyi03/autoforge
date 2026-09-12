"""Tests for the v0.2.0 mechanisms: evolution, adversarial gate, fuzzing,
composition, persistence, and the autonomy layer. All offline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.autonomy.policy import FULL_FREEDOM, SUPERVISED, AutonomyPolicy
from autoforge.autonomy.selfmod import SelfModifier
from autoforge.autonomy.spawn import ShareMode, Spawner
from autoforge.core.agent import Agent, AgentResult
from autoforge.core.llm import LLMResponse, MockLLMClient, tool_call
from autoforge.forge.adversary import AdversarialGate
from autoforge.forge.evolution import EvolutionEngine
from autoforge.forge.fuzzer import generate_robustness_probes, run_robustness_checks
from autoforge.forge.generator import GeneratedTool, TemplateGenerator
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.store import ToolStore
from autoforge.tools.composition import DepGraph, compose_code, parse_deps, validate_deps
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState, TriggerProbe


# -- helpers ----------------------------------------------------------
def good_code() -> str:
    return "def rev(s=''):\n    return s[::-1]\n"


def brittle_code() -> str:
    """Crashes on empty input — exactly the kind the fuzzer should catch."""
    return (
        "def rev(s=''):\n"
        "    return s[::-1][0]  # IndexError on empty string\n"
    )


def spec_for(code: str, name: str = "rev") -> ToolSpec:
    return ToolSpec(
        name=name, description="Reverse a string.",
        parameters={"type": "object", "properties": {"s": {"type": "string"}}},
        fn=lambda **_: "", code=code, source="generated",
        probes=[TriggerProbe(query="reverse this string", negative_query="capital of Peru?")],
    )


def reverse_model(messages, tools, **kw) -> LLMResponse:
    last = next((m.content for m in reversed(messages) if m.role == "user"), "")
    names = [t["function"]["name"] for t in (tools or [])]
    if "reverse" in last.lower() and "rev" in names:
        return LLMResponse(tool_calls=[tool_call("rev", {"s": last})])
    return LLMResponse(content="direct answer")


# ======================================================================
# fuzzer
# ======================================================================
class TestFuzzer:
    def test_generates_probes_for_params(self):
        probes = generate_robustness_probes(spec_for(good_code()))
        assert len(probes) > 5
        labels = [p.label for p in probes]
        assert any("''" in l for l in labels)          # empty string probed

    def test_catches_brittle_tool(self):
        r = run_robustness_checks(spec_for(brittle_code()), sandbox=Sandbox(timeout=8))
        assert not r.passed
        assert r.survival_rate < 1.0
        assert any("IndexError" in f["error"] for f in r.failures)

    def test_passes_robust_tool(self):
        r = run_robustness_checks(spec_for(good_code()), sandbox=Sandbox(timeout=8))
        assert r.passed

    def test_probes_cover_multiple_types(self):
        spec = ToolSpec(
            name="f", description="x",
            parameters={"type": "object", "properties": {
                "a": {"type": "string"}, "b": {"type": "integer"},
            }},
            fn=lambda **_: "", code="def f(a='',b=0):\n    return 1\n",
        )
        probes = generate_robustness_probes(spec)
        labels = " ".join(p.label for p in probes)
        assert "a=" in labels and "b=" in labels


# ======================================================================
# adversarial gate
# ======================================================================
class TestAdversarial:
    def _attacker(self, payload: str):
        return MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=payload))

    def _attacker_json(self, attacks: list[dict]):
        import json as _json
        return self._attacker(_json.dumps(attacks))

    def test_attacker_breaks_brittle_tool(self):
        attacker = self._attacker('[{"args": {"s": ""}, "rationale": "empty string"}]')
        gate = AdversarialGate(attacker, execution_sandbox=Sandbox(timeout=8))
        report = gate.attack(spec_for(brittle_code()))
        assert not report.passed
        assert report.survived == 0

    def test_robust_tool_survives(self):
        attacker = self._attacker_json([
            {"args": {"s": ""}, "rationale": "empty"},
            {"args": {"s": "x" * 5000}, "rationale": "long"},
        ])
        gate = AdversarialGate(attacker, execution_sandbox=Sandbox(timeout=8))
        report = gate.attack(spec_for(good_code()))
        assert report.passed
        assert report.survived == 2

    def test_unparseable_attacker_output_falls_back(self):
        attacker = self._attacker("I cannot generate attacks.")
        gate = AdversarialGate(attacker, execution_sandbox=Sandbox(timeout=8))
        report = gate.attack(spec_for(good_code()))
        assert report.total_attacks >= 1  # fallback attack ran

    def test_reports_attack_details(self):
        attacker = self._attacker('[{"args": {"s": ""}, "rationale": "empty string"}]')
        gate = AdversarialGate(attacker, execution_sandbox=Sandbox(timeout=8))
        report = gate.attack(spec_for(brittle_code()))
        d = report.to_dict()
        assert d["attacks"][0]["args"] == {"s": ""}
        assert "IndexError" in d["attacks"][0]["error"]


# ======================================================================
# evolution
# ======================================================================
class TestEvolution:
    def _engine(self, mutants_json: str):
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=mutants_json))
        verifier = ToolVerifier(
            llm, sandbox=Sandbox(timeout=8),
            run_adversarial_check=False, run_robustness_check=False,
        )
        return EvolutionEngine(llm, verifier, population_size=3)

    def test_selects_the_working_mutant(self):
        payload = (
            'FIX-1 {"name": "rev", "description": "reverse", "code": '
            '"def rev(s=\'\'):\\n    raise RuntimeError(\'bad\')\\n", '
            '"entry": "rev", "parameters": {"type":"object","properties":{"s":{"type":"string"}}}, '
            '"probes": [{"query": "reverse this"}]}\n'
            'FIX-2 {"name": "rev", "description": "reverse", "code": '
            '"def rev(s=\'\'):\\n    return s[::-1]\\n", '
            '"entry": "rev", "parameters": {"type":"object","properties":{"s":{"type":"string"}}}, '
            '"probes": [{"query": "reverse this", "negative_query": "capital of Peru?"}]}'
        )
        engine = self._engine(payload)
        result = engine.evolve(spec_for(brittle_code()), "IndexError on empty input")
        assert result.best_mutant is not None
        assert result.best_mutant.fitness > 0

    def test_no_mutants_keeps_original(self):
        engine = self._engine("I could not generate fixes.")
        result = engine.evolve(spec_for(good_code()), "whatever")
        assert result.kept_existing

    def test_fitness_ranks_pass_over_fail(self):
        report_pass = type("R", (), {"checks": [type("C", (), {"name": "execution", "passed": True})(),
                                                type("C", (), {"name": "negative", "passed": True})()]})()
        report_fail = type("R", (), {"checks": [type("C", (), {"name": "execution", "passed": False})()]})()
        f_pass = EvolutionEngine._compute_fitness(report_pass)
        f_fail = EvolutionEngine._compute_fitness(report_fail)
        assert f_pass > f_fail


# ======================================================================
# composition
# ======================================================================
class TestComposition:
    def test_parse_deps_from_signature(self):
        s = spec_for(good_code())
        s.effect_signature = "uses:normalize_isbn"
        assert parse_deps(s) == ["normalize_isbn"]

    def test_parse_deps_from_code(self):
        s = spec_for(good_code())
        s.code = "# uses:helper_one\ndef rev(s=''):\n    return s\n"
        assert "helper_one" in parse_deps(s)

    def test_validate_missing_dep(self):
        s = spec_for(good_code())
        s.effect_signature = "uses:ghost"
        errors = validate_deps(s, ToolRegistry())
        assert errors and "not found" in errors[0]

    def test_validate_found_dep(self):
        reg = ToolRegistry()
        reg.register(spec_for(good_code(), name="helper"))
        reg.promote("helper")
        s = spec_for(good_code())
        s.effect_signature = "uses:helper"
        assert validate_deps(s, reg) == []

    def test_compose_code_inlines_dep(self):
        s = spec_for("def main_fn():\n    return helper()\n", name="main_fn")
        composed = compose_code(s, {"helper": "def helper():\n    return 42\n"})
        assert "def helper()" in composed and "def main_fn()" in composed
        ns: dict = {}
        exec(composed, ns)
        assert ns["main_fn"]() == 42

    def test_dep_graph_cycle_detection(self):
        g = DepGraph()
        g.add("a", ["b"])
        g.add("b", ["a"])
        assert g.has_cycle("a")

    def test_dep_graph_no_cycle(self):
        g = DepGraph()
        g.add("a", ["b"])
        g.add("b", [])
        assert not g.has_cycle("a")

    def test_reverse_deps(self):
        g = DepGraph()
        g.add("a", ["shared"])
        g.add("b", ["shared"])
        assert set(g.reverse_deps("shared")) == {"a", "b"}


# ======================================================================
# persistence
# ======================================================================
class TestStore:
    def test_save_and_load_tool(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        reg = ToolRegistry()
        reg.register(spec_for(good_code()))
        reg.promote("rev")
        store.save_tool(reg.get("rev"))
        loaded = store.load_all_tools()
        assert "rev" in loaded
        assert loaded["rev"].state == ToolState.ACTIVE
        assert loaded["rev"].code == good_code()
        store.close()

    def test_roundtrip_preserves_stats(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        reg = ToolRegistry()
        reg.register(spec_for(good_code()))
        reg.promote("rev")
        reg.call("rev", {"s": "abc"})
        store.save_tool(reg.get("rev"))
        loaded = store.load_all_tools()["rev"]
        assert loaded.stats["calls"] == 1
        assert loaded.stats["successes"] == 1
        store.close()

    def test_version_archive(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        reg = ToolRegistry()
        reg.register(spec_for(good_code()))
        store.save_tool(reg.get("rev"))
        store.archive_version("rev", "old_code", {"passed": False})
        rec = store.load_all_tools()["rev"]
        assert rec.version == 2
        assert rec.old_versions[0]["code"] == "old_code"
        store.close()

    def test_dependencies_persist(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        store.save_deps("a", ["b", "c"])
        assert set(store.get_deps("a")) == {"b", "c"}
        assert "a" in store.get_reverse_deps("b")
        store.close()

    def test_events_logged(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        store.log_event("forge_done", {"tool": "x", "ok": True})
        store.log_event("run", {"task": "y"})
        events = store.get_events()
        assert len(events) == 2
        assert events[-1]["kind"] == "run"
        store.close()

    def test_delete_cascades_deps(self, tmp_path):
        store = ToolStore(str(tmp_path / "t.db"))
        store.save_deps("a", ["b"])
        store.delete_tool("b")
        assert store.get_deps("a") == []
        store.close()


# ======================================================================
# autonomy — policy
# ======================================================================
class TestPolicy:
    def test_full_freedom_denies_nothing(self):
        assert FULL_FREEDOM.denied == []
        assert "full autonomy" in FULL_FREEDOM.describe()

    def test_supervised_denies_specific_things(self):
        d = SUPERVISED.denied
        assert "may_modify_own_prompt" in d
        assert "may_spawn_agents" in d
        assert "unlimited_turns" in d

    def test_to_dict_roundtrip(self):
        p = AutonomyPolicy(may_forge_tools=False)
        assert p.to_dict()["may_forge_tools"] is False
        assert p.denied == ["may_forge_tools"]


# ======================================================================
# autonomy — self-modification
# ======================================================================
class TestSelfModifier:
    def test_amend_records_change(self):
        class Host:
            prompt = "old"
        h = Host()
        sm = SelfModifier()
        a = sm.amend(h, "prompt", "new", "because it helps")
        assert a.accepted and h.prompt == "new"
        assert sm.accepted_count() == 1

    def test_requires_rationale(self):
        class Host:
            prompt = "old"
        h = Host()
        sm = SelfModifier(require_rationale=True)
        a = sm.amend(h, "prompt", "new", "")
        assert not a.accepted and h.prompt == "old"
        assert "rationale required" in a.rejected_reason

    def test_noop_rejected(self):
        class Host:
            prompt = "same"
        h = Host()
        sm = SelfModifier()
        a = sm.amend(h, "prompt", "same", "no actual change")
        assert not a.accepted and "no-op" in a.rejected_reason

    def test_nested_amendment(self):
        class Inner:
            max_rounds = 3
        class Host:
            forge_config = Inner()
        h = Host()
        sm = SelfModifier()
        a = sm.amend(h, "forge_max_rounds", 5, "need more tries",
                     attr="max_rounds", nested=("forge_config",))
        assert a.accepted and h.forge_config.max_rounds == 5

    def test_veto_blocks_change(self):
        class Host:
            prompt = "old"
        h = Host()
        sm = SelfModifier(veto=lambda a: "policy forbids this")
        a = sm.amend(h, "prompt", "new", "trying")
        assert not a.accepted and h.prompt == "old"
        assert sm.rejected_count() == 1

    def test_log_is_serialisable(self):
        class Host:
            prompt = "old"
        h = Host()
        sm = SelfModifier()
        sm.amend(h, "prompt", "new", "reason")
        log = sm.log()
        assert log[0]["target"] == "prompt" and log[0]["accepted"] is True


# ======================================================================
# autonomy — spawning
# ======================================================================
class _FakeChildAgent:
    def __init__(self, registry):
        self.registry = registry

    def run(self, task):
        if "forge" in task.lower():
            self.registry.register(spec_for(good_code(), name="child_made_this"))
            self.registry.promote("child_made_this")
        return AgentResult(f"child did: {task}", [], 1)


class TestSpawner:
    def _spawner(self, mode=ShareMode.SHARED):
        reg = ToolRegistry()
        return Spawner(registry=reg, agent_factory=lambda s, r: _FakeChildAgent(r)), reg

    def test_shared_spawn_adds_to_parent_library(self):
        sp, reg = self._spawner()
        rec = sp.spawn("please forge something")
        assert rec.error is None
        assert "child_made_this" in rec.tools_forged
        assert "child_made_this" in reg.names()  # visible in the shared library

    def test_isolated_spawn_does_not_leak(self):
        sp, reg = self._spawner()
        rec = sp.spawn("please forge something", mode=ShareMode.ISOLATED)
        assert "child_made_this" in rec.tools_forged
        assert "child_made_this" not in reg.names()  # parent library untouched

    def test_merge_back_pulls_isolated_tools(self):
        sp, reg = self._spawner()
        sp.spawn("please forge something", mode=ShareMode.ISOLATED)
        child_reg = ToolRegistry()
        child_reg.register(spec_for(good_code(), name="child_made_this"))
        child_reg.promote("child_made_this")
        merged = sp.merge_back(child_reg)
        assert "child_made_this" in merged and "child_made_this" in reg.names()

    def test_depth_cap_is_reported_not_silent(self):
        reg = ToolRegistry()
        sp = Spawner(registry=reg, agent_factory=lambda s, r: _FakeChildAgent(r), max_depth=0)
        rec = sp.spawn("anything")
        assert rec.error and "max spawn depth" in rec.error

    def test_summary_lists_children(self):
        sp, _ = self._spawner()
        sp.spawn("a")
        sp.spawn("b")
        assert sp.summary()["spawned"] == 2


# ======================================================================
# agent loop — unbounded + self-termination
# ======================================================================
class TestAgentLoop:
    def test_unbounded_by_default(self):
        reg = ToolRegistry()
        # Model keeps calling a tool forever; the agent must not impose a cap
        # on its own, so we impose one at the call site to end the test.
        calls = {"n": 0}

        def loop(messages, tools, **kw):
            calls["n"] += 1
            if calls["n"] > 12:
                return LLMResponse(content="finally done")
            return LLMResponse(tool_calls=[tool_call("list_tools")])

        reg.register(ToolSpec(
            name="list_tools", description="x",
            parameters={"type": "object", "properties": {}}, fn=lambda **_: "ok",
        ))
        reg.promote("list_tools")
        agent = Agent(MockLLMClient(handler=loop), reg, max_turns=20, allow_self_terminate=False)
        res = agent.run("go")
        assert res.content == "finally done"
        assert res.turns == 13

    def test_self_termination_ends_loop(self):
        reg = ToolRegistry()
        llm = MockLLMClient(script=[
            LLMResponse(tool_calls=[tool_call("terminate", {"summary": "all done", "reason": "finished"})]),
        ])
        agent = Agent(llm, reg)
        res = agent.run("do something")
        assert res.self_terminated
        assert res.content == "all done"
        assert res.termination_reason == "finished"

    def test_terminate_not_swallowed_by_registry(self):
        """The registry catches Exception at the tool boundary; the terminate
        signal must escape it (it subclasses BaseException for this reason)."""
        reg = ToolRegistry()
        agent = Agent(MockLLMClient(script=[
            LLMResponse(tool_calls=[tool_call("terminate", {"summary": "bye"})]),
        ]), reg)
        res = agent.run("x")
        assert res.self_terminated and res.content == "bye"

    def test_turn_cap_is_visible_when_imposed(self):
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="t", description="x",
            parameters={"type": "object", "properties": {}}, fn=lambda **_: "ok",
        ))
        reg.promote("t")
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(tool_calls=[tool_call("t")]))
        res = Agent(llm, reg, max_turns=3, allow_self_terminate=False).run("go")
        assert "turn cap of 3" in res.content


# ======================================================================
# integration: the composed ForgeAgent
# ======================================================================
class TestForgeAgentIntegration:
    def _agent(self):
        from autoforge.agent import ForgeAgent
        return ForgeAgent(MockLLMClient(handler=reverse_model))

    def test_registers_all_meta_tools(self):
        agent = self._agent()
        for name in ("forge_tool", "evolve_tool", "spawn_agent", "amend_self",
                     "set_autonomy", "list_tools", "evaluate_tool", "find_gaps"):
            assert name in agent.registry, f"{name} missing"

    def test_full_freedom_by_default(self):
        agent = self._agent()
        assert agent.policy.denied == []
        assert agent.max_turns is None

    def test_denied_tool_returns_message_not_crash(self):
        from autoforge.agent import ForgeAgent
        agent = ForgeAgent(MockLLMClient(), policy=AutonomyPolicy(may_forge_tools=False))
        r = agent.registry.call("forge_tool", {"need": "anything"})
        assert r.ok and "Denied" in r.output

    def test_self_amendment_via_tool(self):
        agent = self._agent()
        r = agent.registry.call("amend_self", {
            "target": "forge_max_rounds", "new_value": "5",
            "rationale": "some tools need more attempts",
        })
        assert r.ok and agent.forge_config.max_rounds == 5
        assert agent.selfmod.accepted_count() == 1

    def test_amendment_without_rationale_is_rejected(self):
        agent = self._agent()
        before = agent.system_prompt
        r = agent.registry.call("amend_self", {
            "target": "system_prompt", "new_value": "new prompt", "rationale": "",
        })
        assert agent.system_prompt == before
        assert "rejected" in r.output.lower()

    def test_set_autonomy_via_tool(self):
        agent = self._agent()
        r = agent.registry.call("set_autonomy", {
            "freedom": "may_access_network", "enabled": False,
            "rationale": "tightening for this task",
        })
        assert r.ok
        assert agent.policy.may_access_network is False
        assert "may_access_network" in agent.policy.denied

    def test_agent_can_read_its_own_policy(self):
        agent = self._agent()
        assert agent.policy.expose_policy_to_self is True
        assert agent.report()["policy"]["may_forge_tools"] is True

    def test_report_shape(self):
        agent = self._agent()
        rep = agent.report()
        assert "policy" in rep and "tools" in rep and "amendments" in rep

    def test_spawn_via_tool(self):
        agent = self._agent()
        r = agent.registry.call("spawn_agent", {"task": "reverse something"})
        assert r.ok and "Child" in r.output

    def test_persistence_wired(self, tmp_path):
        from autoforge.agent import ForgeAgent
        store = ToolStore(str(tmp_path / "agent.db"))
        agent = ForgeAgent(MockLLMClient(), store=store)
        agent.registry.call("set_autonomy", {
            "freedom": "may_forge_tools", "enabled": False, "rationale": "test",
        })
        events = store.get_events()
        assert any(e["kind"] == "amendment" for e in events)
        store.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-o", "addopts=", "-p", "no:cacheprovider"]))
