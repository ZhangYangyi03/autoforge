"""Tests for the agent-topology module: self-designing multi-agent architectures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.autonomy.topology import (
    AgentNode,
    RoleType,
    Topology,
    TopologyDesigner,
    TopologyEdge,
)
from autoforge.core.llm import LLMResponse, MockLLMClient


def designer(llm: MockLLMClient) -> TopologyDesigner:
    return TopologyDesigner(llm=llm)


def good_topology_json() -> str:
    return (
        '{"rationale": "coordinator delegates to a worker and a critic", '
        '"nodes": ['
        '{"id": "coord", "role": "coordinator", "system_prompt_hint": "decompose"} ,'
        '{"id": "worker", "role": "worker", "system_prompt_hint": "execute"},'
        '{"id": "critic", "role": "critic", "system_prompt_hint": "review"}'
        '], '
        '"edges": ['
        '{"source": "coord", "target": "worker", "channel": "tool_call"},'
        '{"source": "worker", "target": "critic", "channel": "pipe"}'
        ']}'
    )


class TestTopology:
    def test_single_agent_default(self):
        t = Topology.single_agent()
        assert len(t.nodes) == 1
        assert t.nodes[0].role == RoleType.COORDINATOR
        assert t.validate() == []

    def test_validate_catches_bad_edge(self):
        t = Topology.single_agent()
        t.connect("ghost", "also_ghost", "tool_call")
        errors = t.validate()
        assert len(errors) == 2

    def test_validate_passes_valid_topology(self):
        t = Topology(
            nodes=[AgentNode(id="a", role=RoleType.WORKER),
                   AgentNode(id="b", role=RoleType.CRITIC)]
        )
        t.connect("a", "b", "pipe")
        assert t.validate() == []

    def test_add_and_connect_fluent(self):
        t = (Topology()
             .add(AgentNode(id="coord", role=RoleType.COORDINATOR))
             .add(AgentNode(id="w", role=RoleType.WORKER))
             .connect("coord", "w"))
        assert len(t.nodes) == 2 and len(t.edges) == 1

    def test_to_dict_shape(self):
        t = Topology.single_agent("be concise")
        d = t.to_dict()
        assert d["nodes"][0]["id"] == "main"
        assert "rationale" in d and "trials" in d


class TestTopologyDesigner:
    def test_design_parses_triple_agent_topology(self):
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=good_topology_json()))
        t = designer(llm).design("solve this hard problem")
        assert len(t.nodes) == 3
        roles = {n.role for n in t.nodes}
        assert RoleType.COORDINATOR in roles
        assert RoleType.WORKER in roles
        assert RoleType.CRITIC in roles
        assert len(t.edges) == 2
        assert t.validate() == []

    def test_design_degradation_on_unparseable_output(self):
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="I refuse."))
        t = designer(llm).design("anything")
        assert len(t.nodes) == 1  # graceful single-agent fallback

    def test_design_degradation_on_invalid_topology(self):
        # edges referencing nodes that don't exist -> must degrade
        payload = (
            '{"rationale":"x","nodes":[{"id":"a","role":"worker"}],'
            '"edges":[{"source":"a","target":"missing","channel":"tool_call"}]}'
        )
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=payload))
        t = designer(llm).design("task")
        assert len(t.nodes) == 1  # degraded to single agent

    def test_design_includes_failure_report(self):
        seen = {}

        def handler(messages, tools, **kw):
            user_text = next((m.content for m in messages if m.role == "user"), "")
            seen["user"] = user_text
            return LLMResponse(content=good_topology_json())

        llm = MockLLMClient(handler=handler)
        designer(llm).design("task", failure_report="worker crashed on edge case")
        assert "worker crashed" in seen["user"]

    def test_analyse_parses_bottleneck(self):
        payload = '{"bottleneck":"worker","root_cause":"too much work","suggested_change":"split"}'
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=payload))
        result = designer(llm).analyse(Topology.single_agent(), [{"kind": "error"}])
        assert result["bottleneck"] == "worker"
        assert "split" in result["suggested_change"]

    def test_analyse_degrades_on_bad_output(self):
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content="nope"))
        result = designer(llm).analyse(Topology.single_agent(), [])
        assert result["root_cause"] == "analysis failed"

    def test_parse_stores_prompt_hints_and_whitelist(self):
        llm = MockLLMClient(handler=lambda m, t, **k: LLMResponse(content=good_topology_json()))
        t = designer(llm).design("task")
        worker = next(n for n in t.nodes if n.id == "worker")
        assert worker.system_prompt_hint == "execute"
        assert worker.tools_whitelist == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-o", "addopts=", "-p", "no:cacheprovider"]))
