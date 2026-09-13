"""A role has to change something, or it is a label.

`RoleType` shipped with five members and one effect: the name was used to look
up a node's `tools_whitelist`, so a node with an empty whitelist ran
unrestricted and "critic" meant nothing. These tests pin the two things that
make a role real — the brief that reaches the child's prompt, and the ceiling
that stops it changing the thing it is judging — and pin the totality of the
table so a sixth role cannot be added without deciding what it may do.
"""
from __future__ import annotations

import pytest

from autoforge.agent import ForgeAgent
from autoforge.autonomy.roles import (ROLE_BRIEF, ROLE_MAY, brief_for_node,
                                      refuses, role_brief, role_ceiling)
from autoforge.autonomy.spawn import Spawner
from autoforge.autonomy.topology import AgentNode, RoleType
from autoforge.core.llm import LLMResponse, MockLLMClient
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec


def tool(name: str, scope: str, fn=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} ({scope})",
        parameters={"type": "object", "properties": {}},
        fn=fn or (lambda **kw: f"{name} ran"),
        effect_signature=scope,
        source="human",
    )


def registry(**tools: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name, scope in tools.items():
        reg.register(tool(name, scope))
        reg.promote(name)
    return reg


class ProbeChild:
    """A child that does the two things a role is supposed to govern."""

    def __init__(self, reg):
        self.reg = reg
        self.brief = ""
        self.role = ""
        self.visible: list[str] = []
        self.calls: dict[str, object] = {}

    def apply_role(self, brief: str, role: str = "") -> None:
        self.brief = brief
        self.role = role

    def run(self, task: str):
        self.visible = [s["function"]["name"] for s in self.reg.schemas()]
        for name in ("reader", "writer", "runner"):
            if name in self.reg:
                self.calls[name] = self.reg.call(name, {})

        class R:
            content = "done"
        return R()


def spawn_as(role: str, reg: ToolRegistry) -> tuple[ProbeChild, object]:
    """Spawn with a role string and nothing else — no brief, no ceiling.

    Deliberate: the point is that naming a role is sufficient.
    """
    child_box: list[ProbeChild] = []

    def factory(_s, r):
        child_box.append(ProbeChild(r))
        return child_box[-1]

    sp = Spawner(registry=reg, agent_factory=factory)
    rec = sp.spawn("do the thing", role=role)
    return child_box[0], rec


# ======================================================================
# the table is total, and every entry says something
# ======================================================================
class TestTheTableIsDecided:
    def test_every_role_has_a_brief(self):
        assert set(ROLE_BRIEF) == set(RoleType)

    def test_every_role_has_a_decided_ceiling(self):
        # Totality is the point: a role that is absent would fall through to
        # "unrestricted", which is how a new role silently becomes a worker.
        assert set(ROLE_MAY) == set(RoleType)

    def test_briefs_are_specific_not_placeholders(self):
        for role, brief in ROLE_BRIEF.items():
            assert len(brief) > 60, role
        assert len(set(ROLE_BRIEF.values())) == len(RoleType)

    def test_an_unknown_role_has_no_brief_and_no_ceiling(self):
        assert role_brief("banana") == ""
        assert role_ceiling("banana") is None
        assert refuses("banana", "privileged") == ()


# ======================================================================
# the ceiling: what a role may reach
# ======================================================================
class TestTheCeiling:
    @pytest.mark.parametrize("role", [RoleType.CRITIC, RoleType.GATE])
    def test_a_judging_role_may_read_and_not_change(self, role):
        assert refuses(role, "read_only") == ()
        assert refuses(role, "local_write")
        assert refuses(role, "system")

    @pytest.mark.parametrize("role", [RoleType.COORDINATOR, RoleType.WORKER,
                                      RoleType.FORGE])
    def test_a_doing_role_may_reach_the_world(self, role):
        for scope in ("read_only", "local_write", "system", "privileged"):
            assert refuses(role, scope) == (), scope

    def test_an_undeclared_scope_is_not_assumed_harmless(self):
        # Mirrors the confirmation gate: if it cannot be told what a tool does,
        # a role that may not change things does not get to guess.
        assert refuses(RoleType.CRITIC, "")
        assert refuses(RoleType.CRITIC, "who knows")

    def test_the_denied_capabilities_are_named(self):
        missing = refuses(RoleType.CRITIC, "local_write")
        assert "filesystem_write" in missing


# ======================================================================
# the ceiling bites at the registry, not just in a table
# ======================================================================
class TestTheCeilingBites:
    @pytest.mark.parametrize("role", [RoleType.CRITIC, RoleType.GATE])
    def test_a_judging_child_cannot_see_or_run_a_writer(self, role):
        child, _ = spawn_as(role.value, registry(reader="read_only",
                                                 writer="local_write",
                                                 runner="system"))
        # Not visible: the model never asks for what it cannot have.
        assert "reader" in child.visible
        assert "writer" not in child.visible
        assert "runner" not in child.visible
        # And not runnable if it asks anyway. Both ends, or the model learns to
        # ask for things it cannot have.
        assert child.calls["reader"].ok
        assert not child.calls["writer"].ok
        assert not child.calls["runner"].ok
        assert "may not use" in child.calls["writer"].error

    def test_a_working_child_can_still_work(self):
        child, _ = spawn_as(RoleType.WORKER.value, registry(
            reader="read_only", writer="local_write", runner="system"))
        assert {"reader", "writer", "runner"} <= set(child.visible)
        assert all(child.calls[n].ok for n in ("reader", "writer", "runner"))

    def test_a_refusal_is_recorded_rather_than_swallowed(self):
        """A refusal is the evidence the ceiling is live rather than decorative,
        so it is reported instead of quietly returning nothing."""
        child, rec = spawn_as(RoleType.CRITIC.value,
                              registry(reader="read_only", writer="local_write"))
        assert rec.refusals == ["writer"]

    def test_the_ceiling_is_derived_without_the_caller(self):
        """The bug this prevents: a ceiling only the agent.py call site knows
        about is a ceiling every other caller of `spawn` silently lacks. This
        spawns with a role string and nothing else."""
        child, rec = spawn_as(RoleType.CRITIC.value,
                              registry(reader="read_only", writer="local_write"))
        assert rec.to_dict()["brief_chars"] > 0      # brief derived too
        assert not child.calls["writer"].ok


# ======================================================================
# the brief: what the child is told it is
# ======================================================================
class TestTheBrief:
    def test_the_brief_names_what_the_role_is_and_keeps_the_designers_hint(self):
        node = AgentNode(id="rev", role=RoleType.CRITIC,
                         system_prompt_hint="focus on the retry path")
        brief = brief_for_node(node)
        assert "critic" in brief.lower()
        assert "focus on the retry path" in brief      # the hint was inert before

    def test_a_child_is_handed_the_brief(self):
        child, rec = spawn_as(RoleType.CRITIC.value, registry(reader="read_only"))
        assert child.brief == role_brief(RoleType.CRITIC)
        assert child.role == "critic"
        assert rec.brief == child.brief

    def test_the_trace_shows_the_brief_reached_someone(self):
        child, rec = spawn_as(RoleType.GATE.value, registry(reader="read_only"))
        assert rec.to_dict()["brief_chars"] == len(role_brief(RoleType.GATE)) > 0

    def test_a_real_child_puts_the_brief_in_the_prompt_it_sends(self):
        """The end of the chain: not "the record says a role", but "the string
        the model receives contains the directive"."""
        a = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        a.apply_role(role_brief(RoleType.FORGE), RoleType.FORGE.value)
        prompt = a._effective_prompt()
        assert "You are the forge" in prompt
        assert "Your role in this team (forge)" in prompt

    def test_a_child_without_a_role_sends_no_role_section(self):
        a = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
        assert "Your role in this team" not in a._effective_prompt()
