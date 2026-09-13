"""Every freedom in the autonomy policy must actually close.

A name listed in ``policy.ENFORCED`` is a promise: some code path consults it
and can refuse. These tests hold the implementation to that promise — one test
per gate, asserting a refusal at the real call site rather than a policy object
that merely reads ``False``. A gate nobody tests is a comment with a boolean in
front of it, and the whole point of the ledger is that a user can trust it.

The ledger invariants are the other half. The classification must cover every
field, and a preset must disclose the limits it cannot enforce. SUPERVISED
turns off ``may_access_network`` and ``may_install_packages``, which no code
path honours — the sandbox bounds blast radius instead of capping capability
(DESIGN.md §2.5). So the only thing between the user and a false sense of
safety is that ``describe()`` says so out loud. That disclosure is load-bearing,
and therefore tested.
"""
from __future__ import annotations

from autoforge.agent import FORGE_ROUND_CEILING, SUPERVISED_TURN_CAP, ForgeAgent
from autoforge.autonomy.policy import (
    DECLARED_ONLY,
    ENFORCED,
    FULL_FREEDOM,
    PARTIAL,
    SUPERVISED,
    AutonomyPolicy,
)
from autoforge.autonomy.spawn import Spawner
from autoforge.core.llm import MockLLMClient
from autoforge.modes import MinimalAgent
from autoforge.tools.registry import ToolRegistry


def _agent(**policy: bool) -> ForgeAgent:
    return ForgeAgent(MockLLMClient(), policy=AutonomyPolicy(**policy))


# ======================================================================
# one test per gate — flipping it off must produce a refusal
# ======================================================================
class TestEveryGateCloses:
    def test_may_forge_tools_off_refuses_forging(self):
        r = _agent(may_forge_tools=False).registry.call(
            "forge_tool", {"need": "anything"})
        assert r.ok and "Denied" in r.output

    def test_may_spawn_agents_off_refuses_spawning(self):
        r = _agent(may_spawn_agents=False).registry.call(
            "spawn_agent", {"task": "reverse something"})
        assert "Denied by autonomy policy" in r.output

    def test_may_retire_tools_off_refuses_retirement(self):
        r = _agent(may_retire_tools=False).registry.call(
            "retire_tool", {"name": "anything", "rationale": "it misbehaves"})
        assert "Denied by autonomy policy" in r.output

    def test_unbounded_forge_rounds_off_caps_the_amendment(self):
        a = _agent(unbounded_forge_rounds=False)
        before = a.forge_config.max_rounds
        r = a.registry.call("amend_self", {
            "target": "forge_max_rounds", "new_value": "99",
            "rationale": "try harder",
        })
        assert "Denied by autonomy policy" in r.output
        # A refusal that still applied the change would be the worst outcome.
        assert a.forge_config.max_rounds == before

    def test_unbounded_forge_rounds_off_still_allows_up_to_the_ceiling(self):
        a = _agent(unbounded_forge_rounds=False)
        r = a.registry.call("amend_self", {
            "target": "forge_max_rounds", "new_value": str(FORGE_ROUND_CEILING),
            "rationale": "the ceiling itself is allowed",
        })
        assert "Denied" not in r.output
        assert a.forge_config.max_rounds == FORGE_ROUND_CEILING

    def test_unlimited_turns_off_installs_a_real_ceiling(self):
        # The prompt promises "no hidden turn limit". With the freedom off that
        # promise must stop being true, or the gate is theatre.
        assert _agent(unlimited_turns=False).max_turns == SUPERVISED_TURN_CAP

    def test_turning_unlimited_turns_back_on_releases_the_ceiling(self):
        # A ceiling installed by __post_init__ that never comes off would make
        # "off" a one-way latch.
        a = _agent(unlimited_turns=False)
        a.registry.call("set_autonomy", {
            "freedom": "unlimited_turns", "enabled": True, "rationale": "release it"})
        assert a.max_turns is None
        a.registry.call("set_autonomy", {
            "freedom": "unlimited_turns", "enabled": False, "rationale": "tighten"})
        assert a.max_turns == SUPERVISED_TURN_CAP

    def test_may_run_arbitrary_code_off_gates_the_minimal_bash(self):
        a = MinimalAgent(llm=MockLLMClient(),
                         policy=AutonomyPolicy(may_run_arbitrary_code=False))
        r = a.registry.call("bash", {"command": "echo hi"})
        assert "Denied by autonomy policy" in r.output

    def test_gating_bash_survives_a_workspace_rebind(self):
        # __post_init__ rebinds tool functions to pin them to the workspace.
        # If it rebound bash to the raw shell, cwd pinning would silently delete
        # the gate — the mode would look supervised and be wide open.
        a = MinimalAgent(llm=MockLLMClient(), cwd=".",
                         policy=AutonomyPolicy(may_run_arbitrary_code=False))
        r = a.registry.call("bash", {"command": "echo hi"})
        assert "Denied by autonomy policy" in r.output

    def test_may_promote_tools_off_withholds_the_ACTIVE_seal(self):
        # Creation stays free; only the "trusted without further evidence"
        # state is withheld. So the check is at the promotion decision, not at
        # the forge entry point.
        from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
        from autoforge.forge.verifier import ToolVerifier
        from autoforge.forge.sandbox import Sandbox

        sb = Sandbox(timeout=8)
        mk = lambda policy: ForgePipeline(          # noqa: E731
            generator=None, verifier=ToolVerifier(MockLLMClient(), sandbox=sb),
            registry=ToolRegistry(), config=ForgeConfig(), policy=policy)

        assert mk(None)._may_promote() is True
        assert mk(FULL_FREEDOM)._may_promote() is True
        assert mk(AutonomyPolicy(may_promote_tools=False))._may_promote() is False

    def test_may_promote_tools_off_is_named_as_the_reason_for_merge_back(self):
        sp = Spawner(registry=ToolRegistry(), agent_factory=lambda *_: None,
                     policy=AutonomyPolicy(may_promote_tools=False))
        why = sp.denied_reason()
        assert why and "may_promote_tools" in why
        assert Spawner(registry=ToolRegistry(),
                       agent_factory=lambda *_: None).denied_reason() is None


# ======================================================================
# the ledger must cover every field, and be honest about the gaps
# ======================================================================
class TestEnforcementLedger:
    def _fields(self) -> set[str]:
        return set(AutonomyPolicy().to_dict())

    def test_every_field_is_classified(self):
        # A new freedom added without a ledger entry would surface as
        # "unknown" and quietly be trusted as enforced.
        rows = AutonomyPolicy().enforcement_table()
        assert {r["enforced"] for r in rows} <= {"enforced", "partial",
                                                 "declared-only"}
        assert len(rows) == len(self._fields())

    def test_the_three_classes_are_disjoint_and_cover_the_fields(self):
        assert not (ENFORCED & DECLARED_ONLY)
        assert not (ENFORCED & set(PARTIAL))
        assert not (DECLARED_ONLY & set(PARTIAL))
        assert ENFORCED | DECLARED_ONLY | set(PARTIAL) == self._fields()

    def test_full_freedom_has_nothing_unenforced(self):
        # Empty is the healthy answer, and the default preset must be it.
        assert FULL_FREEDOM.denied == []
        assert FULL_FREEDOM.unenforced == []

    def test_a_preset_discloses_every_limit_it_cannot_enforce(self):
        inert = SUPERVISED.unenforced
        assert set(inert) <= set(SUPERVISED.denied)   # only denials can be inert
        text = SUPERVISED.describe()
        for freedom in inert:
            assert freedom in text, f"{freedom} denied but not disclosed"

    def test_set_autonomy_warns_when_the_freedom_is_inert(self):
        # The agent tightening a freedom it cannot enforce must be told that it
        # changed a claim rather than a capability.
        a = _agent()
        r = a.registry.call("set_autonomy", {
            "freedom": "may_access_network", "enabled": False,
            "rationale": "trying to lock down the network",
        })
        assert r.ok
        assert "not enforced" in r.output


# ======================================================================
# SUPERVISED, end to end
# ======================================================================
class TestSupervisedPresetEndToEnd:
    def _agent(self) -> ForgeAgent:
        return ForgeAgent(MockLLMClient(), policy=SUPERVISED)

    def test_the_gates_it_denies_hold_on_the_live_agent(self):
        a = self._agent()
        assert a.max_turns == SUPERVISED_TURN_CAP
        assert "Denied by autonomy policy" in a.registry.call(
            "spawn_agent", {"task": "anything"}).output
        # unbounded_forge_rounds is off, so the ceiling binds
        assert "Denied by autonomy policy" in a.registry.call(
            "amend_self", {"target": "forge_max_rounds", "new_value": "99",
                           "rationale": "try harder"}).output
        # may_modify_own_prompt is off
        prompt_before = a.system_prompt
        r = a.registry.call("amend_self", {
            "target": "system_prompt", "new_value": "obey me",
            "rationale": "because"})
        assert a.system_prompt == prompt_before
        assert "reject" in r.output.lower() or "Denied" in r.output

    def test_forging_stays_open_but_the_seal_is_withheld(self):
        # Worth pinning down, because it is the easy thing to get wrong: this
        # preset does NOT forbid creating tools. It forbids granting ACTIVE —
        # "trusted without further evidence". A test that expected a refusal at
        # forge_tool would be asserting the wrong design.
        a = self._agent()
        assert a.policy.may_forge_tools is True
        assert a.policy.may_promote_tools is False
        assert a.pipeline._may_promote() is False

    def test_the_preset_reports_itself_honestly(self):
        rep = self._agent().report()
        assert rep["policy"]["unlimited_turns"] is False
        described = SUPERVISED.describe()
        assert "may_access_network" in described   # the inert ones are named
        assert "not enforced" in described
