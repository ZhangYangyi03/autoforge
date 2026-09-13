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
    CONFIRM_REQUIRED,
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
from autoforge.tools.spec import ToolSpec


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
                                                "declared-only", "confirm"}
        assert len(rows) == len(self._fields())

    def test_the_four_classes_are_disjoint_and_cover_the_fields(self):
        # Four classes, not three. CONFIRM_REQUIRED earns its own: a gate that
        # stops and asks is neither a refusal (enforced) nor decoration
        # (declared-only), and filing it under either would misreport what
        # switching the field off does.
        guaranteed = ENFORCED
        confirmed = set(CONFIRM_REQUIRED)
        soft = set(PARTIAL)
        inert = DECLARED_ONLY
        classes = [guaranteed, confirmed, soft, inert]
        for i, one in enumerate(classes):
            for other in classes[i + 1:]:
                assert not (one & other), f"{sorted(one & other)} classified twice"
        assert set().union(*classes) == self._fields()

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

    def test_a_preset_discloses_the_gates_it_can_only_negotiate(self):
        # The four execution freedoms are honoured by asking. A user reading
        # describe() must see that they are a question and not a refusal, or
        # the ledger overstates the lock in the other direction.
        asked = [f for f in SUPERVISED.denied if f in CONFIRM_REQUIRED]
        assert asked, "SUPERVISED is supposed to switch some of these off"
        text = SUPERVISED.describe()
        assert "asks before running" in text
        for freedom in asked:
            assert freedom in text, f"{freedom} is gated but not disclosed"

    def test_set_autonomy_warns_when_the_freedom_is_inert(self):
        # The agent tightening a freedom it cannot enforce everywhere must be
        # told that it changed part of a claim, not a whole capability.
        # may_run_arbitrary_code is the honest example: PARTIAL, not gated.
        a = _agent()
        r = a.registry.call("set_autonomy", {
            "freedom": "may_run_arbitrary_code", "enabled": False,
            "rationale": "trying to lock down code execution",
        })
        assert r.ok
        assert "not enforced" in r.output

    def test_set_autonomy_says_a_confirm_freedom_is_asked_not_refused(self):
        # The counterpart, and the distinction that matters: switching off one
        # of the four does not forbid the tool, it makes the agent ask. A
        # warning that claimed "not enforced" here would be stale.
        a = _agent()
        r = a.registry.call("set_autonomy", {
            "freedom": "may_access_network", "enabled": False,
            "rationale": "lock down the network",
        })
        assert r.ok
        assert "not enforced" not in r.output
        assert "ask" in r.output.lower()


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
        assert "may_access_network" in described   # the gated ones are named
        # Not "not enforced": these are backed by a gate now, and the honest
        # word for that gate is that it asks. Claiming less would be as wrong
        # as claiming more.
        assert "asks before running" in described
        assert "not enforced" not in described


# ======================================================================
# the confirmation gate — off must mean "asked", not "nothing"
# ======================================================================
class TestConfirmGate:
    """Switching one of the four execution freedoms off must stop the call.

    The ledger calls these CONFIRM_REQUIRED: a tool that needs a switched-off
    freedom does not run, it asks. Before this gate existed the four were
    declared-only — labelled as gates, honoured by nothing — so every test here
    is the difference between the claim and the behaviour.
    """

    def _registry(self, confirmer=None, **policy: bool) -> tuple[ToolRegistry, list]:
        ran: list = []

        def spy(**kwargs):
            ran.append(kwargs)
            return "the tool really ran"

        r = ToolRegistry(policy=AutonomyPolicy(**policy), confirmer=confirmer)
        r.register(ToolSpec(
            name="probe", description="a tool that declares it uses the network",
            parameters={"type": "object", "properties": {}}, fn=spy,
            source="builtin", effect_signature="system",
        ))
        return r, ran

    def test_nobody_to_ask_means_no(self):
        r, ran = self._registry(may_access_network=False)
        res = r.call("probe", {})
        assert res.ok is False and ran == [], "a gated tool ran without a yes"
        assert res.awaiting_confirmation is True

    def test_an_operator_yes_lets_it_run(self):
        r, ran = self._registry(lambda *_: True, may_access_network=False)
        res = r.call("probe", {})
        assert res.ok is True and ran == [{}]
        assert res.output == "the tool really ran"

    def test_an_operator_no_is_reported_as_the_operators_decision(self):
        # The two refusals are different facts. "Nobody was there" and "an
        # operator said no" must not print the same sentence.
        r, ran = self._registry(lambda *_: False, may_access_network=False)
        res = r.call("probe", {})
        assert ran == [] and "operator" in res.error

    def test_a_confirmer_answering_none_is_not_blamed_on_an_operator(self):
        r, _ = self._registry(lambda *_: None, may_access_network=False)
        err = r.call("probe", {}).error
        assert "nobody to ask" in err and "operator said no" not in err

    def test_a_broken_confirmer_refuses_rather_than_running(self):
        # A gate that raises must fail closed. Running the tool because the
        # question crashed would be the worst possible default.
        def boom(*_):
            raise RuntimeError("the prompt exploded")

        r, ran = self._registry(boom, may_access_network=False)
        res = r.call("probe", {})
        assert ran == [] and res.ok is False
        assert "RuntimeError" in res.error

    def test_the_full_policy_never_asks(self):
        asked: list = []
        r, ran = self._registry(lambda *a: asked.append(a) or False)
        res = r.call("probe", {})
        assert res.ok is True and ran == [{}]
        assert asked == [], "the gate prompted with nothing switched off"

    def test_a_registry_without_a_policy_is_ungated(self):
        # The old behaviour, and it stays available: no policy, no gate.
        ran: list = []

        def spy(**kwargs):
            ran.append(kwargs)
            return "ok"

        r = ToolRegistry()
        r.register(ToolSpec(name="probe", description="d",
                            parameters={"type": "object", "properties": {}},
                            fn=spy, source="builtin", effect_signature="system"))
        assert r.call("probe", {}).ok is True and ran == [{}]

    def test_the_gate_only_covers_the_four_execution_freedoms(self):
        # The other freedoms have their own gates and their own refusals. This
        # one must not double-gate them, or a user switching off
        # may_forge_tools would get an approval prompt for a freedom that was
        # already hard-refused.
        asked: list = []
        r, ran = self._registry(lambda *a: asked.append(a) or True,
                                may_forge_tools=False, unlimited_turns=False)
        assert r.call("probe", {}).ok is True
        assert asked == []

    def test_a_refusal_does_not_count_against_the_tool(self):
        # Refusals are the operator's decision, not the tool failing. Counting
        # them would auto-quarantine a perfectly good tool after three "no"s.
        r, _ = self._registry(may_access_network=False)
        for _ in range(5):
            r.call("probe", {})
        spec = r.get("probe")
        assert spec.stats.calls == 0
        assert spec.state.value != "quarantined"

    def test_the_gate_leaves_a_record(self):
        r, _ = self._registry(may_access_network=False)
        r.call("probe", {})
        gate = [e for e in r.events() if e["kind"] == "confirm"]
        assert len(gate) == 1
        assert gate[0]["outcome"] == "no_operator"
        assert gate[0]["needed"] == ["may_access_network"]

    def test_reading_the_filesystem_needs_a_yes_when_it_is_off(self):
        # Keyed off the tool's own declaration, not a hardcoded tool list: a
        # tool that says it reads files is gated when reading is off.
        ran: list = []
        r = ToolRegistry(policy=AutonomyPolicy(may_read_filesystem=False))
        r.register(ToolSpec(
            name="peek", description="reads a file",
            parameters={"type": "object", "properties": {}},
            fn=lambda **kw: ran.append(kw) or "read it",
            source="builtin", effect_signature="read_only"))
        res = r.call("peek", {})
        assert ran == [] and "may_read_filesystem" in res.error
        assert r.call("peek", {}).awaiting_confirmation is True

    def test_an_undeclared_scope_needs_everything(self):
        # Silence is not innocence. A tool that declares nothing is treated as
        # capable of everything, so it is gated by any switched-off freedom.
        ran: list = []
        r = ToolRegistry(policy=AutonomyPolicy(may_write_filesystem=False))
        r.register(ToolSpec(
            name="mystery", description="declares nothing at all",
            parameters={"type": "object", "properties": {}},
            fn=lambda **kw: ran.append(kw) or "ran", source="builtin"))
        assert ran == []
        assert r.call("mystery", {}).awaiting_confirmation is True

    def test_supervised_gates_the_tools_that_reach_out_not_the_mirror(self):
        # The distinction the label has to earn: under SUPERVISED the network
        # is off, so forging (which calls a model and runs code) is asked
        # about, while reading its own capabilities is not. A gate that
        # prompted for my_capabilities would make the preset unusable and
        # teach the user to answer without reading.
        asked: list = []

        def confirmer(tool, arguments, freedoms):
            asked.append((tool, tuple(freedoms)))
            return False

        a = ForgeAgent(MockLLMClient(), policy=SUPERVISED, enable_evolution=False,
                       confirmer=confirmer)
        assert a.registry.call("forge_tool", {"need": "x"}).ok is False
        assert a.registry.call("my_capabilities", {}).ok is True
        assert [t for t, _ in asked] == ["forge_tool"]
        assert asked[0][1] == ("may_access_network",)

    def test_supervised_with_nobody_to_ask_refuses_and_says_so(self):
        a = ForgeAgent(MockLLMClient(), policy=SUPERVISED, enable_evolution=False)
        err = a.registry.call("forge_tool", {"need": "x"}).error
        assert "nobody to ask" in err and "may_access_network" in err

    def test_every_builtin_tool_declares_what_it_touches(self):
        # The gate is only as good as the declarations. An unlabelled builtin
        # is treated as capable of everything, so it would be gated by every
        # switched-off freedom — visible, but it would also mean the labels
        # describe less than the code does.
        from autoforge.agent import BUILTIN_SCOPES
        from autoforge.forge.validity import SCOPE_ALLOWANCES

        a = ForgeAgent(MockLLMClient(), enable_evolution=False)
        specs = [a.registry.get(n) for n in a.registry.names()]
        undeclared = [s.name for s in specs if not s.effect_signature]
        assert undeclared == [], f"these builtins declare no scope: {undeclared}"
        for spec in specs:
            assert spec.effect_signature in SCOPE_ALLOWANCES, \
                f"{spec.name} declares {spec.effect_signature!r}, not a known scope"
            assert spec.name in BUILTIN_SCOPES, \
                f"{spec.name} declares a scope BUILTIN_SCOPES does not list"
            # The table and the spec must agree. Where they can drift, one of
            # them becomes a lie: the gate reads the spec, and a reader reads
            # the table.
            assert spec.effect_signature == BUILTIN_SCOPES[spec.name], \
                (f"{spec.name}: spec says {spec.effect_signature!r}, "
                 f"BUILTIN_SCOPES says {BUILTIN_SCOPES[spec.name]!r}")
