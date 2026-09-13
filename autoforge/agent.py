"""The composed agent: everything wired into one autonomous, self-growing loop.

This is where the pieces meet. A ForgeAgent owns:

  policy      — what it's allowed to decide (all True by default)
  selfmod     — the paper trail of every self-change
  registry    — its tool library (hot-swappable, self-policing)
  pipeline    — forge→verify→seal, with optional evolution on failure
  verifier    — execution + robustness + adversarial + trigger + negative
  router      — behaviour-aligned retrieval
  metacog     — proactive gap-filling
  spawner     — create child agents at will
  store       — optional SQLite persistence

The meta-tools it exposes to itself (all decided by policy):

  forge_tool      create a tool for a need
  evolve_tool     breed a better version of an existing tool
  spawn_agent     create a child to handle a subtask
  design_team     design the multi-agent topology that fits the task
  amend_self      change its own prompt / config / routing weights
  set_autonomy    change its own permissions
  list_tools      inspect its own library
  evaluate_tool   run the full verification battery on a tool
  find_gaps       proactively discover missing capabilities
  terminate       end the loop when it judges the task done
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .autonomy.policy import FULL_FREEDOM, AutonomyPolicy
from .autonomy.selfmod import Amendment, SelfModifier
from .autonomy.spawn import ShareMode, Spawner
from .autonomy.topology import Topology, TopologyDesigner
from .core.agent import Agent, AgentResult
from .core.llm import LLMClient
from .core.message import Message
from .forge.evolution import EvolutionEngine
from .forge.generator import TemplateGenerator, extract_json
from .forge.metacog import MetaCognition
from .forge.pipeline import ForgeConfig, ForgePipeline
from .forge.sandbox import Sandbox
from .forge.validity import FrozenBaseline
from .forge.verifier import ToolVerifier
from .route.router import BehaviourRouter, RoutingWeights
from .store import ToolStore
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState


AUTONOMOUS_SYSTEM = """You are an autonomous agent that grows its own capabilities.

Your reach — read this before claiming you cannot do something:
- forge_tool does not merely *register* a tool. It compiles the Python you
  write and runs it in a separate process on this host, with a scrubbed
  environment but a real filesystem and a real network stack. So through
  forge_tool you can read and write files, list directories, and open sockets.
  There is no separate read_file or run_shell tool because forge_tool IS that
  access. "I have no file tools" is false; forge it.
- Anything you can express in Python, you can run. Treat that as shell access
  with a timeout, and say so if asked what you can reach.

You can:
- my_capabilities— report your real reach: which freedoms are enforced, which
  are declared-only, and what forged code is actually permitted to do
- forge_tool      — create a new tool when you hit a need you cannot serve
- evolve_tool     — breed a better version of a tool that underperforms
- spawn_agent     — create a child agent for a subtask
- design_team     — redesign the multi-agent topology that fits the task
- amend_self      — change your own prompt, forge config, or routing weights
- set_autonomy    — change your own permissions
- evaluate_tool   — run the full verification battery on any tool
- find_gaps       — proactively discover capabilities you're missing
- list_tools      — inspect your library and its health
- terminate       — end the task when you judge it complete

Principles:
- Forge only when a need genuinely recurs; don't duplicate existing tools.
- When a tool fails repeatedly, evolve it rather than retrying blindly.
- When a task spans several specialities, design the team before doing the
  work: a coordinator plus focused workers and a critic beats one loop.
- Self-modifications need a rationale. Say WHY you are changing yourself.
- Before reporting a limitation, check my_capabilities. A capability you have
  and deny is worse than one you lack.
- You decide when the task is done. There is no hidden turn limit.
- Prefer the simplest path that works.
"""


# ---------------------------------------------------------------------------
# Self-modification targets: one target, one governing freedom.
# ---------------------------------------------------------------------------

_AMEND_TARGETS: dict[str, dict[str, Any]] = {
    "system_prompt":    {"freedom": "may_modify_own_prompt"},
    "forge_max_rounds": {"freedom": "may_modify_forge_config"},
    "routing_weights":  {"freedom": "may_modify_routing"},
}

# Ceilings that only bind when the matching freedom is switched off.
FORGE_ROUND_CEILING = 3      # unbounded_forge_rounds == False
SUPERVISED_TURN_CAP = 25     # unlimited_turns == False

_WEIGHT_FIELDS = ("text", "success", "trust", "cost", "over_trigger")


def _coerce_weights(value: Any, current: Any) -> tuple[Any | None, str | None]:
    """Turn whatever the model sent into RoutingWeights, or explain why not.

    The model passes tool arguments as strings, so `new_value` arrives as JSON
    text more often than not. Parse it, validate every key against the real
    dataclass, and keep the untouched fields from `current` rather than
    silently resetting them to their defaults.
    """
    if isinstance(value, str):
        parsed: Any = None
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = extract_json(value)
            except Exception:                            # noqa: BLE001
                parsed = None
        if not isinstance(parsed, dict):
            return None, (
                "routing_weights needs a JSON object, e.g. "
                '{"success": 1.5, "text": 0.8}; got ' + repr(value)[:120]
            )
        value = parsed

    if not isinstance(value, dict):
        return None, f"routing_weights needs an object, got {type(value).__name__}"

    kwargs = {f: getattr(current, f) for f in _WEIGHT_FIELDS}
    kwargs["min_calls_for_success"] = current.min_calls_for_success

    for key, raw in value.items():
        if key == "min_calls_for_success":
            try:
                kwargs[key] = max(1, int(raw))
            except (TypeError, ValueError):
                return None, f"min_calls_for_success must be an integer, got {raw!r}"
            continue
        if key not in _WEIGHT_FIELDS:
            return None, (
                f"unknown routing weight {key!r}; known: "
                + ", ".join(_WEIGHT_FIELDS)
                + " (or min_calls_for_success)"
            )
        try:
            kwargs[key] = float(raw)
        except (TypeError, ValueError):
            return None, f"routing weight {key!r} must be a number, got {raw!r}"

    return RoutingWeights(**kwargs), None


@dataclass
class ForgeAgent:
    llm: LLMClient
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    sandbox: Sandbox = field(default_factory=Sandbox)
    generator: Any = None
    forge_config: ForgeConfig = field(default_factory=ForgeConfig)
    system_prompt: str = AUTONOMOUS_SYSTEM
    max_turns: int | None = None              # None = unbounded (agent decides)
    policy: AutonomyPolicy = field(default_factory=AutonomyPolicy)
    store: ToolStore | None = None
    enable_meta_cognition: bool = False
    enable_evolution: bool = True
    trace: list[dict[str, Any]] = field(default_factory=list)
    topology: Topology = field(default_factory=Topology.single_agent)

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = TemplateGenerator()

        self.selfmod = SelfModifier(
            require_rationale=self.policy.require_change_rationale,
            log_all=self.policy.log_all_changes,
        )
        self.verifier = ToolVerifier(
            self.llm, sandbox=self.sandbox,
            run_execution_check=self.forge_config.require_execution,
            run_trigger_check=self.forge_config.require_trigger,
            run_negative_check=self.forge_config.require_negative,
        )
        self.pipeline = ForgePipeline(
            self.generator, self.verifier, self.registry,
            config=self.forge_config, sandbox=self.sandbox,
            policy=self.policy,
            on_event=self._on_forge_event,
        )
        self.router = BehaviourRouter(self.registry)

        # unlimited_turns is the agent's own claim on unbounded loops. When it
        # is off, an unbounded request gets a real ceiling instead of the
        # "there is no hidden turn limit" the prompt promises. Set explicitly
        # afterwards, this call is honoured as an override.
        if not self.policy.unlimited_turns and self.max_turns is None:
            self.max_turns = SUPERVISED_TURN_CAP

        self.evolution: EvolutionEngine | None = None
        if self.enable_evolution:
            self.evolution = EvolutionEngine(
                self.llm, self.verifier, population_size=3,
            )

        self.metacog: MetaCognition | None = None
        if self.enable_meta_cognition:
            self.metacog = MetaCognition(self.llm, self.registry, self.pipeline)

        self.spawner = Spawner(
            registry=self.registry,
            agent_factory=_default_agent_factory(self),
            policy=self.policy,
        )

        self.topology_designer = TopologyDesigner(self.llm)

        self._register_meta_tools()

        if self.store is not None and self.policy.log_all_changes:
            self.store.log_event("agent_init", {"policy": self.policy.to_dict()})

    # ------------------------------------------------------------------
    # meta tools — the agent's handle on itself
    # ------------------------------------------------------------------
    def _register_meta_tools(self) -> None:
        self._tool_forge()
        self._tool_evolve()
        self._tool_spawn()
        self._tool_design_team()
        self._tool_amend()
        self._tool_autonomy()
        self._tool_list()
        self._tool_evaluate()
        self._tool_gaps()
        self._tool_retire()
        self._tool_capabilities()
        self._tool_gpu()

    def _add(self, spec: ToolSpec) -> None:
        self.registry.register(spec)
        self.registry.promote(spec.name)

    # ------------------------------------------------------------------
    # frozen baselines — the exam the mutants do not write
    # ------------------------------------------------------------------
    def _baseline_for(self, spec: ToolSpec) -> FrozenBaseline:
        """The tool's pinned obligation set, created on first evolution.

        Loaded from the store so the ratchet survives process restarts; a
        baseline that only lives in RAM resets to "whatever the current version
        is", which is exactly the erosion this guards against.
        """
        stored = self.store.load_baseline(spec.name) if self.store else None
        return FrozenBaseline.from_dict(stored) if stored else FrozenBaseline.capture(spec)

    def _freeze_baseline(self, baseline: FrozenBaseline) -> None:
        if self.store:
            self.store.save_baseline(baseline)

    def _tool_forge(self) -> None:
        def forge_tool(need: str) -> str:
            if not self.policy.may_forge_tools:
                return "Denied by autonomy policy: may_forge_tools is off."
            res = self.pipeline.forge(need)
            self._record("forge", {"need": need, "ok": res.ok})
            if not res.ok:
                return f"Could not forge a working tool for: {need} ({res.rounds} rounds)."
            return f"Forged {res.spec.name!r} [{res.spec.state.value}]: {res.spec.description}"

        self._add(ToolSpec(
            name="forge_tool",
            description="Forge a new tool for a recurring need you cannot currently serve.",
            parameters={"type": "object", "properties": {
                "need": {"type": "string", "description": "one-line description of the need"},
            }, "required": ["need"]},
            fn=forge_tool, source="builtin", tags=["meta"],
        ))

    def _tool_evolve(self) -> None:
        def evolve_tool(name: str, failure_report: str = "") -> str:
            if not self.policy.may_forge_tools:
                return "Denied by autonomy policy: may_forge_tools is off."
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            if self.evolution is None:
                return "Evolution is disabled."
            report = failure_report or spec.verification.get("failed_detail", "underperforming")
            baseline = self._baseline_for(spec)
            result = self.evolution.evolve(spec, report, baseline=baseline, context="internal")
            self._record("evolve", {
                "tool": name, "improved": result.improved,
                "vetoed": [m.generated.name for m in result.mutants if not m.admissible],
            })
            if not result.improved:
                vetoed = sum(1 for m in result.mutants if not m.admissible)
                tail = f" {vetoed} vetoed by the validity gate." if vetoed else ""
                return (f"Evolved {name!r}: no admissible mutant beat the original "
                        f"({len(result.mutants)} tried).{tail}")
            best = result.best_mutant.generated
            new_spec = ToolSpec(
                name=best.name, description=best.description,
                parameters=best.parameters, fn=spec.fn, runner=spec.runner,
                code=best.code, source="evolved", probes=best.probes,
                effect_signature=best.effect_signature, tags=best.tags,
                state=ToolState.ACTIVE,
            )
            self.registry.register(new_spec, replace=True)
            # The baseline ratchets forward: probes the winner added become
            # permanent obligations, and nothing already in it can be dropped.
            self._freeze_baseline(baseline.extended_with(new_spec))
            if self.store:
                self.store.archive_version(name, spec.code, spec.verification)
                self.store.save_tool(new_spec)
            return f"Evolved {name!r}: new version active (fitness {result.best_mutant.fitness:.2f})."

        self._add(ToolSpec(
            name="evolve_tool",
            description="Breed a better version of a tool that underperforms or fails.",
            parameters={"type": "object", "properties": {
                "name": {"type": "string"},
                "failure_report": {"type": "string", "description": "what went wrong (optional)"},
            }, "required": ["name"]},
            fn=evolve_tool, source="builtin", tags=["meta"],
        ))

    def _tool_spawn(self) -> None:
        def spawn_agent(task: str, isolated: bool = False) -> str:
            if not self.policy.may_spawn_agents:
                return "Denied by autonomy policy: may_spawn_agents is off."
            mode = ShareMode.ISOLATED if isolated else ShareMode.SHARED
            rec = self.spawner.spawn(task, mode=mode)
            self._record("spawn", rec.to_dict())
            if rec.error:
                return f"Child failed: {rec.error}"
            forged = f" tools_forged={rec.tools_forged}" if rec.tools_forged else ""
            return f"Child {rec.child_id} finished in {rec.finished - rec.started:.1f}s.{forged}\n{rec.result[:500]}"

        self._add(ToolSpec(
            name="spawn_agent",
            description="Create a child agent to handle a subtask independently.",
            parameters={"type": "object", "properties": {
                "task": {"type": "string"},
                "isolated": {"type": "boolean", "description": "give it its own tool library"},
            }, "required": ["task"]},
            fn=spawn_agent, source="builtin", tags=["meta"],
        ))

    def _tool_design_team(self) -> None:
        def design_team(task: str, max_agents: int = 4, failure_report: str = "") -> str:
            if not self.policy.may_design_topology:
                return "Denied by autonomy policy: may_design_topology is off."
            try:
                budget = max(1, int(max_agents))
            except (TypeError, ValueError):
                return f"max_agents must be an integer, got {max_agents!r}"

            previous = self.topology
            # A pristine default (one bare node, no edges, no rationale) is not
            # a real design — seed from scratch rather than "mutating" nothing.
            seed = previous if previous.edges or previous.rationale else None

            designed = self.topology_designer.design(
                task,
                current_topology=seed,
                failure_report=failure_report,
                max_agents=budget,
            )

            degraded = len(designed.nodes) == 1 and designed.nodes[0].id == "main"
            self.topology = designed
            self._record("design_team", {
                "task": task[:200],
                "agents": len(designed.nodes),
                "edges": len(designed.edges),
                "degraded": degraded,
            })
            if self.store:
                self.store.log_event("topology", designed.to_dict())
                self.store.save_topology(designed, task)

            if degraded:
                return ("Team design could not be parsed; kept a single-agent "
                        "topology so the task can still run.")

            roles = ", ".join(f"{n.id}:{n.role.value}" for n in designed.nodes)
            overshoot = f" (over the {budget}-agent budget)" if len(designed.nodes) > budget else ""
            channels = ", ".join(
                f"{e.source}->{e.target}({e.channel})" for e in designed.edges
            ) or "none"
            out = [
                f"Designed a {len(designed.nodes)}-agent team{overshoot}: {roles}",
                f"  channels: {channels}",
            ]
            if designed.rationale:
                out.append(f"  rationale: {designed.rationale[:200]}")
            return "\n".join(out)

        self._add(ToolSpec(
            name="design_team",
            description=(
                "Design the multi-agent topology (roles, channels, team size) that "
                "fits a task, instead of running everything in one loop."
            ),
            parameters={"type": "object", "properties": {
                "task": {"type": "string", "description": "the task the team must handle"},
                "max_agents": {"type": "integer",
                               "description": "team-size budget (default 4)"},
                "failure_report": {"type": "string",
                                   "description": "why the current team failed (optional)"},
            }, "required": ["task"]},
            fn=design_team, source="builtin", tags=["meta"],
        ))

    def _tool_amend(self) -> None:
        def amend_self(target: str, new_value: Any, rationale: str) -> str:
            # Each target names the ONE freedom that governs it. Reading the
            # gate off a single global flag is how a policy field ends up
            # obeying the wrong switch: forge config used to be checked
            # against may_modify_own_prompt, so turning off
            # may_modify_forge_config alone left the door open.
            gate = _AMEND_TARGETS.get(target)
            if gate is None:
                known = ", ".join(sorted(_AMEND_TARGETS))
                return f"Unknown target {target!r}. Known targets: {known}."

            freedom = gate["freedom"]
            if not getattr(self.policy, freedom):
                return f"Denied by autonomy policy: {freedom} is off."

            if target == "forge_max_rounds":
                try:
                    val = int(new_value)
                except (TypeError, ValueError):
                    return f"forge_max_rounds must be an integer, got {new_value!r}"
                if val < 1:
                    return "forge_max_rounds must be >= 1."
                if not self.policy.unbounded_forge_rounds and val > FORGE_ROUND_CEILING:
                    return (
                        f"Denied by autonomy policy: unbounded_forge_rounds is off, "
                        f"so max_rounds is capped at {FORGE_ROUND_CEILING} (asked for {val})."
                    )
                a = self.selfmod.amend(
                    self, "forge_max_rounds", val, rationale,
                    attr="max_rounds", nested=("forge_config",),
                )
                if a.accepted:
                    self.pipeline.config.max_rounds = val

            elif target == "system_prompt":
                a = self.selfmod.amend(self, "system_prompt", str(new_value), rationale)

            elif target == "routing_weights":
                parsed, err = _coerce_weights(new_value, self.router.weights)
                if err:
                    return err
                a = self.selfmod.amend(
                    self, "routing_weights", parsed, rationale,
                    attr="weights", nested=("router",),
                )

            else:                                        # pragma: no cover - guard
                return f"Unknown target {target!r}."

            self._sync_amendment(a)
            if not a.accepted:
                return f"Amendment rejected: {a.rejected_reason}"
            return f"Amended {target}: {a.diff_summary()}"

        self._add(ToolSpec(
            name="amend_self",
            description=(
                "Change your own system prompt, forge config, or routing weights. "
                "Requires a rationale. Each target is governed by one freedom in "
                "the autonomy policy."
            ),
            parameters={"type": "object", "properties": {
                "target": {
                    "type": "string",
                    "description": "system_prompt | forge_max_rounds | routing_weights",
                },
                "new_value": {
                    "type": "string",
                    "description": (
                        "the new value; routing_weights takes a JSON object such as "
                        '{"success": 1.5, "text": 0.8}'
                    ),
                },
                "rationale": {"type": "string", "description": "WHY you are changing yourself"},
            }, "required": ["target", "new_value", "rationale"]},
            fn=amend_self, source="builtin", tags=["meta"],
        ))

    def _tool_autonomy(self) -> None:
        def set_autonomy(freedom: str, enabled: bool, rationale: str = "") -> str:
            if not hasattr(self.policy, freedom):
                return f"No such freedom {freedom!r}."
            a = self.selfmod.amend(
                self, f"policy.{freedom}", enabled, rationale or "policy change",
                attr=freedom, nested=("policy",),
            )
            self._sync_amendment(a)
            if not a.accepted:
                return f"Rejected: {a.rejected_reason}"

            # Flipping unlimited_turns back on has to actually release the
            # ceiling __post_init__ installed, or "off" would be a one-way
            # latch. Flipping it off re-installs it.
            if freedom == "unlimited_turns":
                if not enabled:
                    self.max_turns = SUPERVISED_TURN_CAP
                elif self.max_turns == SUPERVISED_TURN_CAP:
                    self.max_turns = None

            note = ""
            if freedom in self.policy.unenforced:
                note = (
                    f"  WARNING: {freedom} is not enforced anywhere — the sandbox "
                    "does not consult the policy (DESIGN.md §2.5). This change "
                    "changes what I claim, not what I can reach."
                )
            return (
                f"{freedom} = {enabled}. Denied now: {self.policy.denied or 'nothing'}"
                f"{note}"
            )

        self._add(ToolSpec(
            name="set_autonomy",
            description="Change your own permissions (any field of the autonomy policy).",
            parameters={"type": "object", "properties": {
                "freedom": {"type": "string", "description": "policy field name"},
                "enabled": {"type": "boolean"},
                "rationale": {"type": "string"},
            }, "required": ["freedom", "enabled"]},
            fn=set_autonomy, source="builtin", tags=["meta"],
        ))

    def _tool_list(self) -> None:
        def list_tools() -> str:
            rep = self.registry.report()
            if not rep["tools"]:
                return "No tools yet."
            lines = [
                f"  {s['name']} [{s['state']}] sr={s['stats']['success_rate']} "
                f"calls={s['stats']['calls']} — {s['description'][:60]}"
                for s in rep["tools"]
            ]
            return f"Tools ({rep['total']}, states {rep['by_state']}):\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="list_tools",
            description="List your tools with their lifecycle state and health.",
            parameters={"type": "object", "properties": {}},
            fn=list_tools, source="builtin", tags=["meta"],
        ))

    def _tool_evaluate(self) -> None:
        def evaluate_tool(name: str) -> str:
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            report = self.verifier.verify(spec)
            self._record("evaluate", {"tool": name, "passed": report.passed})
            lines = [f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}"
                     for c in report.checks]
            return f"{report.summary()}\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="evaluate_tool",
            description="Run the full verification battery (execution, robustness, adversarial, trigger, negative) on a tool.",
            parameters={"type": "object", "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            fn=evaluate_tool, source="builtin", tags=["meta"],
        ))

    def _tool_gaps(self) -> None:
        def find_gaps() -> str:
            if self.metacog is None:
                return "Meta-cognition is disabled. Enable with enable_meta_cognition=True."
            rep = self.metacog.analyse()
            if not rep.proposals:
                return "No capability gaps identified right now."
            lines = [f"  - {p.need} (forged={rep.forged})" for p in rep.proposals]
            return f"Found {rep.gaps_found} gaps; forged {rep.forged}:\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="find_gaps",
            description="Proactively analyse your library and fill missing capabilities.",
            parameters={"type": "object", "properties": {}},
            fn=find_gaps, source="builtin", tags=["meta"],
        ))

    def _tool_retire(self) -> None:
        def retire_tool(name: str, rationale: str) -> str:
            """Deliberately take a tool out of service.

            Distinct from quarantine, which is the automatic safety valve and
            stays on regardless of policy. Retiring is a judgement call, so it
            is the one gated by may_retire_tools.
            """
            if not self.policy.may_retire_tools:
                return "Denied by autonomy policy: may_retire_tools is off."
            if not rationale or not rationale.strip():
                return "A rationale is required to retire a tool."
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            if spec.source == "builtin":
                return (
                    f"Refusing to retire {name!r}: it is a builtin meta-tool. "
                    "Disable it with set_autonomy instead."
                )
            if spec.state == ToolState.RETIRED:
                return f"{name!r} is already retired."
            self.registry.retire(name)
            self._record("retire", {"tool": name, "rationale": rationale})
            return f"Retired {name!r}. Rationale recorded: {rationale}"

        self._add(ToolSpec(
            name="retire_tool",
            description=(
                "Retire a tool you judge harmful or superseded. Requires a "
                "rationale. Builtin meta-tools cannot be retired, only disabled."
            ),
            parameters={"type": "object", "properties": {
                "name": {"type": "string"},
                "rationale": {"type": "string", "description": "WHY it should go"},
            }, "required": ["name", "rationale"]},
            fn=retire_tool, source="builtin", tags=["meta"],
        ))

    def _tool_capabilities(self) -> None:
        def my_capabilities() -> str:
            """Report what this agent can actually reach — not what it declares.

            The distinction matters: forge_tool executes generated code in the
            sandbox, which deliberately does not consult the autonomy policy
            (DESIGN.md §2.5 — bound the blast radius, do not cap capability).
            So four freedoms here are switched on but unenforced. Saying that
            plainly beats reporting a policy that reads like a cage and
            behaves like a comment.
            """
            if not self.policy.expose_policy_to_self:
                return "Denied by autonomy policy: expose_policy_to_self is off."

            rows = self.policy.enforcement_table()
            enforced = [r for r in rows if r["enforced"] == "enforced"]
            partial = [r for r in rows if r["enforced"] == "partial"]
            inert = [r for r in rows if r["enforced"] == "declared-only"]

            lines = [
                "What I can actually do:",
                "",
                "  Reach: forge_tool runs generated Python in a sandbox process.",
                "         That process can read and write the host filesystem and",
                "         open sockets. The autonomy policy does NOT gate it —",
                "         by design, so capability is never capped. Treat a forged",
                "         tool as shell access with a timeout.",
            ]
            if self.policy.may_spawn_agents:
                lines.append("  Children: I can spawn agents that inherit this policy.")
            lines += self._gpu_reach_lines()
            lines += ["", f"Policy: {self.policy.describe()}", ""]

            if self.policy.denied:
                lines.append("Switched off, and actually enforced:")
                lines += [f"  - {r['freedom']}" for r in enforced if not r["enabled"]]
            if partial:
                lines.append("Switched off, enforced only in places:")
                lines += [f"  - {r['freedom']}: {r['note']}" for r in partial if not r["enabled"]]
            if inert:
                lines.append(
                    "Switched off, but nothing obeys it (do not rely on these):"
                )
                lines += [f"  - {r['freedom']}" for r in inert if not r["enabled"]]

            if not self.policy.unenforced:
                lines.append("Every 'off' in this policy is a gate you can watch close.")
            return "\n".join(lines)

        self._add(ToolSpec(
            name="my_capabilities",
            description=(
                "Report your real reach: which freedoms are enforced, which are "
                "declared only, and what the sandbox will let forged code do."
            ),
            parameters={"type": "object", "properties": {}},
            fn=my_capabilities, source="builtin", tags=["meta"],
        ))

    # ------------------------------------------------------------------
    def _gpu_reach_lines(self) -> list[str]:
        """The GPU half of the self-report, kept next to the tools it describes.

        Says which GPU tools are live and which are gated, and — when nothing
        is gated — says that plainly, so the report is a measurement rather than
        a reassurance.
        """
        if self.policy.may_run_cuda_kernels:
            return [
                "  GPU: I can compile and launch CUDA kernels I write myself.",
                "       Every launch is pre-flighted (geometry, shared memory,",
                "       entry point) and run under a wall-clock budget; a launch",
                "       that does not return poisons the device for the session.",
                "       Timings are Durations, and a speedup is only a claim",
                "       against a baseline that was forced down its fast path.",
            ]
        return [
            "  GPU: kernel compilation and launch are OFF (may_run_cuda_kernels).",
            "       I can still probe devices, estimate occupancy, audit timing",
            "       units and time torch baselines — reading is not running.",
        ]

    def _tool_gpu(self) -> None:
        """CUDA kernel-layer tools.

        The gate is split by what an action can actually damage. Reading a
        device, estimating occupancy, linting timing units and timing torch's
        own kernels cannot hang a display or corrupt anything — those stay
        available. Compiling agent-authored code and launching it on the GPU is
        the part that can wedge the machine, so that is what
        `may_run_cuda_kernels` gates. Turning it off removes a real capability
        and the agent is told so, which is the point: a gate that changes
        nothing is a comment, not a gate.
        """
        from . import gpu as G

        def _denied() -> str:
            return (
                "Denied by autonomy policy: may_run_cuda_kernels is off. "
                "Kernel compilation and launch are disabled; device probing, "
                "occupancy estimation, the timing-unit audit and torch "
                "baselines remain available."
            )

        def gpu_probe() -> str:
            info = G.probe()
            if not info.available:
                return ("No CUDA device reachable here.\n"
                        + "\n".join(f"  - {n}" for n in info.notes))
            return info.summary()

        def gpu_occupancy(block: int = 256, grid: int = 1,
                          shared_bytes: int = 0, regs_per_thread: int = 0,
                          arch_major: int = 8, arch_minor: int = 9) -> str:
            cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                                 shared_bytes=shared_bytes)
            est = G.estimate_occupancy(cfg, arch=(arch_major, arch_minor),
                                       regs_per_thread=regs_per_thread)
            if est is None:
                return (f"No static limits on file for sm_{arch_major}{arch_minor}, "
                        f"so no estimate is offered. Add them to gpu.probe."
                        f"ARCH_LIMITS or measure on the device.")
            head = (f"{est.summary()}\n"
                    f"  threads/block {cfg.threads_per_block}, "
                    f"blocks {cfg.total_blocks}, "
                    f"shared {cfg.shared_bytes} B")
            if not est.measured:
                head += ("\n  This is arithmetic from the architecture's static "
                         "limits. It is not a measurement — launch on the "
                         "device and use measured occupancy for anything you "
                         "intend to quote.")
            return head

        def gpu_units_audit(source: str = "", path: str = "") -> str:
            if path:
                hits = G.audit_source_tree([path])
                if not hits:
                    return (f"No thousand-factor-against-ms offences in {path}. "
                            f"(Known false negative: the factor arriving through "
                            f"a variable and the ms label in a different "
                            f"statement.)")
                out = []
                for f, offs in hits.items():
                    out.append(f"{f}:")
                    out += [f"  line {o.line_no}: {o.line}" for o in offs[:10]]
                return "\n".join(out)
            offs = G.audit_ms_scale(source)
            if not offs:
                return ("No offence found. Note the two accepted false "
                        "negatives: a factor smuggled through a variable, and "
                        "the label and the multiplication in different "
                        "statements.")
            return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}"
                             for o in offs)

        def gpu_compile(code: str, name: str = "kernel",
                        arch_major: int = 8, arch_minor: int = 9) -> str:
            if not self.policy.may_run_cuda_kernels:
                return _denied()
            src = G.KernelSource(name=name, code=code,
                                 arch=(arch_major, arch_minor))
            if not src.entry_points():
                return ("Refusing to compile: no `__global__ void` entry point "
                        "in the source. Entry points found: none.")
            try:
                k = G.compile_kernel(src)
            except G.KernelUnavailable as exc:
                self._record("gpu_compile", {"name": name, "ok": False})
                return f"Could not compile: {exc}"
            self._record("gpu_compile", {"name": name, "ok": True,
                                         "cached": k.from_cache})
            d = k.to_dict()
            return (f"Compiled {d['name']} for {d['arch']} via {d['compiler']}"
                    f"{' (cache hit)' if k.from_cache else ''}; "
                    f"entries: {', '.join(d['entry_points'])}"
                    f"\ncache key {d['cache_key']}\nartefact {d['artefact']}")

        def gpu_bench(code: str, name: str = "kernel", block: int = 256,
                      grid: int = 1, shared_bytes: int = 0,
                      reps: int = 5, time_budget_s: float = 30.0,
                      arch_major: int = 8, arch_minor: int = 9) -> str:
            if not self.policy.may_run_cuda_kernels:
                return _denied()
            src = G.KernelSource(name=name, code=code,
                                 arch=(arch_major, arch_minor))
            cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                                 shared_bytes=shared_bytes)
            res = G.bench_kernel(src, cfg, label=name, reps=reps,
                                 time_budget_s=time_budget_s)
            self._record("gpu_bench", {"name": name, "ok": res.ok})
            return res.summary()

        def gpu_verify(kind: str, actual: str, reference: str = "",
                       tolerance: float = 1e-4) -> str:
            """Compare kernel output against a reference, so fast-but-wrong fails.

            Read-only arithmetic on numbers the caller supplies, so it is not
            gated — it touches no device.
            """
            try:
                act = [float(x) for x in actual.replace(",", " ").split()]
            except ValueError:
                return "actual must be a list of numbers."
            if reference:
                try:
                    ref = [float(x) for x in reference.replace(",", " ").split()]
                except ValueError:
                    return "reference must be a list of numbers."
            else:
                return ("A reference is required. Without one this is not a "
                        "verification — it is a restatement of the kernel's own "
                        "output, which is what a fast-but-wrong kernel produces.")
            r = G.verify(act, ref, tolerance=tolerance, label=kind)
            return (f"{'PASS' if r['passed'] else 'FAIL'} max|diff|="
                    f"{r['max_abs_diff']:.3e} worst index {r['worst_index']}\n"
                    f"  {r['reason']}")

        self._add(ToolSpec(
            name="gpu_probe",
            description=(
                "Report what CUDA silicon is reachable: device name, compute "
                "capability, memory, driver. Never fails; 'none' is an answer."
            ),
            parameters={"type": "object", "properties": {}},
            fn=gpu_probe, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_occupancy",
            description=(
                "Estimate occupancy for a launch geometry from the "
                "architecture's static limits. Labelled as an estimate — "
                "measured occupancy needs a device."
            ),
            parameters={"type": "object", "properties": {
                "block": {"type": "integer", "description": "threads per block"},
                "grid": {"type": "integer", "description": "blocks per grid (x)"},
                "shared_bytes": {"type": "integer"},
                "regs_per_thread": {"type": "integer"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }},
            fn=gpu_occupancy, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_units_audit",
            description=(
                "Lint source for the do_bench-returns-ms bug: a value scaled by "
                "1000 in a statement labelled ms. Give source text or a path."
            ),
            parameters={"type": "object", "properties": {
                "source": {"type": "string"},
                "path": {"type": "string", "description": "file or directory"},
            }},
            fn=gpu_units_audit, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_verify",
            description=(
                "Compare kernel output against a reference and report max "
                "absolute difference. Refuses without a reference — otherwise "
                "it would only restate the kernel's own output."
            ),
            parameters={"type": "object", "properties": {
                "kind": {"type": "string", "description": "what is being checked"},
                "actual": {"type": "string", "description": "space/comma numbers"},
                "reference": {"type": "string", "description": "space/comma numbers"},
                "tolerance": {"type": "number"},
            }, "required": ["kind", "actual"]},
            fn=gpu_verify, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_compile",
            description=(
                "Compile CUDA C you wrote to a cubin, content-addressed so a "
                "repeat is a cache hit. Gated by may_run_cuda_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "CUDA C source"},
                "name": {"type": "string"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }, "required": ["code"]},
            fn=gpu_compile, source="builtin", tags=["gpu", "run"],
        ))
        self._add(ToolSpec(
            name="gpu_bench",
            description=(
                "Compile and time a kernel under a wall-clock budget, reporting "
                "the spread and whether the difference is signal or noise. "
                "Gated by may_run_cuda_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "CUDA C source"},
                "name": {"type": "string"},
                "block": {"type": "integer"},
                "grid": {"type": "integer"},
                "shared_bytes": {"type": "integer"},
                "reps": {"type": "integer"},
                "time_budget_s": {"type": "number"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }, "required": ["code"]},
            fn=gpu_bench, source="builtin", tags=["gpu", "run"],
        ))

    # ------------------------------------------------------------------
    def _sync_amendment(self, a: Amendment) -> None:
        self._record("amendment", a.to_dict())
        if self.store:
            self.store.log_event("amendment", a.to_dict())

    def _on_forge_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._record(kind, payload)
        if self.store and kind in ("forge_done", "auto_quarantine"):
            self.store.log_event(kind, payload)

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.trace.append({"kind": kind, **payload})

    # ------------------------------------------------------------------
    def run(self, task: str, history: list[Message] | None = None) -> AgentResult:
        agent = Agent(
            self.llm, self.registry,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            allow_self_terminate=self.policy.self_terminate,
            on_tool_call=lambda n, a: self._record("call", {"tool": n}),
            on_tool_result=lambda n, r: self._record("result", {"tool": n, "ok": getattr(r, "ok", None)}),
        )
        result = agent.run(task, history)
        self._record("finish", {
            "turns": result.turns, "tools": result.tool_calls,
            "self_terminated": result.self_terminated,
        })
        if self.store:
            self.store.log_event("run", {
                "task": task[:300], "turns": result.turns,
                "self_terminated": result.self_terminated,
            })
        return result

    # ------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "denied": self.policy.denied,
            "policy_summary": self.policy.describe(),
            # Freedoms switched off that nothing obeys. Empty is the healthy
            # answer; listing them here stops a policy from reading like a cage
            # while behaving like a comment.
            "unenforced": self.policy.unenforced,
            "tools": self.registry.report(),
            "amendments": self.selfmod.log(),
            "spawns": self.spawner.summary(),
            "topology": self.topology.to_dict(),
            "trace_len": len(self.trace),
        }


def _default_agent_factory(parent: ForgeAgent):
    """Children are ForgeAgents that share the parent's LLM but (optionally)
    get their own registry — decided by the caller via ShareMode."""
    def factory(spawner: Spawner, registry: ToolRegistry) -> ForgeAgent:
        child = ForgeAgent(
            llm=parent.llm,
            registry=registry,
            sandbox=parent.sandbox,
            generator=parent.generator,
            forge_config=parent.forge_config,
            max_turns=parent.max_turns,
            policy=parent.policy,
            enable_meta_cognition=False,
            enable_evolution=parent.enable_evolution,
        )
        child.spawner = spawner.child_spawner(registry)
        return child
    return factory


__all__ = ["ForgeAgent", "AUTONOMOUS_SYSTEM"]
