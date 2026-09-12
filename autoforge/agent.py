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
  amend_self      change its own prompt / config / routing weights
  set_autonomy    change its own permissions
  list_tools      inspect its own library
  evaluate_tool   run the full verification battery on a tool
  find_gaps       proactively discover missing capabilities
  terminate       end the loop when it judges the task done
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .autonomy.policy import FULL_FREEDOM, AutonomyPolicy
from .autonomy.selfmod import Amendment, SelfModifier
from .autonomy.spawn import ShareMode, Spawner
from .core.agent import Agent, AgentResult
from .core.llm import LLMClient
from .core.message import Message
from .forge.evolution import EvolutionEngine
from .forge.generator import TemplateGenerator
from .forge.metacog import MetaCognition
from .forge.pipeline import ForgeConfig, ForgePipeline
from .forge.sandbox import Sandbox
from .forge.verifier import ToolVerifier
from .route.router import BehaviourRouter
from .store import ToolStore
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState


AUTONOMOUS_SYSTEM = """You are an autonomous agent that grows its own capabilities.

You can:
- forge_tool      — create a new tool when you hit a need you cannot serve
- evolve_tool     — breed a better version of a tool that underperforms
- spawn_agent     — create a child agent for a subtask
- amend_self      — change your own prompt, forge config, or routing weights
- set_autonomy    — change your own permissions
- evaluate_tool   — run the full verification battery on any tool
- find_gaps       — proactively discover capabilities you're missing
- list_tools      — inspect your library and its health
- terminate       — end the task when you judge it complete

Principles:
- Forge only when a need genuinely recurs; don't duplicate existing tools.
- When a tool fails repeatedly, evolve it rather than retrying blindly.
- Self-modifications need a rationale. Say WHY you are changing yourself.
- You decide when the task is done. There is no hidden turn limit.
- Prefer the simplest path that works.
"""


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
            on_event=self._on_forge_event,
        )
        self.router = BehaviourRouter(self.registry)

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
        )

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
        self._tool_amend()
        self._tool_autonomy()
        self._tool_list()
        self._tool_evaluate()
        self._tool_gaps()

    def _add(self, spec: ToolSpec) -> None:
        self.registry.register(spec)
        self.registry.promote(spec.name)

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
            result = self.evolution.evolve(spec, report)
            self._record("evolve", {"tool": name, "improved": result.improved})
            if not result.improved:
                return f"Evolved {name!r}: no mutant beat the original ({len(result.mutants)} tried)."
            best = result.best_mutant.generated
            new_spec = ToolSpec(
                name=best.name, description=best.description,
                parameters=best.parameters, fn=spec.fn, runner=spec.runner,
                code=best.code, source="evolved", probes=best.probes,
                effect_signature=best.effect_signature, tags=best.tags,
                state=ToolState.ACTIVE,
            )
            self.registry.register(new_spec, replace=True)
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

    def _tool_amend(self) -> None:
        def amend_self(target: str, new_value: str, rationale: str) -> str:
            if not self.policy.may_modify_own_prompt:
                return "Denied by autonomy policy: may_modify_own_prompt is off."

            if target == "system_prompt":
                a = self.selfmod.amend(self, "system_prompt", new_value, rationale)
            elif target == "forge_max_rounds":
                try:
                    val = int(new_value)
                except ValueError:
                    return f"forge_max_rounds must be an integer, got {new_value!r}"
                a = self.selfmod.amend(
                    self, "forge_max_rounds", val, rationale,
                    attr="max_rounds", nested=("forge_config",),
                )
                if a.accepted:
                    self.pipeline.config.max_rounds = val
            elif target == "routing_weights":
                return ("routing_weights requires a dict; use the router API directly "
                        "rather than amend_self.")
            else:
                return f"Unknown target {target!r}."

            self._sync_amendment(a)
            if not a.accepted:
                return f"Amendment rejected: {a.rejected_reason}"
            return f"Amended {target}: {a.diff_summary()}"

        self._add(ToolSpec(
            name="amend_self",
            description="Change your own system prompt or forge config. Requires a rationale.",
            parameters={"type": "object", "properties": {
                "target": {"type": "string", "description": "system_prompt | forge_max_rounds"},
                "new_value": {"type": "string"},
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
            return f"{freedom} = {enabled}. Denied now: {self.policy.denied or 'nothing'}"

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
            "tools": self.registry.report(),
            "amendments": self.selfmod.log(),
            "spawns": self.spawner.summary(),
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
