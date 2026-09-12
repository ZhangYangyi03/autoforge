"""The composed agent: an Agent + ForgePipeline + router, wired into one loop.

This is the framework's opinion about how autonomy and verification coexist:

  * the agent may forge a tool mid-task when it hits a need it cannot serve
  * a forged tool is verified before it earns context budget
  * every call feeds the ledger, which can quarantine and rehab
  * routing picks tools by behaviour when the library outgrows the context

Nothing here locks the model in: it drives one `LLMClient`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.agent import Agent, AgentResult
from ..core.llm import LLMClient
from ..core.message import Message
from ..forge.pipeline import ForgeConfig, ForgePipeline
from ..forge.generator import TemplateGenerator
from ..forge.sandbox import Sandbox
from ..forge.verifier import ToolVerifier
from ..route.router import BehaviourRouter
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec


FORGE_SYSTEM = (
    "You are an agent that can grow new tools.\n"
    "When the task needs a capability you do not have, you may forge one.\n"
    "To forge: call `forge_tool` with a one-line description of the need.\n"
    "Forged tools are verified (execution + trigger) before they become "
    "available. Do not forge a tool that already exists.\n"
    "Prefer normal tools when they suffice."
)


@dataclass
class ForgeAgent:
    llm: LLMClient
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    sandbox: Sandbox = field(default_factory=Sandbox)
    generator: Any = None
    forge_config: ForgeConfig = field(default_factory=ForgeConfig)
    system_prompt: str = FORGE_SYSTEM
    max_turns: int = 12
    trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = TemplateGenerator()
        self.verifier = ToolVerifier(
            self.llm,
            sandbox=self.sandbox,
            run_execution_check=self.forge_config.require_execution,
            run_trigger_check=self.forge_config.require_trigger,
            run_negative_check=self.forge_config.require_negative,
        )
        self.pipeline = ForgePipeline(
            self.generator, self.verifier, self.registry,
            config=self.forge_config, sandbox=self.sandbox, on_event=self._on_forge_event,
        )
        self.router = BehaviourRouter(self.registry)
        self._register_meta_tools()

    # -- meta tools ------------------------------------------------------
    def _register_meta_tools(self) -> None:
        def forge_tool(need: str, **_: Any) -> str:
            res = self.pipeline.forge(need)
            self._record("forge", {"need": need, "ok": res.ok,
                                   "tool": res.spec.name if res.spec else None})
            if not res.ok:
                return f"Could not forge a working tool for: {need}. Rounds: {res.rounds}"
            spec = res.spec
            return (
                f"Forged tool {spec.name!r} ({spec.state.value}). "
                f"Description: {spec.description}"
            )

        def list_tools(**_: Any) -> str:
            rep = self.registry.report()
            lines = [f"{s['name']} [{s['state']}] sr={s['stats']['success_rate']}"
                     for s in rep["tools"]]
            return "Tools:\n" + "\n".join(lines) if lines else "No tools."

        self.registry.register(ToolSpec(
            name="forge_tool",
            description="Forge a new tool for a recurring need the agent cannot currently serve.",
            parameters={
                "type": "object",
                "properties": {
                    "need": {"type": "string", "description": "one-line description of the need"}
                },
                "required": ["need"],
            },
            fn=forge_tool,
            source="builtin",
            tags=["meta"],
        ))
        self.registry.promote("forge_tool")

        self.registry.register(ToolSpec(
            name="list_tools",
            description="List available tools and their health.",
            parameters={"type": "object", "properties": {}},
            fn=list_tools,
            source="builtin",
            tags=["meta"],
        ))
        self.registry.promote("list_tools")

    # -- run ---------------------------------------------------------------
    def run(self, task: str, history: list[Message] | None = None) -> AgentResult:
        agent = Agent(
            self.llm, self.registry,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            on_tool_call=lambda n, a: self._record("call", {"tool": n, "args": a}),
            on_tool_result=lambda n, r: self._record(
                "result", {"tool": n, "ok": getattr(r, "ok", None)}
            ),
        )
        result = agent.run(task, history)
        self._record("finish", {"turns": result.turns, "tools": result.tool_calls})
        return result

    # -- observability ------------------------------------------------------
    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.trace.append({"kind": kind, **payload})

    def _on_forge_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._record(kind, payload)

    def report(self) -> dict[str, Any]:
        return {
            "tools": self.registry.report(),
            "router_top": [c.to_dict() for c in self.router.rank("")][:5],
            "trace_len": len(self.trace),
        }


__all__ = ["ForgeAgent", "FORGE_SYSTEM"]
