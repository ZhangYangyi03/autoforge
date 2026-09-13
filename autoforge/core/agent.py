"""The agent loop.

Deliberately thin in mechanism, deliberately free in policy.

The two things that make this loop different from every other agent loop:

1. **No hidden turn cap.** `max_turns=None` (the default) means the agent runs
   until it produces a text answer or calls `terminate`. A cap exists only if
   the caller imposes one — and when one is imposed, it is logged, not silent.
   The agent decides when a task is done, not an arbitrary number baked into
   the framework.

2. **Self-termination is a first-class tool.** The agent can end its own turn
   loop, returning a summary and a reason. Combined with `unlimited_turns`,
   this means the stopping condition is *semantic* (the agent judges it's done)
   rather than *mechanical* (the counter hit N).

Everything else — tool dispatch, message threading — is ordinary, because that
part isn't what's broken in other frameworks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLMClient
from .message import Message

DEFAULT_SYSTEM = (
    "You are a capable agent. Call a tool when it genuinely helps; answer "
    "directly when none is needed. Do not call a tool you do not need.\n"
    "When you have fully completed the task, either answer in plain text or "
    "call terminate with a summary of what you did."
)


@dataclass
class AgentResult:
    content: str
    messages: list[Message]
    turns: int
    tool_calls: list[str] = field(default_factory=list)
    self_terminated: bool = False
    termination_reason: str = ""

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class _TerminateSignal(BaseException):
    """Raised by the terminate tool to end the loop cleanly.

    Inherits BaseException (not Exception) so that the registry's
    `except Exception` tool-boundary handler does NOT swallow it — a
    self-termination must propagate all the way back to the loop.
    """

    def __init__(self, summary: str, reason: str = "") -> None:
        super().__init__(summary)
        self.summary = summary
        self.reason = reason


def _notify(hook: Callable[..., None] | None, *args: Any) -> None:
    """Call an observer hook, swallowing whatever it throws.

    Every hook here is a *reporter* — progress lines, trace records, UI. None of
    them decide anything, so a broken reporter must not take the run down with
    it. Before this, a CLI writing to a closed stream killed the task mid-loop.
    """
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001 — reporting is never load-bearing
        pass


class Agent:
    """A tool-using loop over an LLMClient and a ToolRegistry.

    Parameters
    ----------
    max_turns:
        `None` (default) means unbounded — the agent stops when it answers or
        self-terminates. An integer imposes a cap, which is recorded on the
        result so it is visible, never silent.
    allow_self_terminate:
        Registers a `terminate` tool the model can call to end the loop with
        a summary. Requires a registry that accepts registration.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: Any,
        *,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        allow_self_terminate: bool = True,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, Any], None] | None = None,
        on_turn: Callable[[int, Message], None] | None = None,
        on_request: Callable[[int], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.max_turns = max_turns
        self.allow_self_terminate = allow_self_terminate
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_turn = on_turn
        self.on_request = on_request
        self._terminated: _TerminateSignal | None = None
        if allow_self_terminate:
            self._register_terminate_tool()

    def _register_terminate_tool(self) -> None:
        from ..tools.spec import ToolSpec, ToolState

        registry = self.registry
        if not hasattr(registry, "register") or "terminate" in getattr(registry, "_tools", {}):
            return

        def terminate(summary: str = "", reason: str = "") -> str:
            raise _TerminateSignal(summary or "task complete", reason)

        registry.register(
            ToolSpec(
                name="terminate",
                description=(
                    "End the task early and return a summary. Use this when the "
                    "task is fully complete or cannot proceed further."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "what was accomplished"},
                        "reason": {"type": "string", "description": "why stopping now"},
                    },
                },
                fn=terminate,
                source="builtin",
                tags=["meta"],
                state=ToolState.ACTIVE,
            )
        )

    def run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        msgs = list(history or [])
        if not msgs or msgs[0].role != "system":
            msgs.insert(0, Message.system(self.system_prompt))
        msgs.append(Message.user(task))

        used: list[str] = []
        turn = 0
        self._terminated = None

        while True:
            turn += 1
            if self.max_turns is not None and turn > self.max_turns:
                return AgentResult(
                    f"(turn cap of {self.max_turns} reached before completion)",
                    msgs, turn - 1, used,
                )

            _notify(self.on_request, turn)
            resp = self.llm.chat(msgs, tools=self.registry.schemas())
            msgs.append(Message.assistant(resp.content, resp.tool_calls))
            _notify(self.on_turn, turn, msgs[-1])

            if not resp.tool_calls:
                return AgentResult(resp.content, msgs, turn, used)

            for tc in resp.tool_calls:
                used.append(tc.name)
                _notify(self.on_tool_call, tc.name, tc.arguments)
                try:
                    result = self.registry.call(tc.name, tc.arguments)
                except _TerminateSignal as sig:
                    self._terminated = sig
                    return AgentResult(
                        sig.summary, msgs, turn, used,
                        self_terminated=True, termination_reason=sig.reason,
                    )
                _notify(self.on_tool_result, tc.name, result)
                msgs.append(Message.tool(result.output, tc.id, tc.name))


__all__ = ["Agent", "AgentResult", "DEFAULT_SYSTEM", "_TerminateSignal"]