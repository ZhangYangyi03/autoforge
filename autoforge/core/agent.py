"""The agent loop.

Intentionally thin. All the interesting machinery lives in the registry and
the forge pipeline, so the loop stays readable and the framework stays
hackable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLMClient
from .message import Message

DEFAULT_SYSTEM = (
    "You are a capable agent. Call a tool when it genuinely helps; answer "
    "directly when none is needed. Do not call a tool you do not need."
)


@dataclass
class AgentResult:
    content: str
    messages: list[Message]
    turns: int
    tool_calls: list[str] = field(default_factory=list)

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class Agent:
    """A tool-using loop over an LLMClient and a ToolRegistry."""

    def __init__(
        self,
        llm: LLMClient,
        registry: Any,
        *,
        system_prompt: str | None = None,
        max_turns: int = 20,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.max_turns = max_turns
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

    def run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        msgs = list(history or [])
        if not msgs or msgs[0].role != "system":
            msgs.insert(0, Message.system(self.system_prompt))
        msgs.append(Message.user(task))

        used: list[str] = []
        for turn in range(1, self.max_turns + 1):
            resp = self.llm.chat(msgs, tools=self.registry.schemas())
            msgs.append(Message.assistant(resp.content, resp.tool_calls))
            if not resp.tool_calls:
                return AgentResult(resp.content, msgs, turn, used)
            for tc in resp.tool_calls:
                used.append(tc.name)
                if self.on_tool_call:
                    self.on_tool_call(tc.name, tc.arguments)
                result = self.registry.call(tc.name, tc.arguments)
                if self.on_tool_result:
                    self.on_tool_result(tc.name, result)
                msgs.append(Message.tool(result.output, tc.id, tc.name))
        return AgentResult("(max turns reached)", msgs, self.max_turns, used)
