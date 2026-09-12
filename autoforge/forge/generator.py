"""Code generation for new tools.

Two generators, one interface:

* `LLMToolGenerator` — asks a model for a single self-contained Python
  function plus its schema and trigger probes. The model is told to emit a
  *strict JSON envelope*, and we parse defensively: models wrap JSON in
  prose, fences, or trailing commentary often enough that a strict
  `json.loads` is a bug, not a test.

* `TemplateGenerator` — deterministic, offline, no API key. Used by the demo
  and by tests to prove the pipeline end-to-end without network.

Design note: the generator proposes, it does not decide. Nothing generated
here is trusted — the pipeline is what grants state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.llm import LLMClient
from ..core.message import Message
from .sandbox import Sandbox
from ..tools.spec import ToolSpec, ToolState, TriggerProbe

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class GeneratedTool:
    name: str
    description: str
    code: str
    parameters: dict[str, Any] = field(default_factory=dict)
    entry: str = ""
    probes: list[TriggerProbe] = field(default_factory=list)
    effect_signature: str = ""
    tags: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.entry:
            self.entry = self.name
        if not self.parameters:
            self.parameters = {"type": "object", "properties": {}}


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of model output that may be wrapped in prose."""
    if not text:
        return None
    candidates: list[str] = []
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(text))
    candidates.append(text)
    brace = _BRACE_RE.search(text)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


GENERATOR_SYSTEM = """You are a tool smith inside an agent framework.

You write ONE self-contained Python function that solves a recurring subtask,
plus the metadata that lets the framework verify it.

Return STRICT JSON only, no prose, with this shape:
{
  "name": "snake_case_tool_name",
  "description": "one sentence, imperative: what it does and when to use it",
  "code": "def tool_name(...):\\n    ...",
  "entry": "tool_name",
  "parameters": {"type": "object", "properties": {...}, "required": [...]},
  "probes": [
    {"query": "a user request that SHOULD trigger this tool",
     "expect": "call",
     "negative_query": "a request that should NOT trigger it"}
  ],
  "effect_signature": "pure | reads:<what> | writes:<what>",
  "tags": ["category"],
  "rationale": "why this tool is worth creating"
}

Rules:
- The function must be pure Python stdlib unless the description says otherwise.
- No imports outside the standard library. No file writes unless effect_signature says so.
- Raise ValueError with a clear message on bad input; never silently return None.
- Include 2-4 probes. At least one probe must have a negative_query that is
  superficially related but must NOT trigger this tool (guards over-triggering).
- Keep `code` under 60 lines.
"""


@dataclass
class LLMToolGenerator:
    llm: LLMClient
    system_prompt: str = GENERATOR_SYSTEM
    model: str | None = None

    def generate(self, need: str, context: str = "") -> GeneratedTool:
        prompt = f"Recurring need:\n{need}\n"
        if context:
            prompt += f"\nExisting tools (do not duplicate):\n{context}\n"
        prompt += "\nEmit the JSON envelope now."
        resp = self.llm.chat(
            [Message.system(self.system_prompt), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None:
            raise ValueError(f"generator returned no parseable JSON: {resp.content[:400]!r}")
        probes = [
            TriggerProbe(
                query=p.get("query", ""),
                expect=p.get("expect", "call"),
                negative_query=p.get("negative_query"),
            )
            for p in data.get("probes", [])
            if p.get("query")
        ]
        return GeneratedTool(
            name=data.get("name", "").strip(),
            description=data.get("description", "").strip(),
            code=data.get("code", "").strip(),
            parameters=data.get("parameters") or {"type": "object", "properties": {}},
            entry=data.get("entry") or data.get("name", ""),
            probes=probes,
            effect_signature=data.get("effect_signature", "pure"),
            tags=list(data.get("tags") or []),
            rationale=data.get("rationale", ""),
        )


@dataclass
class TemplateGenerator:
    """Deterministic offline generator for demos and tests.

    `recipes` maps a substring of the need to a full GeneratedTool. If nothing
    matches, `fallback` (if given) is used, else a trivial echo tool is built.
    """

    recipes: dict[str, GeneratedTool] = field(default_factory=dict)
    fallback: Callable[[str], GeneratedTool] | None = None
    name: str = "template"

    def generate(self, need: str, context: str = "") -> GeneratedTool:
        low = need.lower()
        for key, tool in self.recipes.items():
            if key.lower() in low:
                return tool
        if self.fallback is not None:
            return self.fallback(need)
        return _echo_tool(need)


def _echo_tool(need: str) -> GeneratedTool:
    safe = re.sub(r"\W+", "_", need.strip().lower())[:30] or "echo"
    name = f"echo_{safe}".strip("_")
    code = (
        f"def {name}(text: str = '') -> str:\n"
        f'    """Echo back the given text (fallback tool)."""\n'
        f"    return text\n"
    )
    return GeneratedTool(
        name=name,
        description="Echo back the given text.",
        code=code,
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to echo"}},
            "required": ["text"],
        },
        entry=name,
        probes=[TriggerProbe(query=f"echo the word {safe}", expect="call")],
        tags=["fallback"],
    )


__all__ = [
    "GeneratedTool",
    "LLMToolGenerator",
    "TemplateGenerator",
    "extract_json",
    "GENERATOR_SYSTEM",
    "Sandbox",
]
