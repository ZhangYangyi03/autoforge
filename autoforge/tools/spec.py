"""Tool specification, lifecycle state, and the reliability ledger.

This is where the framework's thesis lives. A tool is not a function you
register; it is a *claim* with a contract:

    ToolSpec = implementation + schema + trigger probes + effect signature
               + provenance + verification record + live reliability ledger

States form a lifecycle, not a boolean flag:

    DRAFT ──verify──> PROBATION ──earn──> ACTIVE ──decay──> QUARANTINED
                         ^                    │                  │
                         └──────rehab─────────┴──────────────────┘
                                            └──retire──> RETIRED

Free to create (DRAFT, PROBATION are immediately callable if `strict=False`),
but ACTIVE status — the thing that earns context budget and trust — must be
earned by passing probes and then *kept* by a track record.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolState(str, Enum):
    DRAFT = "draft"            # generated, not yet verified
    PROBATION = "probation"    # verified, on trial — callable but flagged
    ACTIVE = "active"          # earned trust, injected into context by default
    QUARANTINED = "quarantined"  # degraded — hidden by default, still executable
    RETIRED = "retired"        # removed from service


@dataclass
class TriggerProbe:
    """A test of the *trigger* question, not the *execution* question.

    `query` should demand this tool. `expect` states what a well-behaved agent
    should do. `negative_query` (optional) should NOT trigger it — a tool that
    fires on everything is worse than a tool that never fires.
    """

    query: str
    expect: str = "call"
    negative_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStats:
    """Append-only ledger of how a tool behaves in the wild."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    trigger_hits: int = 0
    trigger_misses: int = 0
    consecutive_failures: int = 0
    first_seen: float = field(default_factory=time.time)
    last_called: float = 0.0
    last_failure: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def trigger_rate(self) -> float:
        n = self.trigger_hits + self.trigger_misses
        return self.trigger_hits / n if n else 0.0

    def record_call(self, ok: bool, error: str | None = None) -> None:
        self.calls += 1
        self.last_called = time.time()
        if ok:
            self.successes += 1
            self.consecutive_failures = 0
        else:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_failure = time.time()
        self.history.append(
            {"t": self.last_called, "ok": ok, "error": error}
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate, 4)
        d["trigger_rate"] = round(self.trigger_rate, 4)
        return d


_JSON_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)


def normalise_parameters(params: Any) -> dict[str, Any]:
    """Force a JSON-Schema object whose ``properties`` values are schemas.

    Two habits — one from models, one from hand-written JSON — break every
    downstream consumer with ``AttributeError: 'str' object has no attribute
    'get'``, raised three modules away from the cause::

        {"properties": {"isbn": "string"}}    # bare type name, not a schema
        {"properties": "isbn: string"}        # properties as free text

    The verifier, the fuzzer and the adversarial gate each read
    ``schema.get("type", ...)`` per property, so the shape is settled once, at
    the trust boundary where untrusted parameters enter (LLM output, stored
    JSON) — not in ``ToolSpec`` itself, whose many construction sites and
    hash-dependent lifecycle want exactly what the caller passed.
    """
    if not isinstance(params, dict) or not params:
        return {"type": "object", "properties": {}}

    out = dict(params)
    out.setdefault("type", "object")

    props = out.get("properties")
    props = props if isinstance(props, dict) else {}
    fixed: dict[str, Any] = {}
    for name, schema in props.items():
        if isinstance(schema, str):
            token = schema.strip().lower()
            schema = {"type": token if token in _JSON_TYPES else "string"}
        elif not isinstance(schema, dict):
            schema = {"type": "string"}
        fixed[str(name)] = schema
    out["properties"] = fixed

    required = out.get("required")
    if required is not None:
        out["required"] = (
            [str(r) for r in required] if isinstance(required, list)
            else [str(required)] if isinstance(required, str) else []
        )
    return out


@dataclass
class ToolSpec:
    """A self-made tool, contract and all."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    code: str = ""
    source: str = "human"          # human | generated | composed
    generator: str = ""            # model that wrote it
    probes: list[TriggerProbe] = field(default_factory=list)
    effect_signature: str = ""     # what it touches; used for re-verification
    invariances: list[str] = field(default_factory=list)  # token types whose
    # decoration must not change the answer (e.g. "isbn"). Declared at birth by
    # the generator, then frozen into FrozenBaseline — a later mutant cannot
    # drop one without failing the regression gate.
    state: ToolState = ToolState.DRAFT
    stats: ToolStats = field(default_factory=ToolStats)
    verification: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    cost_hint: str = "cheap"       # cheap | moderate | expensive
    created_at: float = field(default_factory=time.time)
    # Optional out-of-process executor. When set, the registry calls
    # `runner(name, args) -> ToolResult-ish` instead of `fn(**args)`. Forged
    # tools set this to a sandbox bridge so untrusted code never runs in-loop.
    runner: Callable[[str, dict[str, Any]], Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def hash(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "code": self.code,
                "parameters": self.parameters,
                "description": self.description,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def is_trusted(self) -> bool:
        return self.state in (ToolState.PROBATION, ToolState.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "source": self.source,
            "generator": self.generator,
            "hash": self.hash,
            "tags": self.tags,
            "effect_signature": self.effect_signature,
            "stats": self.stats.to_dict(),
            "verification": self.verification,
            "probes": [p.to_dict() for p in self.probes],
        }


__all__ = ["ToolState", "ToolStats", "TriggerProbe", "ToolSpec"]
