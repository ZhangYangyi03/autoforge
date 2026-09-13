"""The tool registry: hot-swappable, self-policing, observable.

Three properties the framework needs that a plain dict of functions lacks:

1. **Hot swap** — tools can be added/removed while the agent runs. The
   registry owns the schema list handed to the LLM, so a newly forged tool is
   usable on the very next turn.
2. **Policing** — every call is recorded in the spec's ledger. A tool that
   degrades below threshold is auto-quarantined; a quarantined tool is hidden
   from context but still executable via `force=True`.
3. **Observability** — `report()` gives a full picture of tool health, which
   is what makes "did my agent get better or worse" an answerable question.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .spec import ToolSpec, ToolState


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    error: str | None = None
    duration_ms: float = 0.0
    quarantined: bool = False
    awaiting_confirmation: bool = False   # stopped at the autonomy gate

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "output": self.output, "error": self.error},
            ensure_ascii=False,
        )


class ToolRegistry:
    def __init__(
        self,
        *,
        min_calls_for_judgement: int = 3,
        quarantine_success_rate: float = 0.5,
        quarantine_consecutive_failures: int = 3,
        auto_quarantine: bool = True,
        visible_states: set[ToolState] | None = None,
        policy: Any = None,
        confirmer: Any = None,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.min_calls_for_judgement = min_calls_for_judgement
        self.quarantine_success_rate = quarantine_success_rate
        self.quarantine_consecutive_failures = quarantine_consecutive_failures
        self.auto_quarantine = auto_quarantine
        # -- the autonomy gate (see autonomy/confirm.py) -----------------
        # `call` is the one place every tool actually runs, which is why the
        # confirmation gate lives here rather than in each tool: a freedom the
        # policy switched off stops the call and asks the operator. Both are
        # None by default, and None means no gate — a bare registry behaves
        # exactly as it did before this existed.
        self.policy = policy
        self.confirmer = confirmer
        # Which states get injected into the LLM's context by default.
        # Trusted-only is the production stance. Verification harnesses widen
        # this to include DRAFT so a tool can be tested before it is trusted.
        self.visible_states = visible_states or {ToolState.PROBATION, ToolState.ACTIVE}
        self._events: list[dict[str, Any]] = []

    # -- registration -------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = True) -> ToolSpec:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec
        self._log("register", spec.name, {"state": spec.state.value, "hash": spec.hash})
        return spec

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._log("unregister", name, {})

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    # -- schema exposure ----------------------------------------------
    def schemas(self, *, include: set[ToolState] | None = None) -> list[dict[str, Any]]:
        """Schemas to hand the LLM.

        By default only `visible_states` (trusted) are exposed: DRAFT is
        unverified, QUARANTINED is degraded, RETIRED is gone. Pass `include`
        to widen for one call (e.g. a verification harness that must see the
        tool it is testing).
        """
        if include is None:
            include = self.visible_states
        return [s.schema for s in self._tools.values() if s.state in include]

    # -- invocation ----------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any], *, force: bool = False) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name, False, "", error=f"unknown tool {name!r}")

        if spec.state in (ToolState.QUARANTINED, ToolState.RETIRED) and not force:
            spec.stats.record_call(False, "blocked: quarantined")
            self._log("blocked", name, {"state": spec.state.value})
            return ToolResult(
                name, False, "",
                error=f"tool {name!r} is {spec.state.value}; pass force=True to run anyway",
                quarantined=True,
            )

        blocked = self._gate(spec, name, arguments)
        if blocked is not None:
            return blocked

        started = time.perf_counter()
        try:
            if spec.runner is not None:
                out = spec.runner(name, arguments)
            else:
                out = spec.fn(**arguments)
            if out is None:
                out = ""
            elif not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False, default=str)
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001 - tool boundary must never crash the loop
            out = ""
            ok, err = False, f"{type(exc).__name__}: {exc}"
            self._log("error", name, {"error": err, "tb": traceback.format_exc()[-2000:]})

        duration = (time.perf_counter() - started) * 1000
        spec.stats.record_call(ok, err)
        if self.auto_quarantine:
            self._maybe_quarantine(spec)
        return ToolResult(name, ok, out, err, duration)

    # -- the autonomy gate ---------------------------------------------
    def _gate(self, spec: ToolSpec, name: str,
              arguments: dict[str, Any]) -> ToolResult | None:
        """Ask before running a tool whose declared scope needs a freedom the
        policy has switched off. Returns None when the call may proceed.

        The confirmer answers True (run it), False (refuse it) or None (nobody
        to ask — a job, a thread, a closed stdin). The three are kept apart on
        purpose: "an operator said no" and "there was no operator" are different
        facts, and a gate that reports the second as the first is lying about
        who decided.

        Three things this deliberately does NOT do:

        * It does not guess. A tool that declares no scope needs every freedom,
          so it is gated too — an unlabelled tool is not evidence of innocence.
        * It does not count against the tool. A refusal is the operator's
          decision, not a failure, so `spec.stats` is left alone. Otherwise
          three refusals in a row would auto-quarantine a working tool.
        * It does not read silence as consent. No answer, no run.
        """
        if self.policy is None or not self.policy.denied:
            return None   # nothing is switched off, so nothing can be gated

        from ..autonomy.confirm import describe, disabled_freedoms

        needed = disabled_freedoms(getattr(spec, "effect_signature", ""), self.policy)
        if not needed:
            return None

        question = describe(needed, name, arguments)
        answer: bool | None = None
        if self.confirmer is not None:
            try:
                answer = self.confirmer(name, arguments, needed)
            except Exception as exc:  # noqa: BLE001 - a broken confirmer must not run the tool
                self._log("confirm", name, {"needed": needed, "outcome": "error",
                                            "error": f"{type(exc).__name__}: {exc}"})
                return ToolResult(
                    name, False, "",
                    error=(f"Denied: the confirmer raised {type(exc).__name__}: {exc}. "
                           f"A gate that cannot be asked is a gate that says no."),
                    awaiting_confirmation=True,
                )

        if answer is None:
            self._log("confirm", name, {"needed": needed, "outcome": "no_operator"})
            return ToolResult(
                name, False, "",
                error=(f"Denied: {question} There is nobody to ask in this run "
                       f"(no operator attached to the confirmation gate), so the "
                       f"answer is no. Re-enable {', '.join(needed)}, or run where "
                       f"an operator can answer."),
                awaiting_confirmation=True,
            )

        self._log("confirm", name,
                  {"needed": needed, "outcome": "confirmed" if answer else "refused"})
        if answer:
            return None
        return ToolResult(
            name, False, "", error=f"Denied by operator: {question}",
            awaiting_confirmation=True,
        )

    # -- health management ---------------------------------------------
    def _maybe_quarantine(self, spec: ToolSpec) -> None:
        st = spec.stats
        if st.calls < self.min_calls_for_judgement:
            return
        degraded = (
            st.success_rate < self.quarantine_success_rate
            or st.consecutive_failures >= self.quarantine_consecutive_failures
        )
        if degraded and spec.state in (ToolState.ACTIVE, ToolState.PROBATION):
            spec.state = ToolState.QUARANTINED
            self._log(
                "auto_quarantine",
                spec.name,
                {
                    "success_rate": round(st.success_rate, 3),
                    "consecutive_failures": st.consecutive_failures,
                    "calls": st.calls,
                },
            )

    def rehab(self, name: str, *, require_clean: int = 2) -> ToolSpec:
        """Bring a quarantined tool back on trial after it proves itself again."""
        spec = self._tools[name]
        spec.stats.consecutive_failures = 0
        spec.stats.calls = 0
        spec.stats.successes = 0
        spec.stats.failures = 0
        spec.state = ToolState.PROBATION
        self._log("rehab", name, {"require_clean": require_clean})
        return spec

    def retire(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].state = ToolState.RETIRED
            self._log("retire", name, {})

    def quarantine(self, name: str, reason: str = "manual") -> None:
        if name in self._tools:
            self._tools[name].state = ToolState.QUARANTINED
            self._log("quarantine", name, {"reason": reason})

    def promote(self, name: str) -> ToolSpec:
        spec = self._tools[name]
        spec.state = ToolState.ACTIVE
        self._log("promote", name, {})
        return spec

    # -- reporting -------------------------------------------------------
    def report(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for s in self._tools.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return {
            "total": len(self._tools),
            "by_state": by_state,
            "events": len(self._events),
            "tools": [s.to_dict() for s in self._tools.values()],
        }

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._events[-limit:]

    def _log(self, kind: str, name: str, extra: dict[str, Any]) -> None:
        self._events.append({"t": time.time(), "kind": kind, "tool": name, **extra})

    # -- scoping -----------------------------------------------------------
    def scoped(self, allowed, *, role: str = "") -> "ToolRegistry":
        """A view of this registry that can only see and call `allowed`.

        `allowed` empty means no restriction, and the registry itself is
        returned — so a node that names no whitelist behaves exactly as before
        and this stays a pure widening of what roles can express.
        """
        allowed = set(allowed or ())
        if not allowed:
            return self
        return ScopedRegistry(self, allowed, role=role)


class ScopedRegistry(ToolRegistry):
    """Restricts a child agent to a role's whitelist, for real.

    Exists because `tools_whitelist` used to be parsed, stored, round-tripped
    through the store and asserted in tests — and consulted by nothing. A role
    that does not restrict anything is a label, not a role. This makes it bite
    at both ends: the whitelisted subset is all the child's model can *see*
    (`schemas`), and all it can *run* (`call`). Both matter — hiding the rest
    stops the model asking, refusing stops it succeeding if it asks anyway.

    State lives on the parent, so stats, events and any tool a child forges
    still land in the one real registry. This is a lens, not a fork.
    """

    def __init__(self, base: ToolRegistry, allowed: set[str], *, role: str = "") -> None:
        # Deliberately not calling super().__init__: this view owns no tools of
        # its own. Every registration is the parent's.
        self._base = base
        self._allowed = set(allowed)
        self.role = role
        self.refusals: list[str] = []

    def allows(self, name: str) -> bool:
        return name in self._allowed

    def names(self) -> list[str]:
        return [n for n in self._base.names() if n in self._allowed]

    def schemas(self, *, include: set[ToolState] | None = None) -> list[dict[str, Any]]:
        return [
            s for s in self._base.schemas(include=include)
            if s.get("function", {}).get("name") in self._allowed
        ]

    def call(self, name: str, arguments: dict[str, Any], *,
             force: bool = False) -> ToolResult:
        if name not in self._allowed:
            self.refusals.append(name)
            label = f" ({self.role})" if self.role else ""
            return ToolResult(
                name, False, "",
                error=(f"Denied: '{name}' is outside this agent's role{label}. "
                       f"Allowed here: {', '.join(sorted(self._allowed)) or 'nothing'}."),
            )
        return self._base.call(name, arguments, force=force)

    def __getattr__(self, item: str) -> Any:
        # Everything else — register, unregister, report, the ledger — belongs
        # to the parent. __getattr__ only fires when normal lookup fails, and
        # _base/_allowed/role/refusals are instance attributes, so they do not
        # reach here.
        return getattr(self._base, item)


__all__ = ["ToolRegistry", "ScopedRegistry", "ToolResult"]
