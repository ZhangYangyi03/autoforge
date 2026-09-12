"""Verification: the gate between "forged" and "trusted".

Two orthogonal questions, both tested (see DESIGN.md §2.4):

  A. Does it RUN?      — execution check: call it, does it do the right thing
                         without crashing?
  B. Does it FIRE?     — trigger check: when the need arises, will the agent
                         actually call it? And will it stay quiet when the need
                         is absent?

Question B is the one mainstream frameworks skip, and it is exactly where the
"Constraint Tax" lives: a tool can be perfectly correct and still never be
invoked. We test it by handing the probe query to a live agent with the tool
registered, and observing whether the tool call appears.

We also test the NEGATIVE case. A tool that fires on everything is worse than
one that never fires, because it steals calls from better tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.agent import Agent
from ..core.llm import LLMClient
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState
from .sandbox import Sandbox


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    tool: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.passed)
        return f"{self.tool}: {ok}/{len(self.checks)} checks passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "evidence": c.evidence}
                for c in self.checks
            ],
        }


class ToolVerifier:
    """Runs the check battery against a freshly forged tool."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: Sandbox | None = None,
        run_execution_check: bool = True,
        run_trigger_check: bool = True,
        run_negative_check: bool = True,
        trigger_trials: int = 1,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox or Sandbox()
        self.run_execution_check = run_execution_check
        self.run_trigger_check = run_trigger_check
        self.run_negative_check = run_negative_check
        self.trigger_trials = trigger_trials

    # -- individual checks --------------------------------------------
    @staticmethod
    def _infer_args(spec: ToolSpec) -> dict[str, Any]:
        """Generate plausible sample arguments from the parameter schema."""
        props = spec.parameters.get("properties") or {}
        args = {}
        for name, schema in props.items():
            t = schema.get("type", "string")
            if t == "string":
                args[name] = "978-0-306-40615-7"
            elif t in ("number", "integer"):
                args[name] = 1
            elif t == "boolean":
                args[name] = True
            elif t == "object":
                args[name] = {}
            elif t == "array":
                args[name] = []
            else:
                args[name] = "test"
        return args

    def check_execution(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> CheckResult:
        """A. Run it out-of-process with sample args."""
        if sample_args is None:
            sample_args = self._infer_args(spec)
        result = self.sandbox.run(spec.code, spec.name, sample_args)
        if result.timed_out:
            return CheckResult("execution", False, "timed out", {"timeout": self.sandbox.timeout})
        if not result.ok:
            return CheckResult("execution", False, result.error or "failed", {})
        return CheckResult(
            "execution", True, "ran clean",
            {"output": str(result.output)[:200], "duration_ms": round(result.duration_ms, 1)},
        )

    def check_trigger(self, spec: ToolSpec, query: str) -> CheckResult:
        """B+. Register the tool alone, ask the query, see if it fires."""
        registry = self._probe_registry()
        registry.register(spec)
        hits = 0
        seen: list[str] = []
        for _ in range(self.trigger_trials):
            agent = Agent(self.llm, registry, max_turns=2)
            res = agent.run(query)
            seen.extend(res.tool_calls)
            # One trial = one vote. A tool called twice in a turn still counts
            # once; otherwise a chatty agent inflates the trigger score.
            if spec.name in res.tool_calls:
                hits += 1
        passed = hits > 0
        return CheckResult(
            "trigger", passed,
            f"fired {hits}/{self.trigger_trials} on the positive probe",
            {"query": query, "tool_calls": seen},
        )

    def check_negative(self, spec: ToolSpec, query: str) -> CheckResult:
        """B-. Ask an unrelated-but-adjacent query; the tool must stay quiet."""
        registry = self._probe_registry()
        registry.register(spec)
        agent = Agent(self.llm, registry, max_turns=2)
        res = agent.run(query)
        over_fired = spec.name in res.tool_calls
        return CheckResult(
            "negative", not over_fired,
            "stayed quiet" if not over_fired else "over-fired on a negative probe",
            {"query": query, "tool_calls": res.tool_calls},
        )

    @staticmethod
    def _probe_registry() -> ToolRegistry:
        """A harness registry that can SEE draft tools — otherwise the trigger
        probe would be testing a tool the agent is not allowed to call."""
        return ToolRegistry(
            auto_quarantine=False,
            visible_states={ToolState.DRAFT, ToolState.PROBATION, ToolState.ACTIVE},
        )

    # -- battery --------------------------------------------------------
    def verify(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> VerificationReport:
        checks: list[CheckResult] = []

        if self.run_execution_check:
            checks.append(self.check_execution(spec, sample_args))

        if self.run_trigger_check:
            positive = [p for p in spec.probes if p.expect == "call"]
            if not positive:
                checks.append(CheckResult("trigger", False, "no positive probes defined"))
            else:
                for probe in positive[:2]:
                    checks.append(self.check_trigger(spec, probe.query))

        if self.run_negative_check:
            negatives = [p.negative_query for p in spec.probes if p.negative_query]
            if negatives:
                checks.append(self.check_negative(spec, negatives[0]))

        passed = bool(checks) and all(c.passed for c in checks)
        return VerificationReport(spec.name, passed, checks)


def register_if_verified(
    registry: ToolRegistry,
    spec: ToolSpec,
    verifier: ToolVerifier,
    *,
    sample_args: dict[str, Any] | None = None,
    promote_on_pass: bool = False,
) -> VerificationReport:
    """Verify, then place the tool in the lifecycle accordingly.

    Passing verification earns PROBATION (callable + context-visible). Failing
    leaves it DRAFT (registered, but not injected). ACTIVE is reserved for tools
    that have also held up in live use, or for `promote_on_pass`.
    """
    report = verifier.verify(spec, sample_args)
    spec.verification = report.to_dict()
    if report.passed:
        spec.state = ToolState.ACTIVE if promote_on_pass else ToolState.PROBATION
    else:
        spec.state = ToolState.DRAFT
    registry.register(spec)
    return report


__all__ = [
    "ToolVerifier",
    "VerificationReport",
    "CheckResult",
    "register_if_verified",
    "json",
]
