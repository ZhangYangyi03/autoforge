"""Verification: the gate between "forged" and "trusted".

Five orthogonal checks, in order of increasing cost:

  A. execution     — does the code run on a standard input without crashing?
  B. robustness    — does it survive edge-case inputs? (deterministic fuzzer)
  C. adversarial   — can it survive an LLM attacker? (adversarial gate)
  D. trigger       — when the need arises, will the agent call this tool?
  E. negative      — when the need is absent, will it stay quiet?

Question D+E is the "Constraint Tax" and "over-triggering" problems. Question B
guards against the exact bug found in the live demo (ISBN prefix contamination).
Question C is unique to autoforge — no framework runs an attacker against its
own tools before trusting them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.agent import Agent
from ..core.llm import LLMClient
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState, normalise_parameters
from .adversary import AdversarialGate, AdversarialReport
from .fuzzer import RobustnessResult, run_robustness_checks
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
    """Runs the full check battery against a freshly forged tool.

    Checks (in order of increasing cost):
      A. execution — does the code run without crashing on a standard input?
      B. robustness — does it survive edge-case inputs? (deterministic fuzzer)
      C. adversarial — can it survive an LLM attacker?
      D. trigger — does the agent call it when appropriate?
      E. negative — does it stay quiet when inappropriate?
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: Sandbox | None = None,
        run_execution_check: bool = True,
        run_robustness_check: bool = True,
        run_adversarial_check: bool = True,
        run_trigger_check: bool = True,
        run_negative_check: bool = True,
        trigger_trials: int = 1,
        adversary: AdversarialGate | None = None,
        require_robustness_rate: float = 0.8,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox or Sandbox()
        self.run_execution_check = run_execution_check
        self.run_robustness_check = run_robustness_check
        self.run_adversarial_check = run_adversarial_check
        self.run_trigger_check = run_trigger_check
        self.run_negative_check = run_negative_check
        self.trigger_trials = trigger_trials
        self.adversary = adversary
        self.require_robustness_rate = require_robustness_rate

    # -- individual checks (A–E) ----------------------------------------
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

    def check_robustness(self, spec: ToolSpec) -> CheckResult:
        """B. Deterministic edge-case fuzzing — survives the ISBN prefix bug?"""
        result = run_robustness_checks(
            spec, sandbox=self.sandbox, require_survival_rate=self.require_robustness_rate,
        )
        # Name the probes that failed. Without this the retry only sees "0/19"
        # and cannot learn which inputs broke it, so the feedback loop is inert.
        detail = result.summary()
        if result.failures:
            shown = "; ".join(
                f"{f.get('label', '?')} -> {str(f.get('error', ''))[:60]}"
                for f in result.failures[:5]
            )
            detail += f". Failing probes: {shown}"
        return CheckResult(
            "robustness", result.passed,
            detail,
            {"survival_rate": round(result.survival_rate, 3),
             "survived": result.survived, "total": result.total,
             "invariance": (result.invariance.to_dict()
                            if result.invariance is not None else None)},
        )

    def check_adversarial(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> CheckResult:
        """C. An LLM attacker tries to break the tool."""
        gate = self.adversary or AdversarialGate(self.llm, execution_sandbox=self.sandbox)
        report = gate.attack(spec)
        return CheckResult(
            "adversarial", report.passed,
            report.summary(),
            {"survived": report.survived, "total": report.total_attacks},
        )

    def check_trigger(self, spec: ToolSpec, query: str) -> CheckResult:
        """D. Register the tool alone, ask the query, see if it fires."""
        registry = self._probe_registry()
        registry.register(spec)
        hits = 0
        seen: list[str] = []
        for _ in range(self.trigger_trials):
            # No terminate tool here: the probe measures whether THIS tool
            # fires, so the context must contain nothing else that competes.
            agent = Agent(self.llm, registry, max_turns=2, allow_self_terminate=False)
            res = agent.run(query)
            seen.extend(res.tool_calls)
            if spec.name in res.tool_calls:
                hits += 1
        passed = hits > 0
        return CheckResult(
            "trigger", passed,
            f"fired {hits}/{self.trigger_trials} on the positive probe",
            {"query": query, "tool_calls": seen},
        )

    def check_negative(self, spec: ToolSpec, query: str) -> CheckResult:
        """E. Ask an unrelated-but-adjacent query; the tool must stay quiet."""
        registry = self._probe_registry()
        registry.register(spec)
        agent = Agent(self.llm, registry, max_turns=2, allow_self_terminate=False)
        res = agent.run(query)
        over_fired = spec.name in res.tool_calls
        return CheckResult(
            "negative", not over_fired,
            "stayed quiet" if not over_fired else "over-fired on a negative probe",
            {"query": query, "tool_calls": res.tool_calls},
        )

    @staticmethod
    def _probe_registry() -> ToolRegistry:
        return ToolRegistry(
            auto_quarantine=False,
            visible_states={ToolState.DRAFT, ToolState.PROBATION, ToolState.ACTIVE},
        )

    # -- battery --------------------------------------------------------
    def verify(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> VerificationReport:
        # Settle the parameter shape before any check reads it. The generator
        # and the store already normalise at their trust boundaries; a spec
        # built by hand still reaches here raw, and every check below reads
        # `schema.get("type", ...)` per property. The judge establishes its own
        # preconditions rather than dying three frames deep.
        spec.parameters = normalise_parameters(spec.parameters)

        checks: list[CheckResult] = []

        if self.run_execution_check:
            checks.append(self.check_execution(spec, sample_args))

        if self.run_robustness_check and spec.code:
            checks.append(self.check_robustness(spec))

        if self.run_adversarial_check and spec.code:
            checks.append(self.check_adversarial(spec, sample_args))

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
    """Verify, then place the tool in the lifecycle accordingly."""
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