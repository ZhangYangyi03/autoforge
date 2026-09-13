"""The forge pipeline: the three-stage closed loop.

    FORGE ────────> VERIFY ────────> SEAL
    (generate)      (exec + trigger)   (register in lifecycle)

and the living half:

    OBSERVE ──────> JUDGE ─────────> ACT
    (ledger)        (degrade?)        (quarantine / rehab / retire)

Why a pipeline object rather than loose functions: the stages have policy —
how many tries before giving up, whether to promote on pass, whether to
re-verify on drift. Policy belongs somewhere inspectable, so it lives here
with a full audit log of every forge attempt.

The design bet (DESIGN.md §2): most frameworks stop at FORGE. The failures
that actually hurt show up in the living half — a tool that never fires, or
one that fired well and quietly rotted. autoforge closes both loops.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..autonomy.policy import AutonomyPolicy
from ..core.llm import LLMClient
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState
from .generator import GeneratedTool, TemplateGenerator, UnrecoverableGeneration
from .sandbox import Sandbox
from .verifier import ToolVerifier, VerificationReport


class ToolGenerator(Protocol):  # pragma: no cover - structural
    def generate(self, need: str, context: str = "") -> GeneratedTool: ...


@dataclass
class ForgeAttempt:
    need: str
    round: int
    generated: GeneratedTool | None = None
    report: VerificationReport | None = None
    error: str | None = None
    duration_ms: float = 0.0
    accepted: bool = False


@dataclass
class ForgeResult:
    need: str
    spec: ToolSpec | None
    attempts: list[ForgeAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.spec is not None

    @property
    def rounds(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "need": self.need,
            "ok": self.ok,
            "rounds": self.rounds,
            "tool": self.spec.to_dict() if self.spec else None,
            "attempts": [
                {
                    "round": a.round,
                    "error": a.error,
                    "accepted": a.accepted,
                    "report": a.report.to_dict() if a.report else None,
                }
                for a in self.attempts
            ],
        }


@dataclass
class ForgeConfig:
    max_rounds: int = 3              # generate→verify retries (feeds failure back)
    promote_on_pass: bool = True     # ACTIVE on a clean verification
    sample_args: dict[str, Any] | None = None
    require_execution: bool = True
    require_trigger: bool = True
    require_negative: bool = True


class ForgePipeline:
    def __init__(
        self,
        generator: ToolGenerator,
        verifier: ToolVerifier,
        registry: ToolRegistry,
        *,
        config: ForgeConfig | None = None,
        sandbox: Sandbox | None = None,
        policy: AutonomyPolicy | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.generator = generator
        self.verifier = verifier
        self.registry = registry
        self.config = config or ForgeConfig()
        self.sandbox = sandbox or verifier.sandbox
        # None means "no policy attached": every gate below fails open, which
        # is what the forge pipeline did before the policy existed. Attaching
        # a policy is how a caller makes the verdicts binding.
        self.policy = policy
        self.on_event = on_event
        self.log: list[dict[str, Any]] = []

    def _may_promote(self) -> bool:
        """Whether a clean verification may seal a tool as ACTIVE.

        Turning may_promote_tools off does NOT stop the tool being forged or
        called — creation stays free and PROBATION tools are already in the
        model's context. It only withholds ACTIVE, the state that means
        "trusted without further evidence".
        """
        if not self.config.promote_on_pass:
            return False
        return self.policy is None or self.policy.may_promote_tools

    # -- the loop --------------------------------------------------------
    def forge(self, need: str, context: str = "") -> ForgeResult:
        result = ForgeResult(need, None)
        feedback = ""
        existing = ", ".join(self.registry.names()) or "(none)"

        for round_no in range(1, self.config.max_rounds + 1):
            started = time.perf_counter()
            attempt = ForgeAttempt(need, round_no)
            # Set when retrying cannot possibly change the outcome, so the loop
            # stops after recording the attempt instead of spending another
            # round's budget to observe the same wall.
            futile = False
            try:
                prompt_need = need if not feedback else self._repair_prompt(need, feedback)
                generated = self.generator.generate(prompt_need, context or existing)
                attempt.generated = generated
                spec = self._to_spec(generated)

                report = self.verifier.verify(spec, self.config.sample_args)
                attempt.report = report
                spec.verification = report.to_dict()

                if report.passed:
                    if self._may_promote():
                        spec.state = ToolState.ACTIVE
                    else:
                        # PROBATION tools are already visible to the model, so
                        # nothing is lost but the seal. Record why, so a tool
                        # sitting on probation is never a mystery.
                        spec.state = ToolState.PROBATION
                        self._emit("promote_withheld", {
                            "tool": spec.name,
                            "reason": (
                                "may_promote_tools is off"
                                if self.policy is not None and not self.policy.may_promote_tools
                                else "promote_on_pass is off"
                            ),
                        })
                    self.registry.register(spec)
                    attempt.accepted = True
                    result.spec = spec
                else:
                    spec.state = ToolState.DRAFT
                    feedback = self._feedback(report)
            except UnrecoverableGeneration as exc:
                # Nothing about this failure is round-specific: the model, not
                # the attempt, cannot produce an answer. Record it and stop.
                futile = True
                attempt.error = f"{type(exc).__name__}: {exc}"
                self._emit("forge_error", {
                    "round": round_no,
                    "error": attempt.error,
                    "futile": True,
                    "traceback": traceback.format_exc(),
                })
            except Exception as exc:  # noqa: BLE001
                attempt.error = f"{type(exc).__name__}: {exc}"
                # A framework whose whole point is judging generated code cannot
                # afford to swallow its own tracebacks. Keep them in the log.
                self._emit("forge_error", {
                    "round": round_no,
                    "error": attempt.error,
                    "traceback": traceback.format_exc(),
                })
                feedback = attempt.error

            attempt.duration_ms = (time.perf_counter() - started) * 1000
            result.attempts.append(attempt)
            self._emit("forge_attempt", {
                "need": need, "round": round_no, "accepted": attempt.accepted,
                "error": attempt.error,
                "report": attempt.report.to_dict() if attempt.report else None,
            })
            if attempt.accepted or futile:
                break

        self._emit("forge_done", {"need": need, "ok": result.ok, "rounds": result.rounds})
        return result

    # -- helpers ---------------------------------------------------------
    def _to_spec(self, g: GeneratedTool) -> ToolSpec:
        """Wrap a generated tool as a spec whose execution goes through the sandbox.

        The `fn` is a tripwire: any in-process call is a bug, so it raises
        loudly rather than silently running untrusted code in the agent loop.
        The `runner` is the real path — it shells out to the sandbox.
        """
        sandbox = self.sandbox

        def _bridge(name: str, args: dict[str, Any]) -> Any:
            res = sandbox.run(g.code, g.entry, args)
            if res.timed_out:
                raise TimeoutError(f"tool {name} timed out after {sandbox.timeout}s")
            if not res.ok:
                raise RuntimeError(res.error or "sandbox failure")
            return res.output if res.output is not None else (res.stdout or "")

        def _tripwire(**_: Any) -> Any:
            raise RuntimeError(
                "generated tool attempted in-process execution; use the sandbox runner"
            )

        return ToolSpec(
            name=g.name,
            description=g.description,
            parameters=g.parameters,
            fn=_tripwire,
            runner=_bridge,
            code=g.code,
            source="generated",
            probes=g.probes,
            effect_signature=g.effect_signature,
            tags=g.tags,
            state=ToolState.DRAFT,
        )

    @staticmethod
    def _feedback(report: VerificationReport) -> str:
        lines = [f"- {c.name}: {c.detail}" for c in report.failed]
        return "Failed checks:\n" + "\n".join(lines)

    @staticmethod
    def _repair_prompt(original: str, feedback: str) -> str:
        return (
            f"{original}\n\n"
            f"Your previous tool did not pass verification:\n{feedback}\n"
            "Fix the root cause. Keep the same behaviour, make it robust."
        )

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"t": time.time(), "kind": kind, **payload}
        self.log.append(entry)
        if self.on_event:
            self.on_event(kind, payload)

    # -- the living half -------------------------------------------------
    def judge(self, name: str) -> dict[str, Any]:
        """Inspect a tool's ledger and act if it has degraded."""
        spec = self.registry.get(name)
        if spec is None:
            return {"tool": name, "action": "unknown"}
        st = spec.stats
        before = spec.state
        if spec.state in (ToolState.ACTIVE, ToolState.PROBATION):
            self.registry._maybe_quarantine(spec)
        action = "none"
        if spec.state != before:
            action = f"{before.value}->{spec.state.value}"
        return {
            "tool": name,
            "action": action,
            "state": spec.state.value,
            "success_rate": round(st.success_rate, 3),
            "calls": st.calls,
        }


__all__ = [
    "ForgePipeline",
    "ForgeConfig",
    "ForgeResult",
    "ForgeAttempt",
    "TemplateGenerator",
    "Sandbox",
    "LLMClient",
]
