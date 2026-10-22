"""SEPL — the closed-loop evolution operator, with teeth.

AGP's Self-Evolution Protocol Layer names a closed loop: **propose → assess →
commit**. That naming is the contribution; the enforcement is ours.

The load-bearing design choice, inherited from autoforge and preserved here
without softening, is the *order*:

    1. propose   — a candidate is registered as a new DRAFT node. It is not the
                   live resource yet and cannot be called as one.
    2. assess    — the independent validity gate runs FIRST. A candidate that
                   dropped a baseline probe or introduced an undeclared side
                   effect is refused here, *before* any fitness number exists.
                   Only survivors are verified and scored.
    3. commit    — the best admissible candidate is promoted: a new version
                   advances, the state moves along the enforcement lifecycle,
                   a lineage edge is drawn, and events are appended.

Why the order is the whole point: if fitness were computed first, a mutant that
deleted its own guardrail could score higher than the tool it replaced and win
the competition — it would be optimising its own exam. Running the gate first
means the veto is not a penalty a mutant can out-earn. There is no score to
trade against admissibility.

The operator is deliberately *proposer-agnostic*: it accepts any callable that
turns a base tool + goal into candidate `ToolSpec`s, so the loop is testable
with zero LLM calls (`StubProposer`) and runs for real with autoforge's
`EvolutionEngine` when a model is wired in.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from toolmarket.protocol.events import EventKind, EventLog
from toolmarket.protocol.lifecycle import (
    LifecycleError,
    ResourceState,
    VersionStatus,
    transition,
)
from toolmarket.protocol.lineage import LineageNode
from toolmarket.protocol.resources import ResourceRecord

# -- autoforge imports, defended -------------------------------------------
# The enforcement machinery lives in autoforge; we import it, we do not
# reimplement it. If it is missing we fail loudly rather than silently degrade
# into a gate-less substrate (which would defeat the entire premise).
try:  # pragma: no cover - exercised by import, not by logic
    from autoforge.forge.validity import ValidityGate, ValidityReport
    from autoforge.forge.verifier import ToolVerifier, VerificationReport
    from autoforge.core.llm import MockLLMClient
    from autoforge.tools.spec import ToolSpec
    _HAVE_AUTOFORGE = True
except Exception as exc:  # noqa: BLE001
    _HAVE_AUTOFORGE = False
    _AUTOFORGE_ERR = exc
    ValidityGate = object  # type: ignore
    ValidityReport = object  # type: ignore
    ToolVerifier = object  # type: ignore
    VerificationReport = object  # type: ignore
    MockLLMClient = object  # type: ignore
    ToolSpec = object  # type: ignore


def _require_autoforge() -> None:
    if not _HAVE_AUTOFORGE:
        raise RuntimeError(
            "toolmarket requires autoforge for its enforcement layer, but "
            f"import failed: {_AUTOFORGE_ERR!r}. Install it with: "
            "pip install -e <path-to>/autoforge"
        ) from _AUTOFORGE_ERR


# -- proposer protocol ------------------------------------------------------
class Proposer(Protocol):
    """Anything that can turn (base tool, goal) into candidate specs."""

    def __call__(self, base: Any, goal: str) -> list[Any]:  # -> list[ToolSpec]
        ...


class StubProposer:
    """Deterministic candidates for tests and dry runs. No model, no network.

    Given a goal, it emits a fixed set of candidates, one of which is knowingly
    *inadmissible* (it drops a declared invariance) so the gate's veto path is
    exercised end-to-end without an LLM.
    """

    def __init__(self, variants: Optional[list[str]] = None,
                 sabotage: bool = True) -> None:
        self.variants = variants or ["patch", "reimplement"]
        self.sabotage = sabotage

    def __call__(self, base: Any, goal: str) -> list[Any]:
        _require_autoforge()
        out: list[Any] = []
        for v in self.variants:
            code = (base.code or "") + f"\n# variant: {v} :: goal={goal[:60]}\n"
            if v == "patch":
                code += "def _variant_marker():  # patched\n    return True\n"
            out.append(
                ToolSpec(
                    name=base.name,
                    description=base.description + f" [{v}]",
                    parameters=dict(base.parameters),
                    fn=base.fn,
                    code=code,
                    source="generated",
                    generator=f"stub:{v}",
                    probes=list(base.probes),
                    effect_signature=base.effect_signature,
                    # Deliberately preserve invariances on the good candidates.
                    invariances=list(base.invariances),
                )
            )
        if self.sabotage:
            # A candidate that drops a declared invariance — the gate must catch
            # this BEFORE it is scored.
            bad = ToolSpec(
                name=base.name,
                description=base.description + " [sabotaged]",
                parameters=dict(base.parameters),
                fn=base.fn,
                code=(base.code or "") + "\n# variant: sabotage\n",
                source="generated",
                generator="stub:sabotage",
                probes=[],
                effect_signature=base.effect_signature,
                invariances=[],  # <-- the drop the regression gate must veto
            )
            out.append(bad)
        return out


# -- proposal + assessment records -----------------------------------------
@dataclass
class Proposal:
    proposal_id: str
    resource_id: str
    base_version: str
    goal: str
    candidates: list[ResourceRecord] = field(default_factory=list)
    # candidate index -> rehydrated autoforge ToolSpec (kept off the wire)
    _specs: list[Any] = field(default_factory=list, repr=False)
    created_at: float = field(default_factory=time.time)
    status: str = "proposed"  # proposed | assessed | committed | rejected


@dataclass
class AssessmentReport:
    proposal_id: str
    resource_id: str
    # per-candidate verdicts, in candidate order
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    admissible_indices: list[int] = field(default_factory=list)
    best_index: Optional[int] = None
    admissible: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "resource_id": self.resource_id,
            "verdicts": self.verdicts,
            "admissible_indices": self.admissible_indices,
            "best_index": self.best_index,
            "admissible": self.admissible,
            "summary": self.summary,
        }


class ProposalRejected(Exception):
    """Raised by `commit` when no candidate cleared the gate."""

    def __init__(self, report: AssessmentReport) -> None:
        super().__init__(report.summary or "no admissible candidate")
        self.report = report


# -- the operator -----------------------------------------------------------
class EvolutionOperator:
    """Drives propose → assess → commit over a `ResourceRegistry`."""

    def __init__(
        self,
        registry: Any,  # toolmarket.registry.ResourceRegistry (avoid cycle)
        *,
        gate: Any = None,
        verifier: Any = None,
        proposer: Optional[Proposer] = None,
        on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        self.registry = registry
        self.log: EventLog = registry.log
        self.gate = gate
        self.verifier = verifier
        self.proposer = proposer or StubProposer()
        # Mirror autoforge's choice: the gate is ON unless it is deliberately
        # and visibly replaced. A substrate without a gate is a toy.
        self._gate_defaults = gate is None
        self.on_event = on_event
        self._proposals: dict[str, Proposal] = {}
        self._assessments: dict[str, AssessmentReport] = {}

    # -- wiring -----------------------------------------------------------
    def _ensure_gate(self) -> Any:
        if self.gate is None:
            _require_autoforge()
            self.gate = ValidityGate()
        return self.gate

    def _ensure_verifier(self) -> Any:
        if self.verifier is None:
            _require_autoforge()
            # Prefer the registry's deterministic, network-free verifier so
            # assessment is reproducible offline. Fall back conservatively.
            if hasattr(self.registry, "offline_verifier"):
                self.verifier = self.registry.offline_verifier()
            else:
                try:
                    self.verifier = ToolVerifier(MockLLMClient())
                except Exception:  # noqa: BLE001
                    self.verifier = ToolVerifier(MockLLMClient(),
                                                 run_adversarial_check=False,
                                                 run_trigger_check=False,
                                                 run_negative_check=False)
        return self.verifier

    # -- 1. propose -------------------------------------------------------
    def propose(
        self,
        resource_id: str,
        goal: str,
        *,
        proposer: Optional[Proposer] = None,
    ) -> Proposal:
        """Register a set of DRAFT candidates. Nothing live changes yet."""
        _require_autoforge()
        base_rec = self.registry.get(resource_id)
        if base_rec is None:
            raise KeyError(f"unknown resource: {resource_id}")

        props = proposer or self.proposer
        # Propose against the *current* live spec so enforcement sees exactly
        # what is deployed.
        base_spec = self.registry.to_spec(base_rec)
        candidates = props(base_spec, goal)

        proposal = Proposal(
            proposal_id=f"p_{uuid.uuid4().hex[:12]}",
            resource_id=resource_id,
            base_version=base_rec.version_current.version,
            goal=goal,
        )
        for i, spec in enumerate(candidates):
            rec = ResourceRecord.from_toolspec(spec)
            rec.state = ResourceState.DRAFT
            rec.lineage_parents = [self._node_id(base_rec, base_rec.version_current.version)]
            rec.metadata["proposal_id"] = proposal.proposal_id
            rec.metadata["candidate_index"] = i
            rec.version_current.note = f"proposed for: {goal[:80]}"
            proposal.candidates.append(rec)
            proposal._specs.append(spec)

        self._proposals[proposal.proposal_id] = proposal
        self.log.append(
            EventKind.PROPOSE, resource_id,
            data={
                "proposal_id": proposal.proposal_id,
                "goal": goal,
                "base_version": proposal.base_version,
                "n_candidates": len(proposal.candidates),
                "candidates": [c.name for c in proposal.candidates],
            },
        )
        self._emit("propose", {"proposal_id": proposal.proposal_id,
                               "n": len(proposal.candidates)})
        return proposal

    # -- 2. assess --------------------------------------------------------
    def assess(self, proposal_id: str) -> AssessmentReport:
        """Run the gate FIRST, then verify survivors. The veto precedes the score."""
        _require_autoforge()
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"unknown proposal: {proposal_id}")
        gate = self._ensure_gate()
        verifier = self._ensure_verifier()

        report = AssessmentReport(
            proposal_id=proposal_id,
            resource_id=proposal.resource_id,
        )

        baseline = self.registry.baseline_for(proposal.resource_id)
        # Re-verify in the context the resource is actually deployed in. The
        # context is a property of the resource, not a knob the caller picks —
        # otherwise a proposal could be waved through by asserting a laxer
        # context than the one the tool really runs in.
        base_rec = self.registry.require(proposal.resource_id)
        context = (base_rec.metadata.get("context") or "public")

        for i, (rec, spec) in enumerate(zip(proposal.candidates, proposal._specs)):
            started = time.perf_counter()
            verdict = None
            try:
                verdict = gate.evaluate(spec, baseline=baseline, context=context)
            except Exception as exc:  # noqa: BLE001 - gate failure is a refusal
                report.verdicts.append({
                    "index": i,
                    "candidate": rec.name,
                    "generator": rec.provenance.get("generator", ""),
                    "admissible": False,
                    "reason": f"gate-error: {exc}",
                    "gate": None,
                    "verification": None,
                    "fitness": 0.0,
                    "ms": (time.perf_counter() - started) * 1000,
                })
                continue

            vdict = verdict.to_dict() if hasattr(verdict, "to_dict") else {}

            if not getattr(verdict, "admissible", False):
                violated = [f.gate for f in getattr(verdict, "violated", [])]
                report.verdicts.append({
                    "index": i,
                    "candidate": rec.name,
                    "generator": rec.provenance.get("generator", ""),
                    "admissible": False,
                    "reason": "validity veto: " + ", ".join(violated or ["unknown"]),
                    "gate": vdict,
                    "verification": None,
                    "fitness": 0.0,
                    "ms": (time.perf_counter() - started) * 1000,
                })
                continue

            # Only now does the candidate get to compete.
            vreport = verifier.verify(spec)
            rdict = vreport.to_dict() if hasattr(vreport, "to_dict") else {}
            fitness = self._fitness(vreport)
            rec.contract.verification = rdict
            report.verdicts.append({
                "index": i,
                "candidate": rec.name,
                "generator": rec.provenance.get("generator", ""),
                "admissible": True,
                "reason": "cleared gate",
                "gate": vdict,
                "verification": rdict,
                "fitness": fitness,
                "ms": (time.perf_counter() - started) * 1000,
            })
            report.admissible_indices.append(i)

        if report.admissible_indices:
            best = max(report.admissible_indices,
                       key=lambda j: report.verdicts[j]["fitness"])
            report.best_index = best
            report.admissible = True
            report.summary = (
                f"{len(report.admissible_indices)}/{len(proposal.candidates)} "
                f"admissible; best index {best} "
                f"(fitness {report.verdicts[best]['fitness']:.3f})"
            )
        else:
            report.summary = (
                f"0/{len(proposal.candidates)} admissible — every candidate "
                "was vetoed at the gate; the incumbent stays"
            )

        proposal.status = "assessed"
        self._assessments[proposal_id] = report

        self.log.append(
            EventKind.ASSESS, proposal.resource_id,
            data={
                "proposal_id": proposal_id,
                "admissible": report.admissible,
                "admissible_indices": report.admissible_indices,
                "best_index": report.best_index,
                "vetoed": [v["candidate"] for v in report.verdicts
                           if not v["admissible"]],
            },
        )
        self._emit("assess", {"proposal_id": proposal_id,
                              "admissible": report.admissible})
        return report

    # -- 3. commit --------------------------------------------------------
    def commit(self, proposal_id: str) -> ResourceRecord:
        """Promote the best admissible candidate into the live version."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"unknown proposal: {proposal_id}")
        report = self._assessments.get(proposal_id)
        if report is None:
            report = self.assess(proposal_id)

        if not report.admissible or report.best_index is None:
            proposal.status = "rejected"
            self.log.append(
                EventKind.REJECT, proposal.resource_id,
                data={"proposal_id": proposal_id, "reason": report.summary},
            )
            self._emit("reject", {"proposal_id": proposal_id})
            raise ProposalRejected(report)

        best = proposal.candidates[report.best_index]
        incumbent = self.registry.get(proposal.resource_id)

        # The new version adopts the candidate's contract + verification; the
        # live ledger and lineage continue from the incumbent.
        new = self.registry.replace_live(
            proposal.resource_id,
            best.contract,
            note=f"evolved: {proposal.goal[:80]}",
            mutation="mutate",
            parent_node=self._node_id(incumbent, proposal.base_version),
            verification=best.contract.verification,
        )
        proposal.status = "committed"
        self.log.append(
            EventKind.COMMIT, proposal.resource_id,
            data={
                "proposal_id": proposal_id,
                "from_version": proposal.base_version,
                "to_version": new.version_current.version,
                "best_index": report.best_index,
                "fitness": report.verdicts[report.best_index]["fitness"],
            },
        )
        self._emit("commit", {"proposal_id": proposal_id,
                              "version": new.version_current.version})
        return new

    # -- rollback ---------------------------------------------------------
    def rollback(self, resource_id: str, to_version: str) -> ResourceRecord:
        """Restore an archived version as the live one. History is preserved.

        This is AGP's "atomic rollback" made concrete against the provenance
        axis: the target version's status returns to ACTIVE, the version being
        displaced is ARCHIVED, and a lineage node records the reversal.
        """
        rec = self.registry.get(resource_id)
        if rec is None:
            raise KeyError(f"unknown resource: {resource_id}")
        target = next((v for v in rec.versions if v.version == to_version), None)
        if target is None:
            raise KeyError(f"{resource_id} has no version {to_version}")

        displaced = rec.version_current
        if displaced.version == to_version:
            raise ValueError(f"{to_version} is already live")

        displaced.status = VersionStatus.ARCHIVED
        target.status = VersionStatus.ACTIVE
        rec.version_current = target
        rec.updated_at = time.time()
        self.registry.save(rec)

        node = self._node_id(rec, target.version)
        # A rollback is its own lineage event: a new node whose parent is the
        # version it displaced (so the DAG records *why*, not just *what*).
        self.registry.add_lineage_node(
            LineageNode(
                node_id=f"{node}#rollback-{int(time.time())}",
                resource_id=resource_id,
                version=target.version,
                parents=[self._node_id(rec, displaced.version)],
                reason=f"rollback from {displaced.version}",
                mutation="rollback",
                created_at=time.time(),
            )
        )
        self.log.append(
            EventKind.ROLLBACK, resource_id,
            data={"from_version": displaced.version, "to_version": target.version},
        )
        self._emit("rollback", {"resource_id": resource_id, "to": target.version})
        return rec

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _node_id(rec: ResourceRecord, version: str) -> str:
        return f"{rec.id}@{version}"

    @staticmethod
    def _fitness(report: Any) -> float:
        """Fitness = passed checks / total. Deterministic, no self-report."""
        try:
            checks = list(getattr(report, "checks", []) or [])
            if checks:
                passed = sum(1 for c in checks if getattr(c, "passed", False))
                return passed / len(checks)
        except Exception:  # noqa: BLE001
            pass
        return 1.0 if getattr(report, "passed", True) else 0.0

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(name, payload)
            except Exception:  # noqa: BLE001
                pass
