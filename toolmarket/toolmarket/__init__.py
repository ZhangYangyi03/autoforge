"""toolmarket — a protocol-registered resource platform for evolvable tools.

Positioning (and the honest scope of the claim)
-----------------------------------------------
The Autogenesis Protocol (AGP, arXiv 2604.15034) specifies a *Resource
Substrate Protocol Layer* (RSPL) — prompts, agents, tools, environments and
memory modelled as protocol-registered resources with explicit state, lifecycle,
versioned interfaces and auditable lineage — and a *Self-Evolution Protocol
Layer* (SEPL) that names a propose→assess→commit closed loop for evolving them.

AGP defines the *shape* of such a substrate. It does not, in its reference
implementation, *enforce* it: admissibility is not a hard gate, and a resource
can be regenerated without an independent check that it did not regress.

`toolmarket` is the enforced implementation of that substrate for one resource
type — **tool** — built on top of `autoforge`, which already ships the teeth:

  * a boolean validity gate that runs *before* any fitness score, so a mutant
    that deleted a guardrail cannot out-score its way back in;
  * a frozen baseline ratchet — probes and declared invariances can only grow;
  * a reliability ledger that turns a degraded tool into a quarantined one, and
    a rehabilitation path back;
  * append-only events and archived versions, so every transition is auditable.

Two orthogonal axes, each grounded in a real system (not invented):

  lifecycle.state  ∈ {draft, probation, active, quarantined, retired}
      — `autoforge.tools.spec.ToolState`, the *enforcement* lifecycle.
  version.status   ∈ {active, deprecated, archived}
      — AGP's `VersionStatus` (autogenesis/version/types.py), the *provenance*
        axis: whether a *version* is the current one, superseded, or shelved.

A resource is therefore not a function you upload and call. It is a record with
a contract, a state, a version, a lineage and a ledger — and the state can only
be *earned*, never declared.
"""

__version__ = "0.1.0"

from toolmarket.protocol.events import Event, EventLog
from toolmarket.protocol.lifecycle import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    ResourceState,
    VersionStatus,
    can_transition,
    transition,
)
from toolmarket.protocol.lineage import LineageGraph, LineageNode
from toolmarket.protocol.resources import (
    ResourceRecord,
    ResourceType,
    ResourceVersion,
    ToolContract,
)
from toolmarket.protocol.sepl import (
    AssessmentReport,
    EvolutionOperator,
    Proposal,
    ProposalRejected,
    StubProposer,
)
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore

__all__ = [
    "__version__",
    # resources (RSPL)
    "ResourceRecord", "ResourceType", "ResourceVersion", "ToolContract",
    # lifecycle
    "ResourceState", "VersionStatus", "LEGAL_TRANSITIONS",
    "LifecycleError", "can_transition", "transition",
    # lineage + events
    "LineageGraph", "LineageNode", "Event", "EventLog",
    # evolution (SEPL)
    "EvolutionOperator", "Proposal", "AssessmentReport",
    "ProposalRejected", "StubProposer",
    # facade + persistence
    "ResourceRegistry", "ResourceStore",
]
