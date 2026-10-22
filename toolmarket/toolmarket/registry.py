"""ResourceRegistry — the substrate's front door.

Everything a caller can do to a resource goes through here, and every mutation
leaves both an event and (where relevant) a lineage node. The registry owns the
three pieces of protocol state — records, the event log, the lineage DAG — and
keeps them consistent with each other and with the backing `autoforge` registry.

Enforcement is not optional here. `register` admits a resource at DRAFT; it
becomes callable-as-trusted only by walking the lifecycle. `replace_live` (the
commit step of evolution) re-opens the trial: a new version of an ACTIVE tool
drops back to PROBATION until it has re-earned trust. That single line is the
difference between a registry and a marketplace with a badge.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from toolmarket.protocol.events import EventKind, EventLog
from toolmarket.protocol.lifecycle import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    ResourceState,
    transition,
)
from toolmarket.protocol.lineage import LineageGraph, LineageNode
from toolmarket.protocol.resources import ResourceRecord, ToolContract
from toolmarket.trust import TRUST_ACTOR, TrustDecision, TrustPolicy, assess

try:  # pragma: no cover
    from autoforge.tools.registry import ToolRegistry, ToolResult
    from autoforge.tools.spec import ToolSpec
    from autoforge.forge.validity import FrozenBaseline
    from autoforge.core.llm import MockLLMClient
    from autoforge.forge.verifier import ToolVerifier
    _HAVE = True
except Exception as exc:  # noqa: BLE001
    _HAVE = False
    _ERR = exc


def _require() -> None:
    if not _HAVE:
        raise RuntimeError(
            "toolmarket requires autoforge for its enforcement layer; import "
            f"failed: {_ERR!r}. Install with: pip install -e <path>/autoforge"
        ) from _ERR


class ResourceRegistry:
    """Registers, evolves, invokes and audits protocol resources."""

    def __init__(
        self,
        store: Any = None,
        *,
        tool_registry: Any = None,
        log: Optional[EventLog] = None,
        lineage: Optional[LineageGraph] = None,
        trust_policy: Optional[TrustPolicy] = None,
    ) -> None:
        from toolmarket.store import make_store

        self.store = store if store is not None else make_store()
        # Adopt autoforge's registry so the *same* enforcement object runs on
        # live calls; we never run a second, weaker copy.
        self.tools = tool_registry
        self.log: EventLog = log or self.store.load_events()
        # Every append becomes durable at the sink, so the operator's evolution
        # events are persisted exactly like the registry's own.
        #
        # *Which* sink is the concurrency contract, and getting it wrong is a 500
        # rather than a slow path. The API and the Celery worker each build their
        # own registry over one shared store, so both hold a `_events` mirror that
        # knows only what that process has seen. Let either of them take `seq` from
        # that mirror and both eventually write event N — the first after the
        # worker's evolution appended anything at all. So a store that can allocate
        # does (`reserve_event`), and only a store that cannot falls back to
        # sealing in the log and handing the finished event over.
        reserve = getattr(self.store, "reserve_event", None)
        self.log.on_reserve = reserve
        self.log.on_append = None if reserve else self.store.append_event
        self.lineage: LineageGraph = lineage or self.store.load_lineage()
        self._records: dict[str, ResourceRecord] = {
            r.id: r for r in self.store.load_resources()
        }
        # The high-water mark of what is already loaded, so `refresh` can tell
        # "nothing new" from "not checked yet" without re-reading the table. Set
        # from the records themselves rather than from `max_updated_at()`: a
        # store that cannot answer that query still gets a correct baseline.
        self._loaded_stamp = max(
            (getattr(r, "updated_at", 0.0) for r in self._records.values()),
            default=0.0)
        self._refreshed_at = 0.0
        self._baselines: dict[str, Any] = {}
        # Set by `make_queue` on first request, so every surface built over this
        # registry shares one inline queue. None means "no queue asked for yet".
        self.task_queue: Any = None
        # The `earn`/`decay` edges of the lifecycle (`toolmarket/trust.py`).
        # Default ON, because the alternative is what this shelf looked like
        # without it: every tool published, none ever promoted, and the only two
        # resources that reached ACTIVE got there through a human's keyboard with
        # an empty `reason`. Pass `TrustPolicy(enabled=False)` to keep the ledger
        # as a record that nothing acts on.
        self.trust: TrustPolicy = (
            trust_policy if trust_policy is not None else TrustPolicy()
        )

    # -- lazy autoforge wiring -------------------------------------------
    def _tools(self) -> Any:
        if self.tools is None:
            _require()
            self.tools = ToolRegistry()
        return self.tools

    # -- read -------------------------------------------------------------
    def get(self, resource_id: str) -> Optional[ResourceRecord]:
        return self._records.get(resource_id)

    def require(self, resource_id: str) -> ResourceRecord:
        rec = self._records.get(resource_id)
        if rec is None:
            raise KeyError(f"unknown resource: {resource_id}")
        return rec

    def list(self, *, type: Optional[str] = None,
             state: Optional[str] = None) -> list[ResourceRecord]:
        out = list(self._records.values())
        if type is not None:
            out = [r for r in out if r.type.value == type]
        if state is not None:
            out = [r for r in out if r.state.value == state]
        return sorted(out, key=lambda r: r.id)

    def refresh(self, *, max_age: float = 0.0) -> list[Any]:
        """Re-read the store, so a row another process wrote becomes visible.

        The registry is a *cache* of the store: `__init__` fills `_records`
        once, and `save` keeps it in step with this process's own writes. That
        is correct for one writer and silently wrong for two, which is what this
        deployment is -- the importer holds the sqlite file while the API serves
        :8000, and the API's copy is whatever existed when it booted.

        Measured 2026-09-28: the importer had written to 11,871 resources while
        `/health` on the live shelf said 6,985 and `/resources/tool:<id>` for a
        row written minutes earlier answered 404. Nothing was broken and nothing
        was stale about the *store*; the process was answering faithfully from a
        snapshot it had no reason to know was one.

        What it deliberately does not do: evict. A resource deleted underneath
        this process stays in `_records` until a restart, because dropping a
        record another request may be mid-way through using is a worse failure
        than listing one row too many. New and changed rows are the case that
        matters -- that is the case where a caller is told "no such tool" about a
        tool that plainly exists.

        Zero-cost when the store has nothing new: the `updated_at` high-water
        mark is one indexed query, and a second call within `max_age` seconds
        does not even do that.

        Returns the records that were new or changed -- not a count. The caller
        is usually a search index that has to be told *which* rows to index, and
        making it diff the registry itself to find out would be the same walk
        done twice.
        """
        now = time.time()
        if max_age and (now - getattr(self, "_refreshed_at", 0.0)) < max_age:
            return []
        self._refreshed_at = now
        try:
            stamp = self.store.max_updated_at()
        except Exception:  # noqa: BLE001 - an unsupported store is not a failure
            stamp = None
        if stamp is not None and stamp <= getattr(self, "_loaded_stamp", 0.0):
            return []
        loaded = self.store.load_resources()
        changed: list[ResourceRecord] = []
        for rec in loaded:
            old = self._records.get(rec.id)
            if old is None or getattr(old, "updated_at", 0) != rec.updated_at:
                self._records[rec.id] = rec
                self._invalidate(rec.id)
                changed.append(rec)
        self._loaded_stamp = max([stamp or 0.0] +
                                 [getattr(r, "updated_at", 0.0) for r in loaded])
        return changed

    # -- write ------------------------------------------------------------
    def save(self, rec: ResourceRecord) -> None:
        rec.updated_at = time.time()
        self._records[rec.id] = rec
        self.store.save_resource(rec)
        self._invalidate(rec.id)

    def _invalidate(self, resource_id: str) -> None:
        """Drop the cached view of one resource.

        Called from `save` and nowhere else, because `save` is the only method
        that changes a record — including when the caller is the Celery worker
        in a different process, which is the case this has to work for. With a
        shared Redis, a worker's commit invalidates the API's cache entry even
        though the two share no memory; that property is why invalidation lives
        at the write rather than in the API's route handlers, where it would look
        like it worked in tests and silently not in production.

        Best-effort by design: an unreachable cache must never turn a successful
        write into a failed one. The cost of failing here is a stale read for
        one TTL; the cost of raising is a registry that cannot persist.
        """
        try:
            from toolmarket import metrics as _metrics
            from toolmarket.cache import get_cache, resource_key

            get_cache().delete(resource_key(resource_id))
            _metrics.STORE_OPERATIONS.inc(operation="save_resource")
        except Exception:  # noqa: BLE001 - see docstring
            pass

    def register(self, spec_or_record: Any, *, actor: str = "system",
                 state: Optional[ResourceState] = None) -> ResourceRecord:
        """Admit a tool into the substrate. Always lands at DRAFT (or the
        explicitly requested starting state, which is still a lifecycle node)."""
        _require()
        if isinstance(spec_or_record, ResourceRecord):
            rec = spec_or_record
        else:
            rec = ResourceRecord.from_toolspec(spec_or_record)
        if state is not None:
            rec.state = ResourceState(state)
        elif rec.state == ResourceState.DRAFT:
            rec.state = ResourceState.DRAFT

        if rec.id in self._records:
            raise ValueError(f"resource already registered: {rec.id}")

        self._tools().register(spec_or_record if not isinstance(spec_or_record, ResourceRecord)
                               else rec.to_toolspec(), replace=True)
        self.save(rec)

        node = LineageNode(
            node_id=f"{rec.id}@{rec.version_current.version}",
            resource_id=rec.id,
            version=rec.version_current.version,
            parents=[],
            reason="initial registration",
            mutation="register",
            verification=rec.contract.verification,
            created_at=rec.created_at,
        )
        self.add_lineage_node(node)

        self.log.append(
            EventKind.REGISTER, rec.id,
            data={
                "name": rec.name,
                "state": rec.state.value,
                "version": rec.version_current.version,
                "source": rec.provenance.get("source"),
            },
            actor=actor,
        )
        return rec

    def transition(self, resource_id: str, dst: ResourceState, *,
                   reason: str = "", actor: str = "system") -> ResourceRecord:
        """Move a resource along the *enforcement* lifecycle."""
        rec = self.require(resource_id)
        try:
            transition(rec.state, dst)
        except LifecycleError:
            raise
        src = rec.state
        rec.state = ResourceState(dst)
        self.save(rec)

        kind = {
            ResourceState.QUARANTINED: EventKind.QUARANTINE,
            ResourceState.PROBATION: EventKind.REHAB
            if src == ResourceState.QUARANTINED else EventKind.TRANSITION,
            ResourceState.RETIRED: EventKind.RETIRE,
        }.get(rec.state, EventKind.TRANSITION)
        if src == ResourceState.DRAFT and rec.state == ResourceState.PROBATION:
            kind = EventKind.TRANSITION

        self.log.append(
            kind, rec.id,
            data={"from": src.value, "to": rec.state.value, "reason": reason},
            actor=actor,
        )
        return rec

    def promote(self, resource_id: str, *, actor: str = "system") -> ResourceRecord:
        return self.transition(resource_id, ResourceState.ACTIVE,
                               reason="earned trust", actor=actor)

    def quarantine(self, resource_id: str, *, reason: str = "degraded",
                   actor: str = "system") -> ResourceRecord:
        return self.transition(resource_id, ResourceState.QUARANTINED,
                               reason=reason, actor=actor)

    def rehab(self, resource_id: str, *, reason: str = "rehabilitated",
              actor: str = "system") -> ResourceRecord:
        return self.transition(resource_id, ResourceState.PROBATION,
                               reason=reason, actor=actor)

    def retire(self, resource_id: str, *, reason: str = "withdrawn",
               actor: str = "system") -> ResourceRecord:
        return self.transition(resource_id, ResourceState.RETIRED,
                               reason=reason, actor=actor)

    # -- evolution commit hook -------------------------------------------
    def replace_live(
        self,
        resource_id: str,
        contract: ToolContract,
        *,
        note: str = "",
        mutation: str = "mutate",
        parent_node: Optional[str] = None,
        verification: Optional[dict[str, Any]] = None,
    ) -> ResourceRecord:
        """Advance the live version to `contract` and re-open the trial.

        This is the only sanctioned way a resource's *body* changes. It always:
          * archives the superseded version (provenance axis),
          * advances a new ACTIVE version,
          * draws a lineage edge from the parent,
          * pushes an ACTIVE tool back to PROBATION (enforcement axis) — a new
            version has not yet earned trust, no matter how trusted its parent.
        """
        rec = self.require(resource_id)
        old_state = rec.state
        rec.contract = contract
        if verification is not None:
            rec.contract.verification = verification
        new_version = rec.advance_version(note=note)
        rec.contract.verification = rec.contract.verification or {}
        self.save(rec)

        # Enforcement: a version change re-opens the trial.
        if old_state in (ResourceState.ACTIVE, ResourceState.PROBATION):
            rec.state = ResourceState.PROBATION
            self.save(rec)

        node = LineageNode(
            node_id=f"{rec.id}@{new_version.version}",
            resource_id=rec.id,
            version=new_version.version,
            parents=[parent_node] if parent_node else [],
            reason=note,
            mutation=mutation,
            verification=rec.contract.verification,
            created_at=time.time(),
        )
        self.add_lineage_node(node)
        return rec

    # -- autoforge bridge -------------------------------------------------
    def to_spec(self, rec: ResourceRecord, fn: Any = None) -> Any:
        """Rehydrate the live autoforge spec for enforcement to run against."""
        _require()
        spec = rec.to_toolspec(fn=fn)
        # If the backing registry already holds this tool, prefer its fn so the
        # callable is never lost across a rehydrate.
        existing = self._tools().get(rec.name) if self.tools is not None else None
        if existing is not None and fn is None:
            spec.fn = existing.fn
            spec.runner = getattr(existing, "runner", None)
        # Last resort, and the one that makes a restart survivable: recompile
        # the callable from the stored contract. Without this, `fn` is None for
        # every resource loaded from a durable store, because a function object
        # cannot be serialised into the record -- only its source can. The
        # symptom is a shelf that lists correctly and answers every invoke with
        # "TypeError: 'NoneType' object is not callable", which reads like a
        # broken tool rather than a lost one.
        #
        # `compile_tool_fn` never raises on bad code; it returns a callable that
        # reports the failure when called, so a resource whose body does not
        # compile stays registered and its invoke is logged like any other.
        if spec.fn is None and rec.contract.code:
            from toolmarket.protocol.resources import compile_tool_fn

            spec.fn = compile_tool_fn(rec.contract.code, rec.name)
        return spec

    def baseline_for(self, resource_id: str) -> Any:
        """The frozen baseline a candidate must not regress against."""
        _require()
        if resource_id in self._baselines:
            return self._baselines[resource_id]
        rec = self.require(resource_id)
        spec = self.to_spec(rec)
        baseline = FrozenBaseline.capture(spec)
        self._baselines[resource_id] = baseline
        return baseline

    def enforce_baseline(self, resource_id: str, spec: Any) -> Any:
        """Grow the frozen baseline after a version is accepted (the ratchet)."""
        _require()
        base = self.baseline_for(resource_id)
        grown = base.extended_with(spec)
        self._baselines[resource_id] = grown
        return grown

    def offline_verifier(self) -> Any:
        """A deterministic, network-free verifier: execution + robustness only.

        The adversarial and trigger checks need a real model; offline we run the
        checks that are reproducible so assessment is a function of the code, not
        of the weather.
        """
        _require()
        return ToolVerifier(
            MockLLMClient(),
            run_adversarial_check=False,
            run_trigger_check=False,
            run_negative_check=False,
        )

    # -- invocation -------------------------------------------------------
    def invoke(self, resource_id: str, arguments: dict[str, Any], *,
               force: bool = False, actor: str = "system") -> Any:
        """Call a resource through the backing registry, and log it."""
        rec = self.require(resource_id)
        tool_reg = self._tools()
        # Make sure the live code is what the registry holds.
        tool_reg.register(self.to_spec(rec), replace=True)
        result = tool_reg.call(rec.name, arguments, force=force)
        rec.ledger = self._spec_ledger(rec)
        self.save(rec)
        self.log.append(
            EventKind.INVOKE, rec.id,
            data={"arguments": arguments,
                  "ok": bool(getattr(result, "ok", True)),
                  "error": getattr(result, "error", None)},
            actor=actor,
        )
        # The `earn` edge, taken here rather than in a cron or a sweep. The
        # ledger was updated one statement ago, so this is the only moment the
        # decision is a function of data that is actually fresh; a scheduled
        # sweep would promote on the same numbers, just later and with no way to
        # say which call tipped it. Best-effort and swallowed: a shelf that
        # refuses a transition must never turn a successful call into a failed
        # one, which is the same contract `_invalidate` and the event sink's
        # durability already work under.
        try:
            self.reconcile_trust(rec.id)
        except Exception:  # noqa: BLE001 - see comment
            pass
        return result

    def _spec_ledger(self, rec: ResourceRecord) -> dict[str, Any]:
        spec = self._tools().get(rec.name)
        return spec.stats.to_dict() if spec is not None and hasattr(spec, "stats") else rec.ledger

    # -- trust ------------------------------------------------------------
    def trust_view(self, resource_id: str) -> dict[str, Any]:
        """What the policy makes of this resource right now — changing nothing.

        Read-only on purpose. The console calls this while rendering a resource,
        and an assessment that moved state would make opening a page a
        side-effecting act -- the same reason `chips()` gathers its counts from
        current state instead of incrementing counters.
        """
        rec = self.require(resource_id)
        return assess(rec.id, rec.state, rec.ledger, self.trust).to_dict(
            self.trust.thresholds())

    def reconcile_trust(self, resource_id: str) -> Optional[TrustDecision]:
        """Apply the policy to one resource; returns the decision either way.

        The decision is returned even when nothing moved, so a caller sweeping
        the shelf can print the reason a tool is still on probation instead of
        collecting a list of unexplained silences.
        """
        rec = self.require(resource_id)
        decision = assess(rec.id, rec.state, rec.ledger, self.trust)
        if decision.action == "hold" or not self.trust.enabled:
            return decision
        # Through `transition`, not by writing `rec.state`, so a policy-driven
        # move leaves exactly the audit trail a hand-typed one does -- and one
        # better: `TRUST_ACTOR` names the decider instead of "system", and the
        # reason carries the evidence that earned it.
        self.transition(resource_id, ResourceState(decision.to),
                        reason=decision.reason, actor=TRUST_ACTOR)
        return decision

    def reconcile_all(self) -> list[dict[str, Any]]:
        """Sweep the whole shelf and report every decision, applied or not."""
        out: list[dict[str, Any]] = []
        for rec in self.list():
            decision = self.reconcile_trust(rec.id)
            if decision is not None:
                out.append(decision.to_dict(self.trust.thresholds()))
        return out

    # -- views ------------------------------------------------------------
    def event_log(self, resource_id: Optional[str] = None) -> list[dict[str, Any]]:
        events = (self.log.for_resource(resource_id) if resource_id
                  else list(self.log))
        return [e.to_dict() for e in events]

    # -- lineage ----------------------------------------------------------

    def add_lineage_node(self, node: Any) -> None:
        """Record a lineage node in memory *and* durably.

        Single entry point so no caller can add an in-memory node and forget
        to persist it — which is exactly how an evolution's provenance would
        quietly vanish on restart.
        """
        self.lineage.add(node)
        self.store.save_lineage_node(node)

    def lineage_view(self, resource_id: str) -> dict[str, Any]:
        nodes = self.lineage.of_resource(resource_id)
        return {
            "resource_id": resource_id,
            "nodes": [n.to_dict() for n in nodes],
            "roots": [n.node_id for n in nodes if not n.parents],
            "current": (f"{resource_id}@{self._records[resource_id].version_current.version}"
                        if resource_id in self._records else None),
        }
