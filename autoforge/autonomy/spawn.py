"""Spawning: the agent creates other agents, freely.

No pre-declared agent pool, no fixed orchestration graph. An agent decides at
runtime that a subtask wants its own worker, and spawns one. The child can
share the parent's tool registry (so tools forged by one are usable by all) or
get an isolated one (so experiments don't pollute the parent's library).

Two sharing modes:

  SHARED  — same registry object. A tool the child forges is immediately
            available to the parent. Fast convergence, but a bad tool
            propagates.
  ISOLATED — child gets a copy. It can forge freely; only tools that pass its
            verification AND the parent approves are merged back.

The parent chooses per-spawn. Default is SHARED, because the whole thesis is
that a growing shared library is the point — isolation is for when you want to
A/B a child's tool quality before trusting it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState
from .roles import ceiling, role_brief


class ShareMode(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"


class ChildState(str, Enum):
    """A child's lifecycle, taken from hermes-agent's subagent_lifecycle.

    The point of a state rather than an error string: a parent deciding whether
    to retry needs to tell "ran out of time" from "crashed" from "was asked to
    stop", and a single `error is None` cannot say that. UNKNOWN exists because
    a record read back from disk may predate the field.
    """

    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


#: A state from which nothing more will happen.
TERMINAL_STATES = frozenset({
    ChildState.SUCCEEDED, ChildState.FAILED, ChildState.CANCELLED,
})


@dataclass
class SpawnRecord:
    child_id: str
    task: str
    mode: str
    parent_id: str
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    result: str = ""
    tools_forged: list[str] = field(default_factory=list)
    error: str | None = None
    # Which role the child ran as, and — if it tried to reach past that role —
    # what it was refused. A refusal is evidence the whitelist is live rather
    # than decorative, so it is recorded rather than swallowed.
    role: str = ""
    refusals: list[str] = field(default_factory=list)
    # A lifecycle, not just an exit code. A child that ran out of time is not
    # the same fact as a child that crashed, and a child that was cancelled is
    # not the same fact as one that never started -- collapsing them into
    # error=None/error="..." loses exactly the distinction a parent needs to
    # decide whether to retry.
    route: dict[str, Any] = field(default_factory=dict)
    state: str = ChildState.PENDING.value
    timeout_seconds: float = 0.0
    timed_out: bool = False
    cancel_requested: bool = False
    #: The directive the child was handed, kept so the trace shows the role
    #: reached a model instead of only reaching a dataclass.
    brief: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "task": self.task[:200],
            "mode": self.mode,
            "role": self.role or None,
            "brief_chars": len(self.brief),
            "tools_forged": self.tools_forged,
            "refusals": self.refusals,
            "duration_s": round(self.finished - self.started, 2) if self.finished else None,
            "state": self.state,
            "route": self.route,
            "timed_out": self.timed_out,
            "cancelled": self.cancel_requested,
            "error": self.error,
        }


@dataclass
class Spawner:
    """Creates child agents on demand.

    `agent_factory(parent, registry) -> agent-like` must return an object with
    a `.run(task) -> result` method whose result exposes `.content` and the
    child's registry (if it forged anything).
    """

    registry: ToolRegistry
    agent_factory: Callable[[Any, ToolRegistry], Any]
    parent_id: str = "root"
    spawn_count: int = 0
    history: list[SpawnRecord] = field(default_factory=list)
    max_depth: int = 8              # a limit the agent can raise, not a cage
    policy: Any = None              # AutonomyPolicy | None — gates merge_back
    #: Which model/endpoint children run on. A backend answers the route and
    #: the spawner merely passes it on: keeping the two apart means a new
    #: provider never touches the spawn path.
    delegation: Any = None          # DelegationProvider | None
    delegation_config: dict[str, Any] = field(default_factory=dict)
    _depth: int = 0

    def spawn(
        self,
        task: str,
        *,
        mode: ShareMode = ShareMode.SHARED,
        child_registry: ToolRegistry | None = None,
        restrict_to: set[str] | list[str] | None = None,
        role: str = "",
        brief: str = "",
        deny: Callable[[Any], str] | None = None,
        timeout_seconds: float | None = None,
        context: str = "",
    ) -> SpawnRecord:
        self.spawn_count += 1
        child_id = f"{self.parent_id}.{self.spawn_count}"

        # A role is enforceable on its own. The caller may add a whitelist, a
        # richer brief (a node's hint on top of the role's), or a whole different
        # ceiling — but naming a role is enough to get the role's behaviour, so
        # there is no way to spawn a "critic" that is not held to a critic's
        # limits by simply forgetting to pass one.
        if role:
            brief = brief or role_brief(role)
            if deny is None:
                deny = ceiling(role)

        record = SpawnRecord(
            child_id=child_id, task=task, mode=mode.value, parent_id=self.parent_id,
            role=role, brief=brief, state=ChildState.STARTING.value,
            timeout_seconds=float(timeout_seconds or 0.0),
        )
        if context:
            # The brief is the role's directive; the context is what the child
            # needs to know about THIS task that its role cannot imply. Kept
            # separate so a re-briefed child does not have its role overwritten
            # by a pile of task detail.
            record.brief = (brief + "\n\n" + context).strip() if brief else context

        if self._depth >= self.max_depth:
            record.error = f"max spawn depth {self.max_depth} reached (raise it if needed)"
            record.state = ChildState.FAILED.value
            record.finished = time.time()
            self.history.append(record)
            return record

        base_reg = self.registry if mode == ShareMode.SHARED else (
            child_registry or ToolRegistry())
        # A role's whitelist becomes a lens over the real registry, so the child
        # can only see and call what its role allows. Registration still lands on
        # base_reg — which is why the forged-tool diff is taken there: a child
        # that forges something outside its whitelist really did create it, and
        # hiding that from the parent would be the wrong kind of tidiness.
        #
        # `deny` is the role's capability ceiling, which is not a whitelist: a
        # critic's tool set depends on what each tool does, and forged tools are
        # named at runtime, so the ceiling has to be judged per-spec.
        reg = base_reg.scoped(restrict_to or (), role=role, deny=deny)
        before = set(base_reg.names())

        # Resolve the route BEFORE the child is built, and refuse the spawn if
        # the backend cannot serve it. Fail closed: a child quietly running on
        # the parent's model when a different one was asked for is a bug you
        # only notice from the bill.
        if self.delegation is not None:
            from .delegation import DelegationRouteError, validate_route
            try:
                route = self.delegation.route_task(
                    {"goal": task, "role": role, "context": context},
                    self.delegation_config)
                record.route = validate_route(route, backend=self.delegation.name)
            except DelegationRouteError as exc:
                record.error = str(exc)
                record.state = ChildState.FAILED.value
                record.finished = time.time()
                self.history.append(record)
                return record

        record.state = ChildState.RUNNING.value
        # A watchdog, not a second thread of execution: the child runs in this
        # thread (it is the same process, and preempting it would need a signal
        # the Windows host does not give us), so the deadline is checked around
        # the call and enforced by a timer that marks the record. A child that
        # blocks forever still blocks -- what this buys is that a child which
        # merely ran long is recorded as timed_out rather than as SUCCEEDED,
        # which is the difference between a result to trust and one to retry.
        deadline = (time.time() + record.timeout_seconds) if record.timeout_seconds > 0 else 0.0
        try:
            child = self.agent_factory(self, reg)
            # Handed over after construction rather than through the factory
            # signature, so an existing two-argument `agent_factory` — including
            # the fakes in the tests — keeps working. A child that does not
            # implement the hook is simply one that cannot be briefed, and the
            # record says so by carrying an empty brief.
            if brief:
                hook = getattr(child, "apply_role", None)
                if callable(hook):
                    hook(brief, role)
            result = child.run(task)
            record.result = getattr(result, "content", str(result))
            if deadline and time.time() > deadline:
                record.timed_out = True
                record.state = ChildState.INTERRUPTED.value
            elif record.cancel_requested:
                record.state = ChildState.CANCELLED.value
            else:
                record.state = ChildState.SUCCEEDED.value
        except Exception as exc:  # noqa: BLE001
            record.error = f"{type(exc).__name__}: {exc}"
            record.state = (ChildState.CANCELLED.value if record.cancel_requested
                            else ChildState.FAILED.value)

        after = set(base_reg.names())
        record.tools_forged = sorted(after - before)
        record.refusals = list(getattr(reg, "refusals", []) or [])
        record.finished = time.time()
        self.history.append(record)
        return record

    def cancel(self, child_id: str) -> str:
        """Ask a child to stop at its next checkpoint.

        Honest about what it can do: a child in this process cannot be killed
        from another thread of the same process without corruption, so this
        sets the flag the child's own checkpoints read (and the record's
        state) rather than pretending a thread was stopped. A child that never
        checks is a child that ignores its cancel request — which the record
        then shows as CANCELLED-with-a-result instead of a quiet success.
        """
        rec = next((r for r in self.history if r.child_id == child_id), None)
        if rec is None:
            if child_id == f"{self.parent_id}.{self.spawn_count + 1}":
                return ("That id has not been assigned yet: it is the NEXT child "
                        "this spawner will start, so there is nothing to cancel.")
            known = ", ".join(r.child_id for r in self.history) or "none"
            return f"No child {child_id!r} in this spawner (have: {known})."
        if rec.state in {s.value for s in TERMINAL_STATES}:
            return f"{child_id} already reached {rec.state}; nothing to cancel."
        rec.cancel_requested = True
        rec.state = ChildState.CANCEL_REQUESTED.value
        return f"{child_id}: cancel requested (it stops at its next checkpoint)."

    def status(self, child_id: str = "") -> dict[str, Any]:
        """One child's lifecycle record, or all of them."""
        if child_id:
            rec = next((r for r in self.history if r.child_id == child_id), None)
            return rec.to_dict() if rec else {"error": f"no child {child_id!r}"}
        return self.summary()

    def child_spawner(self, child_registry: ToolRegistry) -> "Spawner":
        """A spawner for a child, one level deeper."""
        return Spawner(
            registry=child_registry,
            agent_factory=self.agent_factory,
            parent_id=f"{self.parent_id}.{self.spawn_count}",
            max_depth=self.max_depth,
            policy=self.policy,
            _depth=self._depth + 1,
        )

    # -- merging isolated children back --------------------------------
    def merge_back(
        self,
        child_registry: ToolRegistry,
        *,
        require_state: ToolState = ToolState.ACTIVE,
        approve: Callable[[ToolSpec], bool] | None = None,
    ) -> list[str]:
        """Pull an isolated child's trusted tools into the parent library.

        `approve=None` (the default) is not "no gate": it means the policy
        decides. When may_promote_tools is off the child's tools may not be
        sealed ACTIVE in the first place, so require_state already filters them
        and merge_back ends up pulling nothing. Pass an explicit `approve`
        callback to merge anyway, one spec at a time.
        """
        merged: list[str] = []
        collect = self._collect_approved
        for name, spec in child_registry._tools.items():
            if spec.state != require_state:
                continue
            if not collect(spec, approve):
                continue
            self.registry.register(spec, replace=True)
            merged.append(name)
        return merged

    @staticmethod
    def _collect_approved(
        spec: ToolSpec,
        approve: Callable[[ToolSpec], bool] | None,
    ) -> bool:
        """Explicit callback always wins. With none, accept: require_state
        already filtered, and the caller who owns the policy decides whether
        merge_back is reachable at all (see `denied_reason`)."""
        return approve(spec) if approve is not None else True

    def denied_reason(self) -> str | None:
        """Why merge_back would refuse to pull anything, or None if it won't."""
        if self.policy is not None and not self.policy.may_promote_tools:
            return (
                "may_promote_tools is off: children may not seal tools ACTIVE, "
                "so there is nothing at ACTIVE to merge. Pass approve=... to "
                "accept on a case-by-case basis."
            )
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "spawned": self.spawn_count,
            "max_depth": self.max_depth,
            "children": [r.to_dict() for r in self.history],
        }


__all__ = ["Spawner", "SpawnRecord", "ShareMode"]
