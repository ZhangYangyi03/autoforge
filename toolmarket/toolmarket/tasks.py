"""Asynchronous evolution: the queue, and what happens when there isn't one.

An evolution is propose (call a model) → assess (run the verifier: execution,
robustness, adversarial, trigger) → commit. The assessment step is *slow and
unbounded* — it runs candidate code in a harness — so serving it on the request
thread means a single evolution can hold a worker for minutes and the API's
latency becomes a function of the slowest candidate anyone submitted. That is
the reason this module exists; it is not architecture for its own sake.

The design decision worth reading: **`InlineQueue` is the default, not a
fallback.** If `TASK_QUEUE` is unset the API still answers `POST .../evolve`
with a task id and `GET /tasks/{id}` still works — the work just happens on a
background thread of the API process and the same task record is written. So
every client, every test and every doc example uses one protocol, and switching
to Celery changes throughput rather than the contract. A design where the async
path only exists when Redis is present produces two code paths, and the one
nobody runs in development is the one that breaks in production.

Task records live in the cache (Redis when there is one) with a TTL, not in the
database. A task record is operational exhaust: it is worthless after an hour and
writing every progress tick to durable storage would put the queue's write volume
through the same connection pool as the audit log.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from toolmarket import metrics as _metrics
from toolmarket import tracing as _tracing

TASK_PREFIX = "task"
DEFAULT_TASK_TTL = 3600.0


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class TaskRecord:
    """One async evolution run. Mirrored in the cache; never authoritative."""

    task_id: str
    resource_id: str
    goal: str = ""
    state: str = TaskState.PENDING.value
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    stage: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    queue: str = "inline"
    # The trace this run belongs to. Recorded on the task so the correlation is
    # discoverable rather than inferred: an operator holding a task id can read
    # the trace id off `/tasks/{id}` and open the waterfall directly, instead of
    # searching a trace backend by timestamp and hoping. Empty when tracing is
    # off, which is also how a caller can tell that it is off.
    #
    # A new field on a cached record is safe because `from_dict` drops unknown
    # keys and defaults missing ones — a task written before this field existed
    # still loads, with `trace_id` empty.
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


# -- task storage -----------------------------------------------------------
def task_key(task_id: str) -> str:
    return f"{TASK_PREFIX}:{task_id}"


class TaskStore:
    """Task records in whatever cache the deployment has."""

    def __init__(self, cache: Any = None, ttl: float = DEFAULT_TASK_TTL) -> None:
        if cache is None:
            from toolmarket.cache import get_cache
            cache = get_cache()
        self.cache = cache
        self.ttl = ttl

    def save(self, rec: TaskRecord) -> None:
        self.cache.set(task_key(rec.task_id), rec.to_dict(), self.ttl)
        self._index().add(rec.task_id)

    def get(self, task_id: str) -> Optional[TaskRecord]:
        raw = self.cache.get(task_key(task_id))
        if isinstance(raw, str):
            raw = json.loads(raw)
        return TaskRecord.from_dict(raw) if raw else None

    def _index(self) -> set[str]:
        """The task ids this store has written, kept on the cache object.

        `save` maintains it rather than the queue, which is a correction: the
        queue used to poke a private attribute into the cache from outside, which
        meant a bare `TaskStore` (the Celery path, or any test) reported all-zero
        counts while holding real records. Whoever writes the record is the only
        party that knows a task exists, so the index belongs here.

        An attribute rather than a cache key on purpose. An index kept as a key
        would be a read-modify-write against Redis, racing between the API and the
        worker for the sake of a dashboard gauge — and the *cost* of that mistake
        (a count that is briefly wrong) is smaller than the cost of the lock that
        would prevent it. The consequence is that the index is per-process, which
        `by_state`'s docstring states out loud for Redis deployments.
        """
        index = getattr(self.cache, "_task_index", None)
        if index is None:
            index = set()
            try:
                self.cache._task_index = index  # noqa: SLF001 - see above
            except Exception:  # noqa: BLE001 - a frozen cache object
                return set()
        return index

    def by_state(self) -> dict[str, int]:
        """Counts per state for the `toolmarket_async_tasks` gauge.

        Only sees tasks this process created or was told about. With Redis the
        board is shared and the number is global; with `MemoryCache` it is
        per-process. The gauge's help text says "as reported by the task store"
        for exactly this reason — a metric that overstates its own scope is worse
        than no metric.
        """
        counts: dict[str, int] = {s.value: 0 for s in TaskState}
        index = getattr(self.cache, "_task_index", None)
        if index is None:
            return counts
        for tid in list(index):
            rec = self.get(tid)
            if rec is None:
                index.discard(tid)
                continue
            counts[rec.state] = counts.get(rec.state, 0) + 1
        return counts


# -- the work ---------------------------------------------------------------
def execute_evolution(
    resource_id: str,
    goal: str,
    *,
    commit: bool = True,
    proposer: str = "stub",
    task_id: Optional[str] = None,
    registry: Any = None,
    store: Optional[TaskStore] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> dict[str, Any]:
    """Run one propose → assess → (commit). Importable, so Celery, the inline
    queue and the tests all call *this*, not three copies of it.

    `registry is None` means "build one from the environment" — which in the
    worker container means Postgres, so the worker sees the resources the API
    wrote. That only works because the store is swappable; with the SQLite
    default the two processes would each have their own private substrate, and
    the async path would be silently broken in a way no test would catch.
    """
    def report(stage: str, pct: float) -> None:
        if progress is not None:
            progress(stage, pct)

    if registry is None:
        from toolmarket.registry import ResourceRegistry
        registry = ResourceRegistry()

    from toolmarket.protocol.sepl import EvolutionOperator, ProposalRejected

    report("loading", 0.05)
    rec = registry.get(resource_id)
    if rec is None:
        raise KeyError(f"unknown resource: {resource_id}")

    op = EvolutionOperator(registry)
    report("proposing", 0.15)
    proposal = op.propose(resource_id, goal)

    report("assessing", 0.35)
    assessment = op.assess(proposal.proposal_id)

    payload: dict[str, Any] = {
        "task_id": task_id,
        "resource_id": resource_id,
        "goal": goal,
        "proposer": proposer,
        "proposal_id": proposal.proposal_id,
        "assessment": assessment.to_dict(),
    }
    report("assessed", 0.8)

    if commit and assessment.admissible:
        try:
            committed = op.commit(proposal.proposal_id)
            payload["committed"] = True
            payload["version"] = committed.version_current.version
            payload["state"] = committed.state.value
        except ProposalRejected as exc:
            payload["committed"] = False
            payload["reject_reason"] = str(exc)
    else:
        payload["committed"] = False
        payload["reject_reason"] = (
            assessment.summary if not assessment.admissible else "commit=False"
        )

    report("done", 1.0)
    return payload


# -- the queues -------------------------------------------------------------
class InlineQueue:
    """Runs the evolution on a daemon thread of the API process.

    Bound, and honest about it: a single-process deployment gets concurrency but
    not durability — a restart loses in-flight work. The task record is still
    written to the cache, so a client polling a task id from before the restart
    gets `pending` forever rather than an error. Documented rather than papered
    over with a fake `failure`.
    """

    backend = "inline"

    def __init__(self, registry: Any = None, store: Optional[TaskStore] = None) -> None:
        self.registry = registry
        self.store = store or TaskStore()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(self, resource_id: str, goal: str, *, commit: bool = True,
               proposer: str = "stub") -> TaskRecord:
        task_id = uuid.uuid4().hex
        rec = TaskRecord(task_id=task_id, resource_id=resource_id, goal=goal,
                         state=TaskState.PENDING.value, queue=self.backend,
                         # Captured before the thread starts, because the thread
                         # gets a *copy* of this context: reading it inside would
                         # work for the inline queue and silently record "" for
                         # Celery, where the consumer's context is a different one.
                         trace_id=_tracing.current_context().trace_id)
        # No index bookkeeping here: `TaskStore.save` owns it, so an inline task
        # and a Celery task are counted by the same code path.
        self.store.save(rec)

        # A PRODUCER span around the handoff, closed before the work runs, and
        # `in_thread` — not a bare `Thread` — so the worker thread starts inside a
        # copy of this context. A plain `Thread` would get an empty context, the
        # ambient request span would be invisible in it, and the evolution would
        # open its own trace: one operation, two traces, and the API's looking
        # like it finished when the enqueue returned. Nothing raises when this is
        # wrong; that is why it is called out here and asserted in
        # `tests/test_tracing.py`.
        with _tracing.span(
            "evolve.enqueue",
            kind=_tracing.SPAN_PRODUCER,
            attributes={"task.id": task_id, "resource.id": resource_id,
                        "queue": self.backend, "messaging.system": "thread"},
        ):
            def run() -> None:
                self._run(task_id, resource_id, goal, commit, proposer)

            thread = _tracing.in_thread(run, name=f"evolve-{task_id[:8]}")
        with self._lock:
            self._threads[task_id] = thread
        thread.start()
        return rec

    def _run(self, task_id: str, resource_id: str, goal: str, commit: bool,
             proposer: str) -> None:
        # A CONSUMER span, not a root: `in_thread` put the producer's span in this
        # thread's context, so this is `start_span` with no explicit parent and it
        # still attaches to the request's trace. Passing the parent explicitly
        # would also work and would hide a broken `in_thread` — which is exactly
        # the bug worth not hiding.
        with _tracing.span(
            "evolve.run",
            kind=_tracing.SPAN_CONSUMER,
            attributes={"task.id": task_id, "resource.id": resource_id,
                        "queue": self.backend},
        ) as span:
            self._run_inner(span, task_id, resource_id, goal, commit, proposer)

    def _run_inner(self, span: Any, task_id: str, resource_id: str, goal: str,
                   commit: bool, proposer: str) -> None:
        rec = self.store.get(task_id) or TaskRecord(task_id=task_id,
                                                    resource_id=resource_id)
        rec.state = TaskState.RUNNING.value
        rec.started_at = time.time()
        rec.stage = "starting"
        self.store.save(rec)

        def progress(stage: str, pct: float) -> None:
            current = self.store.get(task_id) or rec
            current.stage = stage
            current.progress = pct
            self.store.save(current)
            # Mirrored onto the span so a trace shows *where* a slow evolution
            # spent its time without a second lookup: the stage names are the same
            # strings the task record carries, so a trace and a `/tasks/{id}` poll
            # cannot disagree about what was happening.
            span.set_attributes({"evolve.stage": stage, "evolve.progress": pct})

        try:
            result = execute_evolution(
                resource_id, goal, commit=commit, proposer=proposer,
                task_id=task_id, registry=self.registry, store=self.store,
                progress=progress,
            )
            rec.state = TaskState.SUCCESS.value
            rec.result = result
            rec.progress = 1.0
            _metrics.EVOLUTIONS.inc(outcome="committed" if result.get("committed")
                                    else "rejected")
        except Exception as exc:  # noqa: BLE001 - the error *is* the result here
            rec.state = TaskState.FAILURE.value
            rec.error = f"{type(exc).__name__}: {exc}"
            # The traceback goes to the log, not into the API response: a
            # traceback is a map of the filesystem and the code structure, and
            # this field is served to whoever holds the task id.
            traceback.print_exc()
            _metrics.EVOLUTIONS.inc(outcome="error")
        finally:
            rec.finished_at = time.time()
            self.store.save(rec)

    def status(self, task_id: str) -> Optional[TaskRecord]:
        return self.store.get(task_id)

    def join(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskRecord]:
        """Wait for a task. Used by tests; the API never blocks on this."""
        with self._lock:
            thread = self._threads.get(task_id)
        if thread is not None:
            thread.join(timeout)
        return self.store.get(task_id)

    def close(self) -> None:
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(2.0)


class CeleryQueue:
    """Hands the work to a Celery worker, which is a different process.

    The status endpoint reads the task record from the shared cache first and
    falls back to Celery's own result backend. Both are consulted because they
    answer different questions: the cache knows about progress *while* the task
    runs, and the result backend is authoritative about the outcome — a task that
    finished with all workers restarted still has its result there.
    """

    backend = "celery"

    def __init__(self, store: Optional[TaskStore] = None,
                 app: Any = None) -> None:
        self.store = store or TaskStore()
        self._app = app

    @property
    def app(self) -> Any:
        if self._app is None:
            from toolmarket.worker import celery_app
            self._app = celery_app
        return self._app

    def submit(self, resource_id: str, goal: str, *, commit: bool = True,
               proposer: str = "stub") -> TaskRecord:
        task_id = uuid.uuid4().hex
        rec = TaskRecord(task_id=task_id, resource_id=resource_id, goal=goal,
                         state=TaskState.PENDING.value, queue=self.backend,
                         # Captured before the thread starts, because the thread
                         # gets a *copy* of this context: reading it inside would
                         # work for the inline queue and silently record "" for
                         # Celery, where the consumer's context is a different one.
                         trace_id=_tracing.current_context().trace_id)
        self.store.save(rec)
        try:
            # The PRODUCER span covers the handoff only; the consumer's span is
            # created in the worker process and attaches to this same trace. Note
            # that `inject` is called *inside* the span's scope, so what it writes
            # is this span's context — inject outside and a downstream service gets
            # the caller's parent instead of the enqueue operation, which makes the
            # broker hop invisible in the waterfall.
            with _tracing.span(
                "evolve.send",
                kind=_tracing.SPAN_PRODUCER,
                attributes={"task.id": task_id, "resource.id": resource_id,
                            "messaging.system": "celery",
                            "messaging.destination": "toolmarket.evolve"},
            ):
                self.app.send_task(
                    "toolmarket.evolve",
                    args=[resource_id, goal, task_id],
                    kwargs={"commit": commit, "proposer": proposer},
                    task_id=task_id,
                    # Headers, not `kwargs`: the task's own signature stays free of
                    # tracing, so a caller invoking `toolmarket.evolve` by hand — or
                    # a future non-Python producer — is not required to know that
                    # the substrate propagates context at all.
                    headers=_tracing.inject({}),
                )
        except Exception as exc:  # noqa: BLE001
            rec.state = TaskState.FAILURE.value
            rec.error = f"could not enqueue: {type(exc).__name__}: {exc}"
            rec.finished_at = time.time()
            self.store.save(rec)
        return rec

    def status(self, task_id: str) -> Optional[TaskRecord]:
        rec = self.store.get(task_id)
        if rec is not None and rec.state in (TaskState.SUCCESS.value,
                                             TaskState.FAILURE.value):
            return rec
        try:
            async_result = self.app.AsyncResult(task_id)
            if async_result.state == "SUCCESS" and rec is not None:
                rec.state = TaskState.SUCCESS.value
                rec.result = async_result.result
                self.store.save(rec)
            elif async_result.state == "FAILURE" and rec is not None:
                rec.state = TaskState.FAILURE.value
                rec.error = str(async_result.result)
                self.store.save(rec)
        except Exception:  # noqa: BLE001 - cache-only status is still useful
            pass
        return rec

    def join(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskRecord]:
        deadline = time.time() + (timeout or 0)
        while True:
            rec = self.status(task_id)
            if rec is not None and rec.state in (TaskState.SUCCESS.value,
                                                 TaskState.FAILURE.value):
                return rec
            if time.time() >= deadline:
                return rec
            time.sleep(0.25)

    def close(self) -> None:
        return None


def make_queue(registry: Any = None, kind: Optional[str] = None) -> Any:
    """`TASK_QUEUE=celery` for a worker process; anything else runs inline.

    For the inline backend this is *not* a constructor — it is a getter. Asking
    twice for the queue of one registry hands back the same object, and that
    identity is the whole point.

    The bug it fixes: the REST app and the gRPC server each call this for the
    same registry, so a naive `return InlineQueue(registry)` gave a deployment
    two independent queues over one substrate. An evolution submitted through
    gRPC then returned a task id that `GET /tasks/{id}` answered 404 for — and
    nothing about the failure pointed at the queue, it looked like the id was
    wrong. A Celery queue is broker-backed and shares by construction; the
    inline one has no broker, so the sharing has to be explicit.

    Bound to the registry rather than keyed on a module global because the
    registry *is* the substrate: two registries are two substrates and must get
    two queues, which a global would get wrong in the other direction.
    """
    choice = (kind or os.environ.get("TASK_QUEUE") or "inline").strip().lower()
    if choice in ("celery", "worker", "async"):
        return CeleryQueue()
    if registry is not None:
        existing = getattr(registry, "task_queue", None)
        if existing is not None:
            return existing
    queue = InlineQueue(registry=registry)
    if registry is not None:
        # Attached after construction: on the first call `task_queue` is None,
        # which is exactly the "not built yet" answer the next caller needs.
        registry.task_queue = queue
    return queue
