"""The Celery worker process — `celery -A toolmarket.worker:celery_app worker`.

Run as its own container. It builds its **own** `ResourceRegistry` against the
same store the API uses, which is the whole reason the store is swappable: a
worker that could not see the API's resources would be a worker that reports
"unknown resource" for everything, and that failure is silent until the async
path is exercised by hand. With `DATABASE_URL` set to Postgres in both
containers, the worker sees exactly what the API sees; with the default SQLite
the two processes would each have a private substrate and the async endpoints
would be broken in a way no unit test would notice. `tests/test_tasks.py` pins
this by asserting the worker path reads a record the API wrote.

Configuration comes entirely from the environment:

    CELERY_BROKER_URL     default redis://localhost:6379/0
    CELERY_RESULT_BACKEND default: the broker
    TASK_QUEUE            set to `celery` in the API so it enqueues instead of
                          running inline
    OTEL_EXPORTER_OTLP_ENDPOINT
                          unset -> no tracing. Set to a collector to get one
                          trace spanning the API request and this worker's run.

The trace is the interesting part of this module. The API puts a W3C
`traceparent` in the *message headers* (`CeleryQueue.submit`), and this process —
which shares no memory with the API — reads it back out and continues the same
trace. That is the only thing that can survive the boundary: two hex strings in a
header. Everything else about the API's context is gone by the time this code
runs, which is precisely why the header exists and why the span below is created
with an *explicit* parent rather than by inheriting an ambient one.
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from celery import Celery
    _HAVE_CELERY = True
    _CELERY_ERR: Optional[BaseException] = None
except Exception as _exc:  # noqa: BLE001
    Celery = None  # type: ignore[assignment]
    _HAVE_CELERY = False
    _CELERY_ERR = _exc

DEFAULT_BROKER = "redis://localhost:6379/0"


def _require() -> None:
    if not _HAVE_CELERY:
        raise RuntimeError(
            "the worker needs celery, which failed to import: "
            f"{_CELERY_ERR!r}. Install the extra: pip install 'tool-market[worker]'"
        )


def _inbound_traceparent(task: Any) -> Optional[dict[str, str]]:
    """The message headers of the delivery being handled, or None.

    `self.request` is the bound-task context Celery populates per delivery, and
    `headers` is what `send_task(headers=...)` put on the wire. Read
    defensively: called outside a real delivery — a direct `.apply()` in a test, a
    `celery call` from the shell — `request` may be absent or a stub, and a tracing
    lookup must never be the thing that turns a working task into a failure.
    """
    request = getattr(task, "request", None)
    headers = getattr(request, "headers", None)
    if not isinstance(headers, dict):
        return None
    return headers


def _run_evolution(task: Any, span: Any, store: Any, resource_id: str, goal: str,
                   task_id: str, commit: bool, proposer: str) -> dict[str, Any]:
    """The body of one evolution. Split out so the tracing wrapper is legible.

    Takes `span` rather than opening one, so the progress callback can annotate
    the same span the caller is timing — a second span here would report the
    stage changes as a sibling of the run instead of as part of it.
    """
    from toolmarket import metrics as _metrics
    from toolmarket.tasks import TaskRecord, TaskState, execute_evolution
    import time as _time

    def progress(stage: str, pct: float) -> None:
        rec = store.get(task_id) or TaskRecord(task_id=task_id,
                                               resource_id=resource_id)
        rec.state = TaskState.RUNNING.value
        rec.started_at = rec.started_at or _time.time()
        rec.stage = stage
        rec.progress = pct
        rec.queue = "celery"
        store.save(rec)
        task.update_state(state="PROGRESS",
                          meta={"stage": stage, "progress": pct})
        span.set_attributes({"evolve.stage": stage, "evolve.progress": pct})

    rec = store.get(task_id) or TaskRecord(task_id=task_id,
                                          resource_id=resource_id,
                                          queue="celery")
    rec.state = TaskState.RUNNING.value
    rec.started_at = _time.time()
    rec.stage = "queued"
    rec.queue = "celery"
    # The worker learns the trace id from the message, not from the API, so this
    # is where the pollable record gets it. Kept in step with the API's copy:
    # both write the same value because both read the same traceparent.
    rec.trace_id = span.trace_id if span is not None else rec.trace_id
    store.save(rec)

    try:
        result = execute_evolution(
            resource_id, goal, commit=commit, proposer=proposer,
            task_id=task_id, registry=None, store=store, progress=progress,
        )
    except Exception as exc:  # noqa: BLE001
        rec.state = TaskState.FAILURE.value
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.finished_at = _time.time()
        store.save(rec)
        _metrics.EVOLUTIONS.inc(outcome="error")
        # Re-raise so Celery records a FAILURE state too — the result backend
        # is what a restarted API reads, and swallowing here would leave it
        # reporting success for a run that never finished. The span is marked
        # failed by the wrapper's context manager on the way out.
        raise
    rec.state = TaskState.SUCCESS.value
    rec.result = result
    rec.progress = 1.0
    rec.finished_at = _time.time()
    store.save(rec)
    _metrics.EVOLUTIONS.inc(outcome="committed" if result.get("committed")
                            else "rejected")
    return result


def build_celery(broker: Optional[str] = None,
                 backend: Optional[str] = None) -> Any:
    """Construct the Celery app. A function, not just a module global, so tests
    can build one against a throwaway broker without import side effects."""
    _require()
    broker_url = broker or os.environ.get("CELERY_BROKER_URL") or DEFAULT_BROKER
    result_url = (backend or os.environ.get("CELERY_RESULT_BACKEND")
                  or broker_url)
    app = Celery("toolmarket", broker=broker_url, backend=result_url)

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Late acknowledgement + reject-on-worker-lost: the default (ack on
        # receipt) loses a task outright if the worker is killed mid-run. For an
        # evolution that has already spent model calls, re-running is cheaper
        # than dropping it.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        # An evolution runs candidate code; it can hang. A hard ceiling means a
        # wedged candidate costs one task slot for ten minutes, not forever.
        task_time_limit=600,
        task_soft_time_limit=540,
        result_expires=3600,
        timezone="UTC",
        broker_connection_retry_on_startup=True,
    )

    @app.task(bind=True, name="toolmarket.evolve", max_retries=0)
    def evolve(self: Any, resource_id: str, goal: str, task_id: str,
               commit: bool = True, proposer: str = "stub") -> dict[str, Any]:
        """Run one evolution in the worker process.

        A thin wrapper whose only job is the trace, which is why the body lives in
        `_run_evolution` instead: the boundary handling should be the first thing
        visible in this function, not buried under progress bookkeeping.

        Progress is mirrored into the cache as each stage begins, because
        Celery's own states are coarse (`STARTED` / `SUCCESS`) and a client
        polling an evolution wants to know whether it is stuck in *assessing* —
        the stage that runs untrusted code — or waiting in the queue.
        """
        from toolmarket.tasks import TaskStore
        from toolmarket.tracing import SPAN_CONSUMER, extract, span as trace_span

        store = TaskStore()

        # The parent is passed *explicitly* from the message header rather than
        # left to the ambient span, and that is not a style choice: this process
        # has no ambient span to inherit — the API's context died with the API's
        # request — so an implicit parent would silently start a new trace here
        # and the async half of every evolution would be a separate trace in the
        # backend. `extract` returns None for an absent or malformed header, which
        # correctly degrades to "start a fresh trace" for a task enqueued by hand.
        inbound = extract(_inbound_traceparent(self))

        with trace_span(
            "toolmarket.evolve",
            kind=SPAN_CONSUMER,
            parent=inbound,
            attributes={
                "task.id": task_id,
                "resource.id": resource_id,
                "messaging.system": "celery",
                "messaging.destination": "toolmarket.evolve",
            },
        ) as trace:
            return _run_evolution(self, trace, store, resource_id, goal, task_id,
                                  commit, proposer)

    return app


# Module-level app for `celery -A toolmarket.worker:celery_app`. Guarded so
# importing this module on a machine that has no celery (and has no intention of
# running a worker) does not explode — the API's optional-extra pattern, applied
# to the worker.
celery_app: Any = None
if _HAVE_CELERY:  # pragma: no cover - exercised by the worker container
    try:
        celery_app = build_celery()
    except Exception:  # noqa: BLE001
        celery_app = None


__all__ = ["build_celery", "celery_app", "DEFAULT_BROKER"]
