"""The async path: the task record, the inline queue, and the honest failure.

The evolution itself is not tested here — `test_protocol.py` and
`test_ablation_measurement.py` own the gate. What is tested is the *plumbing*
around it, because that is where the async design can be quietly wrong: a task id
that cannot be polled, a queue that reports `success` for work it never ran, a
worker that cannot see the API's resources.

No LLM is called. The failure cases are the interesting ones and they are
reachable without a model: an unknown resource must produce a *terminal* task
carrying the reason, not a task that stays `pending` forever.
"""
from __future__ import annotations

import time

import pytest

from toolmarket.cache import MemoryCache, NullCache, make_cache, reset_cache_singleton
from toolmarket.tasks import (
    InlineQueue,
    TaskRecord,
    TaskState,
    TaskStore,
    make_queue,
    task_key,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_cache_singleton()
    yield
    reset_cache_singleton()


# ------------------------------------------------------------- TaskRecord
def test_task_record_round_trips_through_dict():
    rec = TaskRecord(task_id="t1", resource_id="tool:a", goal="tighten")
    assert rec.state == TaskState.PENDING.value
    assert rec.duration is None
    clone = TaskRecord.from_dict(rec.to_dict())
    assert (clone.task_id, clone.resource_id, clone.goal) == ("t1", "tool:a", "tighten")
    assert clone.state == TaskState.PENDING.value


def test_duration_is_elapsed_while_running_and_exact_when_finished():
    """`duration` is None only before the work starts.

    While a task is running, elapsed-so-far is the useful answer — a client
    polling `/tasks/{id}` wants to know it has been going for 40 seconds, not
    `null`. Once finished it is the exact span, which is what a benchmark or a
    test asserts against.
    """
    rec = TaskRecord(task_id="t1", resource_id="tool:a")
    assert rec.duration is None
    rec.started_at = time.time() - 1.5
    # Sampled on both sides, so the exact bound depends on how long the
    # subtraction itself takes; the tolerance is for that, not for the logic.
    assert rec.duration == pytest.approx(1.5, abs=0.3)
    rec.started_at = 100.0
    rec.finished_at = 102.5
    assert rec.duration == pytest.approx(2.5)


def test_task_key_has_no_namespace_of_its_own():
    # The cache applies the `tm:v1:` namespace itself (see cache.py), so a second
    # one here would produce `tm:v1:task:...` and a key that nothing else can
    # find — the mismatch only shows up as "task not found" across processes.
    assert task_key("abc") == "task:abc"
    assert not task_key("abc").startswith("tm:v1")


# -------------------------------------------------------------- TaskStore
def test_task_store_round_trip():
    store = TaskStore(MemoryCache())
    rec = TaskRecord(task_id="t1", resource_id="tool:a", goal="g")
    store.save(rec)
    got = store.get("t1")
    assert got is not None and got.goal == "g"
    assert store.get("nope") is None


def test_a_cache_that_stores_nothing_makes_tasks_unpollable_not_broken():
    """`REDIS_URL=none` means no task records exist — a documented limit.

    `NullCache` is for deployments that deliberately want no cache, and task
    records live *in* the cache, so with it there is nowhere to put them. The
    behaviour under test is that this is a clean `None` (→ 404 on `/tasks/{id}`)
    rather than an exception: a 500 on the polling endpoint would present as "the
    async API is broken" with no hint that the cause is a cache setting.

    This is also why `MemoryCache`, not `NullCache`, is the default when
    `REDIS_URL` is unset: a single-process container still needs to remember the
    task it just accepted.
    """
    store = TaskStore(NullCache())
    store.save(TaskRecord(task_id="t1", resource_id="tool:a"))
    assert store.get("t1") is None
    assert store.by_state() == {s.value: 0 for s in TaskState}


def test_the_default_cache_can_hold_task_records():
    # The container's bare-run default, asserted rather than assumed.
    from toolmarket.cache import MemoryCache, make_cache

    assert isinstance(make_cache(), MemoryCache)
    store = TaskStore(MemoryCache())
    store.save(TaskRecord(task_id="t1", resource_id="tool:a"))
    assert store.get("t1") is not None


def test_by_state_counts_what_was_saved():
    store = TaskStore(MemoryCache())
    for i, state in enumerate([TaskState.PENDING, TaskState.RUNNING,
                               TaskState.SUCCESS, TaskState.SUCCESS]):
        rec = TaskRecord(task_id=f"t{i}", resource_id="tool:a")
        rec.state = state.value
        store.save(rec)
    counts = store.by_state()
    assert counts.get("success") == 2
    assert counts.get("pending") == 1
    assert counts.get("running") == 1
    # Every state is reported, including the empty ones, so a panel does not
    # disappear when a state momentarily has no members. `TaskState.__members__`
    # is the set of *names* (uppercase); the counts are keyed by value.
    assert set(counts) == {s.value for s in TaskState}
    assert set(TaskState.__members__) == {s.name for s in TaskState}


# ------------------------------------------------------------ InlineQueue
def _registry_without_resources():
    """/tasks against a registry whose store is empty.

    Deliberately not the fixture from test_protocol.py: the point of these cases
    is the failure path, and the failure path needs a resource id that does not
    exist. Reaching the real proposal path here would drag an LLM into a unit test.
    """
    from toolmarket.registry import ResourceRegistry
    from toolmarket.store import ResourceStore

    return ResourceRegistry(ResourceStore(":memory:"))


def test_inline_queue_reports_a_terminal_failure_for_an_unknown_resource():
    """The failure must be *visible and terminal*.

    An async submit whose task never leaves `pending` is indistinguishable from
    a worker that died, and the client polls forever. Carrying the exception into
    the record is what makes the 202 answer debuggable.
    """
    q = InlineQueue(registry=_registry_without_resources())
    rec = q.submit("tool:does-not-exist", "tighten")
    assert rec.state == TaskState.PENDING.value
    assert rec.queue == "inline"

    done = q.join(rec.task_id, timeout=15)
    assert done is not None
    assert done.state == TaskState.FAILURE.value
    assert "unknown resource" in (done.error or "")
    # A traceback is a map of the filesystem and the code; this field is served
    # to whoever holds the task id. The message is enough.
    assert "Traceback" not in (done.error or "")
    assert done.finished_at is not None and done.duration is not None


def test_inline_queue_status_is_none_for_an_unknown_task():
    q = InlineQueue(registry=_registry_without_resources())
    assert q.status("never-submitted") is None


def test_inline_tasks_are_counted_by_the_task_store():
    q = InlineQueue(registry=_registry_without_resources())
    q.submit("tool:missing", "g")
    counts = TaskStore(q.store.cache).by_state()
    assert sum(counts.values()) >= 1


def test_inline_queue_close_joins_without_hanging():
    q = InlineQueue(registry=_registry_without_resources())
    q.submit("tool:missing", "g")
    q.close()  # must return, not block on a stuck thread


# ------------------------------------------------------------- make_queue
def test_make_queue_defaults_to_inline(monkeypatch):
    monkeypatch.delenv("TASK_QUEUE", raising=False)
    assert isinstance(make_queue(), InlineQueue)


def test_make_queue_selects_celery_by_name_or_environment(monkeypatch):
    from toolmarket.tasks import CeleryQueue

    monkeypatch.setenv("TASK_QUEUE", "celery")
    assert isinstance(make_queue(), CeleryQueue)
    monkeypatch.setenv("TASK_QUEUE", "worker")
    assert isinstance(make_queue(), CeleryQueue)
    monkeypatch.setenv("TASK_QUEUE", "inline")
    assert isinstance(make_queue(), InlineQueue)


def test_celery_queue_status_reads_the_shared_store():
    """`CeleryQueue.status` must resolve from the store, not from the broker.

    The API process has no Celery result backend worth trusting for this: the
    record the worker wrote is the same record the API can read, and polling it
    through the broker would make task status depend on a second system agreeing.
    """
    from toolmarket.tasks import CeleryQueue

    store = TaskStore(MemoryCache())
    store.save(TaskRecord(task_id="t1", resource_id="tool:a", state="success"))
    q = CeleryQueue(store=store)
    got = q.status("t1")
    assert got is not None and got.state == "success"
    assert q.status("absent") is None


def test_celery_queue_join_times_out_rather_than_blocking_forever():
    from toolmarket.tasks import CeleryQueue

    q = CeleryQueue(store=TaskStore(MemoryCache()))
    # Nothing was submitted, so there is nothing to wait for; the contract is
    # that this returns None at the deadline instead of spinning.
    assert q.join("never", timeout=0.3) is None


def test_cache_factory_used_by_the_task_store_is_the_env_one(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(TaskStore().cache, MemoryCache)
    monkeypatch.setenv("REDIS_URL", "none")
    assert isinstance(make_cache(), NullCache)
