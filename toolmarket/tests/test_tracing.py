"""Trace continuity across the boundaries that do not carry Python objects.

A tracer is easy to test and easy to test *uselessly*. Asserting that a span
exports, that it has a name, that its duration is positive — all of that passes
for a tracer whose traces are broken, because a broken trace still exports. The
failure this file exists to catch is the one that renders perfectly and is wrong:

    the API's span and the worker's span carry *different* trace_ids.

That happens at exactly two places in this system, both of which start a
concurrency context that does not inherit the caller's:

  * `InlineQueue` runs the evolution on a `threading.Thread`. A new thread gets a
    fresh, empty `contextvars.Context`, so the ambient span is invisible in it.
  * `CeleryQueue` hands the work to another *process*, where nothing is shared at
    all and the only thing that can survive is a string in a message header.

Both are silent. Nothing raises, no span is lost, and the dashboard shows two
short traces instead of one waterfall — with the API's looking like the operation
ended when the enqueue call returned. So the assertions below are on *identity*:
same `trace_id`, correct `parent_span_id`. Each boundary gets a test that shows
the failure as well as the fix, because a test that only shows the fix cannot
tell you the fix is load-bearing.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from toolmarket import tracing
from toolmarket.tracing import (
    SPAN_CONSUMER,
    SPAN_PRODUCER,
    SPAN_SERVER,
    STATUS_ERROR,
    OtlpHttpExporter,
    Span,
    SpanContext,
    Tracer,
)

API_SPAN = "api.evolve.enqueue"
WORKER_SPAN = "worker.evolve"


@pytest.fixture
def tracer():
    """A tracer with no exporter: these cases are about ids, not about export."""
    return Tracer(service_name="test", exporter=None)


@pytest.fixture(autouse=True)
def _isolate_ambient_span():
    """Nothing may leak a parent into the next case.

    The ambient span lives in a ContextVar, which is per-context rather than
    global — so a leak is unlikely and, if it happened, would look like a test
    that passes for the wrong reason. Reset both the variable and the process
    tracer around every case.
    """
    tracing._CURRENT.set(None)  # noqa: SLF001
    yield
    tracing._CURRENT.set(None)  # noqa: SLF001
    tracing.reset_tracer(None)


# -- the wire format -------------------------------------------------------

def test_traceparent_round_trips():
    """The header a peer parses must be the header this module writes."""
    original = SpanContext("a" * 32, "b" * 16, sampled=True)
    header = original.to_traceparent()
    assert header == f"00-{'a' * 32}-{'b' * 16}-01"
    parsed = SpanContext.parse(header)
    assert parsed == original


def test_sampled_flag_survives_the_round_trip():
    """`sampled=False` must not become `True` in transit.

    It is a single bit in one hex octet, and the default in `SpanContext` is
    `True` — so an implementation that parses the flags but forgets to pass them
    to the constructor silently upgrades every unsampled trace to sampled and a
    1% sampling rate quietly becomes 100%.
    """
    for sampled in (True, False):
        ctx = SpanContext("c" * 32, "d" * 16, sampled=sampled)
        assert SpanContext.parse(ctx.to_traceparent()).sampled is sampled


@pytest.mark.parametrize("bad", [
    None,                                  # absent
    "",                                    # empty
    "00-tooshort-" + "e" * 16 + "-01",     # trace id wrong length
    "00-" + "e" * 32 + "-tooshort-01",     # span id wrong length
    "00-" + "0" * 32 + "-" + "e" * 16 + "-01",   # all-zero trace id
    "00-" + "e" * 32 + "-" + "0" * 16 + "-01",   # all-zero span id
    "00-" + "z" * 32 + "-" + "e" * 16 + "-01",   # not hex
    "ff-" + "e" * 32 + "-" + "f" * 16 + "-01",   # reserved version
    "00-" + "e" * 32 + "-" + "f" * 16,           # truncated: no flags
])
def test_invalid_traceparent_is_not_a_parent(bad):
    """Anything malformed degrades to `None`, never to an exception.

    A tracing bug must not be able to fail a request, so the parse contract is
    "valid context or nothing" — and the caller treats nothing as "start a fresh
    trace". The reserved `ff` version is in the list because the spec forbids
    inferring anything from it, and the truncated case because a proxy that mangles
    the header must not produce a half-parsed parent.
    """
    assert SpanContext.parse(bad) is None


def test_unknown_flag_bits_are_ignored_not_rejected():
    """A future version may define more flags; an unknown bit is not an error.

    `02` has the sampled bit clear and an undefined bit set. The spec's forward
    compatibility rule is that unknown flags propagate untouched, so rejecting
    this would make the tracer incompatible with a newer peer for no reason.
    """
    ctx = SpanContext.parse("00-" + "a" * 32 + "-" + "b" * 16 + "-02")
    assert ctx is not None and ctx.sampled is False


# -- nesting ---------------------------------------------------------------

def test_child_shares_the_parents_trace_id(tracer):
    """The base invariant. Everything else in this file is a special case."""
    with tracer.span("root") as root:
        with tracer.span("child") as child:
            assert child.trace_id == root.trace_id
            assert child.parent_span_id == root.span_id
            assert child.span_id != root.span_id


def test_the_ambient_span_is_restored_on_exit(tracer):
    """Nesting must not leak the inner span to a sibling or to the caller.

    A tracer that only ever *sets* the ambient span leaves a finished span
    installed, so every later span in the process becomes its child — one
    enormous trace rooted at whichever request happened to run first. Restoring
    the previous value is the fix, and it is invisible until there are siblings.
    """
    with tracer.span("root") as root:
        with tracer.span("first"):
            pass
        assert tracing.current_span() is root
        with tracer.span("second") as second:
            assert second.trace_id == root.trace_id
    assert tracing.current_span() is None


def test_an_exception_is_recorded_and_re_raised(tracer):
    """Status is set on the way out; the exception is not swallowed.

    Swallowing would turn a failing request into a successful one, which is a
    strictly worse bug than a missing span.
    """
    with pytest.raises(ValueError):
        with tracer.span("work") as s:
            raise ValueError("boom")
    assert s.status == STATUS_ERROR
    assert "ValueError" in s.error_message


# -- boundary 1: the thread, in-process -----------------------------------

def test_a_plain_thread_loses_the_span_and_that_is_the_bug(tracer):
    """Documents the failure that `run_in_context` exists to prevent.

    Not a test of correct behaviour — it asserts the *broken* behaviour, on
    purpose. `threading.Thread` starts with an empty context, so the ambient span
    is gone. Written down because this is what makes the next test's assertion
    meaningful: without it, "the span arrived" could be a coincidence of when the
    thread happened to start.
    """
    seen: dict[str, object] = {}

    def work() -> None:
        seen["span"] = tracing.current_span()

    with tracer.span(API_SPAN):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(timeout=10)
    assert seen["span"] is None


def test_run_in_context_carries_the_span_into_the_thread(tracer):
    """The fix: the worker thread sees the same trace, not a new one.

    `in_thread` builds the thread, so the context is captured here — in the
    request side — at construction. The first version of this helper took a
    callable and copied the context *inside* it, which meant calling it as the
    thread's target (the obvious way) copied the new thread's empty context and
    propagated nothing. That version failed this test, which is why the helper
    hands back a Thread instead of running one.
    """
    seen: dict[str, Span] = {}
    done = threading.Event()

    def work() -> None:
        with tracer.span(WORKER_SPAN, kind=SPAN_CONSUMER) as s:
            seen["worker"] = s
        done.set()

    with tracer.span(API_SPAN, kind=SPAN_PRODUCER) as api:
        tracing.in_thread(work).start()
        assert done.wait(timeout=10)

    worker = seen["worker"]
    assert worker.trace_id == api.trace_id
    assert worker.parent_span_id == api.span_id


def test_in_thread_understands_a_target_with_arguments(tracer):
    """The helper is a thread factory, so it must forward args and kwargs."""
    seen: dict[str, Any] = {}

    def work(a: int, *, b: int) -> None:
        seen["sum"] = a + b

    t = tracing.in_thread(work, 2, b=3)
    t.start()
    t.join(timeout=10)
    assert seen["sum"] == 5


# -- boundary 2: the process, via the broker ------------------------------

def test_injected_headers_carry_the_ambient_trace(tracer):
    """What a producer puts in the message is what a consumer can re-parent on."""
    with tracer.span(API_SPAN, kind=SPAN_PRODUCER) as api:
        headers = tracing.inject({})
        assert headers["traceparent"] == api.context().to_traceparent()

    # The consumer's whole view: a header dict. This is the boundary — the worker
    # has none of the API's memory, so a traceparent string is all that arrives.
    ctx = tracing.extract(headers)
    assert ctx is not None
    with tracer.span(WORKER_SPAN, kind=SPAN_CONSUMER, parent=headers["traceparent"]) as w:
        assert w.trace_id == api.trace_id
        assert w.parent_span_id == api.span_id


def test_the_consumer_accepts_a_bare_traceparent_string(tracer):
    """The Celery task passes a header string, not a `SpanContext`.

    Pinned because the obvious implementation calls `_resolve_parent` with the
    string and any refactor that stops handling `str` breaks propagation in the
    worker only — the API's own tests would stay green.
    """
    header = SpanContext("1" * 32, "2" * 16).to_traceparent()
    with tracer.span(WORKER_SPAN, parent=header) as s:
        assert s.trace_id == "1" * 32
        assert s.parent_span_id == "2" * 16


def test_a_task_started_outside_a_request_gets_its_own_unsampled_trace(tracer):
    """A producer with no ambient span must not fabricate a sampled trace.

    The CLI and the cron path enqueue with no request around them. Inventing a
    sampled context would make every such call its own 100%-sampled trace, which
    is how a tracing bill appears with no requests to explain it.
    """
    fresh = tracing.current_context()
    assert fresh.sampled is False
    assert SpanContext.parse(fresh.to_traceparent()) is not None  # still legal


def test_extract_tolerates_odd_header_shapes():
    """Headers arrive from brokers, proxies and users; none of it may raise."""
    assert tracing.extract({"TraceParent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"})
    assert tracing.extract({"traceparent": 12345}) is None
    assert tracing.extract({"traceparent": "garbage"}) is None
    assert tracing.extract({}) is None
    assert tracing.extract(None) is None


# -- sampling --------------------------------------------------------------

def test_a_child_inherits_an_unsampled_parent_even_at_full_ratio(tracer):
    """The parent's decision wins over the child's own sampler.

    Otherwise a 1%-sampled trace grows back into a fully-sampled one the moment
    it crosses into a service whose ratio is 1.0 — and a broken waterfall, where
    the root is missing and the children are not, is worse than no trace.
    """
    with tracer.span("root", sampled=False) as root:
        with tracer.span("child") as child:
            assert root.trace_id == child.trace_id
            assert child._sampled is False


def test_zero_ratio_samples_nothing(tracer):
    tracer.sample_ratio = 0.0
    with tracer.span("root") as s:
        assert s._sampled is False


def test_an_explicit_parent_is_never_re_sampled():
    """A remote parent's `sampled` bit is authoritative, at any ratio.

    The receiving process has no idea what the sender's sampler decided, and the
    only record of it is the flag octet. Re-rolling the dice locally is what makes
    a trace appear in one service and not the next.
    """
    t = Tracer(service_name="t", exporter=None, sample_ratio=0.0)
    header = SpanContext("3" * 32, "4" * 16, sampled=True).to_traceparent()
    with t.span("child", parent=header) as s:
        assert s._sampled is True


# -- the exporter ----------------------------------------------------------

def test_otlp_payload_shape():
    """The document a receiver is handed, checked field by field.

    OTLP/JSON is strict about types in ways that are easy to get wrong and hard
    to notice: the unix-nano timestamps are *strings* (they exceed float64
    precision) and the ids are lowercase hex. A payload that is merely close is
    rejected by the receiver, which means no traces at all rather than one bad
    span.
    """
    span = Span(name="op", trace_id="a" * 32, span_id="b" * 16,
                parent_span_id="c" * 16, kind=SPAN_SERVER)
    span.set_attributes({"http.route": "/ready", "http.status_code": 200,
                         "ok": True})
    span.finish()
    doc = json.loads(json.dumps(span.to_otlp()))

    assert doc["traceId"] == "a" * 32
    assert doc["spanId"] == "b" * 16
    assert doc["parentSpanId"] == "c" * 16
    assert isinstance(doc["startTimeUnixNano"], str)
    assert isinstance(doc["endTimeUnixNano"], str)
    values = {a["key"]: a["value"] for a in doc["attributes"]}
    assert values["http.route"] == {"stringValue": "/ready"}
    # An int is an int and a bool is a bool — `bool` subclasses `int` in Python,
    # so the wrong check order turns every boolean attribute into the integer 1,
    # which is a legal OTLP value and therefore silently wrong.
    assert values["http.status_code"] == {"intValue": "200"}
    assert values["ok"] == {"boolValue": True}


def test_an_unparented_span_omits_parent_span_id():
    """A root span with `parentSpanId: ""` is a malformed field, not a null one."""
    span = Span(name="root", trace_id="a" * 32, span_id="b" * 16)
    span.finish()
    assert "parentSpanId" not in span.to_otlp()


def test_exporter_posts_to_the_signal_url():
    """Every span reaches `<endpoint>/v1/traces`.

    Deliberately silent on *how many* POSTs. Coalescing depends on when the
    drain thread wakes relative to the next `emit`, so a version of this that
    asserted one batch would pass or fail on scheduler luck — it did, on
    py3.11 only, which is what a race looks like when it is mistaken for a
    platform difference. The batch is a transport detail; delivery is the
    contract. Coalescing is pinned below, deterministically.
    """
    posted: list[tuple[str, dict]] = []

    exporter = OtlpHttpExporter("http://collector:4318", "svc")
    exporter._post = lambda spans: posted.append((exporter.url, spans))  # type: ignore[method-assign]

    t = Tracer(service_name="svc", exporter=exporter)
    with t.span("a"):
        pass
    with t.span("b"):
        pass

    # Wait for delivery rather than for the queue to look empty: a batch is
    # taken off the queue before it is posted, so "empty" can be observed
    # mid-flight.
    delivered: list[str] = []
    deadline = time.time() + 3
    while time.time() < deadline:
        delivered = [s["name"] for _, batch in posted for s in batch]
        if sorted(delivered) == ["a", "b"]:
            break
        time.sleep(0.02)

    assert exporter.url == "http://collector:4318/v1/traces"
    assert posted, "exporter never posted"
    assert sorted(delivered) == ["a", "b"], f"lost spans: {delivered}"
    exporter.shutdown()


def test_exporter_coalesces_what_is_already_queued():
    """The drain step takes everything waiting, so N queued spans cost one POST.

    Driven directly with the thread already stopped, because that is the only
    way to ask "given three spans waiting, how many POSTs?" without racing the
    thread that would answer it.
    """
    exporter = OtlpHttpExporter("http://collector:4318", "svc")
    exporter.shutdown()  # stop the drain thread; the test drives the batch

    for name in ("a", "b", "c"):
        exporter._queue.put_nowait([{"name": name}])

    batch = exporter._collect_batch(exporter._queue.get_nowait())
    assert [s["name"] for s in batch] == ["a", "b", "c"]
    assert exporter._queue.empty()


def test_exporter_does_not_double_append_the_signal_path():
    """A user who pastes the full signal URL must not get `/v1/traces/v1/traces`."""
    assert OtlpHttpExporter("http://c:4318/v1/traces", "s").url == \
        "http://c:4318/v1/traces"
    assert OtlpHttpExporter("http://c:4318/", "s").url == "http://c:4318/v1/traces"


def test_a_full_queue_drops_rather_than_blocking():
    """The request path must never wait on the collector.

    A bounded, dropping queue is the design: the spans are diagnostic, and the
    alternative — backpressure from a slow collector — turns tracing into an
    outage amplifier. What is asserted is that `emit` returns and counts the loss.
    """
    exporter = OtlpHttpExporter("http://127.0.0.1:9", "svc")  # nothing listens
    exporter._queue = __import__("queue").Queue(1)  # type: ignore[assignment]
    exporter._queue.put_nowait([{"filler": True}])
    span = Span(name="overflow", trace_id="a" * 32, span_id="b" * 16)
    span.finish()
    started = time.monotonic()
    exporter.emit(span)
    assert time.monotonic() - started < 1.0
    assert exporter.dropped == 1
    exporter.shutdown()


def test_an_unreachable_collector_is_swallowed_and_does_not_raise():
    """Export failure is invisible to the caller by construction."""
    exporter = OtlpHttpExporter("http://127.0.0.1:9", "svc", timeout=0.5)
    t = Tracer(service_name="svc", exporter=exporter)
    with t.span("x"):
        pass
    exporter.flush(timeout=3)
    exporter.shutdown()


# -- the no-backend default ------------------------------------------------

def test_no_endpoint_means_no_exporter_and_no_thread(monkeypatch):
    """Unset `OTEL_EXPORTER_OTLP_ENDPOINT` costs nothing.

    The no-op path is the default so a deployment with no tracing backend pays
    for an `if` per span rather than for a queue and a background thread.
    """
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert tracing.build_tracer().exporter is None


def test_the_endpoint_env_var_builds_an_exporter(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "toolmarket-api")
    t = tracing.build_tracer()
    assert t.service_name == "toolmarket-api"
    assert t.exporter is not None
    assert t.exporter.url == "http://collector:4318/v1/traces"
    t.exporter.shutdown()


def test_a_bad_sampling_ratio_falls_back_to_always_on(monkeypatch):
    """A typo in an env var must not silently disable tracing."""
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "not-a-number")
    assert tracing.build_tracer().sample_ratio == 1.0
