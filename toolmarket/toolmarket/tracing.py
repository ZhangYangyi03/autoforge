"""Distributed tracing, hand-written: W3C Trace Context + OTLP/HTTP JSON.

Same rationale as `metrics.py`, and the same constraint: **no client library.**
The OTel SDK is a large dependency that pins protobuf, and this module needs
three things out of it — a 55-character header format, an id generator, and one
JSON document. Writing those directly keeps the image's dependency set small
enough to reason about and, more usefully, makes the *propagation* explicit:
the failure mode this file exists to prevent is a trace that looks fine on the
API side and is silently broken at the process boundary.

What a trace is, stated once, because everything below follows from it: a
`trace_id` names one logical operation; every span participating in it carries
that same `trace_id`, and each names one `parent_span_id`. Distribution is
therefore not a feature of the tracer — it is a property of *carrying two hex
strings across a boundary that does not carry Python objects*. There are three
such boundaries in this system and each is handled explicitly:

  1. **The HTTP/gRPC request.** The inbound `traceparent` header, if present,
     becomes the parent. If absent, a new trace starts. That is the only place a
     trace is allowed to begin.
  2. **The thread boundary, in-process.** `InlineQueue` runs the evolution on a
     `threading.Thread`. `contextvars` do **not** cross a thread start — a new
     thread gets a fresh, empty context — so the ambient span would vanish and the
     worker would start a second, unrelated trace. `run_in_context()` copies the
     caller's context into the thread explicitly, which is the whole fix.
  3. **The process boundary, via the broker.** Celery sends a JSON message; the
     tracer puts its `traceparent` in the message *headers* (not the body, so the
     task's own arguments need no tracing-aware changes), and the consumer
     re-parents onto it. The worker then continues the API's trace as a child,
     which is what makes an async evolution appear as one waterfall in Grafana
     rather than two unrelated traces seconds apart.

Boundaries 2 and 3 are the parts that production systems get wrong, and both
fail *silently*: the trace still exports, still renders, and merely shows two
half-traces. So both have a test in `tests/test_tracing.py` that asserts the
worker's `trace_id` equals the API's — not that "some span exists".

Exporter: OTLP/HTTP with JSON encoding, because every receiver that speaks OTLP
(the Collector, Tempo, Jaeger, an OpenTelemetry-instrumented backend) accepts it
over plain HTTP and it needs no protobuf to produce. Unset
`OTEL_EXPORTER_OTLP_ENDPOINT` and this module costs one `if` per span — the
no-op path is the default, so a deployment that has not stood up a backend pays
nothing for the instrumentation.
"""
from __future__ import annotations

import atexit
import contextvars
import json
import os
import queue
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

# -- the wire format --------------------------------------------------------

# W3C Trace Context, section 3.2.2: version "-" trace-id "-" parent-id "-" flags.
# The version is fixed at 00 for what this module emits; a received `ff` version
# must be accepted and its extra fields ignored, and any other version must be
# treated as absent rather than partially parsed — the spec is explicit that an
# unknown version is not a reason to guess.
_TRACEPARENT_VERSION = "00"

# SpanKind, per the OTLP protobuf enum. Named rather than magic-numbered at the
# call sites, because `kind=3` in a request handler reads as a typo.
SPAN_INTERNAL = 1
SPAN_SERVER = 2
SPAN_CLIENT = 3
SPAN_PRODUCER = 4
SPAN_CONSUMER = 5

# Status code, per the OTLP enum: 0 unset, 1 ok, 2 error. Unset rather than ok
# for a span that simply finished, because "ok" is a claim and unset is not.
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2


def new_trace_id() -> str:
    """32 hex chars, 16 bytes. Rejects the all-zero id, which is invalid."""
    while True:
        value = secrets.token_hex(16)
        if value != "0" * 32:
            return value


def new_span_id() -> str:
    """16 hex chars, 8 bytes. Also must not be all zero."""
    while True:
        value = secrets.token_hex(8)
        if value != "0" * 16:
            return value


@dataclass(frozen=True)
class SpanContext:
    """The two ids plus flags that travel across a boundary, and nothing else.

    Deliberately not the `Span`: a boundary receives an identity, not the
    sender's attributes, timings or children. Keeping the two types apart is what
    stops a remote parent's data from being half-merged into a local span.
    """

    trace_id: str
    span_id: str
    sampled: bool = True

    def to_traceparent(self) -> str:
        flags = "01" if self.sampled else "00"
        return f"{_TRACEPARENT_VERSION}-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def parse(cls, header: Optional[str]) -> Optional["SpanContext"]:
        """Parse a `traceparent`, returning None for anything invalid.

        Returning None rather than raising is the spec's intent: a malformed
        header means "no parent", not "fail the request". A tracing bug must not
        be able to turn a working API call into a 500, so every field is
        validated and any failure degrades to starting a fresh trace.
        """
        if not header:
            return None
        parts = header.strip().split("-")
        if len(parts) < 4:
            return None
        version, trace_id, span_id, flags = parts[0], parts[1], parts[2], parts[3]
        if version == "ff":
            # Explicitly invalid per spec -- the version is reserved.
            return None
        try:
            int(trace_id, 16)
            int(span_id, 16)
        except ValueError:
            return None
        if len(trace_id) != 32 or len(span_id) != 16:
            return None
        if trace_id == "0" * 32 or span_id == "0" * 16:
            return None
        # Only the low bit of the flags octet is defined (the sampled flag).
        # A future version may define more; unknown bits are ignored, not
        # rejected, so a newer peer's header still propagates.
        try:
            sampled = bool(int(flags, 16) & 0x01)
        except ValueError:
            return None
        return cls(trace_id=trace_id, span_id=span_id, sampled=sampled)


# -- spans ------------------------------------------------------------------


@dataclass
class Span:
    """One timed operation. Mutable while open, exported once it closes."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    kind: int = SPAN_INTERNAL
    start_ns: int = field(default_factory=lambda: time.time_ns())
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: int = STATUS_UNSET
    error_message: str = ""
    # A `link` is for a relationship that is not parent/child. The Celery retry
    # path uses it: a retried evolution shares the trace but not the parentage,
    # because claiming the original attempt as its parent would draw two
    # concurrent attempts as a sequence.
    links: list[SpanContext] = field(default_factory=list)
    # Sampling is per-trace and decided at the root, so it is a property of the
    # trace's identity rather than of this span. Carried per-span so a child
    # created from a context inherits the root's decision instead of re-sampling.
    _sampled: bool = True

    def context(self) -> SpanContext:
        return SpanContext(self.trace_id, self.span_id, self._sampled)

    def set_attribute(self, key: str, value: Any) -> "Span":
        self.attributes[key] = value
        return self

    def set_attributes(self, values: dict[str, Any]) -> "Span":
        self.attributes.update(values)
        return self

    def record_error(self, exc: BaseException) -> "Span":
        """Mark the span failed, with the exception *type* as the status text.

        The message is kept, but the type is what a query filters on: exception
        strings carry ids and paths that make them useless as a grouping key.
        """
        self.status = STATUS_ERROR
        self.error_message = f"{type(exc).__name__}: {exc}"
        self.set_attribute("error.type", type(exc).__name__)
        return self

    def finish(self) -> None:
        if not self.end_ns:
            self.end_ns = time.time_ns()
        # A span that ends before it began is a clock problem, not a data point.
        if self.end_ns < self.start_ns:
            self.end_ns = self.start_ns

    @property
    def duration_ms(self) -> float:
        end = self.end_ns or time.time_ns()
        return (end - self.start_ns) / 1e6

    def to_otlp(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(self.end_ns or time.time_ns()),
        }
        if self.parent_span_id:
            out["parentSpanId"] = self.parent_span_id
        if self.attributes:
            out["attributes"] = [
                {"key": k, "value": _otlp_value(v)}
                for k, v in self.attributes.items()
            ]
        if self.status == STATUS_ERROR:
            out["status"] = {"code": STATUS_ERROR, "message": self.error_message}
        elif self.status == STATUS_OK:
            out["status"] = {"code": STATUS_OK}
        if self.links:
            out["links"] = [
                {"traceId": l.trace_id, "spanId": l.span_id} for l in self.links
            ]
        return out


def _otlp_value(value: Any) -> dict[str, Any]:
    """Python value -> the OTLP `AnyValue` oneof.

    `bool` is checked before `int` because `bool` is a subclass of `int` in
    Python: getting that order wrong serialises every boolean attribute as the
    integer 1, which is a legal OTLP value and therefore silently wrong.
    """
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


# -- the ambient span -------------------------------------------------------

# The span currently being executed, so instrumentation deep in the stack can
# nest without every function taking a `parent` argument. A ContextVar rather
# than a thread-local because it is the mechanism asyncio composes properly, and
# because `copy_context()` is precisely how it is carried into a new thread.
_CURRENT: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "toolmarket_current_span", default=None
)


def current_span() -> Optional[Span]:
    return _CURRENT.get()


def current_context() -> SpanContext:
    """The context to send to a peer: the ambient span's, or a fresh unsampled one.

    A fresh *unsampled* context when there is no ambient span, rather than an
    exception: a caller enqueueing work outside any request (a CLI, a cron) still
    needs a legal `traceparent`, and inventing a sampled one would make every
    such call a separate 100%-sampled trace in the backend.
    """
    span = _CURRENT.get()
    if span is not None:
        return span.context()
    return SpanContext(new_trace_id(), new_span_id(), sampled=False)


def in_thread(target: Callable[..., Any], *args: Any, name: Optional[str] = None,
              daemon: bool = True, **kwargs: Any) -> threading.Thread:
    """Build a `Thread` that inherits the *caller's* trace context.

    Thread-boundary fix (2). `threading.Thread` starts with an empty context, so
    an ambient span set in the request thread is invisible in the thread the work
    actually runs on. The symptom is not an error: the worker's spans export with
    a *different* `trace_id`, so one trace becomes two and the API's looks like it
    ended when the enqueue returned.

    **The copy has to happen here, in the caller, and that is the entire reason
    this helper builds the Thread instead of running a callable.** The first
    version of this function was `run_in_context(fn)` — it called
    `copy_context()` and then `ctx.run(fn)` — and it was wrong: invoked as the
    thread's target, which is the natural way to call it, it copies the *new*
    thread's empty context and propagates nothing. It looked right, it type-checked,
    and it reproduced the exact bug it was written to fix.

    A helper whose correctness depends on which thread calls it is the bug, not
    the fix. So the API hands back a thread whose context was captured at
    construction: `tracing.in_thread(work).start()` cannot be called from the
    wrong side, because constructing it *is* the parent-side act.
    """
    ctx = contextvars.copy_context()

    def _run() -> None:
        ctx.run(target, *args, **kwargs)

    return threading.Thread(target=_run, name=name, daemon=daemon)


def _resolve_parent(parent: Optional[Any]) -> tuple[str, str, bool]:
    """Normalise the several things a caller may pass as a parent.

    Accepts a `Span`, a `SpanContext`, a raw `traceparent` string, or None, and
    returns `(trace_id, parent_span_id, sampled)`. Accepting the raw string
    matters: the Celery consumer only ever has the header, and making it build a
    `SpanContext` first is an extra conversion for it to get wrong.
    """
    if parent is None:
        parent = _CURRENT.get()
    if parent is None:
        return new_trace_id(), "", True
    if isinstance(parent, Span):
        return parent.trace_id, parent.span_id, parent._sampled
    if isinstance(parent, SpanContext):
        return parent.trace_id, parent.span_id, parent.sampled
    if isinstance(parent, str):
        parsed = SpanContext.parse(parent)
        if parsed is not None:
            return parsed.trace_id, parsed.span_id, parsed.sampled
    # An unparseable parent is treated as no parent, never as a reason to fail.
    return new_trace_id(), "", True


# -- the exporter -----------------------------------------------------------

# The queue is bounded and dropping, on purpose. An exporter that applies
# backpressure to the request path turns a slow collector into a slow API, and
# the trace data it is protecting is diagnostic — worth less than the request.
_MAX_QUEUE = 2048
# One POST per span would be chatty; one POST per hour would be useless. Also
# the reason batching is not a contract: two spans emitted microseconds apart
# land in the same batch or in two depending on when the drain thread wakes.
_MAX_BATCH = 256


class OtlpHttpExporter:
    """Batches finished spans and POSTs them as OTLP/JSON.

    A background daemon thread, because the alternative — export inline at span
    end — adds a network round trip to every request and hands the collector
    control of API latency. `atexit` flush so a short-lived process (the CLI, a
    test) still gets its spans out.
    """

    def __init__(self, endpoint: str, service_name: str,
                 headers: Optional[dict[str, str]] = None,
                 timeout: float = 5.0) -> None:
        # Accept both the base URL and the full signal path. The OTel spec's
        # convention is a base endpoint with `/v1/traces` appended, and a user who
        # pastes the full path should not get a double `/v1/traces/v1/traces`.
        base = endpoint.rstrip("/")
        if base.endswith("/v1/traces"):
            self.url = base
        else:
            self.url = f"{base}/v1/traces"
        self.service_name = service_name
        self.headers = {"Content-Type": "application/json"}
        self.headers.update(headers or {})
        self.timeout = timeout
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(_MAX_QUEUE)
        self._dropped = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="otel-exporter",
                                        daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def emit(self, span: Span) -> None:
        try:
            self._queue.put_nowait([span.to_otlp()])
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def _payload(self, spans: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name",
                     "value": {"stringValue": self.service_name}},
                ]},
                "scopeSpans": [{
                    "scope": {"name": "toolmarket", "version": "0.1.0"},
                    "spans": list(spans),
                }],
            }],
        }

    def _collect_batch(self, first: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`first` plus whatever else is already waiting, up to the ceiling.

        Split out from the loop so it can be driven without a live thread.
        Coalescing is an efficiency, not a promise: whether two spans emitted
        back to back share a POST depends on thread scheduling, so nothing may
        assert on it.
        """
        batch = list(first)
        while len(batch) < _MAX_BATCH:
            try:
                batch.extend(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._post(self._collect_batch(batch))

    def _post(self, spans: Sequence[dict[str, Any]]) -> None:
        body = json.dumps(self._payload(spans)).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers=self.headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            # A failed export must never surface to the caller. There is no
            # retry: an in-memory queue that retries a dead collector is a queue
            # that fills, and the spans are diagnostic.
            pass

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue drains, or `timeout`. For tests and shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.02)

    def shutdown(self) -> None:
        self._stop.set()
        self.flush(timeout=1.0)


# -- the tracer -------------------------------------------------------------


class Tracer:
    """Starts spans. Holds the exporter, the sampler and the service name."""

    def __init__(self, service_name: str = "toolmarket",
                 exporter: Optional[OtlpHttpExporter] = None,
                 sample_ratio: float = 1.0) -> None:
        self.service_name = service_name
        self.exporter = exporter
        self.sample_ratio = sample_ratio

    def _sample(self, parent_sampled: Optional[bool]) -> bool:
        """Decide sampling, and honour a parent's decision.

        A parent that sampled forces its children to sample: sampling is
        per-trace, and a child that drops out leaves a broken waterfall — worse
        than no trace, because it looks like the operation ended there. A parent
        that did *not* sample is likewise honoured, which is what makes a 1%
        sampling rate actually cost 1%.
        """
        if parent_sampled is not None:
            return parent_sampled
        if self.sample_ratio >= 1.0:
            return True
        if self.sample_ratio <= 0.0:
            return False
        return (secrets.randbelow(10_000) / 10_000) < self.sample_ratio

    def start_span(self, name: str, *, kind: int = SPAN_INTERNAL,
                   parent: Optional[Any] = None,
                   attributes: Optional[dict[str, Any]] = None,
                   sampled: Optional[bool] = None) -> Span:
        """Begin a span. Not a context manager — `span()` is that.

        The distinction matters for the async boundaries: a producer wants to
        start a span, inject its context into a message, and end it, without
        holding a `with` block across the broker call.
        """
        if sampled is None:
            explicit_parent = parent is not None
            trace_id, parent_span_id, inherited = _resolve_parent(parent)
            # An explicit parent's decision is inherited even if that parent
            # arrived as a bare traceparent. Only a root consults the sampler.
            sampled = (inherited if explicit_parent or parent_span_id
                       else self._sample(None))
        else:
            trace_id, parent_span_id, _ = _resolve_parent(parent)
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=new_span_id(),
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=dict(attributes or {}),
        )
        span._sampled = sampled
        return span

    def end_span(self, span: Span) -> None:
        span.finish()
        if span._sampled and self.exporter is not None:
            self.exporter.emit(span)

    def span(self, name: str, *, kind: int = SPAN_INTERNAL,
             parent: Optional[Any] = None,
             attributes: Optional[dict[str, Any]] = None,
             sampled: Optional[bool] = None) -> "_SpanScope":
        return _SpanScope(self, name, kind=kind, parent=parent,
                          attributes=attributes, sampled=sampled)


class _SpanScope:
    """`with tracer.span(...) as span:` — starts on enter, ends on exit.

    Exceptions are recorded on the span and re-raised untouched. The `raise`
    is deliberate: instrumentation that swallows an exception to keep the span
    tidy turns a failing request into a successful one.
    """

    def __init__(self, tracer: Tracer, name: str, *, kind: int,
                 parent: Optional[Any] = None,
                 attributes: Optional[dict[str, Any]] = None,
                 sampled: Optional[bool] = None) -> None:
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.parent = parent
        self.attributes = attributes
        self.sampled = sampled
        self.span: Optional[Span] = None
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> Span:
        span = self.tracer.start_span(self.name, kind=self.kind,
                                      parent=self.parent,
                                      attributes=self.attributes,
                                      sampled=self.sampled)
        self.span = span
        # Always ambient, sampled or not. An earlier version set this only for
        # sampled spans, reasoning that an unsampled span has nothing to export
        # and the set/reset is pure cost. That conflates two different things:
        # whether a span is *exported* and whether it *links*. An unsampled parent
        # that is not ambient leaves its children with no parent, so they start
        # their own traces and inherit `sampled=True` from the sampler — the
        # unsampled subtree is not thinned, it is scattered. And a downstream
        # service can only honour "do not sample this" if the flag it propagates
        # came from a span that was installed.
        self._token = _CURRENT.set(span)
        return span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        assert self.span is not None
        if exc is not None:
            self.span.record_error(exc)
        if self._token is not None:
            _CURRENT.reset(self._token)
        self.tracer.end_span(self.span)
        return False


# -- propagation, at the boundaries ----------------------------------------


def inject(headers: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Write the ambient `traceparent` into `headers`, returning them.

    Celery's `send_task(headers=...)` is the consumer of this. The header name is
    the W3C one so that a Celery hop and an HTTP hop carry the same key — a
    broker-specific name like `x-toolmarket-trace` would work and would also
    guarantee that no standard tool could read it.
    """
    headers = {} if headers is None else headers
    headers["traceparent"] = current_context().to_traceparent()
    return headers


def extract(headers: Optional[dict[str, Any]]) -> Optional[SpanContext]:
    """Read a `traceparent` from a header mapping, or None."""
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "traceparent":
            return SpanContext.parse(value if isinstance(value, str) else str(value))
    return None


# -- the process-wide tracer ------------------------------------------------

_TRACER: Optional[Tracer] = None
_TRACER_LOCK = threading.Lock()


def build_tracer(service_name: Optional[str] = None) -> Tracer:
    """Build a tracer from the environment. The OTel variable names, so a
    deployment that already sets them needs no bespoke configuration."""
    name = (service_name or os.environ.get("OTEL_SERVICE_NAME") or "toolmarket")
    endpoint = (os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    ratio_raw = (os.environ.get("OTEL_TRACES_SAMPLER_ARG") or "1.0").strip()
    try:
        ratio = max(0.0, min(1.0, float(ratio_raw)))
    except ValueError:
        ratio = 1.0
    exporter = OtlpHttpExporter(endpoint, name) if endpoint else None
    return Tracer(service_name=name, exporter=exporter, sample_ratio=ratio)


def get_tracer() -> Tracer:
    """The process-wide tracer, built once from the environment.

    A singleton for the same reason `get_cache()` is: the exporter owns a queue
    and a thread, and two tracers would mean two of each and spans split across
    both. Built lazily so importing this module on a machine with no collector
    configured does nothing at all.
    """
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is None:
            _TRACER = build_tracer()
        return _TRACER


def reset_tracer(tracer: Optional[Tracer] = None) -> None:
    """Swap the process tracer. Tests call this; nothing else should."""
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None and _TRACER.exporter is not None:
            _TRACER.exporter.shutdown()
        _TRACER = tracer


def span(name: str, **kwargs: Any) -> _SpanScope:
    """Module-level convenience: `with tracing.span("x", kind=...):`."""
    return get_tracer().span(name, **kwargs)


__all__ = [
    "SPAN_CLIENT", "SPAN_CONSUMER", "SPAN_INTERNAL", "SPAN_PRODUCER",
    "SPAN_SERVER", "STATUS_ERROR", "STATUS_OK", "STATUS_UNSET",
    "OtlpHttpExporter", "Span", "SpanContext", "Tracer",
    "build_tracer", "current_context", "current_span", "extract", "get_tracer",
    "in_thread", "inject", "new_span_id", "new_trace_id", "reset_tracer", "span",
]
