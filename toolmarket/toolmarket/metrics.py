"""Metrics, in the Prometheus text exposition format, with no client library.

A `/metrics` endpoint is only worth having if what it exposes is *true*, and the
failure mode of the usual approach is that it is not: a `prometheus_client`
counter that is incremented in the API process has a different value in each of
N workers, so `rate(http_requests_total[5m])` under a load balancer is a
fraction of the truth with no indication that it is a fraction. Two things
follow, and they are the design of this module:

  * Process-local counters are labelled with the process (`instance`) by
    Prometheus itself at scrape time — so fan-out is handled by the scraper
    summing series, which is what the query language is for.
  * Anything that is genuinely global (how many resources exist, how many
    evolution tasks are queued) is **gathered**, not counted: `render()` takes a
    set of callables that read the current state at scrape time. A gauge that has
    to be incremented by every process that might affect it is a gauge that will
    drift; a gauge that is read from the source of truth cannot.

Dependencies: none. The exposition format is four lines of grammar (`# HELP`,
`# TYPE`, `name{labels} value`), and hand-writing it removes a version-pinning
hazard from the image while producing bytes Prometheus and Grafana cannot tell
apart from the library's.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

# Latency buckets, in seconds. Chosen for an HTTP API backed by a database on a
# LAN: the interesting decisions are all below 250 ms, and anything past 5 s is
# "it failed, slowly" rather than a bucket boundary worth distinguishing.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
)


def _escape_label_value(value: Any) -> str:
    """Per the spec: backslash, double quote and newline are the only escapes."""
    return (str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n"))


def _fmt_float(value: float) -> str:
    """Prometheus wants a number, and wants to be told about the odd ones.

    `inf`/`nan` are legal in the format, but emitting Python's `inf` spelling is
    a parse error — the exposition uses `+Inf`. Getting this wrong makes the
    whole scrape fail, not just one series, which is why it is handled here
    rather than trusted to `str()`.
    """
    if value != value:  # NaN
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _fmt_labels(pairs: Sequence[tuple[str, Any]]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in pairs)
    return "{" + inner + "}"


class _Family:
    """Shared plumbing: a name, label names, and a thread-safe child map.

    The lock is per-family and held only for the duration of a dict lookup or an
    arithmetic update. A global lock would serialise every metric in the process
    against every other, which is how instrumentation becomes the bottleneck.
    """

    type_name = "untyped"

    def __init__(self, name: str, help_text: str,
                 labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()
        self._children: dict[tuple[str, ...], Any] = {}

    def _child(self, **labels: Any) -> Any:
        if set(labels) != set(self.labelnames):
            missing = set(self.labelnames) - set(labels)
            extra = set(labels) - set(self.labelnames)
            raise ValueError(
                f"{self.name}: labels must be exactly {self.labelnames}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        key = tuple(str(labels[n]) for n in self.labelnames)
        with self._lock:
            if key not in self._children:
                self._children[key] = self._new_child()
            return self._children[key]

    def _new_child(self) -> Any:
        raise NotImplementedError

    def _sample_pairs(self) -> Iterator[tuple[tuple[str, ...], Any]]:
        with self._lock:
            items = list(self._children.items())
        yield from items

    def _label_pairs(self, key: tuple[str, ...]) -> list[tuple[str, Any]]:
        return list(zip(self.labelnames, key))

    def _emit(self, out: list[str]) -> None:
        out.append(f"# HELP {self.name} {self.help}")
        out.append(f"# TYPE {self.name} {self.type_name}")

    def collect(self) -> list[str]:
        out: list[str] = []
        self._emit(out)
        self._render_children(out)
        return out

    def _render_children(self, out: list[str]) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        with self._lock:
            self._children.clear()


class Counter(_Family):
    """Monotonic. `inc` never decreases it — a counter that can go down breaks
    every `rate()`/`increase()` query built on it, silently."""

    type_name = "counter"

    def _new_child(self) -> float:
        return 0.0

    def inc(self, amount: float = 1.0, **labels: Any) -> float:
        if amount < 0:
            raise ValueError(f"{self.name}: counters are monotonic; got {amount}")
        with self._lock:
            key = tuple(str(labels[n]) for n in self.labelnames)
            self._children[key] = self._children.get(key, 0.0) + float(amount)
            return self._children[key]

    def value(self, **labels: Any) -> float:
        key = tuple(str(labels[n]) for n in self.labelnames)
        with self._lock:
            return float(self._children.get(key, 0.0))

    def total(self) -> float:
        with self._lock:
            return float(sum(self._children.values()))

    def _render_children(self, out: list[str]) -> None:
        for key, value in self._sample_pairs():
            out.append(f"{self.name}{_fmt_labels(self._label_pairs(key))} "
                       f"{_fmt_float(value)}")
        # An unlabelled counter with no children still needs the family to exist,
        # so a query for it returns 0 rather than "no data" — the difference
        # between "no traffic yet" and "this metric was never wired up".
        #
        # Only when the family is *unlabelled*, though. Emitting `name 0` for a
        # counter declared with labels invents a series that no code can ever
        # write to: the label set is part of the metric's identity, and a sample
        # without it is a different, permanently-zero series that shows up in
        # `sum()` and in every legend, one per labeled metric in the process.
        if not self._children and not self.labelnames:
            out.append(f"{self.name} 0")


class Gauge(_Family):
    """A value that goes up and down. `set` is the honest way to write one that
    mirrors external state; `inc`/`dec` are for things this process owns."""

    type_name = "gauge"

    def _new_child(self) -> float:
        return 0.0

    def set(self, value: float, **labels: Any) -> float:
        with self._lock:
            key = tuple(str(labels[n]) for n in self.labelnames)
            self._children[key] = float(value)
            return self._children[key]

    def inc(self, amount: float = 1.0, **labels: Any) -> float:
        with self._lock:
            key = tuple(str(labels[n]) for n in self.labelnames)
            self._children[key] = self._children.get(key, 0.0) + float(amount)
            return self._children[key]

    def dec(self, amount: float = 1.0, **labels: Any) -> float:
        return self.inc(-amount, **labels)

    def value(self, **labels: Any) -> float:
        key = tuple(str(labels[n]) for n in self.labelnames)
        with self._lock:
            return float(self._children.get(key, 0.0))

    def _render_children(self, out: list[str]) -> None:
        for key, value in self._sample_pairs():
            out.append(f"{self.name}{_fmt_labels(self._label_pairs(key))} "
                       f"{_fmt_float(value)}")


class Histogram(_Family):
    """Cumulative buckets + `_sum` + `_count`.

    Buckets are cumulative *because that is the format* — `le="0.1"` means "at
    most 100 ms", not "between 50 and 100 ms". Storing them any other way and
    emitting the running total at render time is the same arithmetic with one
    more place to get it wrong, so they are kept cumulative in memory.
    """

    type_name = "histogram"

    def __init__(self, name: str, help_text: str,
                 labelnames: Sequence[str] = (),
                 buckets: Sequence[float] = DEFAULT_BUCKETS) -> None:
        super().__init__(name, help_text, labelnames)
        self.buckets = tuple(sorted(float(b) for b in buckets))
        if not self.buckets:
            raise ValueError(f"{name}: a histogram needs at least one bucket")

    def _new_child(self) -> dict[str, Any]:
        return {"buckets": [0] * len(self.buckets), "sum": 0.0, "count": 0}

    def observe(self, value: float, **labels: Any) -> None:
        value = float(value)
        with self._lock:
            key = tuple(str(labels[n]) for n in self.labelnames)
            child = self._children.setdefault(key, self._new_child())
            child["sum"] += value
            child["count"] += 1
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    child["buckets"][i] += 1

    def count(self, **labels: Any) -> int:
        key = tuple(str(labels[n]) for n in self.labelnames)
        with self._lock:
            child = self._children.get(key)
            return int(child["count"]) if child else 0

    def sum(self, **labels: Any) -> float:
        key = tuple(str(labels[n]) for n in self.labelnames)
        with self._lock:
            child = self._children.get(key)
            return float(child["sum"]) if child else 0.0

    def _render_children(self, out: list[str]) -> None:
        for key, child in self._sample_pairs():
            base = self._label_pairs(key)
            for bound, seen in zip(self.buckets, child["buckets"]):
                out.append(
                    f'{self.name}_bucket'
                    f'{_fmt_labels(base + [("le", _fmt_float(bound))])} {seen}'
                )
            # +Inf always equals _count, and must be present: `histogram_quantile`
            # in PromQL reads the +Inf bucket as the denominator and returns
            # nonsense (or nothing) without it.
            out.append(f'{self.name}_bucket'
                       f'{_fmt_labels(base + [("le", "+Inf")])} {child["count"]}')
            out.append(f'{self.name}_sum{_fmt_labels(base)} '
                       f'{_fmt_float(child["sum"])}')
            out.append(f'{self.name}_count{_fmt_labels(base)} {child["count"]}')


class Registry:
    """Holds every family and renders them together, plus gathered gauges."""

    def __init__(self) -> None:
        self._families: list[_Family] = []
        # Keyed, not a list — see `add_gatherer` for why the key is load-bearing.
        self._gatherers: dict[str, Callable[[], Iterable[str]]] = {}
        self._lock = threading.Lock()

    def register(self, family: _Family) -> _Family:
        with self._lock:
            if any(f.name == family.name for f in self._families):
                raise ValueError(f"metric already registered: {family.name}")
            self._families.append(family)
        return family

    def counter(self, name: str, help_text: str,
                labelnames: Sequence[str] = ()) -> Counter:
        return self.register(Counter(name, help_text, labelnames))  # type: ignore[return-value]

    def gauge(self, name: str, help_text: str,
              labelnames: Sequence[str] = ()) -> Gauge:
        return self.register(Gauge(name, help_text, labelnames))  # type: ignore[return-value]

    def histogram(self, name: str, help_text: str,
                  labelnames: Sequence[str] = (),
                  buckets: Sequence[float] = DEFAULT_BUCKETS) -> Histogram:
        return self.register(  # type: ignore[return-value]
            Histogram(name, help_text, labelnames, buckets))

    def add_gatherer(self, fn: Callable[[], Iterable[str]],
                     key: Optional[str] = None) -> str:
        """Register a callable that returns already-formatted exposition lines.

        Used for gauges whose value lives somewhere this process does not own the
        counter for — the registry's own state, the task store's queue depth.
        `fn` must not raise: a scrape that 500s takes out the whole dashboard,
        including the panels that would have shown you why.

        **Keyed, and re-registering the same key replaces rather than appends.**
        This is not defensive padding. `create_app()` is called once to build the
        module-level `app` and again inside `create_served_app()`, so a blind
        append leaves two gatherers emitting `toolmarket_resources` — and a scrape
        containing one duplicated sample is rejected *as a whole* by Prometheus.
        The endpoint goes dark entirely, not partially, and the panels that would
        have explained why are the ones that go dark with it. The default key is
        the function's qualified name, which is the same for every call to the
        same factory, so the common case is idempotent without the caller having
        to think about it.
        """
        key = key or (f"{getattr(fn, '__module__', '?')}."
                      f"{getattr(fn, '__qualname__', None) or id(fn)}")
        with self._lock:
            self._gatherers[key] = fn
        return key

    def remove_gatherer(self, key: str) -> None:
        with self._lock:
            self._gatherers.pop(key, None)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            families = list(self._families)
            gatherers = list(self._gatherers.values())
        for family in families:
            lines.extend(family.collect())
        for fn in gatherers:
            try:
                lines.extend(fn())
            except Exception:  # noqa: BLE001
                # A broken gatherer must not cost you the rest of the scrape.
                # It is visible instead: the family stops appearing, which is a
                # signal, and the API keeps serving.
                continue
        return _dedupe_samples(lines)

    def reset(self) -> None:
        with self._lock:
            families = list(self._families)
        for family in families:
            family.reset()

    def names(self) -> list[str]:
        with self._lock:
            return [f.name for f in self._families]


# -- the substrate's own metrics -------------------------------------------
# A module-level registry, because there is one process and one set of counters
# per process. Tests call `METRICS.reset()`; nothing else should need to.
METRICS = Registry()

BUILD_INFO = METRICS.gauge(
    "toolmarket_build_info",
    "Build metadata. Always 1; the information is in the labels.",
    ("version", "store_backend", "cache_backend"),
)
HTTP_REQUESTS = METRICS.counter(
    "toolmarket_http_requests_total",
    "HTTP requests handled, by matched route template and response status.",
    ("method", "route", "status"),
)
HTTP_LATENCY = METRICS.histogram(
    "toolmarket_http_request_duration_seconds",
    "HTTP request latency in seconds, by matched route template.",
    ("method", "route"),
)
HTTP_IN_FLIGHT = METRICS.gauge(
    "toolmarket_http_in_flight_requests",
    "HTTP requests currently being served by this process.",
)
RATE_LIMITED = METRICS.counter(
    "toolmarket_rate_limited_total",
    "Requests refused by the rate limiter.",
    ("route",),
)
CACHE_LOOKUPS = METRICS.counter(
    "toolmarket_cache_lookups_total",
    "Cache lookups, by outcome.",
    ("result",),
)
EVOLUTIONS = METRICS.counter(
    "toolmarket_evolutions_total",
    "Evolution runs, by outcome.",
    ("outcome",),
)
# `toolmarket_async_tasks` is deliberately *not* a gauge here. Task counts live
# in the task store, which may be shared across processes, so the API gathers
# them at scrape time (see `_gather_dynamic`). A registered gauge of the same
# name would emit a second `# HELP` line for the family, and a duplicate HELP is
# a parse error that costs the whole scrape — the failure mode this file exists
# to avoid.
STORE_OPERATIONS = METRICS.counter(
    "toolmarket_store_operations_total",
    "Persistence operations, by backend operation.",
    ("operation",),
)
# The gRPC surface is a second front door onto the same registry, so its calls
# are counted in the same registry too. The label set mirrors HTTP_REQUESTS
# (`route` is the RPC name, `status` the gRPC code) deliberately: an operator
# reading a dashboard should not need two mental models for one substrate.
# This counter is per-process, so it is only visible to the exposition of the
# process that served the call — run the gRPC server and the API in one process
# (`make grpc-dev`) if you want both protocols on one /metrics.
GRPC_REQUESTS = METRICS.counter(
    "toolmarket_grpc_requests_total",
    "gRPC calls handled, by RPC name and canonical gRPC status code.",
    ("rpc", "status"),
)


def render() -> str:
    return METRICS.render()


def _dedupe_samples(lines: Iterable[str]) -> str:
    """Join exposition lines, keeping the first of any sample rendered twice.

    A Prometheus scrape is accepted or rejected as a whole: one duplicated sample
    and the scraped target is marked down with every line in the response
    discarded. Since gatherers are supplied by callers and a scrape is
    all-or-nothing, the only behaviour that leaves the endpoint saying something
    true is to drop the repeat and keep the first occurrence.

    The sample key is everything before the final space, because a value never
    contains a space but a label value may. Comment lines are keyed on their
    whole text: a repeated `# HELP` for one metric family is itself a parse
    error, so those have to be deduped too, not just the samples.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        key = line if line.startswith("#") else line.rsplit(" ", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out) + "\n"
