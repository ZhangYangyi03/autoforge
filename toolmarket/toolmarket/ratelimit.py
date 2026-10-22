"""Rate limiting and HTTP instrumentation, as pure ASGI middleware.

Pure ASGI rather than `BaseHTTPMiddleware` for one reason that matters: the
Starlette subclass wraps every response in a streaming task and re-raises
exceptions from a different task than the one that produced them, which is a
long-standing source of "the traceback points at the wrong frame" and of
background-task lifetimes that end early. This middleware only needs the
request line, the response status and the clock, so it takes the raw interface
and leaves the response body untouched.

Two behaviours are deliberate and load-bearing:

  * **Unmatched paths are not limited by their own name.** Limiting on the raw
    path lets an attacker mint unbounded counter keys (`/a1`, `/a2`, ...),
    which is a memory-exhaustion vector wearing a protection's clothes. Only
    paths that match a real route get a bucket; everything else shares one.
  * **A limiter that cannot count refuses nothing.** If the cache raises — Redis
    down, timeout — the request is allowed and `toolmarket_rate_limited_total`
    does not move. The alternative is a limiter whose failure mode is a total
    outage, which trades a small abuse risk for a total one. This is a policy
    choice, so it is written down here rather than buried in an `except: pass`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional, Sequence

from toolmarket import metrics as _metrics
from toolmarket import tracing as _tracing
from toolmarket.cache import get_cache

DEFAULT_LIMIT = 120
DEFAULT_WINDOW = 60.0
#: Requests that are never limited. `/health` is here because a load balancer's
#: probe must not be the thing that trips the limit; `/metrics` because
#: Prometheus scrapes on a schedule and being throttled makes the monitoring
#: gap look like an outage.
DEFAULT_EXEMPT: tuple[str, ...] = (
    # Probes and scraping first, because a 429 on any of these is a *silent*
    # failure rather than a visible one: the orchestrator marks the container
    # unhealthy, or Prometheus records a failed scrape, and neither shows up as
    # an error anywhere a human is looking. /ready belongs here for the same
    # reason as /health — a readiness probe that can be rate-limited will
    # eventually rate-limit itself and take the service out of rotation.
    "/health",
    "/ready",
    "/metrics",
    "/",
    "/openapi.json",
    "/docs",
)


def client_key(scope: dict[str, Any]) -> str:
    """Identify the caller.

    Honours `X-Forwarded-For` on the **left-most** entry, which is the original
    client as recorded by the edge proxy. Using the right-most (or the socket
    peer behind a proxy) collapses every user behind one edge into a single
    bucket — the classic bug where the limiter fires for everybody at once.

    Only trust the header when the deployment says a proxy exists
    (`TRUST_PROXY=1`); otherwise a client can spoof it and get an unlimited
    supply of fresh buckets.
    """
    import os

    headers = {k.decode("latin-1").lower(): v.decode("latin-1")
               for k, v in scope.get("headers", [])}
    if os.environ.get("TRUST_PROXY", "").strip() in ("1", "true", "yes"):
        forwarded = headers.get("x-forwarded-for", "")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    return client[0] if client else "unknown"


def _route_label(scope: dict[str, Any]) -> str:
    """The matched route template, or `<other>`.

    Uses the route Starlette already resolved (`scope["route"].path`) so
    `/resources/a1` and `/resources/a2` are one series, not two. Without this,
    any unauthenticated endpoint that takes an id is a cardinality bomb aimed at
    the metrics backend.

    Only usable *after* the router has run. See `_limit_key` for what the limiter
    does instead, and why.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return "<other>"


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    """ASGI `scope["headers"]` -> a `str` dict.

    ASGI carries headers as a list of raw `bytes` pairs in arrival order,
    duplicated where a client or proxy sent a header twice — which is why the
    result is a plain dict and a repeated name keeps its first value rather than
    its last. For `traceparent` specifically that is the right choice: the
    outermost sender's context is the one the trace should continue, and the
    first header in the list is the outermost.
    """
    out: dict[str, str] = {}
    for name, value in scope.get("headers") or ():
        try:
            key = name.decode("latin-1")
            if key.lower() in out:
                continue
            out[key] = value.decode("latin-1")
        except (AttributeError, UnicodeDecodeError):
            # A malformed header must not fail the request. It is dropped.
            continue
    return out


def _limit_key(scope: dict[str, Any]) -> str:
    """A bounded stand-in for the route template, available *before* routing.

    The limiter runs ahead of the router — that is the entire point of it, since
    a request that will be refused should not reach a handler — and at that moment
    `scope["route"]` does not exist yet. The true template is therefore
    unknowable at the only moment it would be useful, and something else has to
    be used. There are three candidates and two of them are wrong:

      * the raw path: `/resources/a1` and `/resources/a2` become different
        buckets, so a client evades the limit by varying the id, and the number of
        keys is unbounded — the cardinality bomb, moved from Prometheus into the
        cache;
      * `"<other>"`: every endpoint of every client collapses into one budget, so
        a burst of cheap health-check-ish reads blocks writes;
      * the first path segment, which is what this returns: coarse (all of
        `/resources/*` shares a budget) and bounded (the number of route
        prefixes, not the number of ids).

    Coarse and honest beats precise and wrong. The *metrics* labels still use the
    real template (they are recorded after the response, when it is known), so a
    dashboard shows per-endpoint detail even though the limit is enforced per
    prefix.
    """
    path = scope.get("path") or "/"
    for segment in path.split("/"):
        if segment:
            return f"/{segment}"
    return "/"


async def _send_json(status: int, payload: dict[str, Any], send: Any,
                     headers: Optional[Sequence[tuple[bytes, bytes]]] = None) -> None:
    """Emit a complete JSON response through raw ASGI `send`.

    Written by hand rather than via `JSONResponse` so this middleware has no
    Starlette import at all — it is the one piece that must keep working even if
    the framework's internals move.
    """
    body = json.dumps(payload).encode()
    raw_headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if headers:
        raw_headers.extend(headers)
    await send({"type": "http.response.start", "status": status,
                "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


class InstrumentedApp:
    """Counts, times and (optionally) limits every HTTP request.

    Order inside `__call__` is the contract: limiter first (a refused request
    should not appear in the latency histogram as a fast success), then the
    in-flight gauge, then the inner app, then the observations in a `finally` so
    a raising handler still records a status of 500 and a duration.

    Metrics and tracing share this one wrapper rather than living in two
    middlewares, because they need the same three facts (the scope, the status
    the handler sent, and the elapsed time) and two wrappers would each compute
    them — with the two answers eventually disagreeing about which requests
    count. The class is named for instrumentation generally, not for the limiter.
    """

    def __init__(
        self,
        app: Any,
        *,
        limit: int = DEFAULT_LIMIT,
        window: float = DEFAULT_WINDOW,
        exempt: Sequence[str] = DEFAULT_EXEMPT,
        enabled: bool = True,
    ) -> None:
        self.app = app
        self.limit = max(0, int(limit))
        self.window = float(window)
        self.exempt = tuple(exempt)
        self.enabled = bool(enabled) and self.limit > 0

    # -- limiting ---------------------------------------------------------
    def _bucket_key(self, scope: dict[str, Any]) -> str:
        # Per (caller, route *prefix*): a burst of id lookups must not spend the
        # budget for registering a tool, and the prefix is the only bounded
        # identifier available before the router has run. See `_limit_key`.
        return f"rl:{client_key(scope)}:{_limit_key(scope)}"

    def _check(self, scope: dict[str, Any]) -> tuple[bool, int, int]:
        """Returns (allowed, used, retry_after_seconds)."""
        try:
            cache = get_cache()
            used = cache.incr(self._bucket_key(scope), ttl=self.window)
        except Exception:  # noqa: BLE001 - fail open, deliberately (see module doc)
            return True, 0, 0
        if used <= self.limit:
            return True, used, 0
        return False, used, int(self.window)

    # -- ASGI -------------------------------------------------------------
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        method = scope.get("method", "GET")

        if self.enabled and path not in self.exempt:
            allowed, _used, retry_after = self._check(scope)
            if not allowed:
                # Labelled with the limiter's own bucket key, not the route
                # template: at refusal time the template is not known yet, and a
                # metric labelled `<other>` for every refusal would be a panel
                # that cannot tell you which endpoint is being hammered.
                _metrics.RATE_LIMITED.inc(route=_limit_key(scope))
                await _send_json(
                    429,
                    {"detail": "rate limit exceeded",
                     "limit": self.limit, "window_seconds": self.window},
                    send,
                    headers=[(b"retry-after", str(retry_after or 1).encode())],
                )
                return

        _metrics.HTTP_IN_FLIGHT.inc()
        started = time.perf_counter()
        recorded: dict[str, Any] = {"status": "500"}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                recorded["status"] = str(message.get("status", 0))
            await send(message)

        # The SERVER span opens here, after the limiter. Both halves of that
        # order are deliberate: a refused request is not a server span (nothing
        # was served, and a span per 429 turns a rate-limit incident into a trace
        # flood), and this is the only place in the process holding an inbound
        # `traceparent`. A trace may be *continued* anywhere, but it may only
        # *begin* at an entry point — opening it deeper in the stack would make
        # every handler the root of its own trace and lose the caller's.
        #
        # A `with` across the `await` rather than manual start/end: the scope
        # restores the previous ambient span on the way out, so a failed request
        # cannot leave a finished span installed for the next one on the same
        # event loop to adopt as a parent.
        with _tracing.span(
            f"{method} {path}",
            kind=_tracing.SPAN_SERVER,
            parent=_tracing.extract(_headers(scope)),
            attributes={
                "http.method": method,
                "url.path": path,
                # No `http.route` here: at this point the router has not run, so
                # `_route_label` can only return `<other>`. The `finally` sets the
                # resolved template. Recording the placeholder would put a value
                # that is known to be wrong on the span and, worse, would look
                # like a resolved route to anything reading the attribute.
            },
        ) as span:
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                elapsed = time.perf_counter() - started
                route = _route_label(scope)
                _metrics.HTTP_IN_FLIGHT.dec()
                _metrics.HTTP_REQUESTS.inc(method=method, route=route,
                                           status=recorded["status"])
                # The body streaming has not finished when `app` returns for a
                # streaming response, so this is time-to-headers for those
                # routes. Noted rather than hidden: a route that streams for a
                # minute would otherwise look like a 2 ms route.
                _metrics.HTTP_LATENCY.observe(elapsed, method=method, route=route)

                # The route *template* is only known now that the router has run.
                # Labelling the span with the raw path instead would make
                # `/resources/tool:add` and `/resources/tool:mul` two unrelated
                # operations in the backend — the same cardinality mistake the
                # metrics route label exists to avoid.
                status = int(recorded["status"])
                span.set_attributes({
                    "http.route": route,
                    "http.status_code": status,
                    "http.duration_ms": round(elapsed * 1000, 3),
                })
                # A 5xx is the server's failure, so the span is an error. A 4xx is
                # not: the server did its job and the client asked for something
                # it could not have, and marking those red makes every trace list
                # look like an incident.
                span.status = (_tracing.STATUS_ERROR if status >= 500
                               else _tracing.STATUS_OK)


def install(app: Any, **kwargs: Any) -> Any:
    """Wrap an app, reading the limit from the environment when present.

    `RATE_LIMIT` / `RATE_LIMIT_WINDOW` are read here so the compose file can tune
    the limiter without a rebuild — the numbers belong to the deployment, not to
    the image.
    """
    import os

    limit = kwargs.pop("limit", None)
    window = kwargs.pop("window", None)
    if limit is None:
        limit = int(os.environ.get("RATE_LIMIT", DEFAULT_LIMIT) or DEFAULT_LIMIT)
    if window is None:
        window = float(os.environ.get("RATE_LIMIT_WINDOW", DEFAULT_WINDOW)
                       or DEFAULT_WINDOW)
    enabled = os.environ.get("RATE_LIMIT_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off")
    return InstrumentedApp(app, limit=limit, window=window, enabled=enabled, **kwargs)
