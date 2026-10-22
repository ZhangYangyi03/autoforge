"""The limiter and the instrumentation, exercised over real ASGI.

These run through `TestClient` against a throwaway app, which is the only way to
test the interesting properties: which *label* a request lands under, whether a
refused request is excluded from the latency histogram, and whether the in-flight
gauge comes back down when a handler raises. All three are invisible to a unit
test of `_check`, and all three are the ones that produce a dashboard that is
confidently wrong.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from toolmarket import metrics as M
from toolmarket.cache import reset_cache_singleton
from toolmarket.ratelimit import (
    DEFAULT_EXEMPT,
    InstrumentedApp,
    client_key,
    install,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # The limiter counts through the process-wide cache, and the metrics are
    # process-wide singletons. Both have to be dropped between cases or the tests
    # inherit each other's request counts — which is exactly the "test order
    # changes the result" class of flake.
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RATE_LIMIT", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW", raising=False)
    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    reset_cache_singleton()
    M.METRICS.reset()
    yield
    reset_cache_singleton()


def _inner() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        return {"ready": True}

    @app.get("/metrics")
    def metrics():
        return {"fake": "exposition"}

    @app.get("/things/{thing_id}")
    def thing(thing_id: str):
        return {"id": thing_id}

    @app.get("/boom")
    def boom():
        raise RuntimeError("handler exploded")

    return app


def _client(*, limit: int = 2, window: float = 60.0, enabled: bool = True,
            exempt=DEFAULT_EXEMPT, raise_server_exceptions: bool = False):
    app = InstrumentedApp(_inner(), limit=limit, window=window, enabled=enabled,
                          exempt=exempt)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _series(text: str, prefix: str) -> list[str]:
    return [l for l in text.splitlines() if l.startswith(prefix)]


# ---------------------------------------------------------------- limiting
def test_allows_up_to_the_limit_then_429s_with_retry_after():
    c = _client(limit=2, exempt=())
    assert c.get("/things/a").status_code == 200
    assert c.get("/things/a").status_code == 200
    blocked = c.get("/things/a")
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "60"
    body = blocked.json()
    assert body["limit"] == 2 and body["window_seconds"] == 60.0


def test_budget_is_per_route_template_not_per_caller_only():
    """A burst of id lookups must not spend the budget for a different endpoint.

    The bucket key is (caller, route template), so exhausting `/things/{id}`
    leaves `/boom` untouched. Without the route component, one hot endpoint would
    rate-limit the whole API for that client — the failure that looks like a load
    problem and is actually a keying bug.
    """
    c = _client(limit=1, exempt=())
    assert c.get("/things/a").status_code == 200
    assert c.get("/things/a").status_code == 429
    assert c.get("/boom").status_code == 500  # its own budget, own failure


def test_path_parameters_share_one_bucket():
    # `/things/a` and `/things/b` are the same route: 200 then 429, rather than
    # two separate budgets keyed on the id.
    c = _client(limit=1, exempt=())
    assert c.get("/things/a").status_code == 200
    assert c.get("/things/b").status_code == 429


def test_exempt_paths_are_never_limited():
    c = _client(limit=1)
    for path in ("/health", "/ready", "/metrics"):
        assert path in DEFAULT_EXEMPT, f"{path} must be exempt or scraper and probe get 429s"
        for _ in range(4):
            assert c.get(path).status_code == 200


def test_disabled_limiter_allows_everything():
    c = _client(limit=1, enabled=False)
    for _ in range(5):
        assert c.get("/things/a").status_code == 200


def test_limit_zero_disables_rather_than_blocks_everything():
    """`RATE_LIMIT=0` must mean "off", not "refuse every request".

    The distinction is worth a test because the obvious implementation —
    `used <= limit` — turns a documented way of disabling the limiter into a
    total outage.
    """
    c = _client(limit=0)
    assert c.get("/things/a").status_code == 200


def test_rate_limited_total_is_labelled_with_the_bounded_bucket():
    """The refusal label is the limiter's own bucket, not the route template.

    At refusal time the router has not run, so the template does not exist yet —
    and a metric labelled `<other>` for every refusal would be a panel that cannot
    tell you which endpoint is being hammered. The label is therefore the same
    bounded prefix the limit is keyed on.
    """
    c = _client(limit=1, exempt=())
    c.get("/things/a")
    c.get("/things/a")
    text = M.render()
    assert any(l == 'toolmarket_rate_limited_total{route="/things"} 1'
               for l in _series(text, "toolmarket_rate_limited_total"))


def test_client_key_prefers_forwarded_for_only_when_trusted(monkeypatch):
    scope = {"client": ("10.0.0.1", 5555), "path": "/", "method": "GET",
             "headers": [(b"x-forwarded-for", b"203.0.113.9, 10.0.0.1")]}
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    # Untrusted: the socket peer is the identity. Trusting XFF by default would
    # let any client pick its own rate-limit bucket by setting a header.
    assert client_key(scope) == "10.0.0.1"
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert client_key(scope) == "203.0.113.9"


# ------------------------------------------------------------ instrumentation
def test_requests_are_labelled_by_route_template():
    c = _client(limit=100)
    c.get("/things/abc")
    c.get("/things/xyz")
    c.get("/health")
    text = M.render()
    assert 'toolmarket_http_requests_total{method="GET",route="/things/{thing_id}",status="200"} 2' in _series(
        text, "toolmarket_http_requests_total")
    # The raw path must not appear as a label value: an unauthenticated endpoint
    # taking an id is a cardinality bomb, and one series per id is how a working
    # /metrics becomes an OOM in Prometheus instead.
    assert 'route="/things/abc"' not in text


def test_latency_is_recorded_with_count_matching_requests():
    c = _client(limit=100)
    for _ in range(3):
        c.get("/health")
    text = M.render()
    assert 'toolmarket_http_request_duration_seconds_count{method="GET",route="/health"} 3' in _series(
        text, "toolmarket_http_request_duration_seconds_count")


def test_a_raising_handler_records_500_and_decrements_in_flight():
    """The `finally` in `InstrumentedApp.__call__` is the whole point.

    A gauge that only decrements on the success path climbs forever after the
    first exception, and the resulting panel says the process is drowning in
    in-flight work while it is idle. The counter must also record 500 rather than
    the "500" placeholder being left as the default — which is why the default is
    the pessimistic one.
    """
    c = _client(limit=100)
    assert c.get("/boom").status_code == 500
    text = M.render()
    assert _series(text, "toolmarket_http_in_flight_requests") == [
        "toolmarket_http_in_flight_requests 0"
    ]
    assert 'toolmarket_http_requests_total{method="GET",route="/boom",status="500"} 1' in _series(
        text, "toolmarket_http_requests_total")


def test_refused_request_is_excluded_from_latency_but_counted_as_a_refusal():
    """A 429 short-circuits before the timer starts.

    Counting a refusal as a 200 with a 0.1 ms latency would make the latency panel
    *improve* as the service degrades — the most misleading correlation available.
    It is also not recorded in `toolmarket_http_requests_total`, because that
    metric is labelled by route template and at refusal time the router has not
    run; refusals have their own counter, labelled with the bounded bucket, so
    nothing is lost by keeping the two apart.
    """
    c = _client(limit=1, exempt=())
    c.get("/things/a")
    assert c.get("/things/a").status_code == 429
    text = M.render()
    requests = _series(text, "toolmarket_http_requests_total")
    assert requests == [
        'toolmarket_http_requests_total{method="GET",route="/things/{thing_id}",status="200"} 1'
    ]
    assert _series(text, "toolmarket_http_request_duration_seconds_count") == [
        'toolmarket_http_request_duration_seconds_count{method="GET",route="/things/{thing_id}"} 1'
    ]
    assert 'toolmarket_rate_limited_total{route="/things"} 1' in _series(
        text, "toolmarket_rate_limited_total")


def test_install_reads_the_limit_from_the_environment(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "1")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "30")
    c = TestClient(install(_inner()), raise_server_exceptions=False)
    assert c.get("/things/a").status_code == 200
    blocked = c.get("/things/a")
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "30"


def test_install_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    monkeypatch.setenv("RATE_LIMIT", "1")
    c = TestClient(install(_inner()), raise_server_exceptions=False)
    for _ in range(4):
        assert c.get("/things/a").status_code == 200


def test_non_http_scopes_pass_through_untouched():
    """Lifespan and websocket scopes must not be treated as requests.

    A limiter that tries to rate-limit the lifespan event either crashes at
    startup or blocks it, and the symptom is a container that never becomes
    ready for a reason no request log shows.
    """
    seen = []

    async def passthrough(scope, receive, send):
        seen.append(scope["type"])

    app = InstrumentedApp(passthrough, limit=1, window=60)
    import asyncio
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        app({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
