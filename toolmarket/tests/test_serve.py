"""The served surface: routes, probes, the cache's edge cases, the async API.

Everything here goes through `create_served_app`, so the rate limiter and the
metrics middleware are in the path — a test that mounted the bare router would
prove the routes work and say nothing about the process that actually runs.

The interesting cases are the ones where a cache, a probe and a queue each have a
way of being subtly wrong: a cached 404, a readiness check that touches nothing,
a 202 whose task id cannot be polled.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from toolmarket import metrics as M
from toolmarket.api.main import create_served_app
from toolmarket.cache import reset_cache_singleton
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


def _sample_spec(name: str = "slugify"):
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    def slugify(text: str = "") -> str:
        return "-".join(text.lower().split())

    return ToolSpec(
        name=name,
        description="Slugify a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=slugify,
        code="def slugify(text=''):\n    return '-'.join(text.lower().split())\n",
        source="human",
        probes=[TriggerProbe(query="slugify this", expect="call")],
        invariances=["text"],
        effect_signature="pure",
    )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("TOOLMARKET_STORE", raising=False)
    monkeypatch.setenv("CACHE_TTL", "30")
    monkeypatch.delenv("TASK_QUEUE", raising=False)
    monkeypatch.setenv("RATE_LIMIT", "100000")  # never interfere with the assertions
    reset_cache_singleton()
    M.METRICS.reset()
    yield
    reset_cache_singleton()


def _client(with_resource: bool = True):
    reg = ResourceRegistry(ResourceStore(":memory:"))
    rid = None
    if with_resource:
        rid = reg.register(_sample_spec()).id
    return TestClient(create_served_app(reg), raise_server_exceptions=False), reg, rid


def _queue(client):
    """The queue the served app built.

    `client.app` is the `InstrumentedApp` wrapper — the limiter and the metrics
    middleware — and FastAPI's `state` lives on the app inside it. Reaching
    through is deliberate here rather than exposing the queue through the wrapper:
    the wrapper's job is to be transparent, and a test that needed it to forward
    attributes would be a test arguing for a worse design.
    """
    return client.app.app.state.queue


# -------------------------------------------------------------------- meta
def test_index_lists_the_endpoints():
    c, _reg, _ = _client()
    body = c.get("/").json()
    assert body["name"] == "toolmarket"
    assert body["docs"] == "/docs"
    assert any("/resources/" in e for e in body["endpoints"])


def test_health_is_process_local_and_reports_the_chain():
    c, reg, _ = _client()
    body = c.get("/health").json()
    assert body["ok"] is True
    assert body["chain_ok"] is True
    assert body["resources"] == 1 and body["events"] >= 1


def test_ready_reports_each_dependency_separately():
    """A readiness answer of `false` with no reason is the least useful thing an
    orchestrator can be told, so every component is named."""
    c, _reg, _ = _client()
    r = c.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert set(body) >= {"ready", "store", "cache", "queue"}
    assert body["store"]["ok"] is True
    assert body["store"]["backend"] == "sqlite"
    assert body["cache"]["backend"] in {"memory", "redis", "none"}


def test_ready_is_503_when_the_store_is_unreachable(monkeypatch):
    c, reg, _ = _client()

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(reg.store, "ping", boom)
    r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["store"]["ok"] is False
    # `/health` must stay green: it touches nothing by design, and a container
    # runtime that restarts the process because a *database* is down turns a
    # recoverable blip into an outage.
    assert c.get("/health").status_code == 200


def test_ready_never_names_a_backend_the_store_did_not_declare(monkeypatch):
    """A store with no `backend` must read as `unknown`, not as some real one.

    This is the regression for a reporting bug that no amount of green tests
    could catch: `/ready` read the backend with `getattr(store, "backend",
    "sqlite")`, and `PostgresStore` — alone among the backends, since the three
    cache classes all declared theirs — never had the attribute. So a stack
    running on Postgres reported `sqlite`, and the one call whose entire job is
    to tell you which store you connected to was the one that could not.

    `test_ready_reports_each_dependency_separately` asserted `== "sqlite"` and
    passed the whole time, because an in-memory store returning `"sqlite"` is
    correct *and* is what the fallback returned. A test cannot catch a default
    by asserting the value the default produces; it has to take the attribute
    away and check what the endpoint says then.
    """
    c, reg, _ = _client()

    class Nameless:
        """A store that answers `ping` and declares nothing about itself."""

        def ping(self) -> bool:
            return True

    monkeypatch.setattr(reg, "store", Nameless())
    body = c.get("/ready").json()
    assert body["store"]["backend"] == "unknown"
    assert body["store"]["backend"] != "sqlite"


def test_every_shipped_store_declares_its_backend():
    """The attribute is part of the store interface, so assert it on the classes.

    Checked at the class rather than through `/ready` because the bug was a
    *missing declaration*, and the endpoint now has an `"unknown"` fallback that
    would hide the next one just as `"sqlite"` hid this one.
    """
    from toolmarket.store import ResourceStore
    from toolmarket.store_pg import PostgresStore

    assert ResourceStore.backend == "sqlite"
    assert PostgresStore.backend == "postgres"


def test_stats_reports_the_backend_and_path():
    c, _reg, _ = _client()
    body = c.get("/stats").json()
    # The store's own numbers are nested under `store` rather than flattened into
    # the registry's, so a field named `resources` cannot mean two different
    # things depending on which layer answered.
    assert body["store"]["backend"] == "sqlite"
    assert body["store"]["resources"] == 1
    assert body["by_state"] == {"draft": 1}


# ----------------------------------------------------------------- metrics
def test_metrics_endpoint_content_type_and_samples():
    c, reg, rid = _client()
    c.get(f"/resources/{rid}")
    r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = r.text
    assert "toolmarket_build_info" in text
    assert 'store_backend="sqlite"' in text
    assert "toolmarket_resources{" in text
    assert "toolmarket_dependency_up{component=\"store\"} 1" in text
    assert "toolmarket_dependency_up{component=\"cache\"} 1" in text
    assert "toolmarket_chain_intact 1" in text


def test_metrics_scrape_is_free_of_duplicate_samples():
    """The endpoint the alerting depends on must be parseable by Prometheus.

    Two `create_app()` calls happen in a running process (the module-level `app`
    and `served_app`), and if both register a gatherer unconditionally the scrape
    carries every gathered sample twice. Prometheus rejects the whole response for
    that, so this asserts the property directly rather than trusting the keying.
    """
    c, _reg, _ = _client()
    lines = [l for l in c.get("/metrics").text.splitlines()
             if l and not l.startswith("#")]
    keys = [l.rsplit(" ", 1)[0] for l in lines]
    assert len(keys) == len(set(keys)), "duplicate samples in the scrape"


def test_requests_are_counted_in_the_served_app():
    c, _reg, _ = _client()
    c.get("/health")
    text = c.get("/metrics").text
    assert 'toolmarket_http_requests_total{method="GET",route="/health",status="200"}' in text


def test_metrics_is_not_rate_limited():
    # A scraper hitting 429s loses data silently: Prometheus records the scrape as
    # failed and the panel goes stale rather than showing an error.
    c, _reg, _ = _client()
    for _ in range(5):
        assert c.get("/metrics").status_code == 200


# ------------------------------------------------------------------- cache
def test_resource_read_is_cached_then_served_from_cache():
    c, reg, rid = _client()
    first = c.get(f"/resources/{rid}")
    assert first.status_code == 200
    second = c.get(f"/resources/{rid}")
    assert second.status_code == 200
    assert second.json() == first.json()
    lookups = [l for l in c.get("/metrics").text.splitlines()
               if l.startswith("toolmarket_cache_lookups_total")]
    assert any('result="miss"' in l for l in lookups)
    assert any('result="hit"' in l for l in lookups)


def test_fresh_true_bypasses_a_warm_cache():
    """The escape hatch that keeps a cache from being a correctness bug.

    The cache entry is poisoned directly rather than by writing a stale record,
    because a write *through the registry* invalidates the entry (see the test
    below) — and the bug this flag exists for is the read that a cache can serve
    on its own. Poisoning states the condition exactly: there is a stale value in
    the cache, and `?fresh=true` must not return it.
    """
    c, _reg, rid = _client()
    c.get(f"/resources/{rid}")  # warm the cache

    from toolmarket.cache import get_cache, resource_key

    get_cache().set(resource_key(rid), {"id": rid, "description": "stale"}, 30)
    assert c.get(f"/resources/{rid}").json()["description"] == "stale"

    M.METRICS.reset()
    fresh = c.get(f"/resources/{rid}", params={"fresh": "true"}).json()
    assert fresh["description"] == "Slugify a string."
    # `fresh` must not consult the cache at all — not merely return the store's
    # answer while still counting a lookup.
    assert not [l for l in c.get("/metrics").text.splitlines()
                if l.startswith("toolmarket_cache_lookups_total")]


def test_a_write_through_the_registry_invalidates_the_cached_view():
    """No stale read after a registry write — including from another process.

    `save` drops the cache entry rather than updating it, which is what makes
    this work for the Celery worker: the worker's `save` reaches the same Redis
    the API reads, so its commit invalidates the API's entry even though the two
    processes share no memory. A test that only ever mutated in-process would pass
    whether or not that held.
    """
    c, reg, rid = _client()
    first = c.get(f"/resources/{rid}").json()
    assert first["description"] == "Slugify a string."

    rec = reg.get(rid)
    assert rec is not None
    rec.description = "mutated"
    reg.save(rec)

    # The next read must see the new value, not the TTL-old copy.
    assert c.get(f"/resources/{rid}").json()["description"] == "mutated"


def test_a_404_is_not_cached():
    """Caching a miss is how a registry contradicts its own `register` response.

    `unknown resource` changes the moment somebody registers the tool. A negative
    cache would keep answering 404 for a resource that now exists, and the person
    who just created it would be the one to see it.
    """
    c, reg, _ = _client(with_resource=False)
    assert c.get("/resources/tool:nope").status_code == 404
    reg.register(_sample_spec("nope"))
    assert c.get("/resources/tool:nope").status_code == 200


def test_cache_can_be_disabled_by_ttl_zero(monkeypatch):
    monkeypatch.setenv("CACHE_TTL", "0")
    c, _reg, rid = _client()
    c.get(f"/resources/{rid}")
    c.get(f"/resources/{rid}")
    assert not [l for l in c.get("/metrics").text.splitlines()
                if l.startswith("toolmarket_cache_lookups_total")]


# ------------------------------------------------------------------- async
def test_async_evolve_on_an_unknown_resource_is_404():
    c, _reg, _ = _client()
    r = c.post("/resources/tool:ghost/evolve/async", json={"goal": "tighten"})
    assert r.status_code == 404


def test_async_evolve_returns_202_with_a_pollable_task_id():
    """202, and a task id that resolves. Both halves matter.

    200 would tell a well-behaved client the resource had already evolved. A task
    id that 404s on the very next request is worse: the client cannot even learn
    that it failed.
    """
    c, _reg, rid = _client()
    r = c.post(f"/resources/{rid}/evolve/async", json={"goal": "tighten"})
    assert r.status_code == 202
    body = r.json()
    assert body["state"] == "pending"
    assert body["queue"] == "inline"
    assert body["status_url"] == f"/tasks/{body['task_id']}"

    status = c.get(body["status_url"])
    assert status.status_code == 200
    payload = status.json()
    assert payload["task_id"] == body["task_id"]
    assert payload["resource_id"] == rid


def test_async_task_reaches_a_terminal_state():
    c, _reg, rid = _client()
    task_id = c.post(f"/resources/{rid}/evolve/async",
                     json={"goal": "add a length guard"}).json()["task_id"]
    state = _queue(c).join(task_id, timeout=120)
    assert state is not None
    # Terminal, whichever one it is: the queue must never leave a task that a
    # client polls forever. Which terminal state it reaches is the gate's call and
    # is tested by test_protocol.py — what this asserts is that it stops.
    assert state.state in ("success", "failure")
    payload = c.get(f"/tasks/{task_id}").json()
    assert payload["terminal"] is True
    assert payload["duration_seconds"] is not None


def test_unknown_task_is_404():
    c, _reg, _ = _client()
    assert c.get("/tasks/does-not-exist").status_code == 404


def test_async_tasks_appear_in_metrics():
    c, _reg, rid = _client()
    task_id = c.post(f"/resources/{rid}/evolve/async",
                     json={"goal": "g"}).json()["task_id"]
    _queue(c).join(task_id, timeout=120)
    text = c.get("/metrics").text
    # The gauge is gathered from the task store at scrape time, so a finished task
    # must show up under a terminal state rather than under `pending` forever.
    assert "toolmarket_async_tasks{" in text
    assert "toolmarket_evolutions_total{" in text
    assert 'toolmarket_async_tasks{state="success"} 1' in text or \
           'toolmarket_async_tasks{state="failure"} 1' in text


# --------------------------------------------------------- sync still works
def test_synchronous_evolve_is_still_available():
    """The async route was added *before* the sync one in registration order.

    Both use `{resource_id:path}`, which is greedy, so a route ordering mistake
    would send `/resources/{id}/evolve` into the async handler (or 404 it). This
    asserts the sync path still resolves after that change.
    """
    c, _reg, rid = _client()
    r = c.post(f"/resources/{rid}/evolve", json={"goal": "tighten"})
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        assert "committed" in r.json()
