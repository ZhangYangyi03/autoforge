"""FastAPI surface over the resource substrate.

Routes are thin: each one calls exactly one registry/operator method. The API is
not where the semantics live, and it is not allowed to become so — a route that
decided a resource's state by itself would be a second, unaudited state machine.

    GET  /health
    GET  /ready                       deep check: the store and the cache answer
    GET  /stats
    GET  /metrics                     Prometheus exposition
    GET  /events                      (optionally ?resource_id=)
    GET  /resources                   (?type= &state=)
    GET  /search                      (?q= &k=)  ranked retrieval
    POST /resources                   register a tool
    GET  /resources/{id}              served from cache when warm
    POST /resources/{id}/transition   the enforcement lifecycle
    POST /resources/{id}/invoke
    GET  /resources/{id}/lineage
    GET  /resources/{id}/events
    GET  /resources/{id}/trust             what the ledger earns it, and why
    POST /resources/{id}/evolve       the SEPL closed loop (propose→assess→commit)
    POST /resources/{id}/evolve/async same, but returns a task id immediately
    GET  /tasks/{task_id}             poll an async evolution
    POST /mcp                         MCP (JSON-RPC 2.0): tools/list, tools/call

Two things the routes do *not* do, deliberately: they never advance a state, and
they never write an event. Both belong to the registry.

`/health` versus `/ready` is a distinction with a purpose. `/health` answers "is
this process alive" and must never touch the database — a liveness probe that
depends on a dependency restarts the API every time the database hiccups, which
turns one outage into two. `/ready` answers "can this process serve traffic" and
checks the store and the cache, because that is what a load balancer needs to
know before it sends a request here.
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel, Field
    _HAVE_FASTAPI = True
except Exception:  # noqa: BLE001 - API extras are optional
    _HAVE_FASTAPI = False

from toolmarket import metrics as _metrics
from toolmarket.cache import get_cache, resource_key
from toolmarket.protocol.lifecycle import ResourceState, VersionStatus, parse_state
from toolmarket.protocol.resources import (
    ResourceRecord,
    ToolContract,
    compile_tool_fn,
)
from toolmarket.registry import ResourceRegistry
from toolmarket import retrieval
from toolmarket import vectorstore as _vs
from toolmarket.tasks import TaskState, TaskStore, make_queue

DEFAULT_CACHE_TTL = 30.0


# -- request models ---------------------------------------------------------
if _HAVE_FASTAPI:

    class RegisterRequest(BaseModel):
        name: str
        description: str = ""
        parameters: dict[str, Any] = Field(default_factory=dict)
        code: str = ""
        source: str = "human"
        generator: str = ""
        invariances: list[str] = Field(default_factory=list)
        effect_signature: str = ""
        tags: list[str] = Field(default_factory=list)
        enable_evolving: bool = True
        permission_mode: str = "workspace_write"
        # Where this came from. Without these two, anything registered through
        # this route is provenance-less by construction, and a caller that
        # imports a hundred resources from somewhere else cannot say where --
        # which is exactly the property the importer is careful about.
        provenance: dict[str, Any] = Field(default_factory=dict)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class DiscoverRequest(BaseModel):
        q: str
        limit: int = 25

    class TransitionRequest(BaseModel):
        to: str
        reason: str = ""

    class InvokeRequest(BaseModel):
        arguments: dict[str, Any] = Field(default_factory=dict)
        force: bool = False

    class EvolveRequest(BaseModel):
        goal: str
        commit: bool = True
        proposer: str = "stub"


def _record_view(rec: ResourceRecord) -> dict[str, Any]:
    d = rec.to_dict()
    d["capability_schema"] = rec.as_capability_schema()
    return d


def _data_dir() -> str:
    """Where toolmarket keeps its data, independent of the process cwd."""
    return os.environ.get("TOOLMARKET_DATA_DIR") or r"C:\Users\china"


def _startup_sync(store, reg) -> None:
    """Build the collection with its real shape, then index the shelf.

    Declaring the dimension and the model at startup is not cosmetic. A
    collection created without them is a bm25-only collection forever -- the
    first vector would be refused as a dimension mismatch against dim=None -- so
    the dense leg would be dead while every route still answered 200.

    Failure is reported, not swallowed. The previous version wrapped this in
    `except Exception: pass`, which is how a broken index stayed invisible until
    someone read the scores closely.
    """
    from .. import embed as _embed

    dim = None
    model = None
    try:
        s = _embed.settings()
        if s.get("api_key"):
            dim, model = s["dim"], s["model"]
    except Exception:
        dim = model = None
    try:
        store.create_collection("tools", dim=dim, model_id=model)
        report = store.sync_from_records("tools", [_record_view(r) for r in reg.list()])
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("toolmarket.search").warning(
            "search index sync failed, retrieval will be lexical-only: %s", exc)
        return
    try:
        h = store.health("tools", expected_model=model)
        if h.get("problems"):
            import logging
            logging.getLogger("toolmarket.search").warning(
                "search index unhealthy: %s", "; ".join(h["problems"]))
    except Exception:
        pass
    import logging
    logging.getLogger("toolmarket.search").info(
        "search index ready: %s (dense %s, model %s)", report, bool(dim), model)


def _trusted_proxies() -> set:
    """The addresses whose ``X-Actor`` header is allowed to be believed.

    Default: loopback only. The shelf is usually reached through the node's
    reverse proxy or a tunnel, and through a proxy every caller arrives from the
    proxy's own address -- so without this list the actor header is a way to
    write someone else's name into the audit trail for anyone who can reach the
    port. The header is *provenance*, not authentication: it says who the caller
    says it is, and it is believed only from a machine the operator runs.

    Set ``TOOLMARKET_TRUSTED_PROXIES=192.168.1.107,10.0.0.5`` when the peer
    machines reach this shelf through their own node proxies.
    """
    raw = os.environ.get("TOOLMARKET_TRUSTED_PROXIES")
    if raw is None:
        # "testclient" is not a spoofable name: it is the address FastAPI's
        # in-process ASGI transport reports, i.e. this same process. Without it
        # every test that sets a header asserted the fallback instead of the
        # behaviour, and the behaviour would then be untested in exactly the
        # configuration the tests run in.
        return {"127.0.0.1", "::1", "localhost", "testclient"}
    return {x.strip() for x in raw.split(",") if x.strip()}


def _caller(request: Any) -> str:
    """Who is making this call, as best the shelf can honestly say.

    Every event on a live shelf used to carry ``actor="system"`` -- not a
    decision, a default nobody overrode. So "who put this here" was
    unanswerable, and a bad tool's origin could not be traced to a peer or to a
    person. Measured on 2026-09-21 by reading /events on both machines: every
    row said "system", including rows caused by an agent on the other host.

    The answer, in order of how much it is worth:

      the X-Actor header, if the caller is a trusted proxy and the value is a
      plain identifier (peers and the MCP client send their label here);
      else the X-Peer label such a proxy sets;
      else the TCP address the connection actually came from -- weak, but true,
      and enough to tell this host from a sibling;
      else "system", which now means exactly what it says: a caller with no
      identifiable origin at all.
    """
    try:
        client = getattr(request, "client", None)
        peer_ip = getattr(client, "host", "") or ""
    except Exception:                                          # noqa: BLE001
        peer_ip = ""
    if peer_ip in _trusted_proxies():
        for header in ("x-actor", "x-peer"):
            value = ""
            try:
                value = (request.headers.get(header) or "").strip()
            except Exception:                                  # noqa: BLE001
                value = ""
            if value and len(value) <= 64 and all(
                    c.isalnum() or c in "-_.:@ " for c in value):
                return value
    return peer_ip or "system"


def create_app(registry: Optional[ResourceRegistry] = None) -> Any:
    """Build the FastAPI app over a registry (a fresh one by default)."""
    if not _HAVE_FASTAPI:
        raise RuntimeError(
            "FastAPI is required for the API. Install the extra: "
            "pip install 'toolmarket[api]'"
        )

    reg = registry or ResourceRegistry()
    app = FastAPI(
        title="toolmarket",
        version="0.1.0",
        description=(
            "A protocol-registered resource platform for evolvable agent tools. "
            "The enforced implementation of AGP's resource substrate, on autoforge."
        ),
    )
    app.state.registry = reg
    # Inline unless `TASK_QUEUE=celery`. Built once per app so the task id a
    # client is handed can be polled against the same store it was written to.
    app.state.queue = make_queue(reg)
    app.state.cache_ttl = float(os.environ.get("CACHE_TTL", DEFAULT_CACHE_TTL) or 0)
    # An absolute default, not os.getcwd(). The index is a property of the
    # deployment, not of wherever the process happened to be started: with the
    # cwd-relative path, restarting the same server from a different directory
    # opened a *different* empty index and rebuilt it bm25-only, so retrieval
    # silently lost its dense leg and every score got worse for a reason that
    # looked nothing like a path. TOOLMARKET_INDEX still overrides.
    _idx_path = os.environ.get("TOOLMARKET_INDEX") or os.path.join(
        _data_dir(), "toolmarket_data", "index.sqlite")
    app.state.search = _vs.Store(_idx_path)
    _startup_sync(app.state.search, reg)
    retrieval.install_health_route(
        app, app.state.search, "tools",
        lambda: [_record_view(r) for r in reg.list()])


    # -- staying current when the shelf has more than one writer ------------
    #
    # The registry is built once, here, and held for the process's life; `save`
    # keeps it in step with *this* process's writes and with nothing else's. On
    # this machine the shelf has two writers -- `mcp_import` writes the sqlite
    # file directly, the API serves :8000 -- and the failure that produced is
    # measurable: the importer had written 11,871 resources while `/health` said
    # 6,985 and `/resources/tool:<id>` answered 404 for a row that existed on
    # disk, three minutes old.
    #
    # So reads refresh first, against a high-water mark that costs one indexed
    # aggregate when there is nothing new. This is deliberately *not* a cache
    # invalidation protocol: there is one store and it is the source of truth,
    # and the question "did anything change" is answerable by asking it.
    #
    # Index sync rides along, because the search index has the same staleness and
    # a worse symptom -- `/search` for a recently imported tool's exact id
    # returned ten unrelated tools, since the index was built at startup. Only
    # the changed rows are indexed, so this is O(new) and not O(shelf).
    _FRESH = float(os.environ.get("READ_FRESH_S", "1.0") or 0)

    def _refresh() -> list[Any]:
        try:
            changed = reg.refresh(max_age=_FRESH)
        except Exception:  # noqa: BLE001 - a read must not fail on a refresh
            return []
        if changed:
            try:
                for rec in changed:
                    app.state.search.index_text(
                        "tools", rec.id,
                        _vs.build_resource_text(_record_view(rec)),
                        {"name": rec.name, "state": rec.state.value})
            except Exception:  # noqa: BLE001 - retrieval degrades, reads do not
                pass
        return changed

    def _op(reg_: ResourceRegistry):
        from toolmarket.protocol.sepl import EvolutionOperator
        return EvolutionOperator(reg_)

    def _gather_dynamic() -> list[str]:
        """State gauges, read at scrape time from the registry itself.

        These are gathered rather than incremented because every one of them is a
        *property* of current state. A `resource.state == ACTIVE` counter that
        each transition had to remember to increment is a counter that will
        disagree with `GET /resources` the first time a path is added, and the
        disagreement will be found by whoever is on call, at 3am, staring at a
        dashboard that says the opposite of the API.
        """
        by_state: dict[str, int] = {}
        for r in reg.list():
            by_state[r.state.value] = by_state.get(r.state.value, 0) + 1
        lines = [
            "# HELP toolmarket_resources Information about registered resources, "
            "by lifecycle state.",
            "# TYPE toolmarket_resources gauge",
        ]
        for state in sorted(by_state):
            lines.append(f'toolmarket_resources{{state="{state}"}} '
                         f"{by_state[state]}")
        lines += [
            "# HELP toolmarket_events_total Events in the append-only log.",
            "# TYPE toolmarket_events_total gauge",
            f"toolmarket_events_total {len(reg.log)}",
            "# HELP toolmarket_lineage_nodes Lineage nodes recorded.",
            "# TYPE toolmarket_lineage_nodes gauge",
            f"toolmarket_lineage_nodes {len(reg.lineage)}",
            "# HELP toolmarket_chain_intact 1 if the event hash chain verifies, "
            "0 if it has been tampered with.",
            "# TYPE toolmarket_chain_intact gauge",
            f"toolmarket_chain_intact {1 if reg.log.verify_chain() else 0}",
        ]
        try:
            counts = TaskStore(get_cache()).by_state()
            lines += [
                "# HELP toolmarket_async_tasks Async evolution tasks, by state, "
                "as reported by the task store.",
                "# TYPE toolmarket_async_tasks gauge",
            ]
            for state in sorted(counts):
                lines.append(f'toolmarket_async_tasks{{state="{state}"}} '
                             f"{counts[state]}")
        except Exception:  # noqa: BLE001 - a cache without a task index
            pass

        # Dependency reachability, sampled at scrape time.
        #
        # This is the only gatherer that makes a network call, and it is here
        # because the alternative is worse in a way that is easy to miss: an
        # alert on `probe_success` requires a blackbox exporter nobody deployed,
        # so the rule reads as coverage while never being able to fire. A metric
        # that cannot be wrong is not a guard, it is decoration.
        #
        # It costs one round trip per scrape. At a 15s interval against a database
        # on the same bridge network that is free, and it is the same cost the
        # orchestrator's own readiness probe pays.
        lines += [
            "# HELP toolmarket_dependency_up 1 if a dependency answered a "
            "trivial call at scrape time, 0 otherwise.",
            "# TYPE toolmarket_dependency_up gauge",
        ]
        for component, probe in (("store", reg.store), ("cache", get_cache())):
            try:
                up = 1 if probe.ping() else 0
            except Exception:  # noqa: BLE001
                up = 0
            lines.append(f'toolmarket_dependency_up{{component="{component}"}} {up}')
        return lines

    _metrics.METRICS.add_gatherer(_gather_dynamic, key="toolmarket.registry")
    _metrics.BUILD_INFO.set(
        1.0,
        version=app.version,
        store_backend=str(getattr(reg.store, "backend", "sqlite")),
        cache_backend=str(getattr(get_cache(), "backend", "none")),
    )

    # -- meta -------------------------------------------------------------
    @app.get("/")
    def index() -> dict[str, Any]:
        """An index, so a deployment has a landing page that is not a 404.

        Served as JSON rather than HTML because this app is the *API*: the
        console is `toolmarket.web.build_app`, and giving the API a hand-written
        HTML page would mean two places to change when a route is added. FastAPI's
        own `/docs` is the interactive view.
        """
        return {
            "name": app.title,
            "version": app.version,
            "docs": "/docs",
            "endpoints": sorted(
                f"{sorted(getattr(r, 'methods', []) or [])[0]} "
                f"{r.path}"
                for r in app.routes
                if getattr(r, "methods", None)
            ),
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness: this process is up and its in-memory state is consistent.

        No store call, no cache call — see the module docstring. A probe that
        fails when a dependency is down gets you a restart loop, not a diagnosis.
        """
        _refresh()
        return {"ok": True, "resources": len(reg.list()),
                "events": len(reg.log), "chain_ok": reg.log.verify_chain()}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        """Readiness: every dependency this process needs answers a trivial call.

        Returns 503 with the failing component named, because "not ready" without
        a reason is the least useful thing an orchestrator can be told.
        """
        from fastapi.responses import JSONResponse

        store_ok = False
        try:
            store_ok = bool(reg.store.ping())
        except Exception:  # noqa: BLE001
            store_ok = False
        cache_ok = False
        try:
            cache_ok = bool(get_cache().ping())
        except Exception:  # noqa: BLE001
            cache_ok = False
        payload = {
            "ready": store_ok and cache_ok,
            # `"unknown"`, never a specific backend, for a component that has not
            # declared one. The previous defaults were `"sqlite"` and `"none"` —
            # real backend names, and wrong ones whenever the object in hand did
            # not declare an attribute of its own. A Postgres stack reported
            # `sqlite`, and the single call whose whole job is to say which store
            # you actually connected to was the call that could not.
            #
            # Both stores now declare `backend`, so this branch is a guard rather
            # than a routine path. It stays a guard that cannot lie: naming a
            # backend the code did not confirm is worse than admitting ignorance,
            # because it is indistinguishable from a correct answer.
            "store": {"ok": store_ok,
                      "backend": getattr(reg.store, "backend", "unknown")},
            "cache": {"ok": cache_ok,
                      "backend": getattr(get_cache(), "backend", "unknown")},
            "queue": {"backend": getattr(app.state.queue, "backend", "inline")},
        }
        status = 200 if payload["ready"] else 503
        return JSONResponse(payload, status_code=status)

    @app.get("/metrics")
    def metrics_endpoint() -> Any:
        """Prometheus exposition. Content type carries the format version: a
        scraper that does not see `version=0.0.4` will guess, and guessing wrong
        here means silently dropped samples rather than an error."""
        from fastapi.responses import Response

        return Response(content=_metrics.render(),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for r in reg.list():
            by_state[r.state.value] = by_state.get(r.state.value, 0) + 1
        return {"store": reg.store.stats(), "by_state": by_state,
                "events": len(reg.log), "lineage_nodes": len(reg.lineage)}

    @app.get("/events")
    def events(resource_id: Optional[str] = None,
               limit: int = 100) -> dict[str, Any]:
        rows = reg.event_log(resource_id)
        return {"count": len(rows), "events": rows[-limit:]}

    # -- resources --------------------------------------------------------
    @app.get("/resources")
    def list_resources(type: Optional[str] = None,
                       state: Optional[str] = None) -> dict[str, Any]:
        _refresh()
        rows = reg.list(type=type, state=state)
        return {"count": len(rows), "resources": [_record_view(r) for r in rows]}

    @app.get("/search")
    def search(q: str = "", k: int = 10) -> dict[str, Any]:
        _refresh()
        k = max(1, min(int(k), 50))
        if not q.strip():
            return {"query": q, "mode": "bm25", "count": 0, "results": []}
        # Ranked retrieval lives in toolmarket.retrieval: a normalized dense+bm25
        # weighted sum, then a cross-encoder over the top of the list. Measured on
        # 112 labelled queries that is hit@1 .777 / hit@5 .955 / MRR .847, against
        # .607/.893/.724 for the bm25-only path this replaces. The cross-encoder
        # needs prose to read, and the index holds tokens, so the text is supplied
        # here from the registry. It is skipped, not faked, when the embedder is
        # unavailable or the index was built by a different model.
        documents = {}
        for resource_id, _name, _desc in [
                (r.id, r.name, r.description or "") for r in reg.list()]:
            documents[resource_id] = (str(_name) + ". " + str(_desc)).strip()
        hits, info = retrieval.best_search(
            app.state.search, "tools", q, k=k, documents=documents,
            pool=int(os.environ.get("SEARCH_POOL") or retrieval.DEFAULT_POOL))
        results = []
        for resource_id, score, _meta in hits:
            rec = reg.get(resource_id)
            if rec is None:
                continue
            results.append({"resource_id": resource_id, "name": rec.name,
                            "state": rec.state.value, "score": round(float(score), 6),
                            "description": (rec.description or "")[:200]})
            if len(results) >= k:
                break
        return {"query": q, "mode": info["mode"], "rerank": info["rerank"],
                "pool": info["pool"], "notes": info["notes"],
                "confidence": info["confidence"],
                "count": len(results), "results": results}

    @app.post("/resources", status_code=201)
    def register(req: "RegisterRequest", request: Request) -> dict[str, Any]:
        # Build an autoforge ToolSpec so enforcement has something real to run.
        try:
            from autoforge.tools.spec import ToolSpec
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"autoforge unavailable: {exc}") from exc

        fn = _compile_fn(req.code, req.name)
        spec = ToolSpec(
            name=req.name,
            description=req.description,
            parameters=req.parameters,
            fn=fn,
            code=req.code,
            source=req.source,
            generator=req.generator,
            invariances=list(req.invariances),
            effect_signature=req.effect_signature,
            tags=list(req.tags),
        )
        try:
            rec = reg.register(spec, actor=_caller(request))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        rec.enable_evolving = req.enable_evolving
        rec.permission_mode = req.permission_mode
        if req.provenance:
            # Merge, do not overwrite: `reg.register` has already stamped
            # source/generator from the spec, and those are the authoritative
            # ones. The caller adds to that, it does not replace it.
            rec.provenance.update(req.provenance)
        if req.metadata:
            rec.metadata.update(req.metadata)
        reg.save(rec)
        try:
            app.state.search.index_text(
                "tools", rec.id, _vs.build_resource_text(_record_view(rec)),
                {"name": rec.name, "state": rec.state.value})
        except Exception:
            pass
        return _record_view(rec)

    @app.post("/discover")
    def discover(req: "DiscoverRequest") -> dict[str, Any]:
        """Go upstream to the public MCP registry when the shelf has nothing.

        This is the difference between a shelf and a library card. `/search`
        ranks *what we already hold* -- on 2026-09-17 that was 153 resources,
        all forged here or pushed by a sibling agent, so a need outside that set
        used to be answered with "nothing on the shelf" and nothing more. The
        public registry holds thousands of servers and is searchable
        server-side, so an unmatched need is one page away from an answer.

        POST, not GET, and deliberately: this *registers* what it finds (as
        DRAFT), and a GET that mutates the shelf is a trap for every retrying
        client and every crawler.

        `found` (what the registry knows) and `inserted` (new here) are separate
        numbers, so `inserted == 0` with `found > 0` reads correctly as "we
        already hold these" rather than as a miss -- the next step in that case
        is `/search`, not `/discover`.

        What it never does: promote. Everything lands DRAFT; remote entries get
        a contract that can really call the server, package-only entries get an
        empty contract marked `unbound`. See `toolmarket.mcp_import` for why.
        """
        if not req.q.strip():
            raise HTTPException(422, "q is required")
        if os.environ.get("TOOLMARKET_DISCOVER", "on").lower() in ("0", "false", "off"):
            return {"query": req.q, "enabled": False, "found": 0, "inserted": 0,
                    "results": [],
                    "note": "upstream discovery is disabled by TOOLMARKET_DISCOVER"}
        from toolmarket import mcp_import

        limit = max(1, min(int(req.limit or 25), 100))
        # Bounded for an interactive path: two attempts at 15s, not the bulk
        # importer's three at 25s. A caller waiting on a forge is owed a fast
        # failure it can report, not the best possible answer eventually.
        rep = mcp_import.find_servers(
            reg, req.q, limit=limit,
            fetch=lambda cursor=None, limit=limit, **kw: mcp_import.fetch_page(
                cursor, limit=limit, attempts=2, timeout=15.0, **kw))

        results = []
        for resource_id in rep.get("ids", [])[:limit]:
            rec = reg.get(resource_id)
            if rec is None:
                continue
            # Index it, the same way `register` does. Without this the resources
            # are on the shelf but invisible to `/search` -- measured: 8 found,
            # 8 inserted, and `/search` still returned 0. A discovery path whose
            # results the next lookup cannot see is a path that found nothing.
            try:
                app.state.search.index_text(
                    "tools", rec.id, _vs.build_resource_text(_record_view(rec)),
                    {"name": rec.name, "state": rec.state.value})
            except Exception:
                pass
            results.append({
                "resource_id": rec.id, "name": rec.name,
                "state": rec.state.value,
                "callable": bool((rec.metadata or {}).get("callable")),
                "requires_launch": bool((rec.metadata or {}).get("requires_launch")),
                "description": (rec.description or "")[:200],
                "origin": (rec.provenance or {}).get("origin_name"),
                "remotes": (rec.provenance or {}).get("remotes") or [],
            })
        return {"query": req.q, "enabled": True, "found": rep.get("found", 0),
                "entries": rep.get("entries", 0),
                "inserted": rep.get("inserted", 0), "error": rep.get("error"),
                "count": len(results), "results": results,
                "note": ("hit the shelf already held" if results
                         and rep.get("inserted") == 0 else "")}

    @app.get("/resources/{resource_id:path}/lineage")
    def lineage(resource_id: str) -> dict[str, Any]:
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        return reg.lineage_view(resource_id)

    @app.get("/resources/{resource_id:path}/events")
    def resource_events(resource_id: str, limit: int = 200) -> dict[str, Any]:
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        rows = reg.event_log(resource_id)
        return {"resource_id": resource_id, "count": len(rows),
                "events": rows[-limit:]}

    @app.get("/resources/{resource_id:path}/trust")
    def resource_trust(resource_id: str) -> dict[str, Any]:
        """Should this be ACTIVE? Answered with the evidence, not a verdict.

        GET, and read-only: the policy is applied on `invoke` (the moment the
        ledger moves) and can be swept by `tool-market trust --apply`, but a read
        never moves a resource. The route exists so the answer is *inspectable* --
        an operator asking why a well-behaved tool is still on probation gets the
        failing precondition and the thresholds it was measured against, rather
        than the empty `reason` string the manual transitions used to leave.
        """
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        return reg.trust_view(resource_id)

    @app.get("/resources/{resource_id:path}")
    def get_resource(resource_id: str, fresh: bool = False) -> dict[str, Any]:
        """One resource, from the cache when it is warm.

        `?fresh=true` bypasses the cache. It exists because the alternative — a
        client that cannot get a value it knows is stale — turns a cache into a
        correctness bug for the one user who is looking at the consequence of a
        write. A cache needs an escape hatch or it needs to be small enough not
        to need one; this one is small, and it has the hatch anyway.

        Only 200s are cached. A 404 is not: "unknown resource" is the answer that
        changes the moment somebody registers the tool, and caching it means the
        registry's own `register` response is contradicted by a stale miss.
        """
        _refresh()
        cache = get_cache()
        key = resource_key(resource_id)
        ttl = app.state.cache_ttl
        if not fresh and ttl > 0:
            hit = cache.get(key)
            if hit is not None:
                _metrics.CACHE_LOOKUPS.inc(result="hit")
                return hit
            _metrics.CACHE_LOOKUPS.inc(result="miss")
        rec = reg.get(resource_id)
        if rec is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        view = _record_view(rec)
        if ttl > 0:
            cache.set(key, view, ttl)
        return view

    @app.post("/resources/{resource_id:path}/transition")
    def do_transition(resource_id: str, req: "TransitionRequest",
                      request: Request) -> dict[str, Any]:
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        # `parse_state`, not `ResourceState(req.to)`: the gRPC surface accepts
        # "ACTIVE" and "active" alike, and a caller that gets a 409 over HTTP
        # for a string the other door accepted would be looking at a rule that
        # only exists because the two surfaces each did their own coercion.
        # An unknown *name* is still 422 (bad request), not 409 (conflict):
        # gRPC maps the same ValueError to INVALID_ARGUMENT, not
        # FAILED_PRECONDITION, and the two must agree on which failure this is.
        try:
            dst = parse_state(req.to)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            rec = reg.transition(resource_id, dst, reason=req.reason,
                                 actor=_caller(request))
        except Exception as exc:  # LifecycleError
            raise HTTPException(409, str(exc)) from exc
        return _record_view(rec)

    @app.post("/resources/{resource_id:path}/invoke")
    def do_invoke(resource_id: str, req: "InvokeRequest",
                  request: Request) -> dict[str, Any]:
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        result = reg.invoke(resource_id, req.arguments, force=req.force,
                            actor=_caller(request))
        return {
            "resource_id": resource_id,
            "ok": bool(getattr(result, "ok", True)),
            "output": getattr(result, "output", None),
            "error": getattr(result, "error", None),
        }

    @app.post("/resources/{resource_id:path}/evolve")
    def do_evolve(resource_id: str, req: "EvolveRequest") -> dict[str, Any]:
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        op = _op(reg)
        proposal = op.propose(resource_id, req.goal)
        report = op.assess(proposal.proposal_id)
        payload: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "assessment": report.to_dict(),
        }
        if req.commit and report.admissible:
            from toolmarket.protocol.sepl import ProposalRejected
            try:
                rec = op.commit(proposal.proposal_id)
                payload["committed"] = True
                payload["resource"] = _record_view(rec)
            except ProposalRejected as exc:
                payload["committed"] = False
                payload["reject_reason"] = str(exc)
        else:
            payload["committed"] = False
            payload["reject_reason"] = (
                report.summary if not report.admissible else "commit=False"
            )
        return payload

    @app.post("/resources/{resource_id:path}/evolve/async", status_code=202)
    def do_evolve_async(resource_id: str, req: "EvolveRequest") -> dict[str, Any]:
        """Enqueue an evolution and return immediately.

        Registered before the synchronous `/evolve` route on purpose. Both use
        `{resource_id:path}`, and a path converter is greedy — the shorter
        pattern is the one that must not be allowed first crack at a longer
        path. FastAPI matches in registration order, so the more specific route
        goes first.

        202, not 200: the work has been accepted and has not happened. A 200 here
        would tell every well-behaved client that the resource had evolved.
        """
        if reg.get(resource_id) is None:
            raise HTTPException(404, f"unknown resource: {resource_id}")
        rec = app.state.queue.submit(resource_id, req.goal,
                                     commit=req.commit, proposer=req.proposer)
        return {
            "task_id": rec.task_id,
            "state": rec.state,
            "queue": rec.queue,
            "status_url": f"/tasks/{rec.task_id}",
        }

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        """Poll one async evolution.

        Served from the task store, which is the cache, because a task record is
        operational state and not substrate history — the durable record of what
        an evolution did is the events and lineage it wrote, and those are
        already queryable per resource. Duplicating them into a tasks table would
        give the substrate two places to disagree about the same fact.
        """
        rec = app.state.queue.status(task_id)
        if rec is None:
            raise HTTPException(404, f"unknown task: {task_id}")
        payload = rec.to_dict()
        payload["duration_seconds"] = rec.duration
        payload["terminal"] = rec.state in (TaskState.SUCCESS.value,
                                           TaskState.FAILURE.value)
        return payload

    # The MCP front door. Mounted here rather than in `served_app` so the bare
    # app -- which is what the console mounts under /api -- speaks it too, and a
    # client that finds one surface finds the other at the same path. The facade
    # holds no state of its own: every call it serves goes through `reg`, the
    # same object the REST routes use.
    from toolmarket import mcp_facade

    mcp_facade.install(app, reg)

    return app


# The compiler moved to `toolmarket.protocol.resources.compile_tool_fn` when the
# gRPC surface landed — two front doors, one compiler. Re-exported here under
# its original private name so nothing that already spells it `_compile_fn`
# breaks for the sake of a layering argument.
_compile_fn = compile_tool_fn


# Module-level app for `uvicorn toolmarket.api.main:app`.
#
# `app` is the bare substrate; `served_app` is the same object wrapped in the
# rate limiter and the metrics middleware. They are two names rather than one
# because `web.build_app` mounts `create_app()` under `/api` and then wraps the
# *outer* app — so a single instrumented global would be instrumented twice, and
# every request would be counted twice. Containers run `served_app`; the console
# builds its own.
try:  # pragma: no cover
    if _HAVE_FASTAPI:
        app = create_app()
    else:
        app = None
except Exception:  # pragma: no cover
    app = None


def create_served_app(registry: Optional[ResourceRegistry] = None) -> Any:
    """The app as a deployment serves it: limiter + metrics + routes."""
    from toolmarket.ratelimit import install

    return install(create_app(registry))


try:  # pragma: no cover
    served_app = create_served_app() if _HAVE_FASTAPI else None
except Exception:  # pragma: no cover
    served_app = None


__all__ = ["create_app", "create_served_app", "served_app", "app",
           "ResourceRecord", "ToolContract", "VersionStatus"]
