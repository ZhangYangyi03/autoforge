"""toolmarket.grpc.server — serve the resource substrate over gRPC.

The constraint that shapes this file, stated once: **this server owns no
protocol state.** It does not know which transitions are legal, it does not
write events, it does not open the store. Every RPC below is a translation
between a protobuf message and exactly one registry/operator call. The moment a
handler here decides something, the substrate has two rulebooks — and the one
that gets enforced is whichever door the caller happened to knock on, which is
not enforcement, it is a coin flip.

Everything the substrate guarantees is therefore *inherited*: a version change
still drops an ACTIVE tool back to PROBATION, an illegal transition still
raises, and the audit chain still grows — because the code that does those
things is the same code the REST route calls. The proof is in
`tests/test_grpc_surface.py`, which asserts the two surfaces agree rather than
asserting this docstring is true.

Why a second surface exists at all, given `toolmarket/api/main.py` already
works: the evolution loop is a typed, cross-process, long-running conversation,
and the caller is not always Python. Over HTTP a Go or Java client learns the
shape of a task by reading a docstring; over gRPC it gets a compile-time
contract from the `.proto`, plus deadlines and streaming when the loop grows a
progress feed. For Python-only callers the HTTP surface is strictly enough, so
this package is an extra, not a dependency.
"""
from __future__ import annotations

import json
import os
from concurrent import futures
from typing import Any, Callable, Optional

from toolmarket import __version__
from toolmarket import metrics as _metrics
from toolmarket.cache import get_cache
from toolmarket.protocol.lifecycle import LifecycleError, parse_state
from toolmarket.protocol.resources import compile_tool_fn
from toolmarket.registry import ResourceRegistry
from toolmarket.tasks import make_queue

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5051
DEFAULT_WORKERS = 8
# The evolution loop ships candidate *code* through this channel, and a file's
# worth of source is not a chat message. The 4 MB gRPC default has bitten
# every project that ever put source on the wire; 16 MB is the smallest number
# that has not needed revisiting.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _json(value: Any) -> str:
    """Render a Python value into a proto string field.

    `default=str` rather than a raise: this is a *projection* of data that is
    already durable elsewhere, and a datetime that cannot be serialised must not
    turn a successful read into a failed RPC. The bytes on the wire are lossy in
    that one case, which is the lesser of the two failures.
    """
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(text: str) -> Any:
    """Parse a proto string field back into Python. Unset -> `{}`, not `None`,
    so a caller that omitted the arguments gets a no-argument call rather than a
    TypeError from inside the registry."""
    if not text:
        return {}
    return json.loads(text)


def import_stubs() -> Any:
    """Import the generated modules, with the install hint in the error.

    Generated code is not committed, so a fresh checkout that forgot to run
    `make grpc-gen` fails here. Saying so beats an ImportError naming a file the
    reader has never heard of.
    """
    try:
        from toolmarket.grpc.gen import toolmarket_pb2 as pb
        from toolmarket.grpc.gen import toolmarket_pb2_grpc as pb_grpc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "gRPC stubs are missing. Generate them first: `make grpc-gen` "
            "(needs grpcio-tools), or install the extra: pip install "
            f"'tool-market[grpc]'. Underlying error: {exc!r}"
        ) from exc
    return pb, pb_grpc


class ResourceSubstrateServicer:
    """One method per RPC: translate, call once, translate back.

    The class body is filled in by `_build_servicer`, which binds it to the
    generated base class at import time. That indirection exists because the
    generated module is not importable until `make grpc-gen` has run, and a
    module that refuses to import is harder to diagnose than one that refuses to
    *serve*.
    """


def _build_servicer() -> Any:
    pb, pb_grpc = import_stubs()

    class _Servicer(pb_grpc.ResourceSubstrateServicer):  # type: ignore[misc]
        """The real implementation, bound to the generated base class."""

        def __init__(self, registry: Optional[ResourceRegistry] = None,
                     queue: Any = None) -> None:
            # Same defaults as the REST app: a fresh registry, and the queue the
            # environment selects. A gRPC process and an HTTP process pointed at
            # the same DATABASE_URL/REDIS_URL therefore see the same substrate.
            self.registry = registry or ResourceRegistry()
            self.queue = queue or make_queue(self.registry)

        # -- translation ---------------------------------------------------
        def _resource(self, rec: Any) -> Any:
            """ResourceRecord -> pb.Resource.

            Deliberately partial: this message carries identity, enforcement
            state and provenance, not the whole record. A field the record does
            not have is left at its proto default instead of being invented, and
            anything a client might want that is missing travels in
            `contract_json`/`capability_schema_json` as-is.
            """
            contract = rec.contract.to_dict()
            return pb.Resource(
                id=rec.id,
                name=rec.name or "",
                type=rec.type.value,
                state=rec.state.value,
                version_status=rec.version_current.status.value,
                version=rec.version_current.version,
                contract_json=_json(contract),
                capability_schema_json=_json(rec.as_capability_schema()),
                updated_at=float(rec.updated_at),
                parents=list(rec.lineage_parents or []),
            )

        def _require(self, resource_id: str) -> Any:
            """A missing resource is a KeyError, which `_call` maps to
            NOT_FOUND. Doing the lookup here rather than inside the registry
            keeps one error vocabulary for both surfaces."""
            rec = self.registry.get(resource_id)
            if rec is None:
                raise KeyError(f"unknown resource: {resource_id}")
            return rec

        # -- the guard -----------------------------------------------------
        def _call(self, rpc: str, context: Any, body: Callable[[], Any]) -> Any:
            """Run one handler body; map failures onto gRPC codes; count it.

            The mapping is the substance of this method. `KeyError` means the
            thing named does not exist (NOT_FOUND). `LifecycleError` means the
            request is well-formed in the abstract and impossible for *this*
            resource right now (FAILED_PRECONDITION) — collapsing that into
            UNKNOWN, which is what a bare try/except produces, throws away the
            only information a caller needs to know whether to retry, fix the
            request, or stop. `ValueError` is a genuinely bad argument
            (INVALID_ARGUMENT).

            Anything else is INTERNAL and carries the exception's type name,
            because "500" with no type is the least debuggable thing a server
            can emit.
            """
            try:
                out = body()
            except KeyError as exc:
                context.abort(  # type: ignore[attr-defined]
                    __import__("grpc").StatusCode.NOT_FOUND, str(exc))
            except LifecycleError as exc:
                context.abort(
                    __import__("grpc").StatusCode.FAILED_PRECONDITION, str(exc))
            except ValueError as exc:
                context.abort(
                    __import__("grpc").StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:  # noqa: BLE001
                context.abort(
                    __import__("grpc").StatusCode.INTERNAL,
                    f"{type(exc).__name__}: {exc}")
            self._count(rpc, "OK")
            return out

        def _count(self, rpc: str, status: str) -> None:
            # Instrumentation must never be able to fail a call, so this is
            # guarded: a metrics registry that has been reset mid-test is not a
            # reason for a client to see INTERNAL.
            try:
                _metrics.GRPC_REQUESTS.inc(rpc=rpc, status=status)
            except Exception:  # noqa: BLE001
                pass

        # -- liveness / readiness ------------------------------------------
        def Health(self, request: Any, context: Any) -> Any:
            """Liveness only. No store call, no cache call — the same rule as
            `GET /health` and for the same reason: a liveness probe that depends
            on a dependency turns one outage into a restart loop."""
            return self._call("Health", context, lambda: pb.HealthResponse(
                status="ok", version=__version__))

        def Ready(self, request: Any, context: Any) -> Any:
            """Readiness: the store and the cache both answer a trivial call.

            Note what is *not* here: an abort. "Not ready" is a normal, expected
            answer, not an error — gRPC has no 503 to carry it, so the boolean
            carries it and the client decides. Aborting with UNAVAILABLE would
            make every orchestrator-side check look like a transport fault.
            """
            store_ok = False
            try:
                store_ok = bool(self.registry.store.ping())
            except Exception:  # noqa: BLE001
                store_ok = False
            cache_ok = False
            try:
                cache_ok = bool(get_cache().ping())
            except Exception:  # noqa: BLE001
                cache_ok = False
            detail = ""
            if not store_ok:
                detail = "store did not answer ping"
            elif not cache_ok:
                detail = "cache did not answer ping"
            return pb.ReadyResponse(
                ready=store_ok and cache_ok,
                store=getattr(self.registry.store, "backend", "sqlite"),
                cache=getattr(get_cache(), "backend", "none"),
                detail=detail,
            )

        # -- resources -----------------------------------------------------
        def Register(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                from autoforge.tools.spec import ToolSpec

                c = request.contract
                spec = ToolSpec(
                    name=c.name,
                    description=c.description,
                    parameters=_loads(c.parameters_json),
                    fn=compile_tool_fn(c.code, c.name),
                    code=c.code,
                    source=c.source or "human",
                    generator=c.generator,
                    invariances=list(c.invariances),
                    effect_signature=c.effect_signature,
                    tags=list(c.tags),
                )
                rec = self.registry.register(spec)
                # Presence, not value: an unset field means "did not ask", and
                # the REST default for evolving is ON. See the .proto note.
                if (not request.HasField("enable_evolving")
                        or request.enable_evolving):
                    rec.enable_evolving = True
                    self.registry.save(rec)
                return pb.RegisterResponse(resource=self._resource(rec))

            return self._call("Register", context, body)

        def Get(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                rec = self._require(request.resource_id)
                return pb.GetResponse(resource=self._resource(rec))

            return self._call("Get", context, body)

        def List(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                rows = self.registry.list(type=request.type or None,
                                          state=request.state or None)
                return pb.ListResponse(
                    resources=[self._resource(r) for r in rows])

            return self._call("List", context, body)

        def Transition(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                self._require(request.resource_id)
                # Same coercion the REST route uses, from the same function —
                # "ACTIVE" and "active" must not mean different things depending
                # on which door the caller knocked on.
                dst = parse_state(request.to)
                rec = self.registry.transition(request.resource_id, dst,
                                               reason=request.reason)
                return pb.TransitionResponse(resource=self._resource(rec))

            return self._call("Transition", context, body)

        def Invoke(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                self._require(request.resource_id)
                result = self.registry.invoke(
                    request.resource_id, _loads(request.arguments_json),
                    force=request.force)
                return pb.InvokeResponse(
                    ok=bool(getattr(result, "ok", True)),
                    result_json=_json(getattr(result, "output", None)),
                    error=str(getattr(result, "error", None) or ""),
                )

            return self._call("Invoke", context, body)

        def Lineage(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                self._require(request.resource_id)
                view = self.registry.lineage_view(request.resource_id)
                nodes = list(view.get("nodes", []))
                return pb.LineageResponse(
                    nodes=[n["node_id"] for n in nodes],
                    # "parent->child", so an edge is readable without a second
                    # lookup and the direction is not a convention the reader
                    # has to guess at.
                    edges=[f"{p}->{n['node_id']}"
                           for n in nodes for p in n.get("parents", [])],
                )

            return self._call("Lineage", context, body)

        def Events(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                if request.resource_id:
                    self._require(request.resource_id)
                rows = self.registry.event_log(request.resource_id or None)
                return pb.EventsResponse(events=[
                    pb.Event(
                        kind=str(e.get("kind", "")),
                        resource_id=str(e.get("resource_id", "")),
                        at=float(e.get("at", 0.0)),
                        detail_json=_json(e.get("data", {})),
                    )
                    for e in rows[-200:]
                ])

            return self._call("Events", context, body)

        # -- evolution -----------------------------------------------------
        def Evolve(self, request: Any, context: Any) -> Any:
            """Enqueue and return. Always asynchronous, always.

            There is no synchronous variant here even though the REST surface
            has one. A blocking RPC has to be bounded by a deadline the server
            does not control, and a candidate assessment can run for minutes; a
            synchronous variant would be a method whose documented contract is
            "may exceed your timeout". Clients that want to wait can poll
            `GetTask` — which is also how they get progress, and how a client in
            another language gets a task id it can hand to a queue.
            """
            def body() -> Any:
                self._require(request.resource_id)
                rec = self.queue.submit(
                    request.resource_id, request.goal,
                    commit=request.commit, proposer=request.proposer or "stub")
                return pb.EvolveResponse(task_id=rec.task_id, state=rec.state)

            return self._call("Evolve", context, body)

        def GetTask(self, request: Any, context: Any) -> Any:
            def body() -> Any:
                rec = self.queue.status(request.task_id)
                if rec is None:
                    raise KeyError(f"unknown task: {request.task_id}")
                return pb.TaskResponse(
                    task_id=rec.task_id,
                    state=rec.state,
                    result_json=_json(rec.result) if rec.result else "",
                    error=str(rec.error or ""),
                )

            return self._call("GetTask", context, body)

    return _Servicer


def serve(registry: Optional[ResourceRegistry] = None, *, queue: Any = None,
          host: Optional[str] = None, port: Optional[int] = None,
          workers: Optional[int] = None) -> Any:
    """Start the server and return it. Not blocking; call `wait_for_termination`.

    `add_insecure_port` is not an oversight. TLS belongs at the ingress or the
    mesh, where the certificate can be rotated without rebuilding this image and
    where the same policy covers the other services beside it. Baking a cert
    path into a service that also runs on a laptop is how "it works locally"
    becomes a production incident.
    """
    global _SERVICER_CLASS
    pb, pb_grpc = import_stubs()
    if _SERVICER_CLASS is None:
        _SERVICER_CLASS = _build_servicer()

    import grpc

    host = host if host is not None else os.environ.get("GRPC_HOST", DEFAULT_HOST)
    # `is not None`, not `or`: port 0 is how a caller asks the OS for an
    # ephemeral port, which is what every test wants and what a `or` would
    # silently turn into "use the default / env port" — a test would then bind
    # the real default and collide with a running server.
    port = port if port is not None else int(
        os.environ.get("GRPC_PORT", DEFAULT_PORT))
    workers = workers if workers is not None else int(
        os.environ.get("GRPC_WORKERS", DEFAULT_WORKERS))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=[("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                 ("grpc.max_send_message_length", MAX_MESSAGE_BYTES)],
    )
    pb_grpc.add_ResourceSubstrateServicer_to_server(
        _SERVICER_CLASS(registry=registry, queue=queue), server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise RuntimeError(
            f"could not bind {host}:{port} — the port is taken or reserved. "
            "Pick another with GRPC_PORT."
        )
    # Exposed so a caller that asked for port 0 can find out what it got. A test
    # that has to guess the port has to probe for it, and a probe is a race.
    server.bound_port = bound  # type: ignore[attr-defined]
    server.start()
    return server


_SERVICER_CLASS: Any = None


def main() -> None:
    """`python -m toolmarket.grpc.server` — run the substrate over gRPC alone."""
    server = serve()
    print(f"toolmarket gRPC listening (version {__version__}); "
          "ctrl-c to stop", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":  # pragma: no cover
    main()
