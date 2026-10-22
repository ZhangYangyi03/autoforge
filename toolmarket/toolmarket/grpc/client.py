"""toolmarket.grpc.client — a thin, typed client for the gRPC surface.

Thin on purpose. Every method here is one stub call plus the JSON decode of
whatever string field came back; there is no retry policy, no connection pool of
its own, no state. The moment this file starts being clever it becomes a second
implementation of the client's own logic — and the caller's logic is the thing
that actually needs to be right, so it must stay in the caller.

What it *does* provide is the one thing a raw stub does not: the string fields
holding JSON come back as Python objects, and the "no arguments" case comes back
as `{}` rather than `None`, so callers never have to remember which of the two a
given RPC produces.

Used by the parity test:

    with SubstrateClient("127.0.0.1:5051") as c:
        rid = c.register(name="add", code=..., description=...)
        c.transition(rid, "PROBATION"); c.transition(rid, "ACTIVE")
        print(c.invoke(rid, {"a": 1, "b": 2}))
"""
from __future__ import annotations

import json
from typing import Any, Optional

from toolmarket.grpc.server import import_stubs

DEFAULT_TARGET = "127.0.0.1:5051"


def _decode(text: str) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A handler that returned a non-JSON string (an error message, say) is
        # not worth an exception in the client. Hand it back verbatim.
        return text


class SubstrateClient:
    """A context manager over one channel to one server.

    Holding one channel for the life of the client is the point: a channel is a
    TCP connection plus HTTP/2 state, and opening one per call turns a typed
    protocol back into a request-per-connection one, which is the cost gRPC
    exists to avoid.
    """

    def __init__(self, target: Optional[str] = None, *,
                 timeout: float = 30.0) -> None:
        pb, pb_grpc = import_stubs()
        import grpc

        self._pb = pb
        self._timeout = timeout
        self._channel = grpc.insecure_channel(target or DEFAULT_TARGET)
        self._stub = pb_grpc.ResourceSubstrateStub(self._channel)

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "SubstrateClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

    # -- substrate ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        r = self._stub.Health(self._pb.HealthRequest(), timeout=self._timeout)
        return {"status": r.status, "version": r.version}

    def ready(self) -> dict[str, Any]:
        r = self._stub.Ready(self._pb.ReadyRequest(), timeout=self._timeout)
        return {"ready": r.ready, "store": r.store, "cache": r.cache,
                "detail": r.detail}

    # -- resources ---------------------------------------------------------
    def register(self, *, name: str, code: str, description: str = "",
                 parameters: Optional[dict[str, Any]] = None,
                 source: str = "human", generator: str = "",
                 invariances: Optional[list[str]] = None,
                 effect_signature: str = "",
                 tags: Optional[list[str]] = None,
                 enable_evolving: Optional[bool] = None) -> str:
        """Register a tool; return its resource id.

        `enable_evolving=None` leaves the field *unset* on the wire, which the
        server reads as "did not ask" and answers with the same default the REST
        surface applies. Passing an explicit bool overrides it. This is the only
        place in the client where None means something other than "empty".
        """
        p = self._pb
        contract = p.ToolContract(
            name=name, description=description,
            parameters_json=json.dumps(parameters or {}, ensure_ascii=False),
            code=code, source=source, generator=generator,
            invariances=list(invariances or []),
            effect_signature=effect_signature, tags=list(tags or []),
        )
        req = p.RegisterRequest(contract=contract)
        if enable_evolving is not None:
            req.enable_evolving = enable_evolving
        r = self._stub.Register(req, timeout=self._timeout)
        return r.resource.id

    def get(self, resource_id: str) -> dict[str, Any]:
        r = self._stub.Get(self._pb.GetRequest(resource_id=resource_id),
                           timeout=self._timeout)
        return self._resource(r.resource)

    def list(self, *, type: str = "", state: str = "") -> list[dict[str, Any]]:
        r = self._stub.List(self._pb.ListRequest(type=type, state=state),
                            timeout=self._timeout)
        return [self._resource(x) for x in r.resources]

    def transition(self, resource_id: str, to: str,
                   reason: str = "") -> dict[str, Any]:
        r = self._stub.Transition(
            self._pb.TransitionRequest(resource_id=resource_id, to=to,
                                       reason=reason),
            timeout=self._timeout)
        return self._resource(r.resource)

    def invoke(self, resource_id: str, arguments: Optional[dict[str, Any]] = None,
               *, force: bool = False) -> dict[str, Any]:
        r = self._stub.Invoke(
            self._pb.InvokeRequest(
                resource_id=resource_id,
                arguments_json=json.dumps(arguments or {}, ensure_ascii=False),
                force=force),
            timeout=self._timeout)
        # `output`, not `result`, to match the REST invoke response key for key.
        # The proto field is `result_json`; the *projection* is what clients
        # switch between, so that is the layer that has to agree.
        return {"ok": r.ok, "output": _decode(r.result_json), "error": r.error}

    def lineage(self, resource_id: str) -> list[dict[str, str]]:
        r = self._stub.Lineage(
            self._pb.LineageRequest(resource_id=resource_id),
            timeout=self._timeout)
        return [{"from": e.split("->")[0], "to": e.split("->")[1]}
                for e in r.edges]

    def events(self, resource_id: str = "") -> list[dict[str, Any]]:
        r = self._stub.Events(
            self._pb.EventsRequest(resource_id=resource_id),
            timeout=self._timeout)
        return [{"kind": e.kind, "resource_id": e.resource_id, "at": e.at,
                 "data": _decode(e.detail_json)} for e in r.events]

    # -- evolution ---------------------------------------------------------
    def evolve(self, resource_id: str, goal: str, *, commit: bool = False,
               proposer: str = "stub") -> dict[str, Any]:
        r = self._stub.Evolve(
            self._pb.EvolveRequest(resource_id=resource_id, goal=goal,
                                   commit=commit, proposer=proposer),
            timeout=self._timeout)
        return {"task_id": r.task_id, "state": r.state}

    def task(self, task_id: str) -> dict[str, Any]:
        r = self._stub.GetTask(self._pb.TaskRequest(task_id=task_id),
                               timeout=self._timeout)
        return {"task_id": r.task_id, "state": r.state,
                "result": _decode(r.result_json), "error": r.error}

    # -- projection --------------------------------------------------------
    def _resource(self, m: Any) -> dict[str, Any]:
        """pb.Resource -> the same dict shape the REST surface returns.

        The field names line up with `ResourceRecord.to_dict()` on purpose: a
        client that switches transports should not also have to switch key
        names, and a parity test comparing the two is then comparing values
        rather than a translation.
        """
        return {
            "id": m.id, "name": m.name, "type": m.type, "state": m.state,
            "version_status": m.version_status, "version": m.version,
            "contract": _decode(m.contract_json),
            "capability_schema": _decode(m.capability_schema_json),
            "updated_at": m.updated_at, "parents": list(m.parents),
        }
