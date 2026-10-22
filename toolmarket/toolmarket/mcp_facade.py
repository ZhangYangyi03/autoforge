"""An MCP front door for the resource substrate.

Why this file exists
--------------------
The shelf has 153 resources and one way in: our own REST (`GET /resources`,
`POST /resources/{id}/invoke`). That is a protocol only we speak. Every other
agent -- Claude Desktop, Cursor, any MCP client -- speaks MCP, and from where
they stand toolmarket is a 404 at `/mcp`: listening, healthy, and unusable.

This module is the translation and nothing else. It does not decide which
resources exist, whether one may run, or what it returns; it asks the registry
exactly the questions the REST routes ask and restates the answers in MCP's
vocabulary. Two front doors, one state machine -- the same rule `api/main.py`
states for REST versus gRPC. In particular a call here reaches enforcement
through `ResourceRegistry.invoke`, so the trust policy and the permission mode
still apply; a facade that called the compiled function directly would be a
second, unaudited way to run a tool, which is the bug the resource/gRPC split
already taught this codebase to avoid.

Transport: MCP "streamable HTTP", POST-only. A client POSTs JSON-RPC 2.0 and
gets `application/json` back, which the spec allows as an alternative to
`text/event-stream`. Server-initiated streams (`GET /mcp`) are not offered: no
operation here needs to push, and a half-implemented SSE endpoint would be worse
than an honest 405.

Scope, deliberately: `tools/list` and `tools/call` only. MCP's separate
`resources/*` family is file-like blobs with URIs -- a different concept that
happens to share a word with our "resource", and mapping one onto the other
would make both harder to reason about.

Auth: none, matching the REST surface. The deployment binds to loopback and the
public path runs through a tunnel that applies its own policy; a token scheme
here would be the second way in the comment in `toolmarket_server.py` warns
about.
"""
from __future__ import annotations

import json
from typing import Any, Optional

# Imported at module scope on purpose, even though only `install` needs them.
# With `from __future__ import annotations` every annotation is a *string*, and
# FastAPI resolves those against the function's module globals -- a `Request`
# imported inside `install` is invisible there, so the annotation stays the
# string "Request" and FastAPI classifies the parameter as a query field: every
# POST answers 422 "missing query: request" while the route looks correct in
# `app.routes`. Optional, as in `api/main.py`, so the core module still imports
# without the API extra.
try:
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response
    _HAVE_FASTAPI = True
except Exception:  # noqa: BLE001 - API extras are optional
    _HAVE_FASTAPI = False

# The revision this facade implements. A client's requested revision is echoed
# back when it asks for one -- MCP's negotiation is "server answers with what it
# supports", and answering with a version the client did not ask for is how a
# client silently switches to a dialect the server does not speak. Only the
# other direction is checked: a *known-old* client gets its version, anything
# unrecognised gets ours.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_PARSE_ERROR = -32700


def _rpc_result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id: Any, code: int, message: str,
               data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _negotiate(params: dict[str, Any]) -> str:
    asked = params.get("protocolVersion")
    if isinstance(asked, str) and asked in SUPPORTED_PROTOCOL_VERSIONS:
        return asked
    return DEFAULT_PROTOCOL_VERSION


# -- resource -> tool projection -------------------------------------------
def tool_view(rec: Any) -> dict[str, Any]:
    """One ResourceRecord as an MCP tool.

    `inputSchema` comes from `as_capability_schema()`: the record already
    projects itself into exactly this shape, with `additionalProperties: false`
    for strict function calling, so building a second schema here would be the
    duplication this facade is supposed to avoid.

    The lifecycle state is carried in the description and in `_meta`, not used
    as a filter. MCP has no way to ask for a subset and a client that cannot see
    a resource cannot decide it wants to promote it; enforcement happens at call
    time, where the ledger is, and saying so in the description is more honest
    than pretending a draft tool does not exist.
    """
    schema = rec.as_capability_schema() if hasattr(rec, "as_capability_schema") \
        else {"name": rec.name, "description": rec.description,
              "parameters": rec.contract.parameters}
    state = getattr(rec.state, "value", str(rec.state))
    desc = (rec.description or "").strip()
    return {
        "name": rec.name,
        "title": rec.name,
        "description": (desc + f"  [toolmarket state: {state}]").strip(),
        "inputSchema": schema.get("parameters") or {
            "type": "object", "properties": {}, "additionalProperties": False},
        "_meta": {"toolmarket/resource_id": rec.id,
                  "toolmarket/state": state,
                  "toolmarket/version": getattr(
                      getattr(rec, "version_current", None), "version", None)},
    }


def _tool_text(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text",
             "text": json.dumps(content, ensure_ascii=False, default=str)}]


# -- paging, and the door that does not depend on an index ------------------
#
# Measured on this shelf 2026-09-28, with 9,533 tools on it: `tools/list` with
# every schema in one reply is 5,365,686 bytes. That number is not a slow reply,
# it is an unusable one -- no client puts 9,533 tool schemas in front of a
# model, and the same shelf seen from the *other* side (autoforge's own prompt
# budget, 6,000 characters for the whole library) says why: a shelf of ten
# thousand tools cannot be an inventory a model holds, only one it queries.
#
# So `tools/list` pages, ordered by id, with an opaque cursor. Absent arguments
# mean the first page, because the MCP default every client already assumes is
# "list what you have"; an over-large `limit` is clamped rather than refused,
# since a client asking for 10,000 has told us what it wants, not made a mistake.
#
# `tools/lookup` is the second half and the more important one. A client that
# cannot enumerate a shelf still has to be able to ask it a question, and the
# shelf's own `/search` cannot answer for a name: measured today, a full-text
# query for the exact id of a tool imported an hour earlier returned ten
# unrelated tools, because the index was built at process start. A lookup that
# runs against the live registry cannot go stale -- there is nothing to rebuild.
# It scores on the same three fields a person would check (name, description,
# tags) by the share of query terms present, which is the honest amount of
# sophistication for a fallback: it cannot be wrong about what it holds, only
# about what it ranks.
DEFAULT_PAGE = 100
MAX_PAGE = 500
#: How many records a lookup scans before it must stop being called cheap.
LOOKUP_POOL = 20000


def _tools(reg: Any) -> list[Any]:
    """Every tool resource, in a stable order.

    Refreshes first, when the registry offers it, because this deployment has
    two writers: `mcp_import` writes the store while the API serves. Without
    this the facade answers from the snapshot taken at process start, which on
    2026-09-28 meant `tools/list` reported 6,985 tools for a shelf of 11,871 and
    `tools/call` for a freshly imported one came back "unknown tool" about
    something plainly present. Free when nothing changed: one indexed aggregate.
    """
    ref = getattr(reg, "refresh", None)
    if callable(ref):
        try:
            ref(max_age=1.0)
        except Exception:  # noqa: BLE001 - a read must not fail on a refresh
            pass
    return sorted(reg.list(type="tool"), key=lambda r: r.id)


def _page(records: list[Any], params: dict[str, Any]) -> dict[str, Any]:
    """One page of `tools/list`, plus `nextCursor` when there is more.

    The MCP spec's cursor is *opaque*: a client must not parse it, and this one
    is the id of the next record rather than an offset on purpose. An offset
    into a list that can change between calls skips or repeats rows when the
    shelf grows; an id-bound cursor resumes at the same place whatever happened
    to the rows before it.
    """
    limit = params.get("limit")
    try:
        limit = int(limit) if limit is not None else DEFAULT_PAGE
    except (TypeError, ValueError):
        limit = DEFAULT_PAGE
    limit = max(1, min(limit, MAX_PAGE))
    cursor = params.get("cursor")
    start = 0
    if isinstance(cursor, str) and cursor:
        # Linear in a list already in memory, which is the whole store: the
        # alternative (bisect on a sorted key) buys microseconds on a scan that
        # happens once per page, and would have to re-sort beneath a live edit.
        start = next((i for i, r in enumerate(records) if r.id == cursor),
                     len(records))
    window = records[start:start + limit]
    out: dict[str, Any] = {"tools": [tool_view(r) for r in window]}
    if start + len(window) < len(records) and window:
        out["nextCursor"] = records[start + len(window)].id
    return out


_SPLIT = None


def _terms(text: str) -> list[str]:
    """Query terms, token-shaped: alphanumeric runs of two or more characters."""
    import re as _re
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = _re.compile(r"[^0-9A-Za-z]+")
    return [w for w in _SPLIT.split((text or "").lower()) if len(w) > 1]


def _tokens(text: str) -> set[str]:
    """The same split applied to a record's text, as a set for membership."""
    return {w for w in _terms(text) if w}


def _matches(term: str, tokens: set[str]) -> bool:
    """Is `term` a thing this field says?

    Equal to a token, or a prefix of one, and nothing else. Substring matching
    was tried and is wrong in a way that is easy to see once you look: on the
    live shelf, `unknown tool: nope` came back offering
    `mcp_com_snopekgames_godai`, because "nope" sits inside "snopekgames". A
    near-miss list that is a coincidence is worse than no near-miss list -- it
    is noise wearing the costume of help.

    A prefix, but only from three characters up: "resume" finding
    `run_resume_report` is the case this is for, and a two-letter prefix would
    match a tenth of a shelf this size.
    """
    if term in tokens:
        return True
    return len(term) >= 3 and any(t.startswith(term) for t in tokens)


#: A term found in the *name* is worth this much against one found only in the
#: prose. Measured on the live shelf: an unweighted substring match let two
#: `mcp_guru_*`/`mcp_io_*` rows tie with `run_resume_report` for the query
#: "resume_run", because "resume" and "run" both appear somewhere in a long
#: generated description. A name is the field a caller actually typed.
_NAME_WEIGHT = 1.0
_DOC_WEIGHT = 0.6


def _fields(rec: Any) -> tuple[set[str], set[str]]:
    """(name tokens, descriptor tokens) — the two fields a lookup scores against."""
    tags = (getattr(rec, "metadata", None) or {}).get("tags") or []
    return (_tokens(rec.name or ""),
            _tokens(" ".join([rec.description or ""] + [str(t) for t in tags])))


def lookup(reg: Any, query: str, limit: int = 10) -> list[tuple[Any, float]]:
    """Tools whose name/description/tags contain the query's terms.

    Share of terms present, weighted by which field matched: a two-term query
    where one term matches scores 0.5 and sorts below one where both do, and a
    term found in the tool's *name* counts for more than one found only in its
    description. Not a cosine and not a learned rank -- this is a fallback, and
    it is honest about being one. Ties break on id, so the same query answers the
    same way twice: a lookup whose order wobbles between calls cannot be tested
    and is unpleasant to use.

    A record with *no* matching term is dropped rather than returned at score 0.
    Returning the near-misses would make "nothing on the shelf" impossible to
    say, and that sentence is the one that tells a caller to go upstream.
    """
    terms = _terms(query)
    if not terms:
        return []
    scored: list[tuple[Any, float]] = []
    for rec in reg.list(type="tool"):
        name, descriptors = _fields(rec)
        weight = 0.0
        for t in terms:
            if _matches(t, name):
                weight += _NAME_WEIGHT
            elif _matches(t, descriptors):
                weight += _DOC_WEIGHT
        if weight:
            scored.append((rec, weight / len(terms)))
    scored.sort(key=lambda pair: (-pair[1], pair[0].id))
    return scored[:max(1, min(int(limit or 10), MAX_PAGE))]


# -- request handling ------------------------------------------------------
def handle(reg: Any, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """One JSON-RPC message in, one response out. None for a notification.

    Returning None rather than a response is what makes `notifications/*` legal
    without a second code path: MCP defines a notification as a message with no
    `id`, and the JSON-RPC spec forbids answering one. A facade that replied to
    `notifications/initialized` would look correct in a hand test and break
    clients that treat an unexpected id as a protocol violation.
    """
    if not isinstance(msg, dict):
        return _rpc_error(None, JSONRPC_INVALID_REQUEST, "not a JSON object")
    if msg.get("jsonrpc") != "2.0":
        return _rpc_error(msg.get("id"), JSONRPC_INVALID_REQUEST,
                          "jsonrpc must be '2.0'")
    method = msg.get("method")
    if not isinstance(method, str):
        return _rpc_error(msg.get("id"), JSONRPC_INVALID_REQUEST,
                          "missing method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    is_notification = "id" not in msg

    if method in ("notifications/initialized", "notifications/cancelled",
                  "notifications/roots/list_changed"):
        return None
    if method == "initialize":
        return _rpc_result(msg_id, {
            "protocolVersion": _negotiate(params),
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "toolmarket", "version": "0.1.0"},
            "instructions": (
                "A shelf of registered agent tools. tools/list returns every "
                "resource with its lifecycle state; tools/call runs one through "
                "the same enforcement the REST surface uses."),
        })
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, _page(_tools(reg), params))
    if method == "tools/lookup":
        return _rpc_result(msg_id, _lookup_view(reg, params))
    if method == "tools/call":
        return _call(reg, msg_id, params)
    if is_notification:
        return None
    return _rpc_error(msg_id, JSONRPC_METHOD_NOT_FOUND,
                      f"unknown method: {method}")


def _lookup_view(reg: Any, params: dict[str, Any]) -> dict[str, Any]:
    """`tools/lookup` — search the shelf by name, without an index to go stale.

    A non-standard method: MCP defines `tools/list` and `tools/call` and nothing
    in between, so this is an extension and is named as one. The alternative was
    to make `tools/list` take a `query` argument, which would be a silent change
    to a standard method's meaning -- a client that sent `query` to a spec-
    compliant server and got everything back could not tell the difference.

    It answers in `tools/list`'s shape (the same objects, the same keys) so a
    client needs one renderer for both, plus its own `score` per entry.
    """
    query = params.get("query") or params.get("q") or ""
    if not isinstance(query, str) or not query.strip():
        return {"tools": [], "query": "", "error": "query is required"}
    limit = params.get("limit") or 10
    hits = lookup(reg, query, limit)
    return {"query": query, "count": len(hits),
            "tools": [{**tool_view(rec), "score": round(score, 4)}
                      for rec, score in hits]}


def _call(reg: Any, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str) or not name:
        return _rpc_error(msg_id, JSONRPC_INVALID_PARAMS, "name is required")
    if not isinstance(arguments, dict):
        return _rpc_error(msg_id, JSONRPC_INVALID_PARAMS,
                          "arguments must be an object")

    # Refresh before resolving: a tool registered by another process since this
    # one started is a tool this door must be able to call. Cheap when nothing
    # changed (one indexed aggregate) and the whole point when something has.
    ref = getattr(reg, "refresh", None)
    if callable(ref):
        try:
            ref(max_age=1.0)
        except Exception:  # noqa: BLE001
            pass

    rid = f"tool:{name}"
    if reg.get(rid) is None:
        # Fall back to the display name: a resource registered under a
        # different id would otherwise be listed and uncallable, which is the
        # worst of the two failure modes because tools/list said it existed.
        match = next((r for r in reg.list(type="tool") if r.name == name), None)
        if match is None:
            # Last resort, and only for a *name* the shelf may hold under a
            # prefixed id. Say what was tried rather than a bare "unknown":
            # `lookup` can answer "did you mean", and a client that gets a
            # near-miss is one step from the right call instead of stuck.
            near = [r.name for r, _ in lookup(reg, name, 3)]
            return _rpc_error(
                msg_id, JSONRPC_INVALID_PARAMS, f"unknown tool: {name}",
                data={"near_misses": near} if near else None)
        rid = match.id

    # An unknown tool is a protocol error (above) and a failed run is not: the
    # spec puts execution failures in the result with `isError`, so a client can
    # tell "you called it wrong" from "it ran and returned an error" and hand
    # the second to the model instead of aborting.
    try:
        result = reg.invoke(rid, arguments)
    except Exception as exc:  # noqa: BLE001 - enforcement refusals land here
        return _rpc_result(msg_id, {
            "content": _tool_text(f"{type(exc).__name__}: {exc}"),
            "isError": True,
        })
    ok = bool(getattr(result, "ok", True))
    error = getattr(result, "error", None)
    output = getattr(result, "output", None)
    if not ok and error is not None:
        return _rpc_result(msg_id, {
            "content": _tool_text(str(error)), "isError": True,
        })
    payload: dict[str, Any] = {"content": _tool_text(output),
                               "isError": not ok}
    if isinstance(output, dict):
        payload["structuredContent"] = output
    return _rpc_result(msg_id, payload)


def install(app: Any, reg: Any, path: str = "/mcp") -> None:
    """Mount the facade on a FastAPI app. POST only, by design."""
    if not _HAVE_FASTAPI:
        raise RuntimeError("FastAPI is required for the MCP facade")

    @app.post(path)
    async def mcp(request: Request) -> Any:
        raw = await request.body()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 - malformed body is a client bug
            return JSONResponse(
                _rpc_error(None, JSONRPC_PARSE_ERROR, "invalid JSON"),
                status_code=400)

        batch = isinstance(payload, list)
        messages = payload if batch else [payload]
        responses: list[dict[str, Any]] = []
        for msg in messages:
            reply = handle(reg, msg)
            if reply is not None:
                responses.append(reply)

        if not responses:
            # Every message was a notification. 202 with no body is the spec's
            # answer; a 200 with an empty object would be read by some clients
            # as a response to a message that never had an id.
            return Response(status_code=202)
        body = responses if batch else responses[0]
        return JSONResponse(body)

    @app.get(path)
    def mcp_get() -> Any:
        """405, not a stub stream. See the module docstring."""
        return JSONResponse(
            {"error": "this facade is POST-only (no server-initiated stream); "
                      "POST JSON-RPC 2.0 here instead"},
            status_code=405)
