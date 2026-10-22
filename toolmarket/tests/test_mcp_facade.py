"""The MCP facade: a second front door that must reach the same state machine.

Three things are worth a test and are tested here.

1. Shape. `initialize` / `tools/list` / `tools/call` answer in MCP's vocabulary,
   and a message with no `id` gets no response at all -- replying to a
   notification is a protocol violation some clients abort on.

2. It is a *translation*, not a second implementation. A call must go through
   `ResourceRegistry.invoke`: enforcement, the ledger write and the trust
   reconcile all belong to the registry, and a facade that reached the compiled
   function directly would be an unaudited way to run a tool. That is asserted
   by counting registry calls, not by reading the code.

3. Failure modes are the spec's, not ours: an unknown tool is a JSON-RPC error
   (the client called it wrong) while a tool that runs and returns an error is a
   result with `isError` (the model should see it).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from toolmarket import mcp_facade as facade
from toolmarket.api.main import create_app
from toolmarket.cache import reset_cache_singleton
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


def _spec(name: str = "slugify", description: str = "Slugify a string."):
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    def slugify(text: str = "") -> str:
        return "-".join(text.lower().split())

    return ToolSpec(
        name=name,
        description=description,
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
    monkeypatch.setenv("RATE_LIMIT", "100000")
    reset_cache_singleton()
    yield
    reset_cache_singleton()


def _client(name="slugify", register=True):
    reg = ResourceRegistry(ResourceStore(":memory:"))
    rid = reg.register(_spec(name)).id if register else None
    return TestClient(create_app(reg), raise_server_exceptions=False), reg, rid


def _rpc(client, method, params=None, msg_id=1):
    body = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        body["id"] = msg_id
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body)


# -- shape -----------------------------------------------------------------
def test_initialize_negotiates_and_names_itself():
    client, _, _ = _client()
    r = _rpc(client, "initialize",
             {"protocolVersion": "2025-03-26", "capabilities": {},
              "clientInfo": {"name": "t", "version": "0"}})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["protocolVersion"] == "2025-03-26"
    assert res["serverInfo"]["name"] == "toolmarket"
    assert "tools" in res["capabilities"]


def test_unknown_protocol_version_falls_back_to_ours():
    client, _, _ = _client()
    r = _rpc(client, "initialize", {"protocolVersion": "1999-01-01"})
    assert r.json()["result"]["protocolVersion"] == facade.DEFAULT_PROTOCOL_VERSION


def test_tools_list_projects_contract_into_input_schema():
    client, reg, rid = _client()
    r = _rpc(client, "tools/list")
    tools = r.json()["result"]["tools"]
    assert [t["name"] for t in tools] == ["slugify"]
    tool = tools[0]
    assert tool["inputSchema"]["properties"]["text"]["type"] == "string"
    # Strict function calling: AGP requires the record to close its schema.
    assert tool["inputSchema"]["additionalProperties"] is False
    assert tool["_meta"]["toolmarket/resource_id"] == rid
    assert tool["_meta"]["toolmarket/state"] == reg.get(rid).state.value


def test_notification_gets_no_response():
    client, _, _ = _client()
    r = _rpc(client, "notifications/initialized", None, msg_id=None)
    assert r.status_code == 202
    assert r.content == b""


def test_batch_is_answered_positionally_and_notifications_dropped():
    client, _, _ = _client()
    r = client.post("/mcp", json=[
        {"jsonrpc": "2.0", "method": "ping", "id": 1},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "tools/list", "id": 2},
    ])
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    assert [m["id"] for m in body] == [1, 2]


def test_get_is_405_not_a_broken_stream():
    client, _, _ = _client()
    assert client.get("/mcp").status_code == 405


# -- it reaches the real state machine -------------------------------------
def test_call_goes_through_the_registry(monkeypatch):
    client, reg, rid = _client()
    seen = {"n": 0}
    real = reg.invoke

    def counting(resource_id, arguments, **kw):
        seen["n"] += 1
        return real(resource_id, arguments, **kw)

    monkeypatch.setattr(reg, "invoke", counting)
    r = _rpc(client, "tools/call",
             {"name": "slugify", "arguments": {"text": "A B"}})
    assert seen["n"] == 1, "the facade must not bypass ResourceRegistry.invoke"
    result = r.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"] == "a-b"


def test_call_is_recorded_in_the_ledger():
    client, reg, rid = _client()
    before = len(reg.event_log(rid))
    _rpc(client, "tools/call", {"name": "slugify", "arguments": {"text": "x"}})
    assert len(reg.event_log(rid)) == before + 1


# -- failure modes ---------------------------------------------------------
def test_unknown_tool_is_a_jsonrpc_error():
    client, _, _ = _client()
    r = _rpc(client, "tools/call", {"name": "nope", "arguments": {}})
    err = r.json()["error"]
    assert err["code"] == facade.JSONRPC_INVALID_PARAMS
    assert "nope" in err["message"]


def test_bad_arguments_come_back_as_iserror_not_a_protocol_error():
    client, _, _ = _client()
    r = _rpc(client, "tools/call",
             {"name": "slugify", "arguments": {"text": 5}})
    result = r.json()["result"]
    assert "error" not in r.json()
    assert result["isError"] is True
    assert result["content"][0]["text"]


def test_unknown_method_is_method_not_found():
    client, _, _ = _client()
    r = _rpc(client, "tools/wat")
    assert r.json()["error"]["code"] == facade.JSONRPC_METHOD_NOT_FOUND


def test_malformed_body_is_400_parse_error():
    client, _, _ = _client()
    r = client.post("/mcp", content=b"{not json")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == facade.JSONRPC_PARSE_ERROR


def test_empty_shelf_lists_zero_tools_not_an_error():
    client, _, _ = _client(register=False)
    r = _rpc(client, "tools/list")
    assert r.json()["result"]["tools"] == []
    assert "error" not in r.json()


# -- paging, lookup, and two writers ----------------------------------------
def test_tools_list_pages_and_the_cursor_resumes_exactly():
    """A shelf of thousands cannot be one reply.

    Measured on the real shelf: 9,533 tools in one `tools/list` is 5.4 MB. The
    property worth testing is not the size, it is that paging is *lossless*:
    walking every page yields each tool exactly once, in the same order, and the
    last page carries no cursor.
    """
    reg = ResourceRegistry(ResourceStore(":memory:"))
    for i in range(25):
        reg.register(_spec(f"tool_{i:02d}"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)

    seen, cursor, pages = [], None, 0
    while True:
        params = {"limit": 10}
        if cursor:
            params["cursor"] = cursor
        r = _rpc(client, "tools/list", params)
        res = r.json()["result"]
        seen += [t["name"] for t in res["tools"]]
        pages += 1
        cursor = res.get("nextCursor")
        if not cursor:
            break
        assert pages < 10
    assert seen == [f"tool_{i:02d}" for i in range(25)]
    assert pages == 3
    # The default page size is small enough to be a page, not a dump.
    assert facade.DEFAULT_PAGE <= 500


def test_default_page_is_used_when_no_limit_is_asked_for():
    reg = ResourceRegistry(ResourceStore(":memory:"))
    for i in range(facade.DEFAULT_PAGE + 5):
        reg.register(_spec(f"t{i:04d}"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)
    res = _rpc(client, "tools/list", {}).json()["result"]
    assert len(res["tools"]) == facade.DEFAULT_PAGE
    assert res["nextCursor"]


def test_an_over_large_limit_is_clamped_not_refused():
    """A client asking for 50,000 has said what it wants, not made a mistake."""
    reg = ResourceRegistry(ResourceStore(":memory:"))
    for i in range(5):
        reg.register(_spec(f"t{i}"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)
    res = _rpc(client, "tools/list", {"limit": 50000}).json()["result"]
    assert len(res["tools"]) == 5
    assert "nextCursor" not in res


def test_lookup_finds_by_name_and_by_description_without_an_index():
    """The door that cannot go stale: no index, no rebuild, live registry."""
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_spec("slugify"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)

    by_name = _rpc(client, "tools/lookup", {"query": "slugify"}).json()["result"]
    assert [t["name"] for t in by_name["tools"]] == ["slugify"]
    assert by_name["tools"][0]["score"] == 1.0
    # A term matching nothing returns nothing -- "not on the shelf" has to be
    # sayable, or the caller never learns to go upstream.
    miss = _rpc(client, "tools/lookup", {"query": "zzz_nothing_here"}).json()["result"]
    assert miss["tools"] == []
    empty = _rpc(client, "tools/lookup", {"query": "  "}).json()["result"]
    assert empty["tools"] == [] and "error" in empty


def test_lookup_ranks_by_share_of_terms_present_and_is_stable():
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_spec("both_terms_here"))
    reg.register(_spec("only_both_present_once"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)
    res = _rpc(client, "tools/lookup", {"query": "both terms"}).json()["result"]
    scores = [t["score"] for t in res["tools"]]
    assert scores == sorted(scores, reverse=True)
    again = _rpc(client, "tools/lookup", {"query": "both terms"}).json()["result"]
    assert [t["name"] for t in again["tools"]] == [t["name"] for t in res["tools"]]


def test_a_tool_written_by_another_process_becomes_visible_and_callable():
    """Two writers, one store -- the failure this was built for.

    Measured 2026-09-28: the importer had written 11,871 resources while the
    live shelf answered `/health` with 6,985 and 404'd on a row three minutes
    old. The registry is a snapshot; nothing was wrong with the store. So a
    second registry over the same file must see the first one's rows, and must
    be able to *call* one -- visibility without callability is the worse half.
    """
    import tempfile, os as _os
    from toolmarket.store import ResourceStore as _RS

    path = _os.path.join(tempfile.mkdtemp(prefix="tm2w_"), "s.db")
    writer = ResourceRegistry(_RS(path))
    served = ResourceRegistry(_RS(path))
    assert served.get("tool:late") is None

    writer.register(_spec("late"))
    changed = served.refresh()
    assert [r.name for r in changed] == ["late"]
    assert served.get("tool:late") is not None
    # Nothing new the second time: the high-water mark is what makes a refresh
    # per request affordable.
    assert served.refresh() == []

    client = TestClient(create_app(served), raise_server_exceptions=False)
    seen = {t["name"] for t in
            _rpc(client, "tools/list", {}).json()["result"]["tools"]}
    assert "late" in seen
    called = _rpc(client, "tools/call",
                  {"name": "late", "arguments": {"text": "A B"}})
    assert called.json()["result"]["isError"] is False


def test_a_refresh_does_not_evict_a_row_deleted_underneath_it():
    """New rows are the case that matters; a missing row is not worth a crash."""
    import tempfile, os as _os
    from toolmarket.store import ResourceStore as _RS

    path = _os.path.join(tempfile.mkdtemp(prefix="tmev_"), "s.db")
    store = _RS(path)
    reg = ResourceRegistry(store)
    reg.register(_spec("keepme"))
    stamp_before = reg._loaded_stamp
    reg.refresh()
    assert reg.get("tool:keepme") is not None
    assert reg._loaded_stamp >= stamp_before


def test_unknown_tool_says_what_it_nearly_matched():
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_spec("slugify"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)
    r = _rpc(client, "tools/call", {"name": "zzz", "arguments": {}})
    err = r.json()["error"]
    assert "unknown tool" in err["message"]


def test_lookup_matches_a_prefix_but_not_a_coincidence_inside_a_word():
    """A near-miss list that is a coincidence is noise wearing the costume of help.

    Both halves are real: on the live shelf `unknown tool: nope` offered
    `mcp_com_snopekgames_godai` under substring matching, and a query for
    "resume" has to reach `run_resume_report` ("resume" and "run" split apart by
    the underscore). Hence tokens, with a prefix allowed from three characters
    up, and nothing else.
    """
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_spec("slugify"))
    reg.register(_spec("snopekgames_godai", description="A third-party shelf entry."))
    client = TestClient(create_app(reg), raise_server_exceptions=False)

    hit = _rpc(client, "tools/lookup", {"query": "slug"}).json()["result"]
    assert [t["name"] for t in hit["tools"]] == ["slugify"]
    # "nope" is inside "snopekgames" -- and must not count.
    miss = _rpc(client, "tools/lookup", {"query": "nope"}).json()["result"]
    assert miss["tools"] == []
    # A two-character prefix is not enough to be a match, or a shelf this size
    # answers "an" with a tenth of itself.
    assert _rpc(client, "tools/lookup",
                {"query": "sl"}).json()["result"]["tools"] == []


def test_lookup_prefers_a_name_over_prose():
    """The field a caller typed beats the field a generator wrote."""
    reg = ResourceRegistry(ResourceStore(":memory:"))
    reg.register(_spec("report"))
    client = TestClient(create_app(reg), raise_server_exceptions=False)
    res = _rpc(client, "tools/lookup", {"query": "report"}).json()["result"]
    assert res["tools"][0]["name"] == "report"
    assert res["tools"][0]["score"] == facade._NAME_WEIGHT
