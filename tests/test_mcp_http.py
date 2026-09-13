"""MCP over streamable HTTP: reaching a server that is already running.

The stdio transport is covered in `test_mcp.py`. This is the other one, and it
exists because the configs this framework imports are full of servers that are
not child processes at all — a hosted MCP endpoint, a company gateway, an
OAuth-protected service. Those cannot be started here, and refusing to read
them would mean "compatible with the other agents on this machine" was true
only for the subset that happens to use stdio.

The fake server is a real HTTP server on a real socket, so what is exercised is
the transport rather than a mock of it. Nothing is stubbed: the request that
leaves here is an HTTP request, and the reply is parsed from bytes.

What must hold:

  * a JSON reply and an SSE reply are both answers, because the server picks
  * a notification arriving before the reply is skipped, not mistaken for it
  * the session id handed out at initialize is sent back on later requests
  * a bearer token is read from the environment at send time and never stored
  * every failure — 5xx, unreachable, no answer — is a sentence naming the
    server, never a hang and never a traceback
  * `close` tells the server the session is over, and does not raise if it
    would rather not hear it
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autoforge.mcp import (
    MCPClient, MCPError, MCPHttpClient, MCPHub, MCPServerConfig, client_for,
)


@pytest.fixture(autouse=True)
def no_ambient_proxies(monkeypatch):
    """Keep a machine-wide proxy from standing between these tests and
    localhost. Without this the suite would pass or fail depending on the
    environment it was run in, which is not a property a test should have."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def cfg(url: str, **kw) -> MCPServerConfig:
    return MCPServerConfig(name=kw.pop("name", "remote"), url=url, **kw)


class FakeMCPHttp:
    """A streamable-HTTP MCP server, on a real socket, that answers by the book."""

    def __init__(self, *, style: str = "json", session: str = "sess-1",
                 tools: list[dict] | None = None, call_result: dict | None = None,
                 handshake_status: int = 200, tools_status: int = 200,
                 answer_nothing: bool = False,
                 error_methods: dict[str, str] | None = None) -> None:
        self.style = style
        self.session = session
        self.tools = tools if tools is not None else [
            {"name": "echo", "description": "say it back",
             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
        ]
        self.call_result = call_result or {
            "content": [{"type": "text", "text": "ok"}]}
        self.handshake_status = handshake_status
        self.tools_status = tools_status
        self.answer_nothing = answer_nothing
        #: method name -> message, answered as a JSON-RPC *error* rather than a
        #: result. A protocol-level error and an HTTP error are different
        #: failures and both have to read as a sentence.
        self.error_methods = dict(error_methods or {})
        self.requests: list[tuple[dict, dict]] = []   # (headers, body)
        self.deleted: list[dict] = []
        self._server: ThreadingHTTPServer | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):    # the suite's output stays readable
                pass

            def _read(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    return json.loads(raw or b"{}")
                except ValueError:
                    return {}

            def do_POST(self):
                msg = self._read()
                outer.requests.append((dict(self.headers), msg))
                outer._answer(self, msg)

            def do_DELETE(self):
                outer.deleted.append(dict(self.headers))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._handler = Handler

    def __enter__(self) -> "FakeMCPHttp":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def headers_sent_for(self, method: str) -> dict:
        for headers, body in self.requests:
            if body.get("method") == method:
                return headers
        raise AssertionError(f"no {method} request was received")

    # -- the server side -------------------------------------------------
    def _answer(self, h: BaseHTTPRequestHandler, msg: dict) -> None:
        method, rid = msg.get("method"), msg.get("id")

        if method == "notifications/initialized":
            # A notification has no reply. 202 with no body is the shape the
            # transport specifies for exactly this.
            h.send_response(202)
            if self.session:
                h.send_header("Mcp-Session-Id", self.session)
            h.send_header("Content-Length", "0")
            h.end_headers()
            return

        session = None
        status = 200
        if method == "initialize":
            status = self.handshake_status
            session = self.session
            result = {"protocolVersion": "2024-11-05",
                      "serverInfo": {"name": "fake-remote", "version": "9"},
                      "capabilities": {}}
        elif method == "tools/list":
            status = self.tools_status
            result = {"tools": self.tools}
        elif method == "tools/call":
            result = self.call_result
        else:
            result = {"value": None}

        if status >= 400:
            self._send(h, status, b"upstream exploded", "text/plain", session)
            return
        if method in self.error_methods:
            err = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": self.error_methods[method]}})
            self._send(h, 200, err.encode(), "application/json", session)
            return
        if self.answer_nothing:
            # A server that takes the request and never replies. The client must
            # bound this by its own timeout rather than hanging with it.
            self._send(h, 200, b"", "application/json", session)
            return
        payload = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
        if self.style == "sse":
            prior = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress",
                                "params": {"percent": 50}})
            body = (f"event: message\ndata: {prior}\n\n"
                    f"event: message\ndata: {payload}\n\n").encode()
            self._send(h, status, body, "text/event-stream", session)
        else:
            self._send(h, status, payload.encode(), "application/json", session)

    def _send(self, h: BaseHTTPRequestHandler, status: int, body: bytes,
              ctype: str, session: str | None) -> None:
        h.send_response(status)
        h.send_header("Content-Type", ctype)
        if session:
            h.send_header("Mcp-Session-Id", session)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        if body:
            h.wfile.write(body)


# ----------------------------------------------------------------------
# the config reads both transports
# ----------------------------------------------------------------------
class TestConfigTransport:

    def test_a_url_config_is_the_http_transport(self):
        c = MCPServerConfig.from_dict("remote", {"url": "https://mcp.example/mcp"})
        assert c.transport == "http"
        assert c.command == ""

    def test_a_command_config_is_stdio(self):
        c = MCPServerConfig.from_dict("local", {"command": "npx", "args": ["-y", "x"]})
        assert c.transport == "stdio"

    def test_every_dialects_timeout_name_is_read(self):
        """Hermes writes `connect_timeout`, Codex writes `startup_timeout_sec`,
        the local file writes `timeout` — all three mean the same thing."""
        for key in ("timeout", "connect_timeout", "startup_timeout_sec"):
            c = MCPServerConfig.from_dict("s", {"command": "x", key: 42})
            assert c.timeout == 42, key
        for key in ("call_timeout", "tool_timeout_sec"):
            c = MCPServerConfig.from_dict("s", {"command": "x", key: 7})
            assert c.call_timeout == 7, key

    def test_a_timeout_that_is_not_a_number_is_refused_with_the_field_named(self):
        with pytest.raises(MCPError, match="timeout.*not a number"):
            MCPServerConfig.from_dict("s", {"command": "x", "timeout": "soon"})

    def test_headers_and_bearer_variable_survive_a_round_trip(self):
        c = MCPServerConfig.from_dict("s", {
            "url": "https://x/mcp", "headers": {"X-Org": "acme"},
            "bearer_token_env_var": "MY_TOKEN"})
        assert c.headers == {"X-Org": "acme"}
        assert c.bearer_token_env_var == "MY_TOKEN"
        assert MCPServerConfig.from_dict("s", c.to_dict()).to_dict() == c.to_dict()

    def test_the_factory_picks_the_client_the_config_describes(self):
        assert isinstance(client_for(MCPServerConfig(name="a", command="x")), MCPClient)
        assert isinstance(client_for(MCPServerConfig(name="b", url="http://x/mcp")),
                          MCPHttpClient)

    def test_enabled_false_written_as_a_string_still_disables(self):
        """A config that switches a server off with "false" must not enable it."""
        c = MCPServerConfig.from_dict("s", {"command": "x", "enabled": "false"})
        assert c.enabled is False


# ----------------------------------------------------------------------
# the handshake
# ----------------------------------------------------------------------
class TestHttpHandshake:

    def test_initialize_over_a_json_reply(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            assert client.start() is True
            assert client.server_info["name"] == "fake-remote"
            assert client.alive is True
            client.close()

    def test_initialize_read_past_a_notification_on_an_sse_stream(self):
        """The server picks SSE; the answer is behind an unrelated event."""
        with FakeMCPHttp(style="sse") as server:
            client = MCPHttpClient(cfg(server.url))
            assert client.start() is True
            assert client.protocol == "2024-11-05"
            client.close()

    def test_the_session_id_from_initialize_is_sent_back(self):
        with FakeMCPHttp(session="abc123") as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            client.list_tools()
            sent = server.headers_sent_for("tools/list")
            assert sent.get("Mcp-Session-Id") == "abc123"
            client.close()

    def test_no_session_id_means_no_header(self):
        with FakeMCPHttp(session="") as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            client.list_tools()
            assert "Mcp-Session-Id" not in server.headers_sent_for("tools/list")
            client.close()

    def test_both_response_types_are_acceptable_to_the_server(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            accept = server.headers_sent_for("initialize").get("Accept", "")
            assert "application/json" in accept and "text/event-stream" in accept
            client.close()

    def test_a_notification_is_sent_as_its_own_request_with_no_id(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            sent = [b for _, b in server.requests
                    if b.get("method") == "notifications/initialized"]
            assert sent and "id" not in sent[0]
            client.close()


# ----------------------------------------------------------------------
# calling
# ----------------------------------------------------------------------
class TestHttpCalling:

    def test_tools_list_reports_what_the_server_offers(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            tools = client.list_tools()
            assert [t["name"] for t in tools] == ["echo"]
            assert tools[0]["inputSchema"]["type"] == "object"
            client.close()

    def test_a_tool_call_round_trips(self):
        with FakeMCPHttp(call_result={"content": [
            {"type": "text", "text": "hello"}]}) as server:
            client = MCPHttpClient(cfg(server.url))
            result = client.call_tool("echo", {"text": "hi"})
            assert result["content"][0]["text"] == "hello"
            sent = server.headers_sent_for("tools/call")
            assert sent.get("Content-Type") == "application/json"
            client.close()

    def test_a_tool_call_over_sse_round_trips(self):
        with FakeMCPHttp(style="sse") as server:
            client = MCPHttpClient(cfg(server.url))
            result = client.call_tool("echo", {})
            assert result["content"][0]["text"] == "ok"
            client.close()

    def test_a_protocol_error_names_the_method_and_the_cause(self):
        """A JSON-RPC error in the body is not an HTTP failure, and the message
        has to carry both what was called and what the server said."""
        with FakeMCPHttp(error_methods={"tools/list": "no such method"}) as server:
            client = MCPHttpClient(cfg(server.url, name="remote"))
            with pytest.raises(MCPError, match="tools/list: no such method"):
                client.list_tools()
            client.close()

    def test_a_body_that_is_not_json_is_a_sentence(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url, name="remote"))
            client.start()

            class Lying:
                """A 200 whose body is not JSON — a misconfigured proxy, most
                often. It must read as a sentence naming the server."""
                status_code = 200
                content = b"<html>hello</html>"
                headers = {"Content-Type": "text/html"}

                def json(self):
                    raise ValueError("Expecting value: line 1 column 1")

                def close(self):
                    pass
            with pytest.raises(MCPError, match="remote.*not.*JSON"):
                client._decode(Lying(), 1)
            client.close()


# ----------------------------------------------------------------------
# misbehaviour
# ----------------------------------------------------------------------
class TestHttpMisbehaviour:

    def test_a_5xx_is_a_sentence_naming_the_server(self):
        with FakeMCPHttp(handshake_status=503) as server:
            client = MCPHttpClient(cfg(server.url, name="gateway"))
            assert client.start() is False
            assert "gateway" in client.error and "503" in client.error

    def test_a_5xx_from_tools_list_names_the_server(self):
        with FakeMCPHttp(tools_status=500) as server:
            client = MCPHttpClient(cfg(server.url, name="gateway"))
            with pytest.raises(MCPError, match="gateway.*500"):
                client.list_tools()
            client.close()

    def test_an_unreachable_server_is_a_sentence_not_a_traceback(self):
        client = MCPHttpClient(cfg("http://127.0.0.1:9/mcp", name="dead"))
        assert client.start() is False
        assert "dead" in client.error and "unreachable" in client.error

    def test_a_server_that_never_answers_times_out_with_words(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url, timeout=0.5))
            client.start()
            # `answer_nothing` is turned on after the handshake so the timeout
            # being measured is the one on the call, not on startup.
            server.answer_nothing = True
            with pytest.raises(MCPError, match="remote"):
                client.list_tools(timeout=0.5)
            client.close()

    def test_calling_a_server_that_was_never_started_starts_it(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            tools = client.list_tools()          # no explicit start()
            assert tools[0]["name"] == "echo"
            client.close()


# ----------------------------------------------------------------------
# secrets and lifecycle
# ----------------------------------------------------------------------
class TestHttpSecurityAndLifecycle:

    def test_a_bearer_token_is_read_from_the_environment_at_send_time(
            self, monkeypatch):
        monkeypatch.setenv("MY_MCP_TOKEN", "s3cret")
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url, bearer_token_env_var="MY_MCP_TOKEN"))
            client.start()
            assert (server.headers_sent_for("initialize")
                    .get("Authorization") == "Bearer s3cret")
            client.close()

    def test_an_unset_token_variable_sends_no_authorization_header(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url,
                                       bearer_token_env_var="NOT_SET_ANYWHERE"))
            client.start()
            assert "Authorization" not in server.headers_sent_for("initialize")
            client.close()

    def test_configured_headers_are_sent(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url, headers={"X-Org": "acme"}))
            client.start()
            assert server.headers_sent_for("initialize").get("X-Org") == "acme"
            client.close()

    def test_the_token_is_not_stored_on_the_client(self, monkeypatch):
        """The value stays in the environment; the config only names it."""
        monkeypatch.setenv("MY_MCP_TOKEN", "s3cret")
        c = cfg("http://127.0.0.1:1/mcp", bearer_token_env_var="MY_MCP_TOKEN")
        assert "s3cret" not in json.dumps(c.to_dict())

    def test_close_tells_the_server_the_session_ended(self):
        with FakeMCPHttp(session="abc123") as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            client.close()
            assert server.deleted and server.deleted[0].get("Mcp-Session-Id") == "abc123"

    def test_close_is_idempotent_and_does_not_raise(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            client.close()
            client.close()               # a second close must be a no-op
            assert client.alive is False

    def test_a_closed_client_does_not_restart(self):
        with FakeMCPHttp() as server:
            client = MCPHttpClient(cfg(server.url))
            client.start()
            client.close()
            assert client.start() is False
            assert "closed" in client.error

    def test_the_context_manager_closes(self):
        with FakeMCPHttp() as server:
            with MCPHttpClient(cfg(server.url)) as client:
                assert client.alive is True
            assert client.alive is False


# ----------------------------------------------------------------------
# through the hub
# ----------------------------------------------------------------------
class TestThroughTheHub:

    def test_the_hub_imports_tools_from_an_http_server(self):
        with FakeMCPHttp() as server:
            hub = MCPHub([cfg(server.url, name="remote")])
            report = hub.start()
            try:
                assert report.servers == 1
                assert report.imported == ["mcp__remote__echo"]
                assert report.problems == []
            finally:
                hub.close()

    def test_a_server_that_will_not_start_costs_only_its_own_tools(self):
        with FakeMCPHttp() as server:
            hub = MCPHub([cfg("http://127.0.0.1:9/mcp", name="dead"),
                          cfg(server.url, name="live")])
            report = hub.start()
            try:
                assert report.servers == 1
                assert report.imported == ["mcp__live__echo"]
                assert len(report.problems) == 1 and "dead" in report.problems[0]
            finally:
                hub.close()

    def test_a_mix_of_transports_imports_from_both(self):
        """The point of the whole exercise: one config, two kinds of server."""
        stdio = MCPServerConfig(
            name="fixture",
            command=os.sys.executable,
            args=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "mcp_server_fixture.py")])
        with FakeMCPHttp() as server:
            hub = MCPHub([stdio, cfg(server.url, name="remote")])
            report = hub.start()
            try:
                assert report.servers == 2
                assert "mcp__remote__echo" in report.imported
                assert any(name.startswith("mcp__fixture__") for name in report.imported)
            finally:
                hub.close()


# ----------------------------------------------------------------------
# tool gating, as the other agents' configs write it
# ----------------------------------------------------------------------
THREE_TOOLS = [
    {"name": "read", "inputSchema": {"type": "object"}},
    {"name": "write", "inputSchema": {"type": "object"}},
    {"name": "delete", "inputSchema": {"type": "object"}},
]


class TestToolGating:
    """`enabled_tools` / `disabled_tools` are honoured rather than ignored.

    Importing a server out of someone else's config and exposing every tool it
    happens to offer is not compatibility — it is quietly undoing the choices
    they made. These tests pin that the choices survive the import.
    """

    def test_enabled_tools_is_an_allowlist(self):
        c = MCPServerConfig.from_dict("s", {
            "url": "http://x/mcp", "enabled_tools": ["read"]})
        assert c.allows("read") is True
        assert c.allows("write") is False

    def test_an_empty_allowlist_means_every_tool(self):
        c = MCPServerConfig.from_dict("s", {"url": "http://x/mcp"})
        assert all(c.allows(t) for t in ("read", "write", "delete"))

    def test_disabled_tools_is_a_denylist(self):
        c = MCPServerConfig.from_dict("s", {
            "url": "http://x/mcp", "disabled_tools": ["delete"]})
        assert c.allows("read") is True
        assert c.allows("delete") is False

    def test_the_denylist_wins_over_the_allowlist(self):
        """If both names are listed, the operator's refusal is the one that
        counts — reading it the other way would run something they named."""
        c = MCPServerConfig.from_dict("s", {
            "url": "http://x/mcp",
            "enabled_tools": ["read", "delete"], "disabled_tools": ["delete"]})
        assert c.allows("read") is True
        assert c.allows("delete") is False

    def test_a_gated_out_tool_is_absent_from_the_registry(self):
        with FakeMCPHttp(tools=THREE_TOOLS) as server:
            hub = MCPHub([cfg(server.url, name="remote", enabled_tools=["read"])])
            report = hub.start()
            try:
                assert report.imported == ["mcp__remote__read"]
                assert report.tools == 1
            finally:
                hub.close()

    def test_a_tool_that_was_switched_off_is_named_in_the_report(self):
        """Silently importing fewer tools than the server offers is the failure
        mode this replaces: the operator reads why, in the log they already see."""
        with FakeMCPHttp(tools=THREE_TOOLS) as server:
            hub = MCPHub([cfg(server.url, name="remote", disabled_tools=["delete"])])
            report = hub.start()
            try:
                assert "mcp__remote__delete" not in report.imported
                said = " ".join(report.problems)
                assert "delete" in said and "not imported" in said
            finally:
                hub.close()

    def test_the_clients_own_tool_list_is_pruned_too(self):
        """The client's own list is pruned, so nothing downstream can call a
        tool the config refused — `install` walks `client.tools`."""
        with FakeMCPHttp(tools=THREE_TOOLS) as server:
            hub = MCPHub([cfg(server.url, name="remote", enabled_tools=["read"])])
            hub.start()
            try:
                assert [t["name"] for t in hub.clients["remote"].tools] == ["read"]
            finally:
                hub.close()

    def test_gating_matches_the_short_name_not_the_namespaced_one(self):
        """The config was written before this framework prefixed anything, so
        `read` is what it says — not `mcp__remote__read`."""
        with FakeMCPHttp(tools=THREE_TOOLS) as server:
            cfg_ = cfg(server.url, name="remote", enabled_tools=["mcp__remote__read"])
            hub = MCPHub([cfg_])
            report = hub.start()
            try:
                assert report.imported == []
            finally:
                hub.close()

    def test_a_gating_list_that_is_not_a_list_is_refused_with_the_field_named(self):
        with pytest.raises(MCPError, match="enabled_tools.*must be a list"):
            MCPServerConfig.from_dict("s", {"url": "http://x/mcp", "enabled_tools": "read"})

    def test_gating_survives_a_round_trip(self):
        c = MCPServerConfig.from_dict("s", {
            "url": "http://x/mcp",
            "enabled_tools": ["read"], "disabled_tools": ["delete"]})
        again = MCPServerConfig.from_dict("s", c.to_dict())
        assert again.enabled_tools == ["read"] and again.disabled_tools == ["delete"]
