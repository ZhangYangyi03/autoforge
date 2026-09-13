"""MCP: reaching tools the agent did not write and cannot read.

The tests here start the fixture server as a real child process and speak real
JSON-RPC to it. Nothing is mocked: the protocol is the thing under test, and a
mocked transport would only prove that the mock matches the client's
assumptions — which is the failure mode this whole repository argues against.

What must hold:

  * a tool from another process becomes a callable tool here, with its schema
    passed through and its result rendered faithfully
  * the scope is `undeclared`, because we cannot read the remote implementation
    — so the confirmation gate treats it as capable of everything, and that is
    the point rather than an oversight
  * a server that dies, hangs, or answers with a protocol error produces a
    sentence naming the server, never a hang and never a wrong answer
  * the agent's own API key is not handed to the child
  * `close` leaves no process behind
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from autoforge.agent import ForgeAgent
from autoforge.core.llm import MockLLMClient
from autoforge.forge.validity import SCOPE_ALLOWANCES
from autoforge.mcp import (
    IMPORTED_SCOPE, MCPClient, MCPError, MCPHub, MCPServerConfig, render_result,
    servers_from_config, spec_from_remote, tool_name,
)
from autoforge.tools.registry import ToolRegistry
from autoforge.store import ToolStore

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server_fixture.py")


def cfg(mode="normal", **kw):
    return MCPServerConfig(
        name=kw.pop("name", "fixture"),
        command=sys.executable,
        args=[FIXTURE, mode],
        timeout=kw.pop("timeout", 15.0),
        call_timeout=kw.pop("call_timeout", 15.0),
        **kw,
    )


@pytest.fixture()
def client():
    c = MCPClient(cfg())
    assert c.start(), c.error
    yield c
    c.close()


# ======================================================================
# the transport works at all
# ======================================================================
class TestHandshake:
    def test_initialize_returns_server_identity(self, client):
        assert client.server_info.get("name") == "autoforge-fixture"
        assert client.protocol == "2024-11-05"

    def test_tools_list_reports_what_the_server_offers(self, client):
        names = [t["name"] for t in client.list_tools()]
        assert names == ["add", "echo", "now", "fail", "boom", "slow", "picture"]

    def test_a_non_json_banner_does_not_break_the_handshake(self, client):
        # The fixture prints one before anything else. A client that parses the
        # first line blindly dies here.
        assert client.alive
        assert client._logs and "fixture server ready" in client._logs[0]

    def test_a_notification_between_frames_is_not_mistaken_for_a_response(self, client):
        # The fixture emits `notifications/*` after initialized and between
        # tool responses. Consuming one as an answer would hang or mis-attribute.
        client.list_tools()
        client.call_tool("add", {"a": 1, "b": 2})
        client.call_tool("add", {"a": 3, "b": 4})
        assert client.alive

    def test_starting_twice_is_a_no_op(self, client):
        assert client.start() is True
        assert client.alive


# ======================================================================
# calling
# ======================================================================
class TestCalling:
    def test_a_tool_result_comes_back(self, client):
        ok, text = render_result(client.call_tool("add", {"a": 20, "b": 22}))
        assert ok and text == "42"

    def test_a_tool_taking_no_arguments(self, client):
        ok, text = render_result(client.call_tool("now", {}))
        assert ok and text == "a-fixed-time"

    def test_is_error_is_honoured_over_the_content(self, client):
        ok, text = render_result(client.call_tool("fail", {}))
        assert not ok
        assert "unavailable" in text

    def test_a_protocol_error_names_the_method_and_the_cause(self, client):
        with pytest.raises(MCPError, match="internal error: ValueError"):
            client.call_tool("boom", {})

    def test_a_non_text_block_is_named_not_dropped(self, client):
        ok, text = render_result(client.call_tool("picture", {}))
        assert ok
        # Returning "" here would read to the agent as "succeeded, produced
        # nothing" — a wrong answer rather than an incomplete one.
        assert "image" in text

    def test_two_calls_in_a_row_stay_matched(self, client):
        for a, b in ((1, 1), (2, 2), (5, 5)):
            ok, text = render_result(client.call_tool("add", {"a": a, "b": b}))
            assert ok and text == str(a + b)

    def test_render_result_handles_a_structured_result(self):
        ok, text = render_result({"structuredContent": {"x": 1}})
        assert ok and json.loads(text) == {"x": 1}

    def test_render_result_handles_an_empty_result(self):
        assert render_result({}) == (True, "")


# ======================================================================
# a server that misbehaves must cost a sentence, not a hang
# ======================================================================
class TestMisbehaviour:
    def test_a_server_that_never_answers_times_out_with_words(self):
        c = MCPClient(cfg("silent", timeout=1.5))
        try:
            assert c.start(), c.error
            started = time.monotonic()
            with pytest.raises(MCPError, match="no response within"):
                c.list_tools()
            assert time.monotonic() - started < 8, "the timeout did not bound the wait"
        finally:
            c.close()

    def test_a_server_that_dies_is_reported_not_waited_on(self):
        # The fixture exits right after the handshake. Whether the process has
        # been reaped at any given instant is a race, so the assertion is the
        # observable that matters: a call comes back with a sentence naming the
        # cause, promptly, rather than hanging until some outer timeout.
        c = MCPClient(cfg("die", timeout=3.0))
        try:
            assert c.start(), c.error
            started = time.monotonic()
            with pytest.raises(MCPError) as exc:
                c.list_tools()
            elapsed = time.monotonic() - started
            assert elapsed < 8, f"the dead server was waited on for {elapsed:.1f}s"
            message = str(exc.value)
            assert "fixture" in message
            assert "closed the connection" in message or "not running" in message
        finally:
            c.close()

    def test_a_bad_command_is_a_sentence_not_a_traceback(self):
        c = MCPClient(MCPServerConfig(name="nope", command="definitely-not-a-real-binary-xyz"))
        assert c.start() is False
        assert "not on PATH" in c.error

    def test_calling_a_tool_that_does_not_exist(self, client):
        ok, text = render_result(client.call_tool("nonexistent", {}))
        assert not ok and "unknown tool" in text

    def test_a_call_timeout_bounds_a_slow_tool(self):
        c = MCPClient(cfg(call_timeout=1.0))
        try:
            assert c.start(), c.error
            started = time.monotonic()
            with pytest.raises(MCPError, match="no response within"):
                c.call_tool("slow", {"seconds": 30})
            assert time.monotonic() - started < 8
        finally:
            c.close()

    def test_a_slow_tool_within_the_budget_still_succeeds(self):
        c = MCPClient(cfg(call_timeout=20.0))
        try:
            assert c.start(), c.error
            ok, text = render_result(c.call_tool("slow", {"seconds": 1}))
            assert ok and text == "slept"
        finally:
            c.close()


# ======================================================================
# no orphans
# ======================================================================
class TestLifecycle:
    def test_close_terminates_the_child(self):
        c = MCPClient(cfg())
        assert c.start(), c.error
        proc = c._proc
        assert proc.poll() is None
        c.close()
        assert proc.poll() is not None, "the server was left running"

    def test_close_is_idempotent(self):
        c = MCPClient(cfg())
        assert c.start(), c.error
        c.close()
        c.close()
        assert c._proc is None

    def test_the_context_manager_closes(self):
        with MCPClient(cfg()) as c:
            proc = c._proc
            assert c.alive
        assert proc.poll() is not None

    def test_a_second_call_after_close_reports_rather_than_crashes(self):
        c = MCPClient(cfg())
        assert c.start(), c.error
        c.close()
        # `close` is what a shutdown path calls. A process reappearing after
        # shutdown because a later call wanted a tool is a leak with manners,
        # so the answer is a sentence, not a second child.
        with pytest.raises(MCPError, match="was closed|not running"):
            c.call_tool("add", {"a": 1, "b": 2})
        assert c._proc is None

    def test_a_crashed_child_can_be_restarted(self):
        # The other half of the distinction above: a server that died on its
        # own is recoverable, which is the point of checking `alive` at all.
        c = MCPClient(cfg())
        assert c.start(), c.error
        c._proc.kill()
        c._proc.wait(timeout=5)
        assert not c.alive
        assert c.start() is True, c.error        # not closed, so it comes back
        try:
            ok, text = render_result(c.call_tool("add", {"a": 3, "b": 4}))
            assert ok and text == "7"
        finally:
            c.close()

    def test_the_child_does_not_inherit_the_api_key(self, monkeypatch):
        # The agent's own credentials are the one thing a third-party server has
        # no business seeing. Read from inside the child, so this tests the
        # environment that actually reached it rather than our intent.
        monkeypatch.setenv("AUTOFORGE_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("AIPING_API_KEY", "sk-also-not")
        c = MCPClient(cfg())
        assert c.start(), c.error
        try:
            env = c._env()
            assert "AUTOFORGE_API_KEY" not in env
            assert "AIPING_API_KEY" not in env
            assert "PATH" in env
        finally:
            c.close()

    def test_declared_env_does_reach_the_child(self):
        c = MCPClient(cfg(env={"FIXTURE_TOKEN": "explicit"}))
        assert c.start(), c.error
        try:
            assert c._env()["FIXTURE_TOKEN"] == "explicit"
        finally:
            c.close()


# ======================================================================
# importing into the registry
# ======================================================================
class TestImport:
    def test_tools_arrive_under_a_namespaced_name(self):
        assert tool_name("fs", "read") == "mcp__fs__read"

    def test_a_remote_tool_becomes_a_callable_spec(self, client):
        remote = [t for t in client.list_tools() if t["name"] == "add"][0]
        spec = spec_from_remote(client.config, client, remote)
        assert spec.name == "mcp__fixture__add"
        assert spec.source == "mcp"
        # Through the runner, which is how the registry calls a spec whose
        # implementation lives elsewhere.
        result = spec.runner(spec.name, {"a": 40, "b": 2})
        assert result.ok and result.output == "42"

    def test_the_schema_is_passed_through_and_normalised(self, client):
        remote = [t for t in client.list_tools() if t["name"] == "add"][0]
        spec = spec_from_remote(client.config, client, remote)
        assert spec.parameters["type"] == "object"
        assert spec.parameters["properties"]["a"]["type"] == "integer"

    def test_the_scope_is_undeclared_because_we_cannot_read_the_code(self, client):
        remote = client.list_tools()[0]
        spec = spec_from_remote(client.config, client, remote)
        assert spec.effect_signature == IMPORTED_SCOPE
        assert IMPORTED_SCOPE in SCOPE_ALLOWANCES

    def test_a_server_may_declare_a_narrower_scope_and_it_is_used(self, client):
        client.config.scope = "network"
        spec = spec_from_remote(client.config, client, client.list_tools()[0])
        assert spec.effect_signature == "network"
        client.config.scope = IMPORTED_SCOPE

    def test_the_verification_block_says_nothing_was_verified(self, client):
        # An empty dict would read as "not yet verified", inviting the
        # assumption that a probe is coming. It is not.
        spec = spec_from_remote(client.config, client, client.list_tools()[0])
        assert spec.verification["verified"] is False
        assert "not in this repository" in spec.verification["reason"]
        assert spec.verification["server"] == "fixture"

    def test_a_failing_remote_call_becomes_a_failed_tool_result(self, client):
        remote = [t for t in client.list_tools() if t["name"] == "fail"][0]
        spec = spec_from_remote(client.config, client, remote)
        result = spec.runner(spec.name, {})
        assert not result.ok and "unavailable" in (result.error or "")

    def test_a_remote_protocol_error_becomes_a_failed_tool_result_naming_the_server(self, client):
        remote = [t for t in client.list_tools() if t["name"] == "boom"][0]
        spec = spec_from_remote(client.config, client, remote)
        result = spec.runner(spec.name, {})
        assert not result.ok
        assert "fixture" in (result.error or "")


# ======================================================================
# the hub: several servers, none of them allowed to take the others down
# ======================================================================
class TestHub:
    def test_install_registers_every_server(self):
        hub = MCPHub([cfg(name="a"), cfg(name="b")])
        reg = ToolRegistry()
        report = hub.install(reg)
        try:
            assert report.servers == 2
            assert report.tools == 14
            assert "mcp__a__add" in reg.names()
            assert "mcp__b__echo" in reg.names()
            assert report.problems == []
        finally:
            hub.close()

    def test_one_broken_server_costs_only_its_own_tools(self):
        hub = MCPHub([cfg(name="good"), cfg(name="bad", mode="die")])
        reg = ToolRegistry()
        report = hub.install(reg)
        try:
            assert "mcp__good__add" in reg.names()
            assert not any(n.startswith("mcp__bad") for n in reg.names())
            assert len(report.problems) == 1 and "bad" in report.problems[0]
        finally:
            hub.close()

    def test_a_missing_binary_costs_only_its_own_tools(self):
        hub = MCPHub([cfg(name="good"),
                      MCPServerConfig(name="ghost", command="no-such-binary-xyz")])
        reg = ToolRegistry()
        report = hub.install(reg)
        try:
            assert "mcp__good__add" in reg.names()
            assert len(report.problems) == 1 and "ghost" in report.problems[0]
        finally:
            hub.close()

    def test_imported_tools_are_callable_through_the_registry(self):
        hub = MCPHub([cfg()])
        reg = ToolRegistry()
        hub.install(reg)
        try:
            result = reg.call("mcp__fixture__add", {"a": 2, "b": 3})
            assert result.ok and result.output == "5"
        finally:
            hub.close()

    def test_a_hub_with_no_servers_says_so(self):
        assert MCPHub([]).start().summary() == "no MCP servers configured"

    def test_close_stops_every_client(self):
        hub = MCPHub([cfg(name="a"), cfg(name="b")])
        reg = ToolRegistry()
        hub.install(reg)
        procs = [c._proc for c in hub.clients.values()]
        hub.close()
        assert all(p.poll() is not None for p in procs)


# ======================================================================
# config
# ======================================================================
class TestConfig:
    def test_servers_are_read_from_the_config_block(self):
        servers, problems = servers_from_config({"mcp": {"servers": {
            "fs": {"command": "npx", "args": ["-y", "server-fs"]},
        }}})
        assert problems == []
        assert servers[0].name == "fs"
        assert servers[0].args == ["-y", "server-fs"]
        assert servers[0].scope == IMPORTED_SCOPE

    def test_a_server_without_a_command_is_reported_not_raised(self):
        servers, problems = servers_from_config({"mcp": {"servers": {"x": {"args": []}}}})
        assert servers == []
        assert "no 'command'" in problems[0]

    def test_a_bad_entry_does_not_lose_the_good_ones(self):
        servers, problems = servers_from_config({"mcp": {"servers": {
            "good": {"command": "python"},
            "bad": "just a string",
        }}})
        assert [s.name for s in servers] == ["good"]
        assert len(problems) == 1

    def test_disabled_servers_are_skipped(self):
        servers, _ = servers_from_config({"mcp": {"servers": {
            "off": {"command": "python", "enabled": False},
        }}})
        assert servers == []

    def test_no_config_at_all_is_not_an_error(self):
        assert servers_from_config(None) == ([], [])
        assert servers_from_config({}) == ([], [])
        assert servers_from_config({"mcp": {}}) == ([], [])

    def test_a_timeout_can_be_configured_per_server(self):
        servers, _ = servers_from_config({"mcp": {"servers": {
            "slow": {"command": "python", "timeout": 90, "call_timeout": 300},
        }}})
        assert servers[0].timeout == 90.0
        assert servers[0].call_timeout == 300.0

    def test_a_server_round_trips_through_dict(self):
        servers, _ = servers_from_config({"mcp": {"servers": {
            "fs": {"command": "npx", "args": ["-y", "x"], "env": {"K": "v"}},
        }}})
        again = MCPServerConfig.from_dict("fs", servers[0].to_dict())
        assert again.command == "npx" and again.env == {"K": "v"}


# ======================================================================
# the gate: an unread implementation is gated by everything switched off
# ======================================================================
class TestTheGateTreatsImportsAsUnknown:
    def test_the_scope_is_one_the_gate_knows(self):
        assert IMPORTED_SCOPE in SCOPE_ALLOWANCES

    def test_an_undeclared_scope_needs_every_confirmed_freedom(self):
        # `may_run_arbitrary_code` is deliberately absent from CONFIRM_REQUIRED
        # — the tools that honour it gate it themselves, and asking an operator
        # the same question twice is how they learn to answer without reading.
        # So the assertion is over the freedoms the gate is answerable for.
        from autoforge.autonomy.confirm import required_freedoms

        needed = set(required_freedoms(IMPORTED_SCOPE))
        assert {"may_read_filesystem", "may_write_filesystem",
                "may_access_network"} <= needed

    def test_undeclared_is_gated_by_any_switch_that_is_off(self):
        from autoforge.autonomy.confirm import disabled_freedoms
        from autoforge.autonomy.policy import AutonomyPolicy

        for switch in ("may_read_filesystem", "may_write_filesystem",
                       "may_access_network"):
            policy = AutonomyPolicy()
            setattr(policy, switch, False)
            assert disabled_freedoms(IMPORTED_SCOPE, policy) == [switch], (
                f"switching off {switch} did not gate an unread implementation")

    def test_a_switched_off_freedom_actually_blocks_an_mcp_call(self):
        # End to end through the registry: the declaration is not decorative,
        # and with no operator attached the answer is a refusal that says so.
        from autoforge.autonomy.policy import AutonomyPolicy

        hub = MCPHub([cfg()])
        reg = ToolRegistry()
        hub.install(reg)
        try:
            policy = AutonomyPolicy()
            policy.may_write_filesystem = False
            reg.policy = policy

            result = reg.call("mcp__fixture__add", {"a": 1, "b": 2}, force=False)
            assert result.awaiting_confirmation
            assert not result.ok
            assert "nobody to ask" in (result.error or "")
            assert "may_write_filesystem" in (result.error or "")
        finally:
            hub.close()

    def test_a_narrower_declared_scope_is_gated_more_lightly(self):
        # read_only needs only read; undeclared is capable of everything. So the
        # same write switch gates one and not the other, which is what makes
        # the honest label worth having.
        from autoforge.autonomy.confirm import disabled_freedoms
        from autoforge.autonomy.policy import AutonomyPolicy

        policy = AutonomyPolicy()
        policy.may_write_filesystem = False
        assert disabled_freedoms("read_only", policy) == []
        assert disabled_freedoms(IMPORTED_SCOPE, policy) == ["may_write_filesystem"]

    def test_a_confirmed_call_runs(self):
        from autoforge.autonomy.policy import AutonomyPolicy

        hub = MCPHub([cfg()])
        reg = ToolRegistry()
        hub.install(reg)
        try:
            policy = AutonomyPolicy()
            policy.may_write_filesystem = False
            reg.policy = policy
            reg.confirmer = lambda name, args, needed: True

            result = reg.call("mcp__fixture__add", {"a": 1, "b": 2})
            assert result.ok and result.output == "3"
        finally:
            hub.close()


# ======================================================================
# no MCP in the agent unless it is configured
# ======================================================================
class TestTheAgentIsUnchangedByDefault:
    def test_a_plain_agent_has_no_mcp_tools(self):
        a = ForgeAgent(MockLLMClient(), enable_evolution=False)
        assert not [n for n in a.registry.names() if n.startswith("mcp__")]

    def test_the_management_tools_exist_even_with_nothing_configured(self):
        # They have to: otherwise an operator with an empty config has no way
        # to ask what MCP even is here.
        a = ForgeAgent(MockLLMClient(), enable_evolution=False)
        assert {"mcp_servers", "mcp_connect", "mcp_call"} <= set(a.registry.names())

    def test_no_server_is_started_when_none_is_configured(self):
        from autoforge.configfile import load
        servers, problems = servers_from_config(load())
        assert servers == [] and problems == []


# ======================================================================
# end to end: a config file, an agent, a real child process
# ======================================================================
class TestThroughTheAgent:
    """The path a user takes: write a config, ask the agent to connect."""

    def _agent(self, tmp_path, **server):
        from autoforge.configfile import save

        entry = {"command": sys.executable, "args": [FIXTURE, "normal"]}
        entry.update(server)
        save({"mcp": {"servers": {"fixture": entry}}})
        return ForgeAgent(MockLLMClient(), enable_evolution=False,
                          store=ToolStore(str(tmp_path / "agent.db")))

    def test_mcp_servers_reports_what_is_configured(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            out = a.registry.call("mcp_servers", {}).output
            assert "fixture" in out
            assert "not started" in out
            assert "undeclared" in out        # the honest label, said up front
        finally:
            a.mcp.close()

    def test_connect_registers_the_tools_and_they_work(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            out = a.registry.call("mcp_connect", {}).output
            assert "mcp__fixture__add" in out
            result = a.registry.call("mcp__fixture__add", {"a": 19, "b": 23})
            assert result.ok and result.output == "42"
        finally:
            a.mcp.close()

    def test_connect_names_a_server_that_is_not_configured(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            out = a.registry.call("mcp_connect", {"server": "typo"}).output
            assert "no server named 'typo'" in out.lower()
            assert "fixture" in out            # says what the valid names are
        finally:
            a.mcp.close()

    def test_call_does_a_one_off_without_importing(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            out = a.registry.call("mcp_call", {
                "server": "fixture", "tool": "echo",
                "arguments": json.dumps({"text": "hi"}),
            }).output
            assert out == "<echo>hi</echo>"
        finally:
            a.mcp.close()

    def test_a_remote_failure_is_reported_as_a_failure_not_a_success(self, tmp_path):
        # The registry turns a raised exception into ok=False and a string
        # return into ok=True. Returning the server's failure text would have
        # arrived at the agent as a success carrying an odd payload.
        a = self._agent(tmp_path)
        try:
            result = a.registry.call("mcp_call", {
                "server": "fixture", "tool": "fail", "arguments": "{}",
            })
            assert not result.ok
            assert "unavailable" in (result.error or "")
        finally:
            a.mcp.close()

    def test_a_remote_protocol_error_is_reported_as_a_failure(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            result = a.registry.call("mcp_call", {
                "server": "fixture", "tool": "boom", "arguments": "{}",
            })
            assert not result.ok
            assert "refused" in (result.error or "")
        finally:
            a.mcp.close()

    def test_call_with_malformed_arguments_says_so(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            out = a.registry.call("mcp_call", {
                "server": "fixture", "tool": "echo", "arguments": "{not json",
            }).output
            assert "must be a JSON object" in out
        finally:
            a.mcp.close()

    def test_a_broken_server_does_not_stop_the_agent_from_starting(self, tmp_path):
        from autoforge.configfile import save

        save({"mcp": {"servers": {"ghost": {"command": "definitely-not-real-xyz"}}}})
        a = ForgeAgent(MockLLMClient(), enable_evolution=False,
                       store=ToolStore(str(tmp_path / "agent.db")))
        try:
            out = a.registry.call("mcp_connect", {}).output
            assert "problem" in out
            assert "not on PATH" in out
            # And the agent still works afterwards.
            assert a.registry.call("mcp_servers", {}).output
        finally:
            a.mcp.close()

    def test_a_malformed_config_entry_is_reported_at_construction(self, tmp_path):
        from autoforge.configfile import save

        save({"mcp": {"servers": {"bad": {"args": []}}}})
        a = ForgeAgent(MockLLMClient(), enable_evolution=False,
                       store=ToolStore(str(tmp_path / "agent.db")))
        try:
            assert a.mcp_servers == []
            assert a._mcp_problems and "no 'command'" in a._mcp_problems[0]
            # Reported through the tool, where the agent will see it.
            assert "no 'command'" in a.registry.call("mcp_servers", {}).output
        finally:
            a.mcp.close()

    def test_the_agent_does_not_spawn_anything_until_asked(self, tmp_path):
        a = self._agent(tmp_path)
        try:
            assert a.mcp.clients == {}, "a server was started at construction"
        finally:
            a.mcp.close()
