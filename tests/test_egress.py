"""The egress gateway — a boundary that refuses, records, and admits what it covers.

The tests that matter are the refusals and the coverage claim. A gateway that
only ever said yes would pass a happy-path test, so every check here is paired
with the case it must not admit, and `coverage()` is asserted to name the call
sites it does *not* route — a boundary whose documentation lists only its
victories invites the reader to assume the rest is inside it too.
"""
from __future__ import annotations

import pytest

from autoforge import audit, egress
from autoforge.egress import Gateway, GatewayDenied


@pytest.fixture(autouse=True)
def clean_gateway(monkeypatch):
    monkeypatch.delenv("AUTOFORGE_EGRESS", raising=False)
    monkeypatch.delenv("AUTOFORGE_EGRESS_ALLOW", raising=False)
    egress.reset()
    yield
    egress.reset()


def closed(*hosts):
    return Gateway(allow=hosts or egress.DEFAULT_ALLOW, mode="closed")


# -- opening the door ----------------------------------------------------

def test_open_mode_admits_anything():
    g = Gateway(mode="open")
    assert g.check("https://anything.example/x")["allowed"]


def test_closed_mode_is_closed():
    g = closed("api.aiping.cn")
    assert g.check("https://api.aiping.cn/v1")["allowed"]
    assert not g.check("https://google.com/")["allowed"]
    assert g.check("https://google.com/")["rule"] == "egress.deny: no allow rule"


def test_loopback_stays_reachable_when_closed():
    """The market, the MCP servers and the bus are all on 127.0.0.1.

    A boundary that cut them off would be measuring the wrong thing.
    """
    g = closed()
    for url in ("http://127.0.0.1:8000/resources", "http://localhost:8000/",
                "http://[::1]:8000/"):
        assert g.check(url)["allowed"], url


def test_port_and_userinfo_do_not_confuse_the_host():
    g = closed("api.aiping.cn")
    assert g.check("https://api.aiping.cn:8443/v1")["allowed"]
    assert not g.check("https://api.aiping.cn@evil.test/")["allowed"]
    assert egress._host_of("https://api.aiping.cn@evil.test/") == "evil.test"
    assert egress._host_of("https://api.aiping.cn:8443/v1") == "api.aiping.cn"


def test_wildcard_does_not_admit_an_address():
    """`*.example.com` must not be satisfiable by an IP literal."""
    g = closed("*.example.com")
    assert g.check("https://x.example.com/")["allowed"]
    assert not g.check("https://93.184.216.34/")["allowed"]


def test_an_empty_allow_list_refuses(tmp_path):
    assert not Gateway(allow=[], mode="closed").check("https://x/")["allowed"]
    assert not Gateway(allow=[], mode="open").check("https://x/")["allowed"]
    # ...and the two cases are distinguishable, because they mean different things
    assert Gateway(allow=[], mode="open").check("https://x/")["mode"] == "open"
    assert Gateway(allow=[], mode="closed").check("https://x/")["mode"] == "closed"


def test_admit_raises_with_the_rule_in_the_message():
    g = closed("api.aiping.cn")
    with pytest.raises(GatewayDenied) as info:
        g.admit("https://google.com/")
    assert info.value.rule == "egress.deny: no allow rule"
    assert "google.com" in str(info.value)


def test_an_unreadable_mode_is_not_a_silently_closed_door(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_EGRESS", "yes-please")
    assert Gateway().closed is False


def test_env_can_close_and_list(monkeypatch):
    monkeypatch.setenv("AUTOFORGE_EGRESS", "closed")
    monkeypatch.setenv("AUTOFORGE_EGRESS_ALLOW", "only.test, *.example.com")
    g = Gateway()
    assert g.check("https://only.test/")["allowed"]
    assert g.check("https://a.example.com/")["allowed"]
    assert not g.check("https://other.test/")["allowed"]


# -- the record ----------------------------------------------------------

def test_admissions_and_refusals_land_on_the_chain(tmp_path):
    from autoforge.store import ToolStore
    st = ToolStore(str(tmp_path / "chain.db"))
    try:
        egress.reset(conn=st._conn)
        g = egress.gateway()
        g.allow, g.mode = ("api.aiping.cn",), "closed"
        g.admit("https://api.aiping.cn/v1")
        with pytest.raises(GatewayDenied):
            g.admit("https://google.com/")
        from autoforge import chaining
        assert chaining.verify_chain(st._conn).get("ok")
        rows = audit._rows(st._conn)
        assert [r["event_type"] for r in rows] == ["network_allow", "network_deny"]
        assert rows[0]["resource"] == "api.aiping.cn"
        assert rows[1]["rule"] == "egress.deny: no allow rule"
    finally:
        st.close()


def test_a_broken_ledger_does_not_eat_the_call(tmp_path, monkeypatch):
    """Losing the log is bad. Losing the work because the log is bad is worse."""
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("database is locked")
        def commit(self): pass
        def rollback(self): pass

    g = Gateway(allow=("api.aiping.cn",), mode="closed", conn=Broken())
    receipt = g.admit("https://api.aiping.cn/v1")
    assert receipt["allowed"] and receipt["recorded"]["recorded"] is False


# -- the claim -----------------------------------------------------------

def test_coverage_names_what_it_does_not_route():
    cov = egress.coverage()
    assert "mcp._post" in cov["covered_sites"]
    assert "webtools.safe_redirects" in cov["covered_sites"]
    # The omission is a fact to report, not a thing to hide.
    assert "llm.chat" in cov["known_unrouted"]
    assert "browser" in cov["known_unrouted"]
    assert "not stopped by this" in cov["not_covered"]


def test_the_two_routed_sites_really_call_admit():
    """Read the source: a table that says "covered" and a call site that does not
    is how documentation drifts away from behaviour."""
    import inspect
    from autoforge import mcp, webtools
    assert "egress.admit(" in inspect.getsource(mcp.MCPHttpClient._post)
    assert "egress.admit(" in inspect.getsource(webtools.safe_redirects)


def test_mcp_refusal_surfaces_as_mcperror(monkeypatch):
    """One error path, not two: a caller catching GatewayDenied *and* MCPError
    would be handling the same problem twice."""
    from autoforge import mcp
    from autoforge.mcp import MCPError, MCPHttpClient, MCPServerConfig
    egress.reset(allow=("allowed.test",), mode="closed")

    server = MCPHttpClient(MCPServerConfig(name="vendor",
                                           url="https://vendor.test/mcp"))
    with pytest.raises(MCPError) as info:
        server._post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, 5.0)
    assert "egress refused" in str(info.value)
    # and it never touched the network: no session, no reply, just the refusal
    assert server._session_id in ("", None)
