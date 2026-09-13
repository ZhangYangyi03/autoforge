"""The web harness: event normalisation, sessions, and the HTTP surface.

Everything here runs against a canned LLM, so no test touches the network.
The HTTP tests start a real server on an ephemeral port and talk to it with
urllib — the same path the browser takes.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from autoforge.core.llm import LLMClient, LLMResponse
from autoforge.core.message import ToolCall
from autoforge.modes import MinimalAgent
from autoforge.web import server as web

# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class Scripted(LLMClient):
    """Hands back pre-built responses, in order. No network, no parsing."""

    name = "scripted"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.seen: list[list] = []

    def chat(self, messages, tools=None, **kwargs):        # noqa: ANN001, ANN003
        self.seen.append(list(messages))
        if self.script:
            return self.script.pop(0)
        return LLMResponse(content="(no more scripted replies)")


def echo_then_done(command: str = "echo hi"):
    return Scripted([
        LLMResponse(content="", tool_calls=[ToolCall("c1", "bash", {"command": command})]),
        LLMResponse(content="ran it", tool_calls=[ToolCall("c2", "terminate",
                                                          {"summary": "done"})]),
    ])


def minimal_factory(cfg, mode="minimal"):
    return MinimalAgent(llm=echo_then_done())


def wait_idle(session: web.Session, timeout: float = 20.0) -> None:
    end = time.time() + timeout
    while time.time() < end and session.busy:
        time.sleep(0.02)
    assert not session.busy, "session never went idle"


# --------------------------------------------------------------------------
# event normalisation
# --------------------------------------------------------------------------

_KINDS = ["call", "result", "finish", "forge_attempt", "forge_done",
          "forge_error", "auto_quarantine", "evolve", "spawn", "amend"]


@pytest.mark.parametrize("kind", _KINDS)
def test_every_known_kind_maps_to_a_source(kind):
    ev = web._normalise({"kind": kind, "t": 1.0})
    assert ev["source"] != kind, f"{kind!r} fell through to its own name"
    assert ev["source"] in web._SOURCE_OF_KIND.values() or ev["source"] == "event"
    assert isinstance(ev["text"], str) and ev["text"]


def test_unknown_kind_is_kept_not_dropped():
    ev = web._normalise({"kind": "brand_new", "text": "hello"})
    assert ev["source"] == "brand_new"
    assert ev["text"] == "hello"


def test_summaries_name_the_thing_that_happened():
    assert "bash" in web._summarise("call", {"tool": "bash", "args": {"a": 1}})
    assert "ok" in web._summarise("result", {"tool": "bash", "ok": True})
    assert "failed" in web._summarise("result", {"tool": "bash", "ok": False})
    assert "self-terminated=True" in web._summarise(
        "finish", {"turns": 2, "tools": ["bash"], "self_terminated": True})
    assert "sealed" in web._summarise("forge_done", {"ok": True, "rounds": 1})
    assert "rejected" in web._summarise("forge_done", {"ok": False, "rounds": 2})
    assert "forge error" in web._summarise("forge_error", {"error": "boom"})
    assert "quarantined" in web._summarise("auto_quarantine",
                                           {"name": "t", "success_rate": 0.2})


def test_jsonable_survives_objects_dataclasses_and_giant_strings():
    class Thing:
        def __init__(self):
            self.a = 1
            self._private = 2

        def to_dict(self):
            return {"a": self.a}

    assert web._jsonable(Thing()) == {"a": 1}
    assert web._jsonable({"x": (1, 2)}) == {"x": [1, 2]}
    assert len(web._jsonable("z" * 99_999)) == 4000
    assert json.dumps(web._jsonable({"nested": [{"deep": object()}]}))  # never raises


def test_jsonable_bounds_recursion_depth():
    deep: dict = {}
    node = deep
    for _ in range(20):
        node["next"] = {}
        node = node["next"]
    json.dumps(web._jsonable(deep))          # must terminate, must serialise


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


def make_harness() -> web.Harness:
    return web.Harness({"model": "m", "base": "b", "fast": True,
                        "max_tokens": 10, "proxy": False},
                       minimal_factory, default_mode="minimal")


def test_session_lifecycle():
    h = make_harness()
    s = h.create("minimal", title="t")
    assert h.get(s.id) is s
    assert [x["id"] for x in h.list()] == [s.id]
    assert h.remove(s.id) is True
    assert h.get(s.id) is None
    assert h.remove(s.id) is False
    assert h.list() == []


def test_send_produces_a_full_trajectory():
    h = make_harness()
    s = h.create("minimal")
    h.send(s, "echo hi please")
    wait_idle(s)

    sources = [e["source"] for e in s.events]
    assert sources[0] == "user"
    assert "system" in sources
    assert "tool_call" in sources
    assert "tool_result" in sources
    assert "assistant" in sources
    assert s.error is None
    assert s.turns == 1
    assert "echo hi please" == s.title

    # the tool actually ran
    call = next(e for e in s.events if e["source"] == "tool_call")
    assert "bash" in call["text"]
    result = next(e for e in s.events if e["source"] == "tool_result")
    assert "ok" in result["text"]


def test_events_are_numbered_from_one_without_gaps():
    h = make_harness()
    s = h.create("minimal")
    h.send(s, "go")
    wait_idle(s)
    seqs = [e["seq"] for e in s.events]
    assert seqs == list(range(1, len(seqs) + 1))


def test_detail_respects_the_after_cursor():
    h = make_harness()
    s = h.create("minimal")
    h.send(s, "go")
    wait_idle(s)
    half = len(s.events) // 2
    rest = s.detail(after=half)["events"]
    assert [e["seq"] for e in rest] == [e["seq"] for e in s.events if e["seq"] > half]


def test_two_sends_in_a_row_are_refused_while_busy():
    """The browser must not be able to double-submit."""
    h = make_harness()
    s = h.create("minimal")
    s.busy = True                                   # pretend a run is in flight
    h.send(s, "second")
    assert any(e["source"] == "error" and "busy" in e["text"] for e in s.events)


def test_minimal_mode_refuses_to_forge_and_says_why():
    h = make_harness()
    s = h.create("minimal")
    h.forge(s, "normalise ISBNs")
    wait_idle(s)
    err = [e for e in s.events if e["source"] == "error"]
    assert err and "cannot forge" in err[0]["text"]


def test_a_raising_agent_becomes_an_error_event_not_a_dead_session():
    def boom(cfg, mode):
        class Bad(MinimalAgent):
            def run(self, task, history=None):
                raise RuntimeError("model exploded")

        return Bad(llm=echo_then_done())

    h = web.Harness({}, boom, default_mode="minimal")
    s = h.create("minimal")
    h.send(s, "go")
    wait_idle(s)
    assert s.error and "model exploded" in s.error
    assert any(e["source"] == "error" for e in s.events)
    # the session is usable again afterwards
    assert s.busy is False


def test_checks_are_flattened_from_a_forge_result():
    class Check:
        def __init__(self, name, passed, detail=""):
            self.name, self.passed, self.detail = name, passed, detail

    class Report:
        checks = [Check("execution", True, "3/3"), Check("robustness", False, "0/2")]

    class Attempt:
        round, error, report = 2, None, Report()

    class Broken:
        round, error, report = 3, "SyntaxError: bad", None

    class Result:
        attempts = [Attempt(), Broken()]

    checks = web.Harness._checks(Result())
    names = [(c["name"], c["passed"]) for c in checks]
    assert ("execution", True) in names
    assert ("robustness", False) in names
    assert ("error", False) in names
    assert all(c["round"] in (2, 3) for c in checks)


def test_tools_reports_the_mode_baseline():
    h = make_harness()
    assert h.tools() == {"tools": [], "total": 0, "by_state": {}}
    h.create("minimal")
    names = sorted(t["name"] for t in h.tools()["tools"])
    assert names == ["bash", "str_replace_editor"]


def test_status_describes_the_running_config():
    h = make_harness()
    st = h.status()
    assert st["ok"] and st["model"] == "m" and st["default_mode"] == "minimal"
    assert st["modes"] == ["standard", "minimal"]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


@pytest.fixture()
def live():
    """A real HTTP server on an ephemeral port, wired to a canned agent."""
    h = make_harness()
    handler = type("H", (web._Handler,), {"harness": h, "token": None})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, h
    finally:
        httpd.shutdown()
        httpd.server_close()


def req(base, path, payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        base + path, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=20) as resp:
        body = resp.read().decode()
        return resp.status, (json.loads(body) if body.strip() else None)


def test_serves_the_ui(live):
    base, _ = live
    with urllib.request.urlopen(base + "/", timeout=20) as r:
        html = r.read().decode()
    assert r.status == 200
    assert "autoforge" in html and "<!DOCTYPE html>" in html
    assert "EventSource" in html                    # the stream is wired up


def test_state_and_session_crud_over_http(live):
    base, _ = live
    code, state = req(base, "/api/state")
    assert code == 200 and state["ok"] is True

    code, s = req(base, "/api/sessions", {"mode": "minimal"})
    assert code == 201 and s["mode"] == "minimal"

    code, listing = req(base, "/api/sessions")
    assert [x["id"] for x in listing["sessions"]] == [s["id"]]

    code, detail = req(base, f"/api/sessions/{s['id']}")
    assert detail["id"] == s["id"]

    code, t = req(base, "/api/tools")
    assert t["total"] == 2

    code, gone = req(base, f"/api/sessions/{s['id']}", method="DELETE")
    assert gone == {"removed": True}


def test_send_over_http_streams_the_whole_turn(live):
    base, h = live
    _, s = req(base, "/api/sessions", {"mode": "minimal"})
    code, _ = req(base, f"/api/sessions/{s['id']}/send", {"text": "echo hi"})
    assert code == 202

    wait_idle(h.get(s["id"]))
    _, detail = req(base, f"/api/sessions/{s['id']}")
    sources = [e["source"] for e in detail["events"]]
    assert "assistant" in sources and "tool_call" in sources
    assert detail["turns"] == 1


def test_send_requires_text(live):
    base, _ = live
    _, s = req(base, "/api/sessions", {})
    with pytest.raises(urllib.error.HTTPError) as err:
        req(base, f"/api/sessions/{s['id']}/send", {"text": "   "})
    assert err.value.code == 400


def test_unknown_routes_and_sessions_are_404(live):
    base, _ = live
    for path in ("/api/nope", "/api/sessions/doesnotexist"):
        with pytest.raises(urllib.error.HTTPError) as err:
            req(base, path)
        assert err.value.code == 404, path


def test_token_gates_every_route_when_set():
    h = make_harness()
    handler = type("H", (web._Handler,), {"harness": h, "token": "s3cret"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            req(base, "/api/state")
        assert err.value.code == 401

        r = urllib.request.Request(base + "/api/state",
                                   headers={"X-Auth-Token": "s3cret"})
        with urllib.request.urlopen(r, timeout=20) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_trailing_and_leading_slashes_do_not_matter(live):
    base, _ = live
    for path in ("/api/state/", "//api//state"):
        code, state = req(base, path)
        assert code == 200 and state["ok"] is True, path
