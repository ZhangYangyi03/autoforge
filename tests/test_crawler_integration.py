"""crawler-stack is a native part of autoforge, not a sibling checkout."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autoforge.agent import BUILTIN_SCOPES, ForgeAgent
from autoforge.core.llm import MockLLMClient

TOOLS = ("crawl_fast", "crawl_stealth", "crawl_render", "crawl_capabilities")


def make_agent():
    return ForgeAgent(MockLLMClient(), enable_evolution=False)


def test_all_four_tools_are_registered_with_a_scope():
    a = make_agent()
    names = set(a.registry.names())
    assert set(TOOLS) <= names
    for n in TOOLS:
        assert BUILTIN_SCOPES[n] in ("network", "read_only")
        assert a.registry.get(n).effect_signature == BUILTIN_SCOPES[n]


def test_capabilities_reports_the_engines_without_network():
    r = make_agent().registry.call("crawl_capabilities", {})
    assert r.ok, r.error
    caps = json.loads(r.output.splitlines()[0])
    assert {"aiohttp", "curl_cffi", "playwright", "selectolax"} <= set(caps)
    assert "autoforge/crawler.py" in r.output


def test_fast_and_stealth_fetch_a_local_page():
    body = (b"<html><head><title>Autoforge Crawler</title></head><body>"
            b"<a href='/a'>A</a><p>integrated</p></body></html>")

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d/" % srv.server_address[1]
        a = make_agent()
        for name in ("crawl_fast", "crawl_stealth"):
            r = a.registry.call(name, {"urls": [url, url], "timeout": 5})
            assert r.ok, r.error
            assert "mode=" in r.output and "Autoforge Crawler" in r.output
            assert "ERR" not in r.output
    finally:
        srv.shutdown()
        srv.server_close()


def test_bad_input_is_refused_not_raised():
    a = make_agent()
    r = a.registry.call("crawl_fast", {"urls": []})
    assert r.ok and r.output.startswith("INVALID:")
    r = a.registry.call("crawl_fast", {"urls": ["http://x"] * 501})
    assert r.ok and "at most 500" in r.output
