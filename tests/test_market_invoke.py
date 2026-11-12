"""The shelf-invoke client: URL shape, payload, and dead-shelf behaviour."""
from __future__ import annotations

import io
import json
import urllib.error
from unittest import TestCase
from unittest.mock import patch

from autoforge.market_invoke import market_tool_invoke


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class InvokeShelfToolTests(TestCase):
    def test_bare_name_is_prefixed_and_arguments_are_wrapped(self):
        seen = {}

        def fake(req, timeout=None):
            seen["url"] = req.full_url
            seen["data"] = json.loads(req.data)
            seen["timeout"] = timeout
            seen["method"] = req.get_method()
            return _Resp({"ok": True, "output": "done", "error": None})

        with patch("urllib.request.urlopen", fake):
            out = market_tool_invoke(
                "mail_inbox_query", {"account": "gmail", "limit": 2},
                timeout=7, base_url="http://shelf:9000/",
            )
        self.assertEqual(out, {"ok": True, "output": "done", "error": None})
        self.assertEqual(seen["url"], "http://shelf:9000/resources/tool:mail_inbox_query/invoke")
        self.assertEqual(seen["data"], {"arguments": {"account": "gmail", "limit": 2}})
        self.assertEqual(seen["timeout"], 7.0)
        self.assertEqual(seen["method"], "POST")

    def test_prefixed_name_is_not_double_prefixed(self):
        seen = {}

        def fake(req, timeout=None):
            seen["url"] = req.full_url
            return _Resp({"ok": True})

        with patch("urllib.request.urlopen", fake):
            market_tool_invoke("tool:foo", base_url="http://x")
        self.assertEqual(seen["url"], "http://x/resources/tool:foo/invoke")

    def test_env_var_supplies_the_base_url(self):
        seen = {}

        def fake(req, timeout=None):
            seen["url"] = req.full_url
            return _Resp({"ok": True})

        with patch.dict("os.environ", {"TOOLMARKET_URL": "http://env:8123"}),              patch("urllib.request.urlopen", fake):
            market_tool_invoke("foo")
        self.assertEqual(seen["url"], "http://env:8123/resources/tool:foo/invoke")

    def test_http_error_body_is_returned_not_raised(self):
        def fake(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"detail":"unknown resource"}'),
            )

        with patch("urllib.request.urlopen", fake):
            out = market_tool_invoke("missing", base_url="http://x")
        self.assertFalse(out["ok"])
        self.assertEqual(out["detail"], "unknown resource")

    def test_dead_shelf_is_a_result_not_a_crash(self):
        def fake(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", fake):
            out = market_tool_invoke("foo", base_url="http://x")
        self.assertFalse(out["ok"])
        self.assertIn("URLError", out["error"])
        self.assertEqual(out["resource_id"], "tool:foo")
