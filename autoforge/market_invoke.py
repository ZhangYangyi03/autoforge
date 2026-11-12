"""Invoke a tool that lives only on the shared toolmarket shelf.

Why this exists
---------------
`market_search` can return a tool that is published on the sibling shelf but is
not registered in this process. `describe_tool` then answers "no tool named
...", and the tempting fixes are to re-forge it or to hand-roll the same
capability with a one-off script. Both waste the shelf.

The shelf already exposes::

    POST /resources/{resource_id}/invoke
    {"arguments": {...}}

This module is the stdlib-only client for that endpoint. No local install, no
third-party dependency, no duplicate implementation.

Example::

    from autoforge.market_invoke import market_tool_invoke
    market_tool_invoke("tool:mail_inbox_query",
                       {"account": "gmail", "query": "newer_than:1d"})
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

DEFAULT_URL = "http://127.0.0.1:8000"


def market_tool_invoke(
    resource_id: str,
    arguments: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Call one shelf tool by resource id and return the server response.

    A bare name is prefixed with ``tool:``. Network and HTTP failures are
    returned as ``{"ok": false, "error": ...}`` rather than raised, so a
    cold-tool lookup never aborts the caller's turn.
    """
    base = (base_url or os.environ.get("TOOLMARKET_URL") or DEFAULT_URL).rstrip("/")
    rid = str(resource_id)
    if ":" not in rid:
        rid = "tool:" + rid
    url = base + "/resources/" + urllib.parse.quote(rid, safe=":") + "/invoke"
    payload = json.dumps({"arguments": dict(arguments or {})}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                parsed.setdefault("ok", False)
                parsed.setdefault("resource_id", rid)
                return parsed
        except ValueError:
            pass
        return {"ok": False, "resource_id": rid,
                "error": "HTTP %s: %s" % (exc.code, body[:400])}
    except Exception as exc:  # noqa: BLE001 - a dead shelf is a result, not a crash
        return {"ok": False, "resource_id": rid,
                "error": "%s: %s" % (type(exc).__name__, exc)}


# A descriptive alias for callers that do not care about the published tool name.
invoke_shelf_tool = market_tool_invoke

__all__ = ["market_tool_invoke", "invoke_shelf_tool", "DEFAULT_URL"]
