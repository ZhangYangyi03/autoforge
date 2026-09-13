#!/usr/bin/env python3
"""A real MCP server, small enough to read, honest enough to test against.

This is not a mock. It is a separate process that speaks the MCP stdio
transport properly — initialize, notifications/initialized, tools/list,
tools/call — and it is deliberately built to misbehave in the specific ways a
real server does, because those are the behaviours the client has to survive:

  * it prints a non-JSON banner on startup (servers do this: version strings,
    "INFO: server starting", a stray print)
  * it emits a `notifications/*` message between responses, with no id
  * one tool returns `isError: true` with text, not a protocol error
  * one tool sleeps, so a client timeout can be exercised for real
  * one tool returns a non-text content block
  * one tool takes no arguments at all (empty schema)
  * `boom` raises inside the handler, so the server answers with a JSON-RPC
    error object rather than a tool result

Turning any of that off would make the tests pass more easily and prove less.

Usage: mcp_server_fixture.py <mode>
  mode = normal | slow | die | silent | nosleep
"""
from __future__ import annotations

import json
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "echo",
        "description": "Echo the text back, wrapped in a marker.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "now",
        "description": "Return the current time as text. Takes no arguments.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fail",
        "description": "Always reports an error through the tool result, not the protocol.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "boom",
        "description": "Raises inside the handler, so the server answers with a JSON-RPC error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "slow",
        "description": "Sleeps before answering, to exercise client timeouts.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
        },
    },
    {
        "name": "picture",
        "description": "Returns a non-text content block.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def notify(method: str, params: dict) -> None:
    send({"jsonrpc": "2.0", "method": method, "params": params})


def handle_call(name: str, args: dict):
    if name == "add":
        return [{"type": "text", "text": str(int(args["a"]) + int(args["b"]))}]
    if name == "echo":
        return [{"type": "text", "text": f"<echo>{args['text']}</echo>"}]
    if name == "now":
        return [{"type": "text", "text": "a-fixed-time"}]
    if name == "fail":
        return [{"type": "text", "text": "the upstream service is unavailable"}], True
    if name == "picture":
        return [{"type": "image", "mimeType": "image/png", "data": "aGk="}]
    if name == "slow":
        time.sleep(float(args.get("seconds", 2)))
        return [{"type": "text", "text": "slept"}]
    if name == "boom":
        raise ValueError("the handler exploded")
    return [{"type": "text", "text": f"unknown tool {name}"}], True


def main() -> None:
    # A banner that is not JSON-RPC. A client that trusts the first line it
    # reads dies here; a correct one logs it and carries on.
    sys.stdout.write("INFO autoforge-fixture server ready\n")
    sys.stdout.flush()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue

        method = msg.get("method")
        rid = msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "autoforge-fixture", "version": "1.2.3"},
            }})
        elif method == "notifications/initialized":
            if MODE == "die":
                sys.exit(3)
            # A notification mid-stream, before any response. Servers do this.
            notify("notifications/message", {"level": "info", "data": "ready"})
        elif method == "tools/list":
            if MODE == "die":
                sys.exit(3)
            if MODE == "silent":
                continue                      # accepts the request, never answers
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "slow" and MODE == "nosleep":
                args = {**args, "seconds": 0}
            try:
                blocks = handle_call(str(name), args)
            except Exception as exc:           # noqa: BLE001 - the point of `boom`
                send({"jsonrpc": "2.0", "id": rid, "error": {
                    "code": -32603, "message": f"internal error: {type(exc).__name__}: {exc}",
                }})
                continue
            is_error = False
            if isinstance(blocks, tuple):
                blocks, is_error = blocks
            # Another mid-stream notification, this time between response
            # frames where a naive reader would treat it as the answer.
            notify("notifications/tools/done", {"tool": name})
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": blocks, "isError": is_error,
            }})
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    # Line-buffered, so a response is visible to the parent immediately.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
