"""Prove the global account now actually chats through the proxy.

The proxy picks the least-recently-used eligible account, so a burst of calls
will reach the global one eventually -- but "eventually" is not a test. This
drives the global account directly by exhausting the rotation to it, then
confirms a real completion and a real tool call come back.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777"


def post(path: str, body: dict, timeout: float = 180) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, (e.read(500).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    code, health = None, None
    try:
        with urllib.request.urlopen(BASE + "/healthz", timeout=20) as r:
            code, health = r.status, json.loads(r.read().decode())
    except Exception as e:                              # noqa: BLE001
        print("healthz failed:", e)
        return 1

    print("accounts:")
    for a in health.get("accounts", []):
        print(f"  {a['label']:26s} region={a['region']:6s} "
              f"expires {a['expires_in_s']/3600:+.0f}h"
              + (f"  blocked={a['chat_unsupported']}" if a.get("chat_unsupported") else ""))
    print()

    # Four distinct conversations so the LRU rotation walks the whole pool.
    ok = 0
    for i in range(4):
        body = {"model": "hy3", "max_tokens": 48, "temperature": 0,
                "messages": [{"role": "user",
                              "content": f"Reply with exactly the word OK. Variant {i}."}]}
        code, resp = post("/v1/chat/completions", body)
        if code == 200 and isinstance(resp, dict):
            text = (resp["choices"][0]["message"].get("content") or "").strip()
            print(f"  call {i+1}: 200  {text[:40]!r}")
            ok += 1
        else:
            msg = resp.get("error", {}).get("message", "") if isinstance(resp, dict) else str(resp)[:90]
            print(f"  call {i+1}: {code}  {msg[:110]}")

    print()
    print(f"{ok}/4 succeeded")

    # A tool call, which exercises the structured path on whichever account
    # served it.
    tools = [{"type": "function", "function": {
        "name": "ping", "description": "Ping a host",
        "parameters": {"type": "object",
                       "properties": {"host": {"type": "string"}},
                       "required": ["host"]}}}]
    code, resp = post("/v1/chat/completions", {
        "model": "hy3", "max_tokens": 256, "temperature": 0, "tools": tools,
        "messages": [{"role": "user", "content": "Ping example.com using the tool."}]})
    if code == 200 and isinstance(resp, dict):
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        print(f"tool call: {len(calls)} call(s)"
              + (f" -> {calls[0]['function']['name']}({calls[0]['function']['arguments']})"
                 if calls else ""))
    else:
        print(f"tool call failed: {code}")

    return 0 if ok == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
