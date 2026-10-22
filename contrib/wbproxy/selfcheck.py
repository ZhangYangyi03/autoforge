"""End-to-end acceptance run against the running wbproxy.

Each check states what it is proving, because "it returned 200" is not the same
as "the contract holds". Prints PASS/FAIL per item and exits non-zero on any
failure, so it can be pasted into a terminal and read at a glance.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def get(path: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def post(path: str, body: dict, timeout: float = 180) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, (e.read(500).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    print("=" * 70)
    print("1. health and catalog")
    print("=" * 70)
    code, health = get("/healthz")
    check("healthz answers 200", code == 200, str(code))
    if isinstance(health, dict):
        accts = health.get("accounts") or []
        check("accounts visible", len(accts) > 0, f"{len(accts)}")
        check("no account has a refresh error",
              all(not a.get("last_error") for a in accts),
              str([a["label"] for a in accts if a.get("last_error")]))

    code, models = get("/v1/models")
    rows = models.get("data", []) if isinstance(models, dict) else []
    check("models list is non-empty", len(rows) > 0, f"{len(rows)}")
    check("every row has an id", all(r.get("id") for r in rows))
    check("rows carry cost metadata",
          any("credits" in r for r in rows),
          "context_window/credits present")
    ids = {r["id"] for r in rows}
    check("hy3 present", "hy3" in ids)
    check("free models identified",
          any(r.get("credits") == 0 for r in rows),
          str([r["id"] for r in rows if r.get("credits") == 0][:4]))

    print()
    print("=" * 70)
    print("2. a plain completion")
    print("=" * 70)
    t0 = time.time()
    code, body = post("/v1/chat/completions", {
        "model": "hy3", "max_tokens": 64, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    })
    dt = time.time() - t0
    check("answers 200", code == 200, f"{code} in {dt:.1f}s")
    if isinstance(body, dict) and code == 200:
        msg = body["choices"][0]["message"]
        check("content is non-empty", bool(msg.get("content")),
              repr(msg.get("content", "")[:40]))
        check("finish_reason is stop",
              body["choices"][0].get("finish_reason") == "stop",
              str(body["choices"][0].get("finish_reason")))
        u = body.get("usage") or {}
        check("usage reported", bool(u), json.dumps(u))
        check("output matches", "PONG" in (msg.get("content") or "").upper())

    print()
    print("=" * 70)
    print("3. structured tool call (round one)")
    print("=" * 70)
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    code, body = post("/v1/chat/completions", {
        "model": "hy3", "max_tokens": 256, "temperature": 0, "tools": tools,
        "tool_choice": "auto",
        "messages": [{"role": "user",
                      "content": "What is the weather in Beijing? Use the tool."}],
    })
    check("answers 200", code == 200, str(code))
    call_id = None
    if isinstance(body, dict) and code == 200:
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        check("model asked for the tool", len(calls) > 0, f"{len(calls)} call(s)")
        check("finish_reason is tool_calls",
              choice.get("finish_reason") == "tool_calls",
              str(choice.get("finish_reason")))
        if calls:
            fn = calls[0]["function"]
            check("function name preserved", fn.get("name") == "get_weather",
                  str(fn.get("name")))
            args = fn.get("arguments") or ""
            try:
                parsed = json.loads(args)
                check("arguments are valid JSON", True, json.dumps(parsed))
                check("arguments carry the city", "city" in parsed)
            except Exception as exc:
                check("arguments are valid JSON", False, f"{exc}: {args[:60]}")
            call_id = calls[0].get("id")

    print()
    print("=" * 70)
    print("4. structured tool result (round two) — the case that used to fail")
    print("=" * 70)
    if call_id is None:
        check("round two skipped", False, "no call id from round one")
    else:
        code, body = post("/v1/chat/completions", {
            "model": "hy3", "max_tokens": 256, "temperature": 0, "tools": tools,
            "messages": [
                {"role": "user",
                 "content": "What is the weather in Beijing? Use the tool."},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": "{\"city\": \"beijing\"}"}}]},
                {"role": "tool", "tool_call_id": call_id,
                 "name": "get_weather", "content": "22C, clear"},
            ],
        })
        check("role:tool accepted", code == 200, str(code))
        if isinstance(body, dict) and code == 200:
            content = body["choices"][0]["message"].get("content") or ""
            check("model used the tool result", "22" in content, repr(content[:80]))

    print()
    print("=" * 70)
    print("5. failures are translated, not swallowed")
    print("=" * 70)
    code, body = post("/v1/chat/completions", {
        "model": "definitely-not-a-model-id",
        "messages": [{"role": "user", "content": "hi"}]})
    check("unknown model fails loudly", code != 200, f"{code}")
    if isinstance(body, dict):
        check("error carries a message", bool((body.get("error") or {}).get("message")),
              str((body.get("error") or {}).get("message", ""))[:70])

    print()
    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
