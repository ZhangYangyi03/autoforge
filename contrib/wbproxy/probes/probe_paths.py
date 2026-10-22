"""Probe which /v2/* endpoints the WorkBuddy account pool actually answers.

The proxy currently posts to /v2/chat/completions, which has no `tool` role --
so multi-turn tool conversations have to be flattened into prose, losing the
structured affordance. If /v2/responses or /v2/messages exists on the
authenticated path, that loss is avoidable.

Run: python probe_endpoints.py
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import Account, load_accounts, CN_CHAT_BASE  # noqa: E402


def post(url: str, headers: dict, body: dict, timeout: float = 60.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2000).decode("utf-8", "replace")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, (exc.read(2000).decode("utf-8", "replace") if exc.fp else "")
    except Exception as exc:                            # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    accounts = load_accounts()
    if not accounts:
        print("no accounts")
        return 1
    acct = next((a for a in accounts if a.region == "cn"), accounts[0])
    token = acct.usable_token()
    if not token:
        print(f"no token for {acct.label}")
        return 1
    h = acct.chat_headers()
    print(f"account: {acct.label}  region={acct.region}  base={acct.chat_base}\n")

    simple = {
        "model": "hy3",
        "messages": [{"role": "user", "content": "say OK"}],
        "stream": True,
    }

    # A tool conversation in each wire's own shape. What we are asking is
    # whether any endpoint accepts a *round two* (a tool result going back).
    chat_tool_turn = {
        "model": "hy3",
        "stream": True,
        "messages": [
            {"role": "user", "content": "count words in 'a b c'"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "count_words",
                              "arguments": "{\"text\": \"a b c\"}"}}]},
            {"role": "tool", "content": "3", "tool_call_id": "call_1"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "count_words", "description": "count words",
            "parameters": {"type": "object",
                           "properties": {"text": {"type": "string"}},
                           "required": ["text"]}}}],
    }

    responses_tool_turn = {
        "model": "hy3",
        "stream": True,
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "count words in 'a b c'"}]},
            {"type": "function_call", "name": "count_words",
             "arguments": "{\"text\": \"a b c\"}", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "3"},
        ],
        "tools": [{"type": "function", "name": "count_words",
                   "description": "count words",
                   "parameters": {"type": "object",
                                  "properties": {"text": {"type": "string"}},
                                  "required": ["text"]}}],
    }

    messages_tool_turn = {
        "model": "hy3",
        "stream": True,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "count words in 'a b c'"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "count_words",
                 "input": {"text": "a b c"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "3"}]},
        ],
        "tools": [{"name": "count_words", "description": "count words",
                   "input_schema": {"type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"]}}],
    }

    cases = [
        ("/v2/chat/completions", simple, "baseline (known good)"),
        ("/v2/chat/completions", chat_tool_turn, "chat + tool role"),
        ("/v2/responses", simple, "responses, simple"),
        ("/v2/responses", responses_tool_turn, "responses + function_call_output"),
        ("/v2/messages", simple, "messages, simple"),
        ("/v2/messages", messages_tool_turn, "messages + tool_result"),
        ("/v1/chat/completions", simple, "v1 path (does it exist here?)"),
        ("/v2/responses", {"model": "hy3", "input": "say OK", "stream": True},
         "responses, string input"),
    ]

    for path, body, label in cases:
        url = f"{acct.chat_base}{path}"
        code, raw = post(url, h, body)
        verdict = "OK " if code == 200 else "ERR"
        snippet = raw[:180].replace("\n", " ")
        print(f"[{verdict}] {code:3d}  {label}")
        print(f"        {path}  -> {snippet}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
