"""Bisect the 400: which field of a real tool-turn request actually trips it.

The previous conclusion -- "the chat wire has no `tool` role, flatten it into
prose" -- was wrong: a minimal tool turn returns 200. So something else in the
forge-shaped request is the trigger, and finding it means the flattened-text
workaround can be deleted instead of maintained.

Bisection: start from the probe that passed, add one field at a time.
"""
import copy
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "count_words", "description": "count words",
    "parameters": {"type": "object",
                   "properties": {"text": {"type": "string"}},
                   "required": ["text"]}}}]

#: The minimal shape that passed the endpoint probe.
BASE = {
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
    "tools": TOOLS,
}


def post(base: str, headers: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{base}/v2/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, resp.read(300).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, (exc.read(400).decode("utf-8", "replace") if exc.fp else "")
    except Exception as exc:                            # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    accts = load_accounts()
    acct = next((a for a in accts if a.region == "cn"), accts[0])
    token = acct.usable_token()
    if not token:
        print("no token")
        return 1
    h = acct.chat_headers()
    base = acct.chat_base

    variants: list[tuple[str, dict]] = []

    def add(label: str, mutate) -> None:
        body = copy.deepcopy(BASE)
        mutate(body)
        variants.append((label, body))

    variants.append(("baseline (minimal tool turn)", copy.deepcopy(BASE)))

    add("+ system message", lambda b: b["messages"].insert(
        0, {"role": "system", "content": "You are a capable agent."}))
    add("+ temperature 0.0", lambda b: b.update({"temperature": 0.0}))
    add("+ max_tokens 32768", lambda b: b.update({"max_tokens": 32768}))
    add("+ max_tokens 512", lambda b: b.update({"max_tokens": 512}))
    add("+ stream_options", lambda b: b.update(
        {"stream_options": {"include_usage": True}}))
    add("+ tool_choice auto", lambda b: b.update({"tool_choice": "auto"}))
    add("+ assistant content text", lambda b: b["messages"][1].update(
        {"content": "I will count the words."}))
    add("+ tool msg with name", lambda b: b["messages"][2].update(
        {"name": "count_words"}))
    add("+ tool msg without tool_call_id", lambda b: b["messages"][2].pop(
        "tool_call_id", None))
    add("+ tool msg bare (no id, no name)", lambda b: (
        b["messages"][2].pop("tool_call_id", None),
        b["messages"][2].pop("name", None)))
    add("all four known-good extras", lambda b: (
        b["messages"].insert(0, {"role": "system", "content": "You are a capable agent."}),
        b.update({"temperature": 0.0, "max_tokens": 32768,
                  "stream_options": {"include_usage": True}})))

    for label, body in variants:
        code, raw = post(base, h, body)
        mark = "OK " if code == 200 else "ERR"
        print(f"[{mark}] {code:3d}  {label}")
        if code != 200:
            print(f"        {raw[:220].replace(chr(10), ' ')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
