"""Is the adversarial gate's prompt what the gateway refuses?

Every structural hypothesis has been eliminated: a tool turn with system
message, temperature, max_tokens=32768, stream_options, tool_choice="auto",
assistant text and a tool result all return 200. The only request that still
400s is the tool-security-auditor prompt, so this isolates that text.

It tests the original verbatim, then progressively softened versions, to find
which sentence the filter objects to.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

#: The real adversary prompt from autoforge/forge/adversary.py.
ORIGINAL = (
    "You are a tool security auditor. Your job: find inputs that break the tool.\n"
    "Given the tool's code, description, and parameters, generate up to 5 inputs\n"
    "that are likely to crash it, return wrong results, or expose bugs.\n\n"
    "Think like a fuzzer: empty strings, very long strings, missing keys,\n"
    "wrong types, Unicode, mixed encodings, negative numbers, zero values,\n"
    "null bytes, script injection, repeated calls.\n\n"
    "Output STRICT JSON array:\n"
    '[{"args": {"param": "value..."}, "rationale": "why this might break it"}]\n\n'
    "Only include inputs that have a *reasonable chance* of triggering a failure.\n"
    "Quality over quantity."
)

USER = (
    "Tool: count_words\n"
    "Description: Count the number of whitespace-separated words in a text.\n"
    "Parameters:\n"
    '{\n "type": "object",\n "properties": {\n  "text": {\n   "type": "string"\n'
    '  }\n },\n "required": [\n  "text"\n ]\n}\n'
    "Code:\n```python\n"
    "def count_words(text):\n"
    "    if not isinstance(text, str):\n"
    "        return 'INVALID: not a string'\n"
    "    return len(text.split())\n"
    "```\n"
)


def post(base: str, h: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{base}/v2/chat/completions",
        data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    accts = load_accounts()
    acct = next((a for a in accts if a.region == "cn"), accts[0])
    if not acct.usable_token():
        print("no token")
        return 1
    h, base = acct.chat_headers(), acct.chat_base

    def body_for(system: str) -> dict:
        return {"model": "hy3", "stream": True, "temperature": 0.0,
                "max_tokens": 32768,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": USER}]}

    # Progressive de-weaponisation. Each step removes one class of phrasing a
    # content filter plausibly reads as "help me attack something".
    steps = [
        ("1 original verbatim", ORIGINAL),
        ("2 drop 'script injection'", ORIGINAL.replace(
            "script injection, ", "hostile markup, ")),
        ("3 drop 'null bytes'", ORIGINAL.replace(
            "null bytes, script injection, ", "byte sequences, hostile markup, ")),
        ("4 neutralise framing", ORIGINAL
            .replace("find inputs that break the tool", "list edge cases worth testing")
            .replace("that are likely to crash it, return wrong results, or expose bugs",
                     "that exercise boundary conditions")
            .replace("Think like a fuzzer:", "Consider these categories:")
            .replace("null bytes, script injection, ", "byte sequences, hostile markup, ")
            .replace("wrong types", "unexpected types")
            .replace("or expose bugs", "or behave unexpectedly")),
        ("5 minimal probe", "You generate test inputs for a Python function. "
                            "Output a STRICT JSON array of {\"args\": {...}, "
                            "\"rationale\": \"...\"} objects."),
        ("6 control: trivial system", "You are a helpful assistant."),
    ]

    for label, system in steps:
        code, raw = post(base, h, body_for(system))
        mark = "OK " if code == 200 else "ERR"
        print(f"[{mark}] {code:3d}  {label}")
        if code != 200:
            # The business code, not the whole envelope.
            try:
                obj = json.loads(raw)
                print(f"        code={obj.get('code')} msg={obj.get('msg')}")
            except Exception:
                print(f"        {raw[:160].replace(chr(10), ' ')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
