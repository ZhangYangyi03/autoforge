"""Why does the gateway return HTTP 200 with an empty body?

Observed against aiping.cn / DeepSeek-V4.1-Flash: two forge rounds, both
`finish_reason='length'`, `0 chars` of content. Not a transient error (retries
did not help), not a client bug (the same client returns text for a one-line
prompt). So look at the raw response instead of guessing.

Prints, for a range of prompt sizes and caps: HTTP status, `finish_reason`,
content length, and every non-standard key in the message (a reasoning or
"thinking" field would consume the output budget and leave `content` empty).
"""
from __future__ import annotations

import json
import os
import sys

import requests
from autoforge import configfile

BASE = os.environ.get("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1").rstrip("/")
MODEL = os.environ.get("AUTOFORGE_MODEL", "DeepSeek-V4.1-Flash")
KEY = os.environ.get("AIPING_API_KEY") or configfile.load().get("api_key", "")
PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}

# A stand-in for the generator's real prompt: long system text, then the need.
SYSTEM = (
    "You write self-contained Python tools. Return ONE JSON object with exactly "
    "the keys name, description, code. The code value is a single function with "
    "every helper defined inline. No prose outside the JSON.\n" + ("Filler line to add bulk. " * 20)
)
NEED = ("I keep needing to validate and normalise ISBN-10 and ISBN-13 identifiers: "
        "strip hyphens, check the check digit, convert a valid ISBN-10 to ISBN-13")


def probe(label: str, max_tokens, messages) -> None:
    payload = {"model": MODEL, "messages": messages, "temperature": 0.0}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        r = requests.post(f"{BASE}/chat/completions",
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {KEY}"},
                          json=payload, timeout=180, proxies=PROXIES)
    except Exception as exc:                                   # transport wobble
        print(f"{label:<28} EXC {type(exc).__name__}: {exc}")
        return

    if r.status_code != 200:
        print(f"{label:<28} HTTP {r.status_code}  {r.text[:90]!r}")
        return

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    extra = {k: (len(v) if isinstance(v, str) else type(v).__name__)
             for k, v in msg.items() if k != "content"}
    usage = data.get("usage") or {}
    print(f"{label:<28} HTTP 200  finish={choice.get('finish_reason')!r}  "
          f"content={len(msg.get('content') or '')}c  "
          f"usage={usage.get('completion_tokens')}/{usage.get('prompt_tokens')}  "
          f"other={extra or '{}'}")


def main() -> int:
    if not KEY:
        print("no API key available"); return 2
    print(f"model {MODEL}  base {BASE}  key {KEY[:5]}...({len(KEY)})  "
          f"proxy {'on' if PROXIES else 'off'}\n")

    short = [{"role": "user", "content": "Reply with exactly: READY"}]
    probe("short prompt, no cap", None, short)
    probe("short prompt, cap 3000", 3000, short)

    forge = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": NEED}]
    for cap in (None, 3000, 8000, 16000):
        probe(f"forge-like prompt, cap {cap}", cap, forge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
