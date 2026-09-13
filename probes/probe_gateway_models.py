"""Which model on this gateway can actually fill a tool envelope?

Established by the two probes before this one:

* `probe_gateway_empty.py` — `DeepSeek-V4.1-Flash` returns HTTP 200 with
  `content` empty and `finish_reason='length'` on a forge-shaped prompt. The
  budget goes to `reasoning_content` instead.
* `probe_gateway_thinking.py` — none of the six documented suppression flags
  stop it, and `/models` lists 147 models.

So the question is no longer "how do we tame this model" but "which model does
the job". This sends the identical forge-shaped request to each candidate with
an identical cap, and grades the reply by whether a JSON tool definition came
back at all.

Grading is deliberately mechanical: did `content` contain a JSON object with a
`code` key. A model that reasons for 10k characters and then answers counts as
usable; a model that reasons for 10k characters and *never* answers does not.

Usage:  python probes/probe_gateway_models.py [cap]
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests
from autoforge import configfile

BASE = os.environ.get("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1").rstrip("/")
KEY = os.environ.get("AIPING_API_KEY") or configfile.load().get("api_key", "")
PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

# Code-tuned and fast/flash tiers first: the workload is "emit one Python
# function inside JSON", which is exactly what these are sold for.
CANDIDATES = [
    "Doubao-Seed-2.0-Code",
    "DeepSeek-V4-Flash-0731",
    "DeepSeek-V4-Pro-0813",
    "Step-3.5-Flash",
    "GLM-5.3-Flash",
    "GLM-5-fast",
    "Qwen3.5-Flash",
    "Qwen3.7-Plus",
    "Kimi-K3",
    "MiniMax-H3",
    "DeepSeek-V4.1-Flash",      # the incumbent, for the comparison to be honest
]

SYSTEM = (
    "You write self-contained Python tools. Return ONE JSON object with exactly "
    "the keys name, description, code. The code value is a single function with "
    "every helper defined inline. No prose outside the JSON."
)
NEED = ("I keep needing to validate and normalise ISBN-10 and ISBN-13 identifiers: "
        "strip hyphens, check the check digit, convert a valid ISBN-10 to ISBN-13")


def grade(content: str) -> str:
    """A tool came back only if it is a JSON object carrying `code`."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text
        text = text.removeprefix("json").strip()
    start = text.find("{")
    if start < 0 or "}" not in text[start:]:
        return "no JSON"
    try:
        obj = json.loads(text[start:text.rfind("}") + 1])
    except ValueError:
        return "JSON unparseable"
    if not isinstance(obj, dict) or "code" not in obj:
        return "no code key"
    return f"OK ({len(obj['code'])}c code)"


def ask(model: str, cap: int) -> dict:
    payload = {"model": model, "temperature": 0.0, "max_tokens": cap,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": NEED}]}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=HEADERS,
                          json=payload, timeout=150, proxies=PROXIES)
    except Exception as exc:
        return {"status": f"EXC {type(exc).__name__}", "reasoning": 0, "content": "",
                "secs": time.time() - t0, "note": str(exc)[:40]}
    secs = time.time() - t0
    if r.status_code != 200:
        note = ""
        try:
            note = str(r.json().get("msg", ""))[:40]
        except Exception:
            note = r.text[:40]
        return {"status": f"HTTP {r.status_code}", "reasoning": 0, "content": "",
                "secs": secs, "note": note}

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    return {"status": str(choice.get("finish_reason")), "content": content,
            "reasoning": len(reasoning), "secs": secs,
            "note": f"{usage.get('completion_tokens')} tok"}


def main(argv: list[str]) -> int:
    cap = int(argv[1]) if len(argv) > 1 else 3000
    if not KEY:
        print("no API key available"); return 2
    print(f"cap {cap}  base {BASE}  proxy socks5\n")
    print(f"  {'model':<24}{'finish':<10}{'think':>7}{'answer':>8}{'secs':>7}   verdict")
    usable = []
    for model in CANDIDATES:
        res = ask(model, cap)
        verdict = grade(res["content"]) if res["content"] else "—"
        if verdict.startswith("OK"):
            usable.append(model)
        print(f"  {model:<24}{res['status']:<10}{res['reasoning']:>7}"
              f"{len(res['content']):>8}{res['secs']:>7.1f}   {verdict}  {res['note']}")
    print(f"\nusable: {', '.join(usable) if usable else 'none'}")
    if len(usable) > 1:
        print("prefer the fastest of these for forging; re-run to confirm, the "
              "gateway 503s intermittently and a single pass can be unlucky.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
