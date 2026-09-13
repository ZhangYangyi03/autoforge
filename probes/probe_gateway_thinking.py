"""Two ways out of the empty-content trap, tested rather than assumed.

`probe_gateway_empty.py` established the cause: the model writes a
`reasoning_content` trace that consumes the whole `max_tokens` budget, so
`content` arrives empty with `finish_reason='length'`. Raising the cap does not
help — the trace simply grows to fill it (10767 chars at 3000; 27092 at 8000).

So either the trace can be switched off, or a different model has to do the
generating. This probe answers both:

  1. `GET /models` — what is actually on offer.
  2. The same forge-like payload with each common thinking-suppression flag,
     reporting `reasoning_content` vs `content` lengths so a working switch is
     visible as "reasoning 0c, content 900c".

Usage:  python probes/probe_gateway_thinking.py [model ...]
"""
from __future__ import annotations

import json
import os
import sys

import requests
from autoforge import configfile

BASE = os.environ.get("AUTOFORGE_BASE_URL", "https://aiping.cn/api/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("AUTOFORGE_MODEL", "DeepSeek-V4.1-Flash")
KEY = os.environ.get("AIPING_API_KEY") or configfile.load().get("api_key", "")
PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

SYSTEM = (
    "You write self-contained Python tools. Return ONE JSON object with exactly "
    "the keys name, description, code. The code value is a single function with "
    "every helper defined inline. No prose outside the JSON."
)
NEED = ("I keep needing to validate and normalise ISBN-10 and ISBN-13 identifiers: "
        "strip hyphens, check the check digit, convert a valid ISBN-10 to ISBN-13")

# Each entry is (label, extra payload keys). Order runs cheapest-first.
SUPPRESSORS = [
    ("baseline", {}),
    ("enable_thinking=False", {"enable_thinking": False}),
    ("thinking=disabled", {"thinking": {"type": "disabled"}}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=min", {"reasoning_effort": "minimal"}),
]


def list_models() -> list[str]:
    try:
        r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=60, proxies=PROXIES)
    except Exception as exc:
        print(f"  /models failed: {type(exc).__name__}: {exc}")
        return []
    if r.status_code != 200:
        print(f"  /models HTTP {r.status_code}: {r.text[:120]!r}")
        return []
    body = r.json()
    items = body.get("data") if isinstance(body, dict) else body
    ids = []
    for item in (items or []):
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    for i, name in enumerate(ids):
        print(f"    {i:>3}. {name}")
    return ids


def attempt(model: str, label: str, extra: dict, cap: int = 3000) -> dict:
    payload = {"model": model, "temperature": 0.0, "max_tokens": cap,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": NEED}]}
    payload.update(extra)
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=HEADERS,
                          json=payload, timeout=180, proxies=PROXIES)
    except Exception as exc:
        return {"label": label, "verdict": f"EXC {type(exc).__name__}", "content": ""}
    if r.status_code != 200:
        # The gateway answers 503 "no provider available" often enough that it
        # must be labelled, not mistaken for a property of the flag.
        return {"label": label, "verdict": f"HTTP {r.status_code}", "content": ""}

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {"label": label, "verdict": str(choice.get("finish_reason")),
            "content": content, "reasoning": reasoning,
            "cap": cap, "usage": (data.get("usage") or {}).get("completion_tokens")}


def evaluable(content: str) -> bool:
    """Did we get something that could be a tool definition?"""
    text = content.strip()
    return "{" in text and "}" in text and len(text) > 40


def main(argv: list[str]) -> int:
    if not KEY:
        print("no API key available"); return 2
    print(f"base {BASE}  key {KEY[:5]}...({len(KEY)})\n")

    print("=== 1. models on offer ===")
    available = list_models()

    models = argv[1:] or [DEFAULT_MODEL]
    for model in models:
        print(f"\n=== 2. thinking suppression — {model} ===")
        print(f"  {'flag':<24}{'finish':<10}{'reason':>8}{'content':>9}   verdict")
        best = None
        for label, extra in SUPPRESSORS:
            res = attempt(model, label, extra)
            reasoning = len(res.get("reasoning", ""))
            print(f"  {label:<24}{res['verdict']:<10}{reasoning:>8}"
                  f"{len(res['content']):>9}   "
                  f"{'USABLE' if evaluable(res['content']) else 'no answer'}")
            if evaluable(res["content"]) and best is None:
                best = label
        print(f"  -> {'first usable flag: ' + best if best else 'NO flag produced a tool'}")

    print("\nNote: a 503 means the gateway had no provider for that request. "
          "Re-run before reading anything into it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
