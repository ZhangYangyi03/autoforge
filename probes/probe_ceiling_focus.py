"""The two cells that decide it, on autoforge's own route, retried through flaps.

Autoforge sends the generator a cap of 3000. Hermes sends the same model
128000 (202 request dumps, every one carrying that number). If the reasoning
trace is the thing eating the budget, then the same prompt on the same route
answers at 128000 and does not at 3000 -- and the fix is a one-line default
rather than a model change.

Retried, because this gateway rotates its failure class: 503 "暂无可用服务商"
inside one window, 401 unauthorized inside another (six identical requests
401'd, then the same request by hand returned 200 seconds later), so a single
attempt measures the gateway's mood, not the cap.

Usage:  python probes/probe_ceiling_focus.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests

sys.path.insert(0, ".")
from autoforge.forge.generator import GENERATOR_SYSTEM       # noqa: E402

BASE = "https://aiping.cn/api/v1"
KEY = os.environ.get("AIPING_API_KEY", "")
MODEL = "DeepSeek-V4.1-Flash"
CAPS = [3000, 128000]
TRIES = 8
GAP = 8

NEED = ("I keep needing to list every .py file under a directory and count "
        "them, with sizes.")
PROMPT = f"Recurring need:\n{NEED}\n\nEmit the JSON envelope now."


def grade(content: str) -> str:
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
    return f"OK ({len(obj['code'])}c of code)"


def once(cap: int) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "temperature": 0.0, "max_tokens": cap,
                  "messages": [{"role": "system", "content": GENERATOR_SYSTEM},
                               {"role": "user", "content": PROMPT}]},
            timeout=900)
    except Exception as exc:                                    # noqa: BLE001
        return {"code": f"EXC {type(exc).__name__}", "secs": time.time() - t0}
    took = time.time() - t0
    if r.status_code != 200:
        return {"code": str(r.status_code), "secs": took,
                "note": r.text[:50].replace("\n", " ")}
    msg = (r.json().get("choices") or [{}])[0]
    return {"code": "200", "secs": took,
            "content": (msg.get("message") or {}).get("content") or "",
            "reasoning": (msg.get("message") or {}).get("reasoning_content") or "",
            "finish": msg.get("finish_reason"),
            "usage": (r.json().get("usage") or {}).get("completion_tokens")}


print(f"route /chat/completions   model {MODEL}   prompt {len(PROMPT)} chars")
print(f"{'cap':>8} {'http':>6} {'secs':>7}  {'finish':>11}  {'content':>8} "
      f"{'reasoning':>9}  {'ctok':>7}  grade")
print("-" * 90)
for cap in CAPS:
    for attempt in range(1, TRIES + 1):
        res = once(cap)
        if res["code"] == "200":
            print(f"{cap:>8} {'200':>6} {res['secs']:7.1f}  "
                  f"{str(res['finish']):>11}  {len(res['content']):>8} "
                  f"{len(res['reasoning']):>9}  {str(res['usage']):>7}  "
                  f"{grade(res['content'])}   (try {attempt})", flush=True)
            break
        if attempt == TRIES:
            print(f"{cap:>8} {res['code']:>6} {res['secs']:7.1f}  "
                  f"{TRIES} tries, last: {res.get('note', '')}", flush=True)
        else:
            time.sleep(GAP)
print("-" * 90)
