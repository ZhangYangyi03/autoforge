"""The same ceiling question, asked through retries, because the gateway flaps.

`probe_reasoning_ceiling.py` came back 503 on all four caps within two minutes
-- "暂无可用服务商", no available provider, on a 128-character prompt. That is
not a length effect: four different caps, four different sizes of budget, one
identical outcome, and the *longest* prompt this repo has ever sent (28963
chars, see `probe_length_vs_route.py`) returned 200 in the same session. The
gateway has windows where it has no provider for a model, and inside one of
those windows every request fails regardless of what is in it.

So the ceiling question needs attempts spread over time:

    cap     3000  -- what autoforge sends today
    cap    16000
    cap   128000  -- what Hermes sends (202 request dumps, every one this cap)

and both routes, because the two programs do not use the same path:

    /chat/completions            autoforge
    /anthropic/v1/messages       Hermes

A cap is reported the first time it returns 200. A cap that only ever 503s is
reported as such, and is *not* evidence for anything about its size.

Usage:  python probes/probe_ceiling_retry.py
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
CAPS = [3000, 16000, 128000]
TRIES = 6
GAP = 12

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


def once(cap: int, anthropic: bool) -> dict:
    headers = {"Content-Type": "application/json"}
    if anthropic:
        headers |= {"x-api-key": KEY, "anthropic-version": "2023-06-01"}
        path, body = "/anthropic/v1/messages", {
            "model": MODEL, "max_tokens": cap,
            "messages": [{"role": "user",
                          "content": GENERATOR_SYSTEM + "\n\n" + PROMPT}]}
    else:
        path, body = "/chat/completions", {
            "model": MODEL, "temperature": 0.0, "max_tokens": cap,
            "messages": [{"role": "system", "content": GENERATOR_SYSTEM},
                         {"role": "user", "content": PROMPT}]}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}{path}", headers=headers, json=body,
                          timeout=600)
    except Exception as exc:                                    # noqa: BLE001
        return {"code": f"EXC {type(exc).__name__}", "secs": time.time() - t0}
    took = time.time() - t0
    if r.status_code != 200:
        return {"code": str(r.status_code), "secs": took,
                "note": r.text[:60].replace("\n", " ")}
    data = r.json()
    if anthropic:
        blocks = data.get("content") or []
        content = "".join(b.get("text", "") for b in blocks
                          if b.get("type") == "text")
        reasoning = "".join(b.get("thinking", "") for b in blocks
                            if b.get("type") == "thinking")
        finish = data.get("stop_reason")
    else:
        msg = (data.get("choices") or [{}])[0]
        content = (msg.get("message") or {}).get("content") or ""
        reasoning = (msg.get("message") or {}).get("reasoning_content") or ""
        finish = msg.get("finish_reason")
    return {"code": "200", "secs": took, "content": content,
            "reasoning": reasoning, "finish": finish}


print(f"model {MODEL}   prompt {len(PROMPT)} chars   "
      f"{TRIES} tries per cap, {GAP}s apart")
print(f"{'route':>14} {'cap':>7}  {'http':>7} {'secs':>6}  {'finish':>10}  "
      f"{'content':>8} {'reasoning':>9}  grade")
print("-" * 92)
for anthropic in (False, True):
    route = "anthropic" if anthropic else "openai-compat"
    for cap in CAPS:
        for attempt in range(1, TRIES + 1):
            res = once(cap, anthropic)
            if res["code"] == "200":
                print(f"{route:>14} {cap:>7}  {'200':>7} {res['secs']:6.1f}  "
                      f"{str(res['finish']):>10}  {len(res['content']):>8} "
                      f"{len(res['reasoning']):>9}  {grade(res['content'])}"
                      f"   (try {attempt})", flush=True)
                break
            if attempt == TRIES:
                print(f"{route:>14} {cap:>7}  {res['code']:>7} "
                      f"{res['secs']:6.1f}  {TRIES} tries: "
                      f"{res.get('note', '')}", flush=True)
            else:
                time.sleep(GAP)
print("-" * 92)
