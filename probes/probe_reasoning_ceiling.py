"""Does the reasoning trace have an end, or does it fill any cap you give it?

This decides why Hermes answers on aiping while autoforge does not.

`probes/FINDINGS.md` concluded from two points -- a trace of 10767 chars at a cap
of 3000 and 27092 at 8000 -- that the trace *scales to fill* the budget, and
therefore that raising the cap cannot help. Two points drawn from caps that are
both small cannot tell "scales to fill" apart from "has a natural length longer
than 8000 tokens", and the second reading makes the fix obvious.

Hermes sends this same model `max_tokens=128000` (verified: 202 request dumps
under the Hermes profile, every one carrying that cap) and gets answers.
autoforge sends 3000. So the sweep runs on either side of the boundary:

    3000   (autoforge today)   -- expect reasoning-only
    16000
    32000
    128000 (Hermes today)      -- if content appears, the cap was the ceiling

Grading is the same mechanical one the other probes use: did content contain a
JSON object with a `code` key.

Usage:  python probes/probe_reasoning_ceiling.py
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
CAPS = [3000, 16000, 32000, 128000]

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


print(f"model {MODEL}  route /chat/completions  prompt {len(PROMPT)} chars")
print(f"{'cap':>8}  {'http':>4} {'secs':>6}  {'finish':>10}  {'content':>9}  "
      f"{'reasoning':>9}  grade")
print("-" * 78)
for cap in CAPS:
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions",
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {KEY}"},
                          json={"model": MODEL, "temperature": 0.0,
                                "max_tokens": cap,
                                "messages": [{"role": "system",
                                              "content": GENERATOR_SYSTEM},
                                             {"role": "user", "content": PROMPT}]},
                          timeout=600)
    except Exception as exc:                                    # noqa: BLE001
        print(f"{cap:>8}  {'EXC':>4} {time.time() - t0:6.1f}  "
              f"{type(exc).__name__}: {str(exc)[:30]}")
        continue
    took = time.time() - t0
    if r.status_code != 200:
        note = r.text[:70].replace("\n", " ")
        print(f"{cap:>8}  {r.status_code:>4} {took:6.1f}  {note}")
        continue
    msg = (r.json().get("choices") or [{}])[0]
    content = (msg.get("message") or {}).get("content") or ""
    reasoning = (msg.get("message") or {}).get("reasoning_content") or ""
    usage = r.json().get("usage") or {}
    print(f"{cap:>8}  {r.status_code:>4} {took:6.1f}  "
          f"{str(msg.get('finish_reason')):>10}  {len(content):>9}  "
          f"{len(reasoning):>9}  {grade(content)}   "
          f"[completion_tokens={usage.get('completion_tokens')}]", flush=True)
print("-" * 78)
