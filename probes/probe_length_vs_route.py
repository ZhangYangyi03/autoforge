"""Why does Hermes answer on aiping while autoforge does not?

Both talk to aiping.cn. The user's question is whether the difference is prompt
*length*, so this measures the one thing that decides it: the same forge-shaped
prompt, the same model, the same seconds -- sent down the two different routes
the two programs actually use.

  route A (autoforge): POST /api/v1/chat/completions   (OpenAI-compat)
  route B (Hermes):    POST /api/v1/anthropic/v1/messages (anthropic_messages)

Plus a length control: a prompt ~7x the size of the forge prompt on route A.
If length were the trigger that one would fail while the small one passed --
and it is the *only* thing it varies.

Run from the autoforge repo root:
    python probes/probe_length_vs_route.py
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
TIMEOUT = 150

#: autoforge's real generator prompt, assembled exactly as generator.py does.
NEED = ("I keep needing to list every .py file under a directory and count "
        "them, with sizes.")
FORGE_PROMPT = f"Recurring need:\n{NEED}\n\nEmit the JSON envelope now."

#: The same prompt plus filler, to reach roughly the size of a Hermes request.
#: Deliberately *not* forge-shaped content -- length is the only variable.
BIG_PROMPT = ("Context (ignored, padding only):\n"
              + ("the quick brown fox jumps over the lazy dog. " * 640)
              + "\n\n" + FORGE_PROMPT)


def post(path: str, payload: dict, anthropic: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if anthropic:
        headers["x-api-key"] = KEY
        headers["anthropic-version"] = "2023-06-01"
        headers["Authorization"] = f"Bearer {KEY}"
    else:
        headers["Authorization"] = f"Bearer {KEY}"
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}{path}", headers=headers, json=payload,
                          timeout=TIMEOUT)
    except Exception as exc:                                    # noqa: BLE001
        return {"status": f"EXC {type(exc).__name__}", "secs": time.time() - t0,
                "note": str(exc)[:60]}
    took = time.time() - t0
    out: dict = {"status": str(r.status_code), "secs": took}
    if r.status_code != 200:
        out["note"] = r.text[:160].replace("\n", " ")
        return out
    data = r.json()
    out["usage"] = data.get("usage", {})
    if anthropic:
        blocks = data.get("content") or []
        out["content"] = "".join(b.get("text", "") for b in blocks
                                 if b.get("type") == "text")
        out["reasoning"] = "".join(b.get("thinking", "") for b in blocks
                                   if b.get("type") == "thinking")
        out["finish"] = data.get("stop_reason")
    else:
        msg = (data.get("choices") or [{}])[0].get("message", {})
        out["content"] = msg.get("content") or ""
        out["reasoning"] = msg.get("reasoning_content") or ""
        out["finish"] = (data.get("choices") or [{}])[0].get("finish_reason")
    return out


def row(label: str, res: dict) -> None:
    if "content" not in res:
        print(f"  {label:44s} {res['status']:>4s} {res['secs']:6.1f}s  "
              f"{res.get('note', '')}", flush=True)
        return
    usage = res.get("usage") or {}
    detail = usage.get("completion_tokens_details") or {}
    print(f"  {label:44s} {res['status']:>4s} {res['secs']:6.1f}s  "
          f"finish={str(res['finish']):6s} content={len(res['content']):6d}c "
          f"reasoning={len(res['reasoning']):6d}c "
          f"completion_tokens={usage.get('completion_tokens', '?')} "
          f"(reasoning {detail.get('reasoning_tokens', '?')})", flush=True)
    if not res["content"].strip():
        print("      ^^ DOC content: the model returned nothing an agent can use",
              flush=True)


print(f"forge prompt: {len(FORGE_PROMPT)} chars  "
      f"(system {len(GENERATOR_SYSTEM)} + user {len(FORGE_PROMPT)})")
print(f"big prompt:   {len(BIG_PROMPT)} chars  "
      f"({len(BIG_PROMPT) / max(1, len(FORGE_PROMPT)):.1f}x the forge prompt)")
print("-" * 100)

# -- the two routes, same prompt, same model ------------------------------
print("route A -- OpenAI-compat, what autoforge uses")
row("forge prompt, max_tokens=3000",
    post("/chat/completions", {"model": MODEL, "temperature": 0.0,
                               "max_tokens": 3000,
                               "messages": [{"role": "system",
                                             "content": GENERATOR_SYSTEM},
                                            {"role": "user",
                                             "content": FORGE_PROMPT}]}))

print("route B -- anthropic_messages, what Hermes uses")
row("forge prompt, max_tokens=3000",
    post("/anthropic/v1/messages", {"model": MODEL, "max_tokens": 3000,
                                    "messages": [{"role": "user",
                                                  "content": FORGE_PROMPT}]},
         anthropic=True))

# -- the length control ---------------------------------------------------
print("length control -- 7x bigger prompt, same route, same cap")
row("big prompt, max_tokens=3000",
    post("/chat/completions", {"model": MODEL, "temperature": 0.0,
                               "max_tokens": 3000,
                               "messages": [{"role": "system",
                                             "content": GENERATOR_SYSTEM},
                                            {"role": "user",
                                             "content": BIG_PROMPT}]}))
print("-" * 100)
