"""The shipped default, on the shipped route: does a forge parse now?

`LLMToolGenerator.max_tokens` was 3000. At 3000, against aiping.cn, the generator's
own prompt came back HTTP 200 with `finish_reason='length'`, 0 chars of content and
10901 chars of `reasoning_content` -- the reasoning trace spent the whole budget and
the answer never began, which reached the operator as "the generator produced no
answer".

`tests/test_reasoning_budget.py` read two such samples (caps of 3000 and 8000, both
empty) as "the trace fills whatever cap it is given", which makes the cap look
unfixable. Two samples below the trace's natural length cannot show that: the same
gateway has been answering 65536-128000 from the same profile, same model, all
along. The default is now `DEFAULT_MAX_TOKENS` (32768).

This probe does not measure the gateway. It runs the *generator*, through the *real*
client, over the *real* prompt, and grades what comes back -- every layer a forge
takes except the sandbox and the retry rounds.

Two routes, because the fix has two halves:

  aiping     generous cap accepted        -> the envelope must parse
  deepseek   generous cap over the ceiling -> its 400 must be survived, at the
                                              number that error names

Usage:  python probes/probe_cap_default.py [route ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")
from autoforge.core import llm as llm_mod                              # noqa: E402
from autoforge.core.llm import DEFAULT_MAX_TOKENS, LLMAborted, OpenAICompatClient  # noqa: E402
from autoforge.forge.generator import (                                # noqa: E402
    LLMToolGenerator, UnrecoverableGeneration,
)

#: aiping.cn is reached through the local SOCKS proxy on this host; the direct
#: endpoints (api.deepseek.com) are not. Kept explicit because the sandbox
#: blocked the proxy for a while and a silent bypass reads as a gateway fault.
PROXY = {"https": "socks5://127.0.0.1:9674", "http": "socks5://127.0.0.1:9674"}

NEED = ("I keep needing to list every .py file under a directory and count "
        "them, with sizes.")


def _config() -> dict:
    path = os.environ.get("AUTOFORGE_CONFIG") or os.path.expanduser(
        "~/.autoforge/config.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def route(name: str) -> dict:
    if name == "aiping":
        return {"base": "https://aiping.cn/api/v1", "key": os.environ["AIPING_API_KEY"],
                "model": "DeepSeek-V4.1-Flash", "proxy": True}
    saved = _config()
    return {"base": saved["base_url"], "key": saved["api_key"],
            "model": saved["model"], "proxy": bool(saved.get("proxy"))}


def trim(text: str, n: int = 70) -> str:
    return text.replace("\n", " ")[:n]


def run(name: str) -> bool:
    cfg = route(name)
    print(f"\n=== {name}  {cfg['model']}  {cfg['base']}  proxy={cfg['proxy']}")

    # Watch the wire: which cap did each attempt actually carry?
    wire: list[dict] = []
    real_post = llm_mod.requests.post

    def spy(url, **kw):
        body = kw.get("json") or {}
        resp = real_post(url, **kw)
        wire.append({"cap": body.get("max_tokens"), "status": resp.status_code,
                     "err": "" if resp.status_code < 400 else trim(resp.text, 90)})
        return resp

    llm_mod.requests.post = spy
    client = OpenAICompatClient(
        cfg["model"], cfg["base"], cfg["key"],
        proxies=PROXY if cfg["proxy"] else None,
        max_attempts=3, retry_backoff=1.0,
    )

    gen = LLMToolGenerator(llm=client)
    print(f"    generator cap = {gen.max_tokens}  (project default "
          f"{DEFAULT_MAX_TOKENS}; reasoning trace needs more than 3000)")
    t0 = time.time()
    try:
        tool = gen.generate(NEED)
        took = time.time() - t0
        print(f"    PARSED in {took:.0f}s: name={tool.name!r} "
              f"code={len(tool.code)}c probes={len(tool.probes)}")
        ok = True
    except UnrecoverableGeneration as exc:
        took = time.time() - t0
        print(f"    NO ANSWER in {took:.0f}s (the 3000 failure, unchanged): {exc}")
        ok = False
    except LLMAborted as exc:
        print(f"    ABORTED: {exc}")
        ok = False
    except ValueError as exc:
        print(f"    UNPARSEABLE in {time.time() - t0:.0f}s: {str(exc)[:200]}")
        ok = False
    except Exception as exc:                                           # noqa: BLE001
        print(f"    {type(exc).__name__} in {time.time() - t0:.0f}s: {str(exc)[:200]}")
        ok = False
    finally:
        llm_mod.requests.post = real_post

    print(f"    wire ({len(wire)} attempt(s)):")
    for i, hop in enumerate(wire, 1):
        print(f"      {i}. cap={hop['cap']}  http={hop['status']}"
              + (f"  {hop['err']!r}" if hop["err"] else ""))
    if len(wire) > 1:
        print("      -> more than one attempt: the clamp or the retry ladder moved "
              "the cap, which is the point for a stricter provider")
    return ok


if __name__ == "__main__":
    names = sys.argv[1:] or ["aiping", "deepseek"]
    results = {n: run(n) for n in names}
    print("\n" + "-" * 70)
    for n, ok in results.items():
        print(f"{n:10s} {'PASS' if ok else 'FAIL'}")
