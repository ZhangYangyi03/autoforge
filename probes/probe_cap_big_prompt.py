"""Is a big cap worse than a small one on aiping, for the *generator's* prompt?

The default is now `DEFAULT_MAX_TOKENS` (32768) instead of 3000. Against
api.deepseek.com that is proven end to end (`probes/probe_cap_default.py` parses in
3s). Against aiping.cn -- the project's out-of-the-box `base_url` -- the same probe
came back 503 "暂无可用服务商" 6 times over two runs, while a short prompt at the
same 32768 answered in 1s in the same minute. Two readings:

  (a) the gateway's pool was empty in both windows, and prompt size is irrelevant;
  (b) aiping routes a large *context plus large cap* combination to a pool that is
      usually empty, so the old 3000 was load-bearing and the new default is worse
      there.

FINDINGS.md holds evidence for both: sweep 2 there sent the real `GENERATOR_SYSTEM`
prompt at caps 2048/3000/4096 and got 200 3/3 each, while sweep 1 got 503 at 8192
and was dismissed as drift; `probe_gateway_context_503.py` found short prompts
200/16 and the generator prompt 503/8.

So the only useful measurement is old-vs-new on the same prompt, *interleaved*, so
that wall-clock drift cannot masquerade as a cap effect -- one cap per cycle rather
than all of them in a row. Whichever way it comes out, one of the two readings dies.

A first pass settled half of it. Nine attempts (6 cycles) at DeepSeek-V4.1-Flash:
cap 3000 answered 3/9, cap 32768 0/9, cap 128000 0/9 -- and all three successes sat
in a cycle where the two larger caps 503'd seconds later. That is not yet a cap effect, because
3000 is always sent *first* in a cycle, so "the window is open at the top of a cycle"
predicts the same rows. Hence `caps=` below: reversing the order is what separates
"small cap is served" from "first request in a burst is served".

Usage:  AIPING_API_KEY=... python probes/probe_cap_big_prompt.py [cycles] [caps=32768,3000,128000]
"""
from __future__ import annotations

import os
import sys
import time

import requests

sys.path.insert(0, ".")
from autoforge.forge.generator import GENERATOR_SYSTEM        # noqa: E402

URL = "https://aiping.cn/api/v1/chat/completions"
PROXY = {"http": "socks5h://127.0.0.1:9674", "https": "socks5h://127.0.0.1:9674"}
MODEL = "DeepSeek-V4.1-Flash"
CAPS = [3000, 32768, 128000]
CYCLES = 3
TIMEOUT = 150

NEED = ("I keep needing to list every .py file under a directory and count "
        "them, with sizes.")
PROMPT = f"Recurring need:\n{NEED}\n\nEmit the JSON envelope now."


def once(cap: int) -> str:
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": cap,
               "messages": [{"role": "system", "content": GENERATOR_SYSTEM},
                            {"role": "user", "content": PROMPT}]}
    t0 = time.time()
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                          json=payload, timeout=TIMEOUT, proxies=PROXY)
    except Exception as exc:                                       # noqa: BLE001
        return f"{time.time() - t0:5.0f}s EXC {type(exc).__name__}"
    took = time.time() - t0
    if r.status_code != 200:
        try:
            note = (r.json() or {}).get("msg", r.text[:30])
        except ValueError:
            note = r.text[:30]
        return f"{took:5.0f}s {r.status_code} {note}"
    body = r.json()
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (f"{took:5.0f}s 200 finish={choice.get('finish_reason')} "
            f"content={len(msg.get('content') or '')}c "
            f"reasoning={len(msg.get('reasoning_content') or '')}c")


if __name__ == "__main__":
    KEY = os.environ.get("AIPING_API_KEY", "")
    if not KEY:
        raise SystemExit("no AIPING_API_KEY")
    order = [a for a in sys.argv[1:] if a.startswith("caps=")]
    if order:
        CAPS[:] = [int(x) for x in order[0][5:].split(",")]
    rest = [a for a in sys.argv[1:] if not a.startswith("caps=")]
    cycles = int(rest[0]) if rest else CYCLES
    print(f"prompt {len(GENERATOR_SYSTEM) + len(PROMPT)} chars (system "
          f"{len(GENERATOR_SYSTEM)}), model {MODEL}, interleaved over "
          f"{cycles} cycles\n", flush=True)
    tally: dict[int, list[str]] = {c: [] for c in CAPS}
    for cycle in range(1, cycles + 1):
        for cap in CAPS:
            out = once(cap)
            tally[cap].append(out)
            print(f"cycle {cycle}  cap {cap:>7}  {out}", flush=True)
    print()
    for cap in CAPS:
        got = tally[cap]
        ok = sum(1 for r in got if " 200 " in r)
        parsed = sum(1 for r in got if "content=" in r and "content=0c" not in r)
        print(f"cap {cap:>7}   200s {ok}/{len(got)}   non-empty content {parsed}/{len(got)}")
