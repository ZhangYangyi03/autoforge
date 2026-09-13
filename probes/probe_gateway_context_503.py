"""Is the gateway's 503 a function of prompt size?

Measured, not assumed. Two earlier observations disagreed:

* short prompts (`"say READY"`) returned 200 on 16 of 16 attempts;
* the generator's real prompt (2873 chars: a system message plus a need)
  returned 503 on 7 of 8.

Same model, same caps, same minute. If 503 is load-dependent on context size
then "just retry harder" is the wrong fix and a shorter prompt is the right one;
if it is random, the retry ladder already handles it. This walks a range of
sizes and counts.

Kept deliberately small — the earlier version of this probe asked for 6 attempts
across 8 sizes and hung past ten minutes, because a 503-ing request can sit in
the queue rather than fail fast.
"""
from __future__ import annotations

import os
import time

import requests

URL = "https://aiping.cn/api/v1/chat/completions"
PROXIES = {"http": "socks5h://127.0.0.1:9674", "https": "socks5h://127.0.0.1:9674"}
SIZES = (200, 600, 1000, 1500, 2000, 2900)
TRIES = 4
TIMEOUT = 40


def main() -> int:
    key = os.environ.get("AIPING_API_KEY", "")
    if not key:
        print("no AIPING_API_KEY", flush=True)
        return 2

    print(f"{'chars':>6}  {'ok':>5}  results", flush=True)
    for size in SIZES:
        # One long user turn, so the only variable is length.
        msgs = [{"role": "user", "content": "x " * (size // 2)}]
        codes = []
        for _ in range(TRIES):
            try:
                r = requests.post(
                    URL, headers={"Authorization": f"Bearer {key}"},
                    json={"model": "Qwen3.5-Flash", "messages": msgs,
                          "max_tokens": 200},
                    timeout=TIMEOUT, proxies=PROXIES)
                codes.append(str(r.status_code))
            except Exception as exc:  # noqa: BLE001
                codes.append(f"EXC:{type(exc).__name__}")
            time.sleep(0.3)
        ok = sum(1 for c in codes if c == "200")
        print(f"{size:>6}  {ok:>3}/{TRIES}  {' '.join(codes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
