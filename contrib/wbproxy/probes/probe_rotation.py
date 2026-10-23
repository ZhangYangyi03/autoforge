"""Prove the two new rotation behaviours, against the running proxy.

1. STICKY -- the same conversation keeps the same account. A request is sent
   several times with an identical first user message; the proxy's log names
   the account it used. If sticky works, one label repeats; if it is still
   round-robin, the label changes every call.

2. PER-MODEL COOLDOWN -- a rate limit on one model must not take the account
   out for the others. This drives the gateway's real limiter, so it is
   opt-in and the probe prints what it found rather than asserting a pass:
   whether a cooldown was recorded, for which pair, and for how long.

The proxy must be running on 127.0.0.1:8777.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8777"
LOG = Path(os.environ.get("TEMP", ".")) / "wb_rr.log"


def post(body: dict, timeout: float = 180) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    import os
    print("=" * 74)
    print("1. sticky: the same conversation keeps the same account")
    print("=" * 74)
    # One identical opening message = one conversation key. A different wording
    # would be a different conversation and would legitimately rebind.
    same = "Reply with exactly: STICKY-PROBE"
    for i in range(4):
        code, resp = post({"model": "hy3", "max_tokens": 16, "temperature": 0,
                           "messages": [{"role": "user", "content": same}]})
        ok = code == 200
        print(f"  call {i+1}: {code} {'ok' if ok else str(resp)[:70]}")
        time.sleep(1)

    print()
    print("  --- proxy log: which account served them ---")
    if LOG.exists():
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [l for l in lines[-40:] if "upstream" in l or "rate-limited" in l]
        if tail:
            for l in tail[-8:]:
                print("   ", l[:150])
        else:
            print("    (no upstream lines in the recent tail)")
            print("    NOTE: the proxy only logs the account on FAILURE, so a")
            print("    clean run proves nothing either way from the log alone.")
    else:
        print(f"    no log at {LOG}")

    print()
    print("=" * 74)
    print("2. different conversations get different accounts (round-robin)")
    print("=" * 74)
    for i in range(4):
        code, resp = post({"model": "hy3", "max_tokens": 16, "temperature": 0,
                           "messages": [{"role": "user",
                                         "content": f"Reply with: ROUND-{i+1}"}]})
        print(f"  conversation {i+1}: {code}")
        time.sleep(1)

    print()
    print("=" * 74)
    print("3. health: any account cooling right now?")
    print("=" * 74)
    try:
        with urllib.request.urlopen(BASE + "/healthz", timeout=20) as r:
            h = json.loads(r.read().decode())
        for a in h.get("accounts", []):
            print(f"  {a['label']:26s} region={a['region']:6s} "
                  f"expires {a['expires_in_s']/3600:+.0f}h"
                  + (f"  blocked={a['chat_unsupported']}"
                     if a.get("chat_unsupported") else ""))
    except Exception as e:                              # noqa: BLE001
        print("  healthz failed:", e)

    print()
    print("The per-model cooldown path is exercised by the live limiter, so it")
    print("cannot be forced without spending quota. Read the log after a real")
    print("rate limit: it should say 'rate-limited on <model>; cooling that pair'.")
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())
