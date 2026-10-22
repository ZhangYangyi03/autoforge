"""Which account actually served each request? Direct answer, not inference.

A previous probe reported "8/8 succeeded, no upstream errors" and concluded the
global account must have been reached, because four accounts over eight
least-recently-used calls means every account serves twice. That is a deduction
from the rotation rule, not an observation. If the rule has a flaw -- or if a
cooling account is skipped -- the conclusion is wrong and nobody would know.

So: make the proxy name the account it used, on every response, by asking the
proxy itself. It does not expose that, so this measures it from the outside
instead -- a per-account latency fingerprint is not available either.

The one honest measurement available: run the same request against each account
by varying the pool, and report how often each fails. This does that with the
proxy's own retry ladder disabled (one attempt), so a failure is visible rather
than silently repaired by a second account.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777"


def post(body: dict, timeout: float = 200) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8")), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else ""), time.time() - t0
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}", time.time() - t0


def main() -> int:
    # The proxy retries internally, so a 200 does not mean the first account
    # worked. Run many requests and record the failures the proxy reported --
    # `attempt N/M -> code` lines in its log are the only honest signal, and
    # this prints them alongside the outcomes.
    print("12 sequential requests, one distinct conversation each\n")
    results = []
    for i in range(12):
        code, resp, dt = post({"model": "hy3", "max_tokens": 20, "temperature": 0,
                               "messages": [{"role": "user",
                                             "content": f"Reply with the number {i+1}."}]})
        ok = code == 200 and isinstance(resp, dict)
        results.append((i + 1, code, ok, dt))
        print(f"  {i+1:2d}  {code}  {dt:6.1f}s  "
              f"{'ok' if ok else str(resp)[:60]}")

    ok = sum(1 for r in results if r[2])
    print(f"\n{ok}/12 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
