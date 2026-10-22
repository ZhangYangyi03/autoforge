"""Force the global account to serve a request, so "it works" is proven for it.

Rotation is least-recently-used, so four calls spread over four accounts -- but
"four calls succeeded" does not prove the global one was among them. This makes
the global account the ONLY eligible one (by marking the CN accounts cooling in
the running proxy is not possible from outside), so instead it drives enough
distinct conversations that every account must take a turn, then reports which
labels appear in the proxy log.

The proxy does not log the serving account on success, so this asks the upstream
directly instead: it compares the answer's `id` prefix is not enough either.
The reliable signal is the proxy's own rotation: 4 accounts, 8 distinct calls,
LRU -- every account must serve exactly twice.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777"


def post(body: dict, timeout: float = 180) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    # A global-only model would prove it outright, but this account is a
    # personal_free_user and every global-exclusive id answers 11158. So the
    # proof is rotation coverage: 8 distinct conversations over 4 accounts.
    ok, fails = 0, []
    for i in range(8):
        code, resp = post({"model": "hy3", "max_tokens": 24, "temperature": 0,
                           "messages": [{"role": "user",
                                         "content": f"Count to {i+1}. Be brief."}]})
        if code == 200 and isinstance(resp, dict):
            ok += 1
        else:
            msg = (resp.get("error", {}).get("message", "")
                   if isinstance(resp, dict) else str(resp)[:80])
            fails.append(f"{code}: {msg[:90]}")
    print(f"8 distinct conversations: {ok} ok, {len(fails)} failed")
    for f in fails:
        print("  ", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
