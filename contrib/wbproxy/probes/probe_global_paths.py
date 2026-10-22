"""The global account reads its catalog fine but every chat 400s.

Catalog 200 + chat 400 means the credential is accepted and the endpoint is
reachable, so the difference is in the request path or the body shape that host
expects. This walks candidates on the global host: paths first (does
/v2/chat/completions even exist there?), then body variants.

dsh-xdpool's `chatBase()` returns the same host it uses for the catalog, and its
global branch reads the catalog from /v3/config -- a different version prefix
from /v2/chat/completions. That mismatch is the first thing to test.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402


def post(url: str, h: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(220).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(220).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    accounts = load_accounts()
    acct = next((a for a in accounts if a.region == "global"), None)
    if acct is None:
        print("no global account")
        return 1
    token = acct.usable_token()
    if not token:
        print("no token")
        return 1
    h = acct.chat_headers()
    base = acct.chat_base
    print(f"account={acct.label} base={base}\n")

    simple = {"model": "hy3", "stream": True,
              "messages": [{"role": "user", "content": "say OK"}]}

    print("--- paths on the global host ---")
    for path in ("/v2/chat/completions", "/v3/chat/completions",
                 "/inference/v1/chat/completions", "/v1/chat/completions",
                 "/v4/chat/completions", "/v2/completions"):
        code, raw = post(f"{base}{path}", h, simple)
        note = ""
        if code != 200:
            try:
                o = json.loads(raw)
                note = f"code={o.get('code')} msg={o.get('msg') or o.get('error_msg')}"
            except Exception:
                note = raw[:60].replace("\n", " ")
        print(f"  [{code:3d}] {path:34s} {note}")

    print("\n--- bodies against /v2/chat/completions ---")
    variants = {
        "as-is (stream)": simple,
        "stream=false": {**simple, "stream": False},
        "no stream key": {"model": "hy3",
                          "messages": [{"role": "user", "content": "say OK"}]},
        "with max_tokens": {**simple, "max_tokens": 512},
        "with temperature": {**simple, "temperature": 0.0},
        "no system, longer text": {
            "model": "hy3", "stream": True,
            "messages": [{"role": "user",
                          "content": "Please reply with the single word OK."}]},
    }
    for label, body in variants.items():
        code, raw = post(f"{base}/v2/chat/completions", h, body)
        note = ""
        if code != 200:
            try:
                o = json.loads(raw)
                note = f"code={o.get('code')}"
            except Exception:
                note = raw[:60].replace("\n", " ")
        print(f"  [{code:3d}] {label:26s} {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
