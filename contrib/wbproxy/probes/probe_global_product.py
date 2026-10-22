"""Is there any request that the global account can make?

The provisioning comparison points at the account, not the request: the global
catalog omits every protocol feature the CLI product needs (`RequestBodyGzip`,
`DeferToolLoading`, `EnableAutoModelTiers`, `CustomModelsJSON`) and reports
`mergeStrategy: None` where the CN account says `merge`.

One last sweep before writing that down, because "unprovisioned" is a strong
claim: try the endpoints that do not depend on those features -- the ones the
desktop client uses for non-chat purposes -- to confirm the credential itself is
healthy and only the chat product is missing.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402


def call(method: str, url: str, h: dict, body: dict | None = None
         ) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:70]}"


def note(code: int, raw: str) -> str:
    if code == 200:
        try:
            o = json.loads(raw)
            d = o.get("data") if isinstance(o, dict) else None
            if isinstance(d, dict):
                return f"ok keys={list(d)[:5]}"
            return f"ok {str(d)[:50]}"
        except Exception:
            return "ok"
    try:
        o = json.loads(raw)
        return f"code={o.get('code')} msg={o.get('msg') or o.get('error_msg')}"
    except Exception:
        return raw[:60].replace("\n", " ")


def main() -> int:
    accts = load_accounts(verbose=False)
    gl = next((a for a in accts if a.region == "global"), None)
    cn = next((a for a in accts if a.region == "cn"), None)
    if gl is None or not gl.usable_token():
        print("no global account")
        return 1
    h = gl.chat_headers()
    base = gl.chat_base
    print(f"global account {gl.label}\n")

    cases = [
        ("GET", "/v3/config", "product config (known good)"),
        ("GET", "/v2/enterprises/personal/models", "CN-style catalog path"),
        ("GET", "/v2/models", "short catalog path"),
        ("GET", "/v2/user/profile", "profile"),
        ("GET", "/v2/userinfo", "userinfo"),
        ("GET", "/v2/enterprises/personal/info", "enterprise info"),
        ("GET", "/v2/balance", "balance"),
        ("GET", "/v2/usage", "usage"),
        ("POST", "/v2/plugin/auth/token/refresh", "refresh (no body)"),
    ]
    for method, path, label in cases:
        code, raw = call(method, f"{base}{path}", h)
        print(f"  [{code:3d}] {label:38s} {note(code, raw)}")

    print("\n--- comparison: the same probes on the CN account ---")
    if cn and cn.usable_token():
        hc = cn.chat_headers()
        for method, path, label in cases[:4]:
            code, raw = call(method, f"{cn.chat_base}{path}", hc)
            print(f"  [{code:3d}] {label:38s} {note(code, raw)}")

    print("\n--- does the global chat endpoint name a reason? ---")
    code, raw = call("POST", f"{base}/v2/chat/completions", h,
                     {"model": "hy3", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]})
    print(f"  [{code:3d}] full body:")
    try:
        print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False)[:900])
    except Exception:
        print(raw[:600])

    return 0


if __name__ == "__main__":
    sys.exit(main())
