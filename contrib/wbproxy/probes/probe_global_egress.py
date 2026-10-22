"""Is the global 11128 an IP/geo decision rather than a request problem?

Every request shape has now been eliminated. What has not been varied is the
network path: this machine reaches the global host from a CN address, while the
account was registered elsewhere. A geo check at the edge would produce exactly
this signature -- the catalog allowed, chat refused, and connections dropped
outright for some variants.

Tests, in order of how much they would explain:
  1. the same request through the local HTTP proxy (a different egress)
  2. the same request with the CN account, against the CN host, as a control
  3. whether the global catalog reports anything about the account's plan or
     region that would predict a refusal
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

PROXY = "http://127.0.0.1:7890"
BODY = {"model": "hy3", "stream": True,
        "messages": [{"role": "user", "content": "say OK"}]}


def post(url: str, headers: dict, via_proxy: bool = False,
         body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body or BODY).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = None
    if via_proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    try:
        with (opener.open(req, timeout=60) if opener
              else urllib.request.urlopen(req, timeout=60)) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:100]}"


def note(code: int, raw: str) -> str:
    if code == 200:
        return "OK"
    try:
        o = json.loads(raw)
        return f"code={o.get('code')} msg={o.get('msg') or o.get('error_msg')}"
    except Exception:
        return raw[:70].replace("\n", " ")


def main() -> int:
    accts = load_accounts(verbose=False)
    gl = next((a for a in accts if a.region == "global"), None)
    cn = next((a for a in accts if a.region == "cn"), None)

    if gl and gl.usable_token():
        print("--- global account, direct vs via local proxy ---")
        url = f"{gl.chat_base}/v2/chat/completions"
        for label, via in (("direct", False), ("via 7890", True)):
            code, raw = post(url, gl.chat_headers(), via_proxy=via)
            print(f"  [{code:3d}] {label:14s} {note(code, raw)}")

        print("\n--- global catalog, direct vs via proxy ---")
        for label, via in (("direct", False), ("via 7890", True)):
            req = urllib.request.Request(
                f"{gl.chat_base}/v3/config", headers=gl.chat_headers())
            try:
                if via:
                    op = urllib.request.build_opener(urllib.request.ProxyHandler(
                        {"http": PROXY, "https": PROXY}))
                    with op.open(req, timeout=40) as r:
                        doc = json.loads(r.read().decode())
                else:
                    with urllib.request.urlopen(req, timeout=40) as r:
                        doc = json.loads(r.read().decode())
                data = doc.get("data") or {}
                feats = data.get("productFeatures") or {}
                print(f"  [{label:14s}] ok  enterpriseId={data.get('enterpriseId')!r} "
                      f"models={len(data.get('models') or [])}")
                print(f"                   features: "
                      f"{', '.join(k for k, v in feats.items() if v)}")
            except Exception as e:                      # noqa: BLE001
                print(f"  [{label:14s}] FAIL {type(e).__name__}: {str(e)[:70]}")

    print()
    if cn and cn.usable_token():
        print("--- control: CN account on the CN host ---")
        code, raw = post(f"{cn.chat_base}/v2/chat/completions", cn.chat_headers())
        print(f"  [{code:3d}] direct         {note(code, raw)}")
        code, raw = post(f"{cn.chat_base}/v2/chat/completions", cn.chat_headers(),
                         via_proxy=True)
        print(f"  [{code:3d}] via 7890       {note(code, raw)}")

        print("\n--- control: CN account against the GLOBAL host ---")
        if gl:
            code, raw = post(f"{gl.chat_base}/v2/chat/completions", cn.chat_headers())
            print(f"  [{code:3d}] CN token -> global host   {note(code, raw)}")

    print()
    if gl and cn and gl.usable_token() and cn.usable_token():
        print("--- cross: GLOBAL token against the CN host ---")
        code, raw = post(f"{cn.chat_base}/v2/chat/completions", gl.chat_headers())
        print(f"  [{code:3d}] global token -> CN host    {note(code, raw)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
