"""On the global host, stream=false reaches business logic; stream=true is 11128.

That asymmetry rules out "wrong path" and "bad token": 11101 (non-stream not
supported) proves the request is being parsed and routed. So something the
gateway checks on the streaming path is unhappy -- most likely client identity,
since dsh-xdpool notes that the global catalog is served only to the DESKTOP
channel, and the identity headers are what declare a channel.

Sweep the identity headers and the two remaining auth-ish fields.
"""
import itertools
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402


def post(url: str, h: dict, body: dict) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read(400).decode("utf-8", "replace") if e.fp else ""
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw[:200]
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    accounts = load_accounts()
    acct = next((a for a in accounts if a.region == "global"), None)
    if acct is None or not acct.usable_token():
        print("no global account")
        return 1
    base = acct.chat_base
    url = f"{base}/v2/chat/completions"
    body = {"model": "hy3", "stream": True,
            "messages": [{"role": "user", "content": "say OK"}]}
    print(f"account={acct.label}  url={url}\n")

    def show(label: str, headers: dict) -> None:
        code, raw = post(url, headers, body)
        if isinstance(raw, dict):
            note = f"code={raw.get('code')} msg={raw.get('msg')}"
        else:
            note = str(raw)[:70].replace("\n", " ")
        mark = "OK " if code == 200 else "ERR"
        print(f"[{mark}] {code:3d}  {label:42s} {note}")

    # 1. baseline
    show("baseline (current headers)", acct.chat_headers())

    # 2. identity header variants
    for name, typ, ver in [("WorkBuddy", "WorkBuddy", "5.5.6"),
                           ("CodeBuddy", "CodeBuddy", "2.63.2"),
                           ("CLI", "CLI", "2.63.2"),
                           ("WorkBuddy", "WorkBuddy", "2.63.2")]:
        h = acct.chat_headers()
        h.update({"X-IDE-Name": name, "X-IDE-Type": typ, "X-IDE-Version": ver})
        show(f"identity {name}/{typ}/{ver}", h)

    # 3. drop the identity headers entirely
    h = acct.chat_headers()
    for k in ("X-IDE-Name", "X-IDE-Type", "X-IDE-Version"):
        h.pop(k, None)
    show("no identity headers", h)

    # 4. user-agent variants (the CLI UA is the CN client's)
    for ua in ("Mozilla/5.0", "WorkBuddy/5.5.6", "CLI/2.63.2 CodeBuddy/2.63.2"):
        h = acct.chat_headers()
        h["User-Agent"] = ua
        show(f"UA {ua}", h)

    # 5. product / domain variants
    for prod in ("SaaS", "Desktop", "CLI"):
        h = acct.chat_headers()
        h["X-Product"] = prod
        show(f"X-Product {prod}", h)

    # 6. drop X-Domain (the .ai domain may only be valid on the desktop client)
    h = acct.chat_headers()
    h.pop("X-Domain", None)
    h["X-No-Department-Info"] = "1"
    show("no X-Domain", h)

    # 7. does a model with an explicit global-only id change anything?
    for mid in ("gpt-5.6-terra", "deepseek-v4.1-flash-sg"):
        code, raw = post(url, acct.chat_headers(),
                         {**body, "model": mid})
        note = f"code={raw.get('code')}" if isinstance(raw, dict) else str(raw)[:60]
        mark = "OK " if code == 200 else "ERR"
        print(f"[{mark}] {code:3d}  {'model ' + mid:42s} {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
