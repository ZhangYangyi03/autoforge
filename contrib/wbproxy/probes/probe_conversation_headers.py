"""Test the conversation-identity headers against the global gateway.

xdpool 2.4.2's source says, in a comment describing a verified experiment:

    `www.workbuddy.ai` turned "channel verification" from a soft check into a
    hard one. A POST /v2/chat/completions missing ANY of X-Conversation-ID,
    X-Conversation-Request-ID or X-Conversation-Message-ID is rejected with
    business code 11128. Verified by elimination against real credentials:
    delete any one of the three and the call 400s; send all three and it 200s
    with normal SSE. The values are NOT validated -- plain non-UUID strings
    pass -- so the gateway is checking for presence and consistency.

Earlier probing of this account tried 15 header combinations and never tried
these three, so "the global account cannot chat" was a conclusion drawn from an
incomplete sweep. This adds the missing dimension.
"""
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

BODY = {"model": "hy3", "stream": True,
        "messages": [{"role": "user", "content": "say OK"}]}


def post(url: str, h: dict, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body or BODY).encode()
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(300).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:90]}"


def note(code: int, raw: str) -> str:
    if code == 200:
        return "OK  <-- " + raw[:110].replace("\n", " ")
    try:
        o = json.loads(raw)
        return f"code={o.get('code')} msg={o.get('msg') or o.get('error_msg')}"
    except Exception:
        return raw[:70].replace("\n", " ")


def main() -> int:
    accts = load_accounts(verbose=False)
    gl = next((a for a in accts if a.region == "global"), None)
    cn = next((a for a in accts if a.region == "cn"), None)
    if gl is None or not gl.usable_token():
        print("no usable global account")
        return 1

    url = f"{gl.chat_base}/v2/chat/completions"
    base = gl.chat_headers()

    # One logical conversation: the three ids must stay stable across calls.
    conv = str(uuid.uuid4())

    def ident(extra: dict | None = None) -> dict:
        h = dict(base)
        h["X-Conversation-ID"] = conv
        h["X-Conversation-Request-ID"] = str(uuid.uuid4())
        h["X-Conversation-Message-ID"] = str(uuid.uuid4())
        if extra:
            h.update(extra)
        return h

    print("--- global account, isolation of the three headers ---")
    print(f"{'variant':40s} result")
    print("-" * 96)

    code, raw = post(url, base)
    print(f"{'no conversation headers (baseline)':40s} {note(code, raw)}")

    h = ident()
    for drop in ("X-Conversation-ID", "X-Conversation-Request-ID",
                 "X-Conversation-Message-ID"):
        hh = dict(h)
        hh.pop(drop, None)
        code, raw = post(url, hh)
        print(f"{'minus ' + drop:40s} {note(code, raw)}")

    code, raw = post(url, h)
    print(f"{'all three present':40s} {note(code, raw)}")

    # Presence-only claim: non-UUID values should also pass.
    hh = dict(base)
    hh["X-Conversation-ID"] = "conv-test"
    hh["X-Conversation-Request-ID"] = "req-test"
    hh["X-Conversation-Message-ID"] = "msg-test"
    code, raw = post(url, hh)
    print(f"{'all three, plain non-UUID strings':40s} {note(code, raw)}")

    # Does the domestic gateway need them too?
    if cn and cn.usable_token():
        print("\n--- control: CN account ---")
        cnurl = f"{cn.chat_base}/v2/chat/completions"
        code, raw = post(cnurl, cn.chat_headers())
        print(f"{'CN without conversation headers':40s} {note(code, raw)}")
        hc = dict(cn.chat_headers())
        hc.update({"X-Conversation-ID": conv,
                   "X-Conversation-Request-ID": str(uuid.uuid4()),
                   "X-Conversation-Message-ID": str(uuid.uuid4())})
        code, raw = post(cnurl, hc)
        print(f"{'CN with conversation headers':40s} {note(code, raw)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
