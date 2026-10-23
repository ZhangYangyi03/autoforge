"""Is `reasoning_effort: "off"` actually accepted by the domestic gateway?

xdpool pins `thinkingLevelMap.off` to `null` (lib/index.js:11891) on the stated
grounds that neither gateway accepts a literal `"off"` -- measured, per its
comment at L1224-1232, with the international GPT family answering 11133. But
the CN catalog publishes `canDisableThinking: true` for
`deepseek-v4.1-flash` and `step-5-preview`, which is the gateway saying the
opposite.

One of the two is wrong, and which one decides whether the picker should offer
"off". This asks the gateway directly, for both models, on the domestic host.
"""
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

MODELS = ["deepseek-v4.1-flash", "step-5-preview", "hy3"]
VARIANTS = [
    ("off (literal)", "off"),
    ("none (literal)", "none"),
    ("omitted", None),
    ("low", "low"),
    ("minimal", "minimal"),
]


def post(url: str, h: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:80]}"


def verdict(code: int, raw: str) -> str:
    if code == 200:
        return "ACCEPTED"
    try:
        o = json.loads(raw)
        return f"refused code={o.get('code')} {str(o.get('msg'))[:40]}"
    except Exception:
        return f"refused {code} {raw[:50]}"


def main() -> int:
    accts = load_accounts(verbose=False)
    acct = next((a for a in accts if a.region == "cn"), None)
    if acct is None or not acct.usable_token():
        print("no usable CN account")
        return 1
    url = f"{acct.chat_base}/v2/chat/completions"
    base = dict(acct.chat_headers())
    conv = str(uuid.uuid4())
    base["X-Conversation-ID"] = conv
    base["X-Conversation-Request-ID"] = str(uuid.uuid4())

    print(f"CN gateway, account {acct.label}\n")
    print(f"{'model':24s} {'variant':16s} result")
    print("-" * 84)
    for mid in MODELS:
        for label, effort in VARIANTS:
            body = {"model": mid, "stream": True, "max_tokens": 32,
                    "messages": [{"role": "user", "content": "say OK"}]}
            if effort is not None:
                body["reasoning_effort"] = effort
            h = dict(base)
            h["X-Conversation-Message-ID"] = str(uuid.uuid4())
            code, raw = post(url, h, body)
            print(f"{mid:24s} {label:16s} {verdict(code, raw)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
