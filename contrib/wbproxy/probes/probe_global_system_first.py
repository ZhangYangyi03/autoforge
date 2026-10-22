"""Second gate: the global gateway wants a system message first.

With the three conversation headers present the error changes from
"request illegal" to "first message is not system prompt" -- so channel
verification passed and a different rule is now in the way. xdpool 2.4.1
describes exactly this: the global gateway rejects a body whose first message is
`user`, and its fix is to prepend a placeholder system message while leaving the
user's own text untouched.

This confirms that, then checks whether the account finally chats.
"""
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402


def post(url: str, h: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:90]}"


def note(code: int, raw: str) -> str:
    if code == 200:
        # Show the first content delta, so "it answered" is visible.
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    f = json.loads(line[5:].strip())
                    d = (f.get("choices") or [{}])[0].get("delta") or {}
                    if d.get("content"):
                        return f"OK  content={d['content']!r}"
                except Exception:
                    pass
        return "OK  (stream opened, no text yet in first 400 bytes)"
    try:
        o = json.loads(raw)
        return f"code={o.get('code')} msg={o.get('msg')}"
    except Exception:
        return raw[:70].replace("\n", " ")


def main() -> int:
    accts = load_accounts(verbose=False)
    gl = next((a for a in accts if a.region == "global"), None)
    if gl is None or not gl.usable_token():
        print("no usable global account")
        return 1
    url = f"{gl.chat_base}/v2/chat/completions"

    conv = str(uuid.uuid4())
    h = dict(gl.chat_headers())
    h["X-Conversation-ID"] = conv
    h["X-Conversation-Request-ID"] = str(uuid.uuid4())
    h["X-Conversation-Message-ID"] = str(uuid.uuid4())

    print(f"{'variant':52s} result")
    print("-" * 108)

    user_only = {"model": "hy3", "stream": True,
                 "messages": [{"role": "user", "content": "say OK"}]}
    code, raw = post(url, h, user_only)
    print(f"{'user first (no system)':52s} {note(code, raw)}")

    with_sys = {"model": "hy3", "stream": True,
                "messages": [{"role": "system", "content": "You are a helpful assistant."},
                             {"role": "user", "content": "say OK"}]}
    code, raw = post(url, h, with_sys)
    print(f"{'placeholder system prepended':52s} {note(code, raw)}")

    # Does a system prompt without the two fingerprinted sentences still pass?
    sys_gui = {"model": "hy3", "stream": True,
               "messages": [{"role": "system",
                             "content": "You are an AI agent powered by DeepSeek Harness."},
                            {"role": "user", "content": "say OK"}]}
    code, raw = post(url, h, sys_gui)
    print(f"{'system with the fingerprinted sentence':52s} {note(code, raw)}")

    # Which models answer, now that the two gates are satisfied?
    print()
    print("--- model sweep, both gates satisfied ---")
    for mid in ("hy3", "deepseek-v4.1-flash", "deepseek-v4.1-flash-sg",
                "gpt-5.6-terra", "gemini-3.5-flash"):
        body = {"model": mid, "stream": True,
                "messages": [{"role": "system", "content": "You are helpful."},
                             {"role": "user", "content": "say OK"}]}
        hh = dict(h)
        hh["X-Conversation-Message-ID"] = str(uuid.uuid4())
        code, raw = post(url, hh, body)
        print(f"  {mid:26s} {note(code, raw)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
