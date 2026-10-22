"""Which global-catalog models can this account actually use?

A request that fails with 11158 ("model is not allowed for user type
[personal_free_user]") is classified by xdpool as `client` -- a defect in the
request, not something worth failing over for. So a user picking from the
21-model global catalog will hit a hard failure on every model their plan does
not include, and the pool will not route around it.

This enumerates the global catalog against the real account and splits it into
usable and refused, so the catalog-vs-plan gap is a list rather than a guess.
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
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:70]}"


def main() -> int:
    accts = load_accounts(verbose=False)
    gl = next((a for a in accts if a.region == "global"), None)
    if gl is None or not gl.usable_token():
        print("no usable global account")
        return 1
    url = f"{gl.chat_base}/v2/chat/completions"

    # The two gates: three conversation headers, and a leading system message.
    conv = str(uuid.uuid4())
    h = dict(gl.chat_headers())
    h["X-Conversation-ID"] = conv
    h["X-Conversation-Request-ID"] = str(uuid.uuid4())
    h["X-Conversation-Message-ID"] = str(uuid.uuid4())

    models = ["default-model", "fast-model", "balanced-model", "primary-model",
              "deep-model", "hy3", "hy4-preview", "deepseek-v4.1-flash",
              "deepseek-v4.1-flash-sg", "kimi-k2.8-preview", "kimi-k2.6",
              "kimi-k3", "glm-5.3", "glm-5.3-flash", "glm-5.2",
              "minimax-m3.1-flash", "gpt-5.6-terra", "gpt-5.6-luna",
              "gpt-5.5", "gpt-5.4", "gemini-3.5-flash"]

    print(f"{'model':26s} {'result':>7s}  detail")
    print("-" * 88)
    usable, refused, other = [], [], []
    for mid in models:
        body = {"model": mid, "stream": True,
                "messages": [{"role": "system", "content": "You are helpful."},
                             {"role": "user", "content": "say OK"}]}
        hh = dict(h)
        hh["X-Conversation-Message-ID"] = str(uuid.uuid4())
        code, raw = post(url, hh, body)
        if code == 200:
            usable.append(mid)
            print(f"{mid:26s} {'200':>7s}  usable")
        else:
            detail = ""
            try:
                o = json.loads(raw)
                detail = f"code={o.get('code')} {o.get('msg')}"
            except Exception:
                detail = raw[:60]
            if "11158" in raw or "not allowed" in raw:
                refused.append(mid)
            else:
                other.append((mid, code, detail))
            print(f"{mid:26s} {str(code):>7s}  {detail[:60]}")

    print()
    print(f"usable  : {len(usable):2d}  {', '.join(usable)}")
    print(f"refused : {len(refused):2d}  {', '.join(refused)}")
    if other:
        print(f"other   : {len(other):2d}  " +
              "; ".join(f"{m}({c})" for m, c, _ in other))
    print()
    print("xdpool lists all of these in the model picker and classifies a "
          "refusal as `client` (HTTP 400), which does not fail over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
